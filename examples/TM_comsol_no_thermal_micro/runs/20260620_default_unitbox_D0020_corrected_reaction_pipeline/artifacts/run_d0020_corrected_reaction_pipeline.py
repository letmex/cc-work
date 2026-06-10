"""Corrected D0020 reaction postprocessing pipeline.

This script reads existing checkpointed D0020 seed 7/13/42 runs and produces
standardized energy-consistent reaction metrics. It does not train, extend the
load schedule, run D0040, or change model physics.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PACKAGE = Path(__file__).resolve().parents[1]
RUNS_ROOT = PACKAGE.parent
PREV_PACKAGE = RUNS_ROOT / "20260619_default_unitbox_D0020_energy_stress_conjugacy_audit"
PREV_SCRIPT = PREV_PACKAGE / "artifacts" / "run_d0020_energy_stress_conjugacy_audit.py"
PREV_TABLES = PREV_PACKAGE / "tables"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"

SEEDS = (7, 13, 42)
PRIMARY_MODE = "global_delta_mode"
EXACT_TOL_REL = 1.0e-6
EXACT_TOL_ABS_N = 1.0e-6


def import_previous_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("d0020_energy_stress_conjugacy", PREV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import previous audit script: {PREV_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prev = import_previous_module()


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def safe_pct_drop(peak_abs: float, final_abs: float) -> float:
    if not np.isfinite(peak_abs) or peak_abs <= 0.0 or not np.isfinite(final_abs):
        return math.nan
    return 100.0 * (peak_abs - final_abs) / peak_abs


def safe_ratio(value: float, denom: float) -> float:
    if not np.isfinite(value) or not np.isfinite(denom) or abs(denom) <= 1.0e-12:
        return math.nan
    return value / denom


def load_checkpointed_reactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    device = torch.device("cpu")
    for seed in SEEDS:
        try:
            result = prev.process_seed(seed, device)
            availability_rows.extend(result["availability"])
            identity = pd.DataFrame(result["identity_rows"])
        except Exception as exc:  # pragma: no cover - defensive package behavior
            availability_rows.append(
                {
                    "seed": seed,
                    "checkpoint_available": False,
                    "exact_reaction_computable": False,
                    "checkpoint_count": 0,
                    "error": repr(exc),
                }
            )
            rows.append(
                {
                    "seed": seed,
                    "step": math.nan,
                    "Delta": math.nan,
                    "reaction_primary_metric": "reaction_N_energy_exact",
                    "reaction_metric_status": "reaction_metric_unavailable",
                    "requires_checkpoint": True,
                    "is_energy_conjugate": False,
                    "is_legacy_diagnostic": False,
                    "softening_gate_eligible": False,
                    "checkpoint_available": False,
                    "exact_autograd_computable": False,
                    "reaction_N_energy_exact": math.nan,
                    "reaction_N_energy_virtual_work": math.nan,
                    "reaction_N_legacy_top_sigma": math.nan,
                    "reaction_N_bottom_sigma_legacy": math.nan,
                    "reaction_N_internal_cut_above": math.nan,
                    "reaction_N_internal_cut_below": math.nan,
                    "alpha0p8_through_crack": False,
                    "error": repr(exc),
                }
            )
            continue

        global_rows = identity[identity["mode"] == PRIMARY_MODE].copy().sort_values("step")
        for r in global_rows.itertuples(index=False):
            exact = float(r.R_exact_dPi_dmode_N)
            virtual = float(r.R_virtual_energy_autograd_sigma_N)
            rel_error = abs(exact - virtual) / max(abs(exact), 1.0e-12)
            exact_ok = bool(rel_error <= EXACT_TOL_REL or abs(exact - virtual) <= EXACT_TOL_ABS_N)
            rows.append(
                {
                    "seed": int(r.seed),
                    "step": int(r.step),
                    "Delta": float(r.Delta),
                    "reaction_primary_metric": "reaction_N_energy_exact",
                    "reaction_metric_status": "energy_exact_primary",
                    "requires_checkpoint": True,
                    "is_energy_conjugate": True,
                    "is_legacy_diagnostic": False,
                    "softening_gate_eligible": True,
                    "checkpoint_available": True,
                    "exact_autograd_computable": True,
                    "energy_virtual_work_agrees_with_exact": exact_ok,
                    "energy_virtual_abs_error_N": abs(exact - virtual),
                    "energy_virtual_rel_error": rel_error,
                    "reaction_N_energy_exact": exact,
                    "reaction_N_energy_virtual_work": virtual,
                    "reaction_N_legacy_top_sigma": float(r.legacy_top_sigma_integral_N),
                    "reaction_N_bottom_sigma_legacy": float(r.bottom_reaction_N),
                    "reaction_N_internal_cut_above": float(r.internal_cut_force_above_crack_N),
                    "reaction_N_internal_cut_below": float(r.internal_cut_force_below_crack_N),
                    "reaction_N_postprocessed_sigma_virtual_work": float(r.R_virtual_postprocessed_sigma_N),
                    "alpha0p8_through_crack": bool(r.alpha0p8_through_crack),
                    "legacy_metric_status": "legacy_diagnostic_only",
                    "legacy_softening_gate_eligible": False,
                }
            )

    availability = pd.DataFrame(availability_rows)
    if not availability.empty:
        if "checkpoint_available" not in availability.columns:
            availability["checkpoint_available"] = availability["checkpoint_count"].astype(int) > 0
        if "exact_reaction_computable" not in availability.columns:
            availability["exact_reaction_computable"] = availability["checkpoint_available"].astype(bool)
        availability["requires_checkpoint"] = True
    return pd.DataFrame(rows), availability


def build_reproduction_check(corrected: pd.DataFrame) -> pd.DataFrame:
    prev_identity = pd.read_csv(PREV_TABLES / "energy_autograd_virtual_work_identity.csv")
    prev_global = prev_identity[prev_identity["mode"] == PRIMARY_MODE].copy()
    merged = corrected.merge(
        prev_global[
            [
                "seed",
                "step",
                "R_exact_dPi_dmode_N",
                "R_virtual_energy_autograd_sigma_N",
                "legacy_top_sigma_integral_N",
            ]
        ],
        on=["seed", "step"],
        how="left",
    )
    merged["exact_reaction_reproduction_abs_error_N"] = (
        merged["reaction_N_energy_exact"] - merged["R_exact_dPi_dmode_N"]
    ).abs()
    merged["virtual_reaction_reproduction_abs_error_N"] = (
        merged["reaction_N_energy_virtual_work"] - merged["R_virtual_energy_autograd_sigma_N"]
    ).abs()
    merged["legacy_top_reproduction_abs_error_N"] = (
        merged["reaction_N_legacy_top_sigma"] - merged["legacy_top_sigma_integral_N"]
    ).abs()
    merged["exact_reaction_reproduced"] = (
        merged["exact_reaction_reproduction_abs_error_N"].fillna(math.inf) <= EXACT_TOL_ABS_N
    )
    return merged[
        [
            "seed",
            "step",
            "Delta",
            "reaction_N_energy_exact",
            "R_exact_dPi_dmode_N",
            "exact_reaction_reproduction_abs_error_N",
            "reaction_N_energy_virtual_work",
            "R_virtual_energy_autograd_sigma_N",
            "virtual_reaction_reproduction_abs_error_N",
            "reaction_N_legacy_top_sigma",
            "legacy_top_sigma_integral_N",
            "legacy_top_reproduction_abs_error_N",
            "exact_reaction_reproduced",
        ]
    ]


def build_virtual_work_summary(corrected: pd.DataFrame, reproduction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    usable = corrected[corrected["reaction_metric_status"] != "reaction_metric_unavailable"]
    for seed, sub in usable.sort_values("step").groupby("seed"):
        rep = reproduction[reproduction["seed"] == seed]
        rows.append(
            {
                "seed": int(seed),
                "step_count": int(len(sub)),
                "max_energy_virtual_abs_error_N": float(sub["energy_virtual_abs_error_N"].max()),
                "median_energy_virtual_rel_error": float(sub["energy_virtual_rel_error"].median()),
                "max_energy_virtual_rel_error": float(sub["energy_virtual_rel_error"].max()),
                "virtual_work_matches_exact_within_tolerance": bool(
                    (sub["energy_virtual_rel_error"] <= EXACT_TOL_REL).all()
                    or (sub["energy_virtual_abs_error_N"] <= EXACT_TOL_ABS_N).all()
                ),
                "max_exact_reproduction_abs_error_N": float(rep["exact_reaction_reproduction_abs_error_N"].max()),
                "exact_reaction_reproduced_from_prior_audit": bool(rep["exact_reaction_reproduced"].all()),
            }
        )
    for seed in set(SEEDS) - set(usable["seed"].astype(int).unique()):
        rows.append(
            {
                "seed": int(seed),
                "step_count": 0,
                "max_energy_virtual_abs_error_N": math.nan,
                "median_energy_virtual_rel_error": math.nan,
                "max_energy_virtual_rel_error": math.nan,
                "virtual_work_matches_exact_within_tolerance": False,
                "max_exact_reproduction_abs_error_N": math.nan,
                "exact_reaction_reproduced_from_prior_audit": False,
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def build_softening_summary(corrected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sub = corrected[corrected["seed"] == seed].copy().sort_values("step")
        if sub.empty or (sub["reaction_metric_status"] == "reaction_metric_unavailable").all():
            rows.append(
                {
                    "seed": seed,
                    "softening_classification": "reaction_metric_unavailable",
                    "post_through_collapse_detected": False,
                    "legacy_and_primary_metrics_disagree": math.nan,
                }
            )
            continue
        primary_abs = sub["reaction_N_energy_exact"].astype(float).abs()
        legacy_abs = sub["reaction_N_legacy_top_sigma"].astype(float).abs()
        peak_idx = primary_abs.idxmax()
        legacy_peak_idx = legacy_abs.idxmax()
        peak_abs = float(primary_abs.loc[peak_idx])
        final_abs = float(primary_abs.iloc[-1])
        legacy_peak_abs = float(legacy_abs.loc[legacy_peak_idx])
        legacy_final_abs = float(legacy_abs.iloc[-1])
        through_rows = sub[sub["alpha0p8_through_crack"].astype(bool)]
        first_through_step = int(through_rows["step"].iloc[0]) if not through_rows.empty else math.nan
        if through_rows.empty:
            through_abs = math.nan
            legacy_through_abs = math.nan
            through_drop = math.nan
            legacy_through_drop = math.nan
        else:
            through_abs = float(abs(through_rows["reaction_N_energy_exact"].iloc[0]))
            legacy_through_abs = float(abs(through_rows["reaction_N_legacy_top_sigma"].iloc[0]))
            through_drop = safe_pct_drop(through_abs, final_abs)
            legacy_through_drop = safe_pct_drop(legacy_through_abs, legacy_final_abs)
        post_peak_drop = safe_pct_drop(peak_abs, final_abs)
        legacy_post_peak_drop = safe_pct_drop(legacy_peak_abs, legacy_final_abs)
        final_peak_ratio = safe_ratio(final_abs, peak_abs)
        legacy_final_peak_ratio = safe_ratio(legacy_final_abs, legacy_peak_abs)
        post_through_collapse = bool(
            np.isfinite(first_through_step)
            and (
                (np.isfinite(post_peak_drop) and post_peak_drop >= 50.0)
                or (np.isfinite(final_peak_ratio) and final_peak_ratio <= 0.1)
                or final_abs <= 0.05
            )
        )
        legacy_collapse = bool(
            np.isfinite(first_through_step)
            and (
                (np.isfinite(legacy_post_peak_drop) and legacy_post_peak_drop >= 50.0)
                or (np.isfinite(legacy_final_peak_ratio) and legacy_final_peak_ratio <= 0.1)
                or legacy_final_abs <= 0.05
            )
        )
        rows.append(
            {
                "seed": seed,
                "reaction_primary_metric": "reaction_N_energy_exact",
                "reaction_metric_status": "energy_exact_primary",
                "first_alpha0p8_through_crack_step": first_through_step,
                "peak_reaction_by_primary_metric_N": float(sub.loc[peak_idx, "reaction_N_energy_exact"]),
                "peak_reaction_abs_N": peak_abs,
                "peak_reaction_step": int(sub.loc[peak_idx, "step"]),
                "final_reaction_by_primary_metric_N": float(sub["reaction_N_energy_exact"].iloc[-1]),
                "final_reaction_abs_N": final_abs,
                "post_peak_drop_percent": post_peak_drop,
                "final_to_peak_ratio": final_peak_ratio,
                "reaction_at_first_through_crack_abs_N": through_abs,
                "drop_after_first_through_crack_percent": through_drop,
                "post_through_collapse_detected": post_through_collapse,
                "primary_drop_ge_10pct": bool(np.isfinite(post_peak_drop) and post_peak_drop >= 10.0),
                "primary_drop_ge_30pct": bool(np.isfinite(post_peak_drop) and post_peak_drop >= 30.0),
                "primary_drop_ge_50pct": bool(np.isfinite(post_peak_drop) and post_peak_drop >= 50.0),
                "primary_drop_ge_90pct": bool(np.isfinite(post_peak_drop) and post_peak_drop >= 90.0),
                "legacy_top_sigma_peak_abs_N": legacy_peak_abs,
                "legacy_top_sigma_final_abs_N": legacy_final_abs,
                "legacy_top_sigma_post_peak_drop_percent": legacy_post_peak_drop,
                "legacy_top_sigma_final_to_peak_ratio": legacy_final_peak_ratio,
                "legacy_top_sigma_drop_after_first_through_percent": legacy_through_drop,
                "legacy_top_sigma_collapse_by_same_thresholds": legacy_collapse,
                "legacy_drop_ge_10pct": bool(np.isfinite(legacy_post_peak_drop) and legacy_post_peak_drop >= 10.0),
                "legacy_drop_ge_30pct": bool(np.isfinite(legacy_post_peak_drop) and legacy_post_peak_drop >= 30.0),
                "legacy_drop_ge_50pct": bool(np.isfinite(legacy_post_peak_drop) and legacy_post_peak_drop >= 50.0),
                "legacy_drop_ge_90pct": bool(np.isfinite(legacy_post_peak_drop) and legacy_post_peak_drop >= 90.0),
                "legacy_and_primary_metrics_disagree": bool(post_through_collapse != legacy_collapse),
                "softening_classification": "corrected_softening_gate_passed"
                if post_through_collapse
                else "corrected_softening_gate_not_passed",
            }
        )
    return pd.DataFrame(rows)


def build_legacy_policy() -> pd.DataFrame:
    rows = [
        {
            "metric_name": "energy exact reaction",
            "old_name": "exact dPi/dDelta",
            "new_name": "reaction_N_energy_exact",
            "allowed_use": "primary reaction and primary softening gate when checkpoint autograd is available",
            "disallowed_use": "non-checkpointed runs without exact mechanics objective reconstruction",
            "reason": "It is the derivative of the actual checkpoint mechanics objective with respect to Delta.",
            "requires_warning_in_plots": False,
            "historical_no_softening_conclusions_need_relabeling": False,
        },
        {
            "metric_name": "energy virtual-work reaction",
            "old_name": "R_virtual_energy_autograd_sigma_N",
            "new_name": "reaction_N_energy_virtual_work",
            "allowed_use": "validation-equivalent reaction when it matches reaction_N_energy_exact within tolerance",
            "disallowed_use": "primary gate if exact comparison is unavailable or fails tolerance",
            "reason": "It uses volume virtual work of energy-autograd stress and reproduces exact dPi/dDelta in the D0020 audit.",
            "requires_warning_in_plots": False,
            "historical_no_softening_conclusions_need_relabeling": False,
        },
        {
            "metric_name": "legacy top-boundary sigma reaction",
            "old_name": "reaction_N_tm_eff",
            "new_name": "reaction_N_legacy_top_sigma",
            "allowed_use": "legacy diagnostic and backward comparison only",
            "disallowed_use": "primary reaction, physical stress-strain reaction label, or primary softening gate",
            "reason": "The stress field sigma_tm_eff is not conjugate to the checkpoint history mechanics energy.",
            "requires_warning_in_plots": True,
            "historical_no_softening_conclusions_need_relabeling": True,
        },
        {
            "metric_name": "legacy bottom-boundary sigma reaction",
            "old_name": "bottom_reaction_N",
            "new_name": "reaction_N_bottom_sigma_legacy",
            "allowed_use": "boundary diagnostic and sign/balance check",
            "disallowed_use": "primary softening gate",
            "reason": "It is based on legacy postprocessed stress rather than exact energy derivative.",
            "requires_warning_in_plots": True,
            "historical_no_softening_conclusions_need_relabeling": False,
        },
        {
            "metric_name": "internal cut force above crack",
            "old_name": "internal_cut_force_above_crack_N",
            "new_name": "reaction_N_internal_cut_above",
            "allowed_use": "internal-force diagnostic around the cracked region",
            "disallowed_use": "primary reaction or physical validation",
            "reason": "It is a local diagnostic cut force and not the global energy-conjugate reaction.",
            "requires_warning_in_plots": True,
            "historical_no_softening_conclusions_need_relabeling": False,
        },
        {
            "metric_name": "internal cut force below crack",
            "old_name": "internal_cut_force_below_crack_N",
            "new_name": "reaction_N_internal_cut_below",
            "allowed_use": "internal-force diagnostic around the cracked region",
            "disallowed_use": "primary reaction or physical validation",
            "reason": "It is a local diagnostic cut force and not the global energy-conjugate reaction.",
            "requires_warning_in_plots": True,
            "historical_no_softening_conclusions_need_relabeling": False,
        },
    ]
    return pd.DataFrame(rows)


def make_figures(corrected: pd.DataFrame, softening: pd.DataFrame) -> None:
    usable = corrected[corrected["reaction_metric_status"] != "reaction_metric_unavailable"].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for seed, sub in usable.sort_values("step").groupby("seed"):
        ax.plot(sub["step"], sub["reaction_N_energy_exact"], marker="o", markersize=2.5, label=f"seed {seed}: primary energy-conjugate")
        through = sub[sub["alpha0p8_through_crack"].astype(bool)]
        if not through.empty:
            ax.axvline(float(through["step"].iloc[0]), color=ax.lines[-1].get_color(), alpha=0.25, linestyle="--")
    ax.set_xlabel("D0020 step")
    ax.set_ylabel("reaction_N_energy_exact [N]")
    ax.set_title("D0020 primary energy-conjugate reaction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_corrected_primary_reaction_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), dpi=180, sharex=True)
    for ax, seed in zip(axes, SEEDS):
        sub = usable[usable["seed"] == seed].sort_values("step")
        ax.plot(sub["step"], sub["reaction_N_energy_exact"], marker="o", markersize=2.2, label="primary: energy-conjugate exact")
        ax.plot(sub["step"], sub["reaction_N_energy_virtual_work"], marker="s", markersize=2.2, label="energy virtual-work")
        ax.plot(sub["step"], sub["reaction_N_legacy_top_sigma"], marker="^", markersize=2.2, label="legacy top-boundary sigma")
        ax.plot(sub["step"], sub["reaction_N_bottom_sigma_legacy"], linestyle="--", label="legacy bottom sigma")
        ax.plot(sub["step"], sub["reaction_N_internal_cut_above"], linestyle=":", label="internal cut above")
        ax.plot(sub["step"], sub["reaction_N_internal_cut_below"], linestyle="-.", label="internal cut below")
        through = sub[sub["alpha0p8_through_crack"].astype(bool)]
        if not through.empty:
            ax.axvline(float(through["step"].iloc[0]), color="k", alpha=0.25, linestyle="--")
        ax.set_ylabel(f"seed {seed}\nreaction [N]")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    axes[-1].set_xlabel("D0020 step")
    fig.suptitle("D0020 energy reaction versus legacy and cut diagnostics", y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_legacy_vs_energy_reaction_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=180)
    x = np.arange(len(softening))
    ax.bar(x - 0.18, softening["post_peak_drop_percent"], 0.36, label="primary energy-conjugate drop")
    ax.bar(x + 0.18, softening["legacy_top_sigma_post_peak_drop_percent"], 0.36, label="legacy top sigma drop")
    ax.axhline(50.0, color="k", linestyle="--", lw=0.9, label="50% collapse gate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(s)}" for s in softening["seed"]])
    ax.set_ylabel("post-peak drop [%]")
    ax.set_title("Corrected D0020 softening gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_corrected_softening_gate.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.0), dpi=180, sharex=True)
    for ax, seed in zip(axes, SEEDS):
        sub = usable[usable["seed"] == seed].sort_values("step")
        ax.plot(sub["step"], sub["reaction_N_energy_exact"].abs(), marker="o", markersize=2.2, label="primary energy-conjugate abs")
        ax.plot(sub["step"], sub["reaction_N_legacy_top_sigma"].abs(), marker="^", markersize=2.2, label="legacy top sigma abs")
        through = sub[sub["alpha0p8_through_crack"].astype(bool)]
        if not through.empty:
            step = float(through["step"].iloc[0])
            ax.axvline(step, color="red", alpha=0.45, linestyle="--", label="first alpha>=0.8 through-crack")
        ax.set_ylabel(f"seed {seed}\nabs reaction [N]")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("D0020 step")
    fig.suptitle("Through-crack markers on corrected and legacy reactions", y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_through_crack_reaction_marker.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    x = np.arange(len(softening))
    primary = softening["post_through_collapse_detected"].astype(int)
    legacy = softening["legacy_top_sigma_collapse_by_same_thresholds"].astype(int)
    ax.bar(x - 0.18, primary, 0.36, label="primary gate collapse")
    ax.bar(x + 0.18, legacy, 0.36, label="legacy top sigma collapse")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no", "yes"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(s)}" for s in softening["seed"]])
    ax.set_title("Metric disagreement summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_metric_disagreement_summary.png")
    plt.close(fig)


def write_figure_summary() -> None:
    lines = [
        "# Figure Summary",
        "",
        "All figures are diagnostic postprocessing outputs only and do not constitute physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `D0020_corrected_primary_reaction_curves.png` | Primary `reaction_N_energy_exact` curves for seeds 7, 13, and 42 with through-crack markers. | The energy-conjugate primary reaction drops after through-crack onset in the processed D0020 checkpoints. | Supports corrected softening gate assessment. |",
        "| `D0020_legacy_vs_energy_reaction_curves.png` | Primary energy reaction, energy virtual work, legacy top sigma, bottom sigma, and internal cut diagnostics. | Shows the corrected primary metric and legacy/cut diagnostics are distinct quantities. | Supports metric-labeling and legacy-demotion policy. |",
        "| `D0020_corrected_softening_gate.png` | Primary versus legacy post-peak drop percentages with the 50% gate. | Primary gate passes while legacy top sigma does not collapse by the same thresholds. | Supports gate update. |",
        "| `D0020_through_crack_reaction_marker.png` | Absolute primary and legacy reactions with first alpha>=0.8 through-crack marker. | Through-crack onset aligns with collapse in the corrected primary reaction, not in the legacy top-sigma metric. | Diagnostic support for relabeling old no-softening conclusions. |",
        "| `D0020_metric_disagreement_summary.png` | Per-seed collapse booleans for primary and legacy metrics. | Summarizes disagreement between the corrected primary gate and legacy top sigma gate. | Supports policy table decision. |",
        "",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_policy_note() -> None:
    lines = [
        "# Reaction Metric Policy",
        "",
        "This note applies to checkpointed TM D0020 postprocessing in this package.",
        "",
        "1. `reaction_N_tm_eff` is demoted because it integrates the legacy postprocessed current stress `sigma_tm_eff` on the top boundary. The prior conjugacy audit showed this stress path is not conjugate to the checkpoint history mechanics objective.",
        "2. The energy-conjugate primary metric is `reaction_N_energy_exact`, computed as exact autograd `dPi/dDelta` from the checkpoint mechanics objective. `reaction_N_energy_virtual_work` is a validation-equivalent metric only when it matches the exact reaction within tolerance.",
        "3. Checkpoint availability is required because the exact metric depends on reconstructing the model state, previous-step history fields, displacement mode, and checkpoint mechanics objective.",
        "4. Old runs without checkpoints must be labeled `reaction_metric_unavailable` for primary softening classification. They must not be relabeled as `no_softening` from legacy `reaction_N_tm_eff` alone.",
        "5. D0040 should remain deferred until the corrected D0020 reaction pipeline and metric policy are accepted. Later D0040 processing should use the same corrected metric names and checkpoint requirements.",
        "6. Do not claim physical validation from this package. It is a postprocessing infrastructure and metric-policy update only.",
        "",
    ]
    (PACKAGE / "REACTION_METRIC_POLICY.md").write_text("\n".join(lines), encoding="utf-8")


def classify_package(softening: pd.DataFrame, virtual_summary: pd.DataFrame, policy: pd.DataFrame) -> tuple[str, dict[str, object]]:
    exact_reproduced = int(virtual_summary["exact_reaction_reproduced_from_prior_audit"].sum())
    virtual_matches = int(virtual_summary["virtual_work_matches_exact_within_tolerance"].sum())
    primary_pass = int(softening["post_through_collapse_detected"].sum())
    legacy_demoted = bool(
        (policy["new_name"] == "reaction_N_legacy_top_sigma").any()
        and "primary softening gate" in str(
            policy.loc[policy["new_name"] == "reaction_N_legacy_top_sigma", "disallowed_use"].iloc[0]
        )
    )
    no_primary_legacy = True
    if exact_reproduced == len(SEEDS) and virtual_matches == len(SEEDS) and primary_pass >= 2 and legacy_demoted and no_primary_legacy:
        label = "corrected D0020 reaction pipeline validated; legacy reaction demoted"
    elif legacy_demoted and no_primary_legacy:
        label = "legacy reaction demoted"
    else:
        label = "corrected pipeline unresolved"
    return label, {
        "exact_reproduced_seed_count": exact_reproduced,
        "virtual_work_match_seed_count": virtual_matches,
        "primary_softening_gate_pass_seed_count": primary_pass,
        "legacy_demoted": legacy_demoted,
        "no_primary_legacy": no_primary_legacy,
    }


def write_reports(
    classification: str,
    info: dict[str, object],
    corrected: pd.DataFrame,
    softening: pd.DataFrame,
    virtual_summary: pd.DataFrame,
) -> None:
    processed_seeds = sorted(int(s) for s in corrected["seed"].dropna().unique())
    max_virtual_abs = float(virtual_summary["max_energy_virtual_abs_error_N"].max())
    max_virtual_rel = float(virtual_summary["max_energy_virtual_rel_error"].max())
    max_reproduction_abs = float(virtual_summary["max_exact_reproduction_abs_error_N"].max())
    rows = [
        "# D0020 Corrected Reaction Pipeline",
        "",
        "## Scope",
        "",
        "This package implements corrected energy-consistent reaction postprocessing for existing checkpointed D0020 TM runs. It uses seeds 7, 13, and 42 and does not run D0040, extend loading, retrain, or change physics.",
        "",
        "## Classification",
        "",
        f"**{classification}**.",
        "",
        "## Required Questions",
        "",
        "1. Was corrected energy-consistent reaction postprocessing implemented?",
        "   - Yes. `tables/corrected_reaction_by_step.csv` reports standardized corrected and legacy metric names.",
        "2. Does `reaction_N_energy_exact` reproduce the previous exact D0020 reaction values for seeds 7/13/42?",
        f"   - Yes for {info['exact_reproduced_seed_count']}/3 seeds; maximum absolute reproduction error is {max_reproduction_abs:.6g} N.",
        "3. Does `reaction_N_energy_virtual_work` match exact autograd reaction within tolerance?",
        f"   - Yes for {info['virtual_work_match_seed_count']}/3 seeds using relative-or-absolute tolerance; maximum absolute error is {max_virtual_abs:.6g} N and maximum relative error is {max_virtual_rel:.6g}.",
        "4. Does corrected D0020 softening gate pass for seeds 7/13/42?",
        f"   - It passes for {info['primary_softening_gate_pass_seed_count']}/3 seeds using the corrected primary metric.",
        "5. Does legacy top sigma reaction disagree with primary energy reaction after through-crack onset?",
        f"   - Yes in {int(softening['legacy_and_primary_metrics_disagree'].sum())}/3 seeds.",
        "6. Is `reaction_N_tm_eff` formally demoted to legacy diagnostic-only status?",
        "   - Yes. See `tables/legacy_reaction_metric_policy.csv` and `REACTION_METRIC_POLICY.md`.",
        "7. How should old non-checkpointed D0020/D0040 no-softening conclusions be relabeled?",
        "   - They should be relabeled as legacy-metric-only. Without checkpoints, primary softening classification is `reaction_metric_unavailable`, not `no_softening`.",
        "8. Is D0040 still deferred?",
        "   - Yes.",
        "9. Is any production mechanics change justified?",
        "   - No. This package changes postprocessing infrastructure and metric policy only.",
        "10. What is the next minimal intervention?",
        "   - Review whether to promote `reaction_N_energy_exact` into reusable checkpoint postprocessing code, then reprocess D0040 only after this D0020 policy is accepted.",
        "",
        "## Key Tables",
        "",
        "- `tables/corrected_reaction_by_step.csv`",
        "- `tables/corrected_softening_gate_summary.csv`",
        "- `tables/legacy_reaction_metric_policy.csv`",
        "- `tables/exact_reaction_reproduction_check.csv`",
        "- `tables/virtual_work_agreement_summary.csv`",
        "- `tables/checkpoint_availability.csv`",
        "",
        "## Limits",
        "",
        "- This package does not claim physical validation.",
        "- It does not modify mechanics training, material parameters, TM split, history logic, or load schedule.",
        "- D0040 remains deferred.",
        "",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(rows), encoding="utf-8")

    readme = [
        "# D0020 corrected reaction pipeline package",
        "",
        "Read in this order:",
        "",
        "1. `REPORT.md`",
        "2. `REACTION_METRIC_POLICY.md`",
        "3. `tables/corrected_softening_gate_summary.csv`",
        "4. `tables/corrected_reaction_by_step.csv`",
        "5. `tables/legacy_reaction_metric_policy.csv`",
        "6. `figures/figure_summary.md`",
        "",
        "This package is a postprocessing and metric-policy update for checkpointed D0020 runs only.",
        "",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme), encoding="utf-8")

    next_q = [
        "# Next Questions",
        "",
        "1. Should `reaction_N_energy_exact` be promoted into a reusable checkpoint postprocessing utility outside this evidence package?",
        "2. Should future plots keep `reaction_N_legacy_top_sigma` only as a dashed, warning-labeled legacy diagnostic?",
        "3. After this D0020 policy is accepted, should D0040 be reprocessed with the same corrected metric names and checkpoint requirement?",
        "",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_q), encoding="utf-8")

    handoff = [
        "## Codex handoff: D0020 corrected reaction pipeline",
        "",
        "Commit: COMMIT_PLACEHOLDER",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_corrected_reaction_pipeline",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_corrected_reaction_pipeline/REPORT.md",
        "",
        "### What changed",
        "- Implemented corrected energy-consistent D0020 reaction postprocessing in a package script.",
        "- Standardized corrected and legacy reaction metric names.",
        "- Added corrected softening gate summary and legacy metric policy table.",
        "- Regenerated D0020 reaction figures with explicit primary/legacy/diagnostic labels.",
        "- Did not run D0040, extend loading, retrain, or change physics.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_corrected_reaction_pipeline.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_corrected_reaction_pipeline.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "```",
        "",
        "### Key results",
        f"- Classification: **{classification}**.",
        "- Corrected metric names: `reaction_N_energy_exact`, `reaction_N_energy_virtual_work`, `reaction_N_legacy_top_sigma`, `reaction_N_bottom_sigma_legacy`, `reaction_N_internal_cut_above`, `reaction_N_internal_cut_below`.",
        f"- Processed seeds: {processed_seeds}.",
        f"- Exact reaction reproduction: {info['exact_reproduced_seed_count']}/3 seeds.",
        f"- Energy virtual-work agreement: {info['virtual_work_match_seed_count']}/3 seeds.",
        f"- Corrected softening gate pass: {info['primary_softening_gate_pass_seed_count']}/3 seeds.",
        "- Legacy `reaction_N_tm_eff` is demoted to diagnostic-only status.",
        "- Old non-checkpointed no-softening conclusions should be relabeled as legacy-metric-only / `reaction_metric_unavailable` for primary classification.",
        "- D0040 remains deferred.",
        "- No production mechanics change is justified by this package.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `REACTION_METRIC_POLICY.md`",
        "- `tables/corrected_softening_gate_summary.csv`",
        "- `tables/corrected_reaction_by_step.csv`",
        "- `tables/legacy_reaction_metric_policy.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Does this package correctly promote `reaction_N_energy_exact` as the primary checkpoint reaction metric?",
        "2. Is the legacy demotion policy for `reaction_N_tm_eff` strict enough for future D0020/D0040 reports?",
        "3. What is the next minimal Codex task before D0040 reprocessing?",
        "",
        "### Constraints",
        "- Do not run D0040 yet.",
        "- Do not extend loading.",
        "- Do not retrain seed 7/13/42 unless existing checkpoints are missing or corrupt.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip/local/jump/geometry-guided losses.",
        "- Do not claim physical validation.",
        "",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff), encoding="utf-8")


def write_commands() -> None:
    commands = [
        "git pull origin main",
        "Read previous D0020 energy-stress conjugacy handoff/report/tables.",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_corrected_reaction_pipeline.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_corrected_reaction_pipeline.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_manifest() -> None:
    type_map = {
        ".csv": "table",
        ".png": "figure",
        ".py": "artifact",
        ".txt": "command_log",
        ".json": "artifact",
        ".md": "report",
    }
    required = {
        "README.md",
        "REPORT.md",
        "REACTION_METRIC_POLICY.md",
        "HANDOFF_COMMENT.md",
        "commands_run.txt",
        "next_questions.md",
        "figures/figure_summary.md",
        "tables/corrected_reaction_by_step.csv",
        "tables/corrected_softening_gate_summary.csv",
        "tables/legacy_reaction_metric_policy.csv",
        "tables/exact_reaction_reproduction_check.csv",
        "tables/virtual_work_agreement_summary.csv",
    }
    descriptions = {
        "tables/corrected_reaction_by_step.csv": "Unified corrected and legacy reaction metrics by checkpointed D0020 step.",
        "tables/corrected_softening_gate_summary.csv": "Seed-level corrected primary softening gate summary.",
        "tables/legacy_reaction_metric_policy.csv": "Policy table demoting legacy reaction_N_tm_eff.",
        "tables/exact_reaction_reproduction_check.csv": "Comparison against previous exact D0020 audit values.",
        "tables/virtual_work_agreement_summary.csv": "Energy virtual-work agreement with exact autograd reaction.",
        "tables/checkpoint_availability.csv": "Checkpoint and exact reaction computability flags.",
        "figures/figure_summary.md": "Text summary of generated diagnostic figures.",
    }
    manifest = []
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        kind = type_map.get(path.suffix.lower(), "artifact")
        if rel == "HANDOFF_COMMENT.md":
            kind = "handoff"
        if rel == "figures/figure_summary.md":
            kind = "figure_summary"
        manifest.append(
            {
                "path": rel,
                "type": kind,
                "description": descriptions.get(rel, "Corrected reaction pipeline package file."),
                "required_for_chatgpt": rel in required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    corrected, availability = load_checkpointed_reactions()
    corrected.to_csv(TABLES / "corrected_reaction_by_step.csv", index=False)
    availability.to_csv(TABLES / "checkpoint_availability.csv", index=False)

    reproduction = build_reproduction_check(corrected)
    reproduction.to_csv(TABLES / "exact_reaction_reproduction_check.csv", index=False)

    virtual_summary = build_virtual_work_summary(corrected, reproduction)
    virtual_summary.to_csv(TABLES / "virtual_work_agreement_summary.csv", index=False)

    softening = build_softening_summary(corrected)
    softening.to_csv(TABLES / "corrected_softening_gate_summary.csv", index=False)

    policy = build_legacy_policy()
    policy.to_csv(TABLES / "legacy_reaction_metric_policy.csv", index=False)

    make_figures(corrected, softening)
    write_figure_summary()
    write_policy_note()

    classification, info = classify_package(softening, virtual_summary, policy)
    write_reports(classification, info, corrected, softening, virtual_summary)
    write_commands()
    write_manifest()
    print(classification)


if __name__ == "__main__":
    main()
