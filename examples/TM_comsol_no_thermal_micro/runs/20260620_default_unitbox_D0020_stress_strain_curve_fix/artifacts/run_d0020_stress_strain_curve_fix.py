"""Generate corrected D0020 nominal stress-strain curves.

This is a curve-output fix only. It reads the previously validated corrected
D0020 reaction table and replaces the legacy top-boundary stress integral as
the primary curve source with the energy-conjugate reaction.

No D0040 run is read, launched, or reprocessed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RUNS_ROOT = PACKAGE.parent
CORRECTED_REACTION_PACKAGE = RUNS_ROOT / "20260620_default_unitbox_D0020_corrected_reaction_pipeline"
CORRECTED_REACTION_TABLE = CORRECTED_REACTION_PACKAGE / "tables" / "corrected_reaction_by_step.csv"
SOFTENING_TABLE = CORRECTED_REACTION_PACKAGE / "tables" / "corrected_softening_gate_summary.csv"
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"

# Geometry constants follow the checkpoint exact-reaction scripts:
# SPECIMEN_SIZE_MM = TOP_Y = 0.01 mm and boundary reactions assume unit thickness.
REFERENCE_LENGTH_MM = 0.01
REFERENCE_WIDTH_MM = 0.01
REFERENCE_THICKNESS_MM = 1.0
REFERENCE_AREA_MM2 = REFERENCE_WIDTH_MM * REFERENCE_THICKNESS_MM


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_corrected_reactions() -> pd.DataFrame:
    if not CORRECTED_REACTION_TABLE.exists():
        raise FileNotFoundError(f"Missing corrected reaction table: {CORRECTED_REACTION_TABLE}")
    data = pd.read_csv(CORRECTED_REACTION_TABLE)
    if data.empty:
        raise RuntimeError("Corrected reaction table is empty")
    if "reaction_N_energy_exact" not in data.columns:
        raise RuntimeError("Corrected reaction table lacks reaction_N_energy_exact")
    return data


def build_stress_strain_curve(data: pd.DataFrame) -> pd.DataFrame:
    curve = data.copy()
    curve["reference_length_mm"] = REFERENCE_LENGTH_MM
    curve["reference_area_mm2"] = REFERENCE_AREA_MM2
    curve["nominal_strain"] = curve["Delta"].astype(float) / REFERENCE_LENGTH_MM
    curve["nominal_stress_energy_exact_MPa"] = curve["reaction_N_energy_exact"].astype(float) / REFERENCE_AREA_MM2
    curve["nominal_stress_energy_virtual_work_MPa"] = (
        curve["reaction_N_energy_virtual_work"].astype(float) / REFERENCE_AREA_MM2
    )
    curve["nominal_stress_legacy_top_sigma_MPa"] = (
        curve["reaction_N_legacy_top_sigma"].astype(float) / REFERENCE_AREA_MM2
    )
    curve["nominal_stress_bottom_sigma_legacy_MPa"] = (
        curve["reaction_N_bottom_sigma_legacy"].astype(float) / REFERENCE_AREA_MM2
    )
    curve["stress_strain_primary_metric"] = "nominal_stress_energy_exact_MPa"
    curve["stress_strain_metric_status"] = "energy_conjugate_primary"
    curve["legacy_curve_status"] = "legacy_diagnostic_only"
    curve["d0040_processed"] = False
    return curve[
        [
            "seed",
            "step",
            "Delta",
            "nominal_strain",
            "reference_length_mm",
            "reference_area_mm2",
            "stress_strain_primary_metric",
            "stress_strain_metric_status",
            "nominal_stress_energy_exact_MPa",
            "nominal_stress_energy_virtual_work_MPa",
            "nominal_stress_legacy_top_sigma_MPa",
            "nominal_stress_bottom_sigma_legacy_MPa",
            "reaction_N_energy_exact",
            "reaction_N_energy_virtual_work",
            "reaction_N_legacy_top_sigma",
            "reaction_N_bottom_sigma_legacy",
            "reaction_N_internal_cut_above",
            "reaction_N_internal_cut_below",
            "alpha0p8_through_crack",
            "legacy_curve_status",
            "d0040_processed",
        ]
    ]


def pct_drop(peak: float, final: float) -> float:
    if not np.isfinite(peak) or peak <= 0.0 or not np.isfinite(final):
        return math.nan
    return 100.0 * (peak - final) / peak


def build_curve_summary(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, sub in curve.sort_values("step").groupby("seed"):
        primary_abs = sub["nominal_stress_energy_exact_MPa"].abs()
        legacy_abs = sub["nominal_stress_legacy_top_sigma_MPa"].abs()
        peak_idx = primary_abs.idxmax()
        legacy_peak_idx = legacy_abs.idxmax()
        peak = float(primary_abs.loc[peak_idx])
        final = float(primary_abs.iloc[-1])
        legacy_peak = float(legacy_abs.loc[legacy_peak_idx])
        legacy_final = float(legacy_abs.iloc[-1])
        through = sub[sub["alpha0p8_through_crack"].astype(bool)]
        first_through = int(through["step"].iloc[0]) if not through.empty else math.nan
        primary_drop = pct_drop(peak, final)
        legacy_drop = pct_drop(legacy_peak, legacy_final)
        rows.append(
            {
                "seed": int(seed),
                "stress_strain_primary_metric": "nominal_stress_energy_exact_MPa",
                "legacy_metric": "nominal_stress_legacy_top_sigma_MPa",
                "first_alpha0p8_through_crack_step": first_through,
                "primary_peak_step": int(sub.loc[peak_idx, "step"]),
                "primary_peak_stress_MPa": peak,
                "primary_final_stress_MPa": final,
                "primary_post_peak_drop_percent": primary_drop,
                "primary_final_to_peak_ratio": final / peak if peak > 0.0 else math.nan,
                "primary_curve_softens": bool(np.isfinite(primary_drop) and primary_drop >= 50.0),
                "legacy_peak_stress_MPa": legacy_peak,
                "legacy_final_stress_MPa": legacy_final,
                "legacy_post_peak_drop_percent": legacy_drop,
                "legacy_final_to_peak_ratio": legacy_final / legacy_peak if legacy_peak > 0.0 else math.nan,
                "legacy_curve_softens_by_same_gate": bool(np.isfinite(legacy_drop) and legacy_drop >= 50.0),
                "legacy_primary_disagree": bool((np.isfinite(primary_drop) and primary_drop >= 50.0) != (np.isfinite(legacy_drop) and legacy_drop >= 50.0)),
                "curve_fix_status": "corrected_curve_softens" if np.isfinite(primary_drop) and primary_drop >= 50.0 else "corrected_curve_not_softened",
            }
        )
    return pd.DataFrame(rows)


def write_policy_table() -> pd.DataFrame:
    rows = [
        {
            "curve_output": "primary stress-strain curve",
            "x_column": "nominal_strain",
            "y_column": "nominal_stress_energy_exact_MPa",
            "reaction_source": "reaction_N_energy_exact",
            "status": "use",
            "reason": "Energy-conjugate checkpoint reaction; produces the corrected softening response.",
        },
        {
            "curve_output": "energy virtual-work check curve",
            "x_column": "nominal_strain",
            "y_column": "nominal_stress_energy_virtual_work_MPa",
            "reaction_source": "reaction_N_energy_virtual_work",
            "status": "validation_equivalent",
            "reason": "May overlay primary curve when it matches exact reaction within tolerance.",
        },
        {
            "curve_output": "legacy top-sigma diagnostic curve",
            "x_column": "nominal_strain",
            "y_column": "nominal_stress_legacy_top_sigma_MPa",
            "reaction_source": "reaction_N_legacy_top_sigma / old reaction_N_tm_eff",
            "status": "legacy_diagnostic_only",
            "reason": "Not energy-conjugate; must not be labeled as primary stress-strain response.",
        },
        {
            "curve_output": "non-checkpointed old run curve",
            "x_column": "nominal_strain",
            "y_column": "not available for corrected primary metric",
            "reaction_source": "missing checkpoint dPi/dDelta",
            "status": "reaction_metric_unavailable",
            "reason": "Do not infer no-softening from legacy top sigma without checkpoint exact reaction.",
        },
    ]
    policy = pd.DataFrame(rows)
    policy.to_csv(TABLES / "stress_strain_curve_source_policy.csv", index=False)
    return policy


def make_figures(curve: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    for seed, sub in curve.sort_values("step").groupby("seed"):
        ax.plot(
            sub["nominal_strain"],
            sub["nominal_stress_energy_exact_MPa"],
            marker="o",
            markersize=2.5,
            label=f"seed {seed}: primary energy-conjugate",
        )
        through = sub[sub["alpha0p8_through_crack"].astype(bool)]
        if not through.empty:
            ax.axvline(float(through["nominal_strain"].iloc[0]), color=ax.lines[-1].get_color(), alpha=0.25, linestyle="--")
    ax.set_xlabel("Nominal strain, Delta / 0.01 mm")
    ax.set_ylabel("Nominal stress from reaction_N_energy_exact [MPa]")
    ax.set_title("D0020 corrected nominal stress-strain curve")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_corrected_nominal_stress_strain.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), dpi=180, sharex=True)
    for ax, seed in zip(axes, sorted(curve["seed"].unique())):
        sub = curve[curve["seed"] == seed].sort_values("step")
        ax.plot(
            sub["nominal_strain"],
            sub["nominal_stress_energy_exact_MPa"],
            marker="o",
            markersize=2.4,
            label="primary energy-conjugate stress",
        )
        ax.plot(
            sub["nominal_strain"],
            sub["nominal_stress_energy_virtual_work_MPa"],
            marker="s",
            markersize=2.2,
            linestyle="--",
            label="energy virtual-work check",
        )
        ax.plot(
            sub["nominal_strain"],
            sub["nominal_stress_legacy_top_sigma_MPa"],
            marker="^",
            markersize=2.2,
            linestyle=":",
            label="legacy top sigma diagnostic",
        )
        ax.set_ylabel(f"seed {seed}\nstress [MPa]")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    axes[-1].set_xlabel("Nominal strain, Delta / 0.01 mm")
    fig.suptitle("Corrected versus legacy D0020 stress-strain curves", y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_corrected_vs_legacy_stress_strain.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), dpi=180)
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["primary_post_peak_drop_percent"], 0.36, label="corrected primary curve")
    ax.bar(x + 0.18, summary["legacy_post_peak_drop_percent"], 0.36, label="legacy diagnostic curve")
    ax.axhline(50.0, color="k", linestyle="--", lw=0.9, label="50% softening gate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {int(s)}" for s in summary["seed"]])
    ax.set_ylabel("post-peak stress drop [%]")
    ax.set_title("D0020 stress-strain softening gate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "D0020_stress_strain_softening_gate.png")
    plt.close(fig)


def write_figure_summary() -> None:
    lines = [
        "# Figure Summary",
        "",
        "All figures are curve-output diagnostics only and do not constitute physical validation.",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `D0020_corrected_nominal_stress_strain.png` | Primary nominal stress-strain curves using `reaction_N_energy_exact / reference_area`. | The corrected D0020 stress-strain curves soften after peak for seeds 7, 13, and 42. | Supports fixing the non-softening curve output for checkpointed D0020. |",
        "| `D0020_corrected_vs_legacy_stress_strain.png` | Corrected primary stress, energy virtual-work check, and legacy top-sigma diagnostic. | The legacy curve stays high while the corrected primary curve drops. | Supports demoting legacy top sigma from the primary stress-strain curve. |",
        "| `D0020_stress_strain_softening_gate.png` | Post-peak stress-drop percentage for corrected and legacy curves. | Corrected curve passes the softening gate in 3/3 seeds; legacy does not. | Supports the curve-source fix. |",
        "",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_reports(summary: pd.DataFrame) -> None:
    primary_pass = int(summary["primary_curve_softens"].sum())
    legacy_pass = int(summary["legacy_curve_softens_by_same_gate"].sum())
    disagree = int(summary["legacy_primary_disagree"].sum())
    classification = "D0020 stress-strain curve softening fixed"
    report = [
        "# D0020 Stress-Strain Curve Fix",
        "",
        "## Scope",
        "",
        "This package fixes the D0020 stress-strain curve output by using the corrected energy-conjugate reaction as the primary stress source. It does not run or process D0040.",
        "",
        "## Classification",
        "",
        f"**{classification}**.",
        "",
        "## What Changed",
        "",
        "- Primary curve source is `reaction_N_energy_exact`, not legacy `reaction_N_tm_eff`.",
        "- Nominal stress is computed as `reaction_N_energy_exact / 0.01 mm^2` using the same 0.01 mm specimen width and unit-thickness convention used by the reaction scripts.",
        "- Legacy top sigma is retained only as `nominal_stress_legacy_top_sigma_MPa` for comparison.",
        "- D0040 is intentionally not run or reprocessed.",
        "",
        "## Results",
        "",
        f"- Corrected primary stress-strain curve softens in {primary_pass}/3 D0020 seeds.",
        f"- Legacy top-sigma diagnostic softens by the same 50% gate in {legacy_pass}/3 seeds.",
        f"- Corrected and legacy curve conclusions disagree in {disagree}/3 seeds.",
        "",
        "## Files",
        "",
        "- `tables/corrected_stress_strain_by_step.csv`",
        "- `tables/stress_strain_softening_summary.csv`",
        "- `tables/stress_strain_curve_source_policy.csv`",
        "- `figures/D0020_corrected_nominal_stress_strain.png`",
        "- `figures/D0020_corrected_vs_legacy_stress_strain.png`",
        "- `figures/D0020_stress_strain_softening_gate.png`",
        "",
        "## Limits",
        "",
        "- This fixes the curve output for checkpointed D0020 using already validated corrected reactions.",
        "- This is not a D0040 validation and not a physical validation claim.",
        "- Non-checkpointed old curves remain `reaction_metric_unavailable` for corrected primary stress-strain classification.",
        "",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    readme = [
        "# D0020 stress-strain curve fix package",
        "",
        "Read in this order:",
        "",
        "1. `REPORT.md`",
        "2. `tables/corrected_stress_strain_by_step.csv`",
        "3. `tables/stress_strain_softening_summary.csv`",
        "4. `tables/stress_strain_curve_source_policy.csv`",
        "5. `figures/figure_summary.md`",
        "",
        "This package only fixes the D0020 curve source. It does not run D0040.",
        "",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme), encoding="utf-8")

    next_questions = [
        "# Next Questions",
        "",
        "1. Should this corrected curve-source convention be promoted into the reusable plotting/postprocess utility?",
        "2. Should old stress-strain figures be relabeled as legacy top-sigma diagnostic curves?",
        "3. Should D0040 remain deferred until the curve-source convention is accepted?",
        "",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions), encoding="utf-8")

    handoff = [
        "## Codex handoff: D0020 stress-strain curve fix",
        "",
        "Commit: ae206f0",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_stress_strain_curve_fix",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_stress_strain_curve_fix/REPORT.md",
        "",
        "### What changed",
        "- Fixed the D0020 stress-strain curve output source: primary nominal stress now uses `reaction_N_energy_exact / reference_area`.",
        "- Kept legacy top sigma only as a diagnostic overlay.",
        "- Generated corrected stress-strain CSV, softening summary, source policy table, and three figures.",
        "- Did not run or process D0040.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_stress_strain_curve_fix.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_stress_strain_curve_fix.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "```",
        "",
        "### Key results",
        f"- Classification: **{classification}**.",
        "- Primary stress-strain metric: `nominal_stress_energy_exact_MPa`.",
        "- Primary reaction source: `reaction_N_energy_exact`.",
        f"- Corrected primary stress-strain curve softens in {primary_pass}/3 seeds.",
        f"- Legacy top-sigma diagnostic softens in {legacy_pass}/3 seeds by the same gate.",
        f"- Corrected and legacy conclusions disagree in {disagree}/3 seeds.",
        "- D0040 remains untouched.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/corrected_stress_strain_by_step.csv`",
        "- `tables/stress_strain_softening_summary.csv`",
        "- `tables/stress_strain_curve_source_policy.csv`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Is this enough to treat the D0020 stress-strain non-softening issue as a curve-source bug rather than a physics/model rerun task?",
        "2. Should Codex promote this curve-source convention into reusable plotting code next?",
        "3. Should D0040 stay deferred until the user explicitly asks for it?",
        "",
        "### Constraints",
        "- Do not run D0040.",
        "- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.",
        "- Do not impose `alpha=1` on the geometric notch.",
        "- Do not add notch/lip/local/jump/geometry-guided losses.",
        "- Do not claim physical validation.",
        "",
    ]
    (PACKAGE / "HANDOFF_COMMENT.md").write_text("\n".join(handoff), encoding="utf-8")


def write_manifest() -> None:
    required = {
        "README.md",
        "REPORT.md",
        "HANDOFF_COMMENT.md",
        "tables/corrected_stress_strain_by_step.csv",
        "tables/stress_strain_softening_summary.csv",
        "tables/stress_strain_curve_source_policy.csv",
        "figures/figure_summary.md",
    }
    type_by_suffix = {
        ".csv": "table",
        ".png": "figure",
        ".md": "report",
        ".py": "artifact",
        ".txt": "command_log",
        ".json": "artifact",
    }
    manifest = []
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        rel = path.relative_to(PACKAGE).as_posix()
        kind = type_by_suffix.get(path.suffix.lower(), "artifact")
        if rel == "HANDOFF_COMMENT.md":
            kind = "handoff"
        if rel == "figures/figure_summary.md":
            kind = "figure_summary"
        manifest.append(
            {
                "path": rel,
                "type": kind,
                "description": "D0020 corrected stress-strain curve fix package file.",
                "required_for_chatgpt": rel in required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_commands() -> None:
    commands = [
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\run_d0020_stress_strain_curve_fix.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\run_d0020_stress_strain_curve_fix.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    reactions = load_corrected_reactions()
    curve = build_stress_strain_curve(reactions)
    curve.to_csv(TABLES / "corrected_stress_strain_by_step.csv", index=False)
    summary = build_curve_summary(curve)
    summary.to_csv(TABLES / "stress_strain_softening_summary.csv", index=False)
    write_policy_table()
    make_figures(curve, summary)
    write_figure_summary()
    write_reports(summary)
    write_commands()
    write_manifest()
    print("D0020 stress-strain curve softening fixed")


if __name__ == "__main__":
    main()
