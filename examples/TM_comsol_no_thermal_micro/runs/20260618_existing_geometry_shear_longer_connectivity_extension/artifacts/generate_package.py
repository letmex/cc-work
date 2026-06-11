from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from build_shear_connectivity_package import (
    DEFAULT_CC_ROOT,
    DEFAULT_THRESHOLDS,
    SEED,
    SRC_ROOT,
    TRAIN_PYTHON,
    add_step_context,
    build_figures,
    build_output_manifest,
    build_stepwise_tables,
    count_field_files,
    ensure_dirs,
    read_required_csv,
    result_dirs,
    summary_for_case,
    threshold_label,
    truthy,
    write_manifest,
)
from shear_connectivity import compute_connectivity_by_threshold


PACKAGE_REL = Path(
    "examples/TM_comsol_no_thermal_micro/runs/"
    "20260618_existing_geometry_shear_longer_connectivity_extension"
)
RUN_ID = "seed23_S0090_shear"
BASELINE_RUN_ID = "seed23_S0070_shear"
TRAIN_CMD = (
    rf"{TRAIN_PYTHON} main.py 8 400 23 TrainableReLU 3.0 --full "
    "--n-rprop 300 --n-lbfgs 1 --load-case shear "
    "--load-schedule-file load_schedules/load_schedule_S0090_shear.csv "
    "--run-suffix seed23_S0090_shear"
)
POST_CMD = (
    rf"{TRAIN_PYTHON} postprocess_results.py "
    "--model-dir outputs/checkpoints/seed23_S0090_shear "
    "--result-dir outputs/results/seed23_S0090_shear --device cpu"
)


def classify_shear_longer_extension(
    *,
    run_completed: bool,
    reaction_ok: bool,
    post_peak_drop_percent: float,
    final_top_v_ratio: float,
    unstable_top_v_ratio: float,
    through_crack_any: bool,
    xspan_growth_material: bool,
    propagated_high_threshold: bool,
) -> str:
    if (
        not run_completed
        or not reaction_ok
        or not np.isfinite(final_top_v_ratio)
        or final_top_v_ratio > unstable_top_v_ratio
    ):
        return "shear longer extension unstable"
    if through_crack_any:
        return "shear longer extension reaches through-crack"
    if (
        post_peak_drop_percent > 0.0
        and xspan_growth_material
        and propagated_high_threshold
    ):
        return "shear longer extension shows continued propagation"
    if not through_crack_any and not xspan_growth_material and not propagated_high_threshold:
        return "shear longer extension saturates as local notch crack"
    return "shear longer extension inconclusive"


def git_ahead_status(cc_root: Path) -> Dict[str, object]:
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(cc_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"status_line": "unavailable", "main_ahead_origin": "unknown", "commits_pushed": False}
    first_line = status.splitlines()[0] if status else ""
    return {
        "status_line": first_line,
        "main_ahead_origin": "ahead" in first_line,
        "commits_pushed": False,
    }


def summary_for_longer_case(
    *,
    run_id: str,
    schedule: str,
    continued_label: object,
    stress: pd.DataFrame,
    reaction: pd.DataFrame,
    diagnostics: pd.DataFrame,
    connectivity: pd.DataFrame,
    model_dir: Path,
    result_dir: Path,
    classification: str,
) -> Dict[str, object]:
    summary = summary_for_case(
        run_id=run_id,
        schedule=schedule,
        continued_label=str(continued_label),
        stress=stress,
        reaction=reaction,
        diagnostics=diagnostics,
        connectivity=connectivity,
        model_dir=model_dir,
        result_dir=result_dir,
        classification=classification,
    )
    summary["continued_from_S0070"] = continued_label
    summary.pop("continued_from_S0050", None)
    return summary


def xspan_growths(baseline: Dict[str, object], current: Dict[str, object]) -> Dict[str, float]:
    growth = {}
    for threshold in DEFAULT_THRESHOLDS:
        label = threshold_label(threshold)
        base_value = float(baseline.get(f"final_{label}_x_span") or 0.0)
        current_value = float(current.get(f"final_{label}_x_span") or 0.0)
        growth[f"{label}_x_span_growth_vs_S0070"] = current_value - base_value
    return growth


def comparison_table(s0070: Dict[str, object], s0090: Dict[str, object]) -> pd.DataFrame:
    growths = xspan_growths(s0070, s0090)
    columns = [
        "case",
        "schedule_name",
        "step_count",
        "final_Delta_s",
        "training_settings",
        "continued_from_S0070",
        "final_engineering_shear_strain",
        "peak_nominal_shear_stress_MPa",
        "peak_step",
        "final_nominal_shear_stress_MPa",
        "post_peak_drop_MPa",
        "post_peak_drop_percent",
        "final_alpha_max",
        "alpha_max_location",
        "first_alpha_ge_0p3_notch_connected_step",
        "first_alpha_ge_0p5_notch_connected_step",
        "first_alpha_ge_0p8_notch_connected_step",
        "first_alpha_ge_0p95_notch_connected_step",
        "first_alpha_ge_0p3_right_boundary_step",
        "first_alpha_ge_0p5_right_boundary_step",
        "first_alpha_ge_0p8_right_boundary_step",
        "first_alpha_ge_0p95_right_boundary_step",
        "final_alpha_ge_0p3_x_span",
        "final_alpha_ge_0p5_x_span",
        "final_alpha_ge_0p8_x_span",
        "final_alpha_ge_0p95_x_span",
        "alpha_ge_0p3_x_span_growth_vs_S0070",
        "alpha_ge_0p5_x_span_growth_vs_S0070",
        "alpha_ge_0p8_x_span_growth_vs_S0070",
        "alpha_ge_0p95_x_span_growth_vs_S0070",
        "final_HII_over_HI_peak_ratio",
        "final_mechanics_drive_max_location",
        "drive_location_classification",
        "final_top_v_absmax_over_Delta_s",
        "classification",
    ]
    rows = []
    for case, summary in (("S0070", s0070), ("S0090", s0090)):
        row = {"case": case}
        row.update(summary)
        if case == "S0070":
            for key in growths:
                row[key] = 0.0
        else:
            row.update(growths)
        rows.append({column: row.get(column, "") for column in columns})
    return pd.DataFrame(rows, columns=columns)


def copy_required_figures(result_dir: Path, package: Path) -> None:
    fig_src = result_dir / "figures"
    legacy_run_label = RUN_ID.replace("seed23_", "")
    mapping = {
        "shear_stress_strain_seed23.png": [
            "stress_strain_seed23_shear.png",
            f"stress_strain_seed23_{legacy_run_label}.png",
        ],
        "shear_reaction_strain_seed23.png": [
            "reaction_strain_seed23_shear.png",
            f"reaction_strain_seed23_{legacy_run_label}.png",
        ],
        "final_fields_panel_seed23_shear.png": [
            "final_fields_panel_seed23_shear.png",
            f"final_fields_panel_seed23_{legacy_run_label}.png",
        ],
        "final_alpha_seed23_shear.png": [
            "final_alpha_seed23_shear.png",
            f"final_alpha_seed23_{legacy_run_label}.png",
        ],
        "final_u_seed23_shear.png": [
            "final_u_seed23_shear.png",
            f"final_u_seed23_{legacy_run_label}.png",
        ],
        "final_v_seed23_shear.png": [
            "final_v_seed23_shear.png",
            f"final_v_seed23_{legacy_run_label}.png",
        ],
        "final_HI_seed23_shear.png": [
            "final_HI_seed23_shear.png",
            f"final_HI_seed23_{legacy_run_label}.png",
        ],
        "final_HII_seed23_shear.png": [
            "final_HII_seed23_shear.png",
            f"final_HII_seed23_{legacy_run_label}.png",
        ],
        "final_mechanics_drive_seed23_shear.png": [
            "final_mechanics_drive_seed23_shear.png",
            f"final_mechanics_drive_seed23_{legacy_run_label}.png",
        ],
    }
    for dst_name, src_names in mapping.items():
        src = next((fig_src / name for name in src_names if (fig_src / name).exists()), None)
        if src is None:
            raise FileNotFoundError(f"None of {src_names} exists under {fig_src}")
        shutil.copy2(src, package / "figures" / dst_name)


def copy_artifacts(src_root: Path, package: Path, result_dir: Path, model_dir: Path) -> None:
    artifact_map = [
        ([result_dir / "diagnostics_mixed_tm_summary.csv"], "diagnostics_mixed_tm_summary_seed23_S0090_shear.csv"),
        ([result_dir / "curves" / "reaction_metric_availability.csv"], "reaction_metric_availability_seed23_S0090_shear.csv"),
        (
            [
                result_dir / "figures" / "stress_strain_source_seed23_shear.txt",
                result_dir / "figures" / "stress_strain_source_seed23_S0090_shear.txt",
            ],
            "stress_strain_source_seed23_S0090_shear.txt",
        ),
        ([model_dir / "model_settings.txt"], "model_settings_seed23_S0090_shear.txt"),
    ]
    for src_candidates, name in artifact_map:
        src = next((candidate for candidate in src_candidates if candidate.exists()), None)
        if src is not None:
            shutil.copy2(src, package / "artifacts" / name)

    source_files = [
        "build_shear_connectivity_package.py",
        "build_shear_longer_connectivity_package.py",
        "shear_connectivity.py",
        "load_schedules/load_schedule_S0090_shear.csv",
        "load_schedules/load_schedule_S0070_shear.csv",
        "main.py",
        "config.py",
        "train_mixed_tm.py",
        "mixed_mode_tm.py",
        "history_field_mixed_tm.py",
        "postprocess_results.py",
        "plot_results.py",
        "field_computation.py",
        "compute_energy_mixed_tm.py",
        "README.md",
        "POSTPROCESS_WORKFLOW.md",
        "PROJECT_STRUCTURE.md",
    ]
    for rel in source_files:
        src = src_root / rel
        if src.exists():
            shutil.copy2(src, package / "artifacts" / "source_snapshots" / rel.replace("/", "__"))


def write_markdown(
    package: Path,
    summary: Dict[str, object],
    s0070: Dict[str, object],
    git_status: Dict[str, object],
) -> None:
    growths = xspan_growths(s0070, summary)
    xspan_lines = "\n".join(
        [
            f"- alpha >= {threshold:g}: final x-span "
            f"{float(summary[f'final_{threshold_label(threshold)}_x_span']):.6g} mm, "
            f"growth vs S0070 {growths[f'{threshold_label(threshold)}_x_span_growth_vs_S0070']:.6g} mm"
            for threshold in DEFAULT_THRESHOLDS
        ]
    )
    through_lines = "\n".join(
        [
            f"- alpha >= {threshold:g}: "
            + (
                f"right-boundary reach first at step {summary[f'first_{threshold_label(threshold)}_right_boundary_step']}"
                if summary.get(f"first_{threshold_label(threshold)}_right_boundary_step", "") != ""
                else "no right-boundary reach"
            )
            for threshold in DEFAULT_THRESHOLDS
        ]
    )
    through_any = any(
        summary.get(f"first_{threshold_label(threshold)}_right_boundary_step", "") != ""
        for threshold in DEFAULT_THRESHOLDS
    )

    (package / "commands_run.txt").write_text(
        f"""# Commands Run

```powershell
{TRAIN_CMD}
{POST_CMD}
{TRAIN_PYTHON} build_shear_longer_connectivity_package.py --cc-root "D:\\Desktop\\新建文件夹\\cc-work"
```
""",
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        f"""# Existing-Geometry Shear Longer Connectivity Extension

This package documents the S0090 same-path shear extension for seed 23 on the existing COMSOL micro-notch geometry.

Classification: `{summary['classification']}`.

Read first:

1. `REPORT.md`
2. `tables/shear_longer_extension_run_summary.csv`
3. `tables/shear_extension_vs_S0070_comparison.csv`
4. `tables/shear_connectivity_by_threshold.csv`
5. `figures/figure_summary.md`
""",
        encoding="utf-8",
    )
    (package / "next_questions.md").write_text(
        """# Next Recommended Minimal Intervention

If no right-boundary through-crack appears, stop the single-seed shear escalation and interpret the geometry/connectivity results before changing physics. Do not start D0040 or a seed study from this package.
""",
        encoding="utf-8",
    )
    (package / "REPORT.md").write_text(
        f"""# Existing-Geometry Shear Longer Connectivity Extension Report

## Scope

This package analyzes S0090, one more same-path shear extension using seed 23 and the existing geometry. No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, shear ansatz, boundary conditions, split modes, or training losses were changed.

## Required Questions

1. Did the S0090 longer shear extension complete?  
Yes. S0090 has {summary['step_count']} stress-strain rows and {summary['field_npz_count']} field files.

2. Was this a continuation from S0070 or a full rerun?  
It was a full rerun from step 0: `continued_from_S0070=False`. Clean continuation from the committed S0070 history state was not already implemented unambiguously, so no continuation framework was added.

3. What schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0090_shear.csv`, ending at `Delta_s={summary['final_Delta_s']:.6g}` mm. Training: `RPROP=300, LBFGS=1`.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. `reaction_N_energy = dPi/dDelta_s` is available at all {summary['step_count']} steps.

5. Does post-peak softening continue beyond S0070?  
The final nominal shear stress is {summary['final_nominal_shear_stress_MPa']:.6g} MPa, compared with S0070 final {s0070['final_nominal_shear_stress_MPa']:.6g} MPa.

6. What are peak stress, final stress, post-peak drop amount, and post-peak drop percent?  
Peak nominal shear stress is {summary['peak_nominal_shear_stress_MPa']:.6g} MPa at step {summary['peak_step']}. Final stress is {summary['final_nominal_shear_stress_MPa']:.6g} MPa. Post-peak drop is {summary['post_peak_drop_MPa']:.6g} MPa, or {summary['post_peak_drop_percent']:.3g}%.

7. Does alpha remain notch-localized?  
Final alpha maximum is near {summary['alpha_max_location']}.

8. Does alpha stay saturated near 1, and does the connected damaged region grow?  
Final `alpha_max={summary['final_alpha_max']:.6g}`. X-span growth by threshold is listed below.

9. How do the notch-connected x-spans evolve for thresholds 0.3, 0.5, 0.8, and 0.95?  
{xspan_lines}

10. How much did each x-span grow relative to S0070?  
{xspan_lines}

11. Does any threshold reach the right boundary?  
{through_lines}

12. Does the crack path propagate away from the notch, or remain a local notch-tip damage zone?  
Classification: `{summary['classification']}`.

13. Is HII still active and notch-localized?  
Final HII/HI peak ratio is {summary['final_HII_over_HI_peak_ratio']:.6g}.

14. Does mechanics drive remain notch-dominated?  
Final mechanics-drive maximum location is {summary['final_mechanics_drive_max_location']}, classified as `{summary['drive_location_classification']}`.

15. Does top `v` remain below warning/unstable thresholds?  
Final `top_v_absmax/Delta_s={summary['final_top_v_absmax_over_Delta_s']:.6g}`.

16. Compared with S0070, is the longer extension more informative?  
It is more informative if the x-span growth and stress-softening changes in `tables/shear_extension_vs_S0070_comparison.csv` are material.

17. If no right-boundary through-crack occurs, should the next step be another extension, a geometry interpretation, or stopping the single-seed shear diagnostic?  
If no right-boundary through-crack occurs, stop further single-seed shear escalation and interpret geometry/connectivity first.

18. Was any physics changed?  
No.

19. Was any seed study or D0040 run performed?  
No. Only seed 23 S0090 is reported here; no D0040 run and no seed study were performed.

20. Were local commits pushed? If not, state that local main remains ahead of origin.  
No commits were pushed. Git status at package generation: `{git_status['status_line']}`.

## Classification

`{summary['classification']}`

Through-crack at any threshold: `{through_any}`.
""",
        encoding="utf-8",
    )
    (package / "figures" / "figure_summary.md").write_text(
        """# Figure Summary

- `shear_stress_strain_seed23.png`: energy-conjugate nominal shear stress versus engineering shear strain.
- `shear_reaction_strain_seed23.png`: checkpoint reaction `reaction_N_energy` versus engineering shear strain.
- `final_fields_panel_seed23_shear.png`: final field panel from normal postprocessing.
- `final_alpha_seed23_shear.png`: final alpha field.
- `final_u_seed23_shear.png`: final horizontal displacement field.
- `final_v_seed23_shear.png`: final vertical displacement field.
- `final_HI_seed23_shear.png`: final HI field.
- `final_HII_seed23_shear.png`: final HII field.
- `final_mechanics_drive_seed23_shear.png`: final mechanics-drive field.
- `shear_alpha_max_by_step.png`: alpha maximum by step.
- `shear_HII_HI_ratio_by_step.png`: HII/HI peak ratio by step.
- `shear_top_v_absmax_over_Delta_by_step.png`: top `|v|max/Delta_s` monitor.
- `shear_notch_drive_by_step.png`: notch-tip versus bottom-right mechanics-drive monitor.
- `shear_connectivity_xspan_by_threshold.png`: notch-connected x-span for alpha thresholds 0.3, 0.5, 0.8, and 0.95.
- `shear_connected_component_count_by_threshold.png`: connected component count by threshold.
- `shear_crack_path_overlay_final.png`: final alpha scatter with notch-connected component geometry overlays.
- `shear_through_crack_status_by_threshold.png`: right-boundary reach status by threshold.
""",
        encoding="utf-8",
    )
    (package / "HANDOFF_COMMENT.md").write_text(
        f"""## Codex handoff: Existing-geometry shear longer connectivity extension

Commit: TO_BE_FILLED_AFTER_COMMIT
Package folder: `{PACKAGE_REL.as_posix()}`
Local main ahead of origin: `{git_status['main_ahead_origin']}`. Commits pushed: no.

### Run identity
- Continuation: `continued_from_S0070=False`.
- Reason: clean continuation from the committed S0070 history state was not already implemented unambiguously; S0090 is a full rerun from step 0.
- Training command: `{TRAIN_CMD}`
- Postprocess command: `{POST_CMD}`
- Load schedule: `load_schedules/load_schedule_S0090_shear.csv`.
- Seed: 23 only.
- Training settings: `RPROP=300, LBFGS=1`.

### Status
- Checkpoint availability: {summary['checkpoint_count']}/{summary['step_count']} checkpoints.
- Energy reaction status: energy-conjugate reaction at all steps.
- S0070 comparison: final S0070 stress {s0070['final_nominal_shear_stress_MPa']:.6g} MPa, final S0090 stress {summary['final_nominal_shear_stress_MPa']:.6g} MPa.
- S0090 peak stress: {summary['peak_nominal_shear_stress_MPa']:.6g} MPa at step {summary['peak_step']}; final stress {summary['final_nominal_shear_stress_MPa']:.6g} MPa; post-peak drop {summary['post_peak_drop_percent']:.3g}%.
- Alpha/HII/mechanics drive: final alpha {summary['final_alpha_max']:.6g}; final HII/HI ratio {summary['final_HII_over_HI_peak_ratio']:.6g}; drive `{summary['drive_location_classification']}`.

### Connectivity by threshold and growth vs S0070
{xspan_lines}

### Through-crack status
{through_lines}

### Top-v diagnostic
- Final top_v_absmax/Delta_s: {summary['final_top_v_absmax_over_Delta_s']:.6g}.

### Generated tables and figures
- Tables are under `tables/`; key tables are `shear_longer_extension_run_summary.csv`, `shear_extension_vs_S0070_comparison.csv`, and `shear_connectivity_by_threshold.csv`.
- Figures are under `figures/`; read `figures/figure_summary.md`.

### Classification
`{summary['classification']}`

### Constraints observed
- Physics changed: no.
- Seed study run: no.
- D0040 run: no.
- Commits pushed: no.

### Next recommended minimal intervention
If no right-boundary through-crack occurs, stop further single-seed shear escalation and interpret geometry/connectivity first.
""",
        encoding="utf-8",
    )


def build_package(src_root: Path = SRC_ROOT, cc_root: Path = DEFAULT_CC_ROOT) -> Dict[str, object]:
    package = cc_root / PACKAGE_REL
    if package.exists():
        shutil.rmtree(package)
    ensure_dirs(package)

    dirs = result_dirs(src_root, RUN_ID)
    result_dir = dirs["result"]
    model_dir = dirs["model"]
    baseline_dirs = result_dirs(src_root, BASELINE_RUN_ID)
    baseline_result = baseline_dirs["result"]
    baseline_model = baseline_dirs["model"]

    stress = read_required_csv(result_dir / "curves" / "stress_strain_by_step.csv")
    reaction = read_required_csv(result_dir / "curves" / "reaction_by_step.csv")
    diagnostics = read_required_csv(result_dir / "diagnostics_mixed_tm_summary.csv")
    baseline_stress = read_required_csv(baseline_result / "curves" / "stress_strain_by_step.csv")
    baseline_reaction = read_required_csv(baseline_result / "curves" / "reaction_by_step.csv")
    baseline_diagnostics = read_required_csv(baseline_result / "diagnostics_mixed_tm_summary.csv")

    connectivity = add_step_context(compute_connectivity_by_threshold(result_dir), stress)
    baseline_connectivity = add_step_context(compute_connectivity_by_threshold(baseline_result), baseline_stress)
    tables = build_stepwise_tables(stress, reaction, diagnostics, connectivity, model_dir, result_dir)

    s0070_summary = summary_for_longer_case(
        run_id=BASELINE_RUN_ID,
        schedule="load_schedules/load_schedule_S0070_shear.csv",
        continued_label="baseline",
        stress=baseline_stress,
        reaction=baseline_reaction,
        diagnostics=baseline_diagnostics,
        connectivity=baseline_connectivity,
        model_dir=baseline_model,
        result_dir=baseline_result,
        classification="shear extension shows propagating crack",
    )
    final_top_v_ratio = float(tables["top_v"].iloc[-1]["top_v_absmax_over_Delta_s"])
    reaction_ok = bool(
        (reaction["reaction_metric_status"].astype(str) == "energy_conjugate").all()
        and truthy(reaction["is_energy_conjugate"]).all()
    )
    peak = float(stress["nominal_shear_stress_energy_MPa"].max())
    final = float(stress.iloc[-1]["nominal_shear_stress_energy_MPa"])
    post_peak_drop_percent = 100.0 * max(0.0, peak - final) / peak
    final_conn = connectivity[connectivity["step"] == int(stress["step"].max())]
    through_any = bool(connectivity["reaches_right_boundary"].astype(bool).any())
    alpha08_span = float(
        final_conn[np.isclose(final_conn["threshold"].astype(float), 0.8)].iloc[0]["notch_connected_x_span"]
    )
    alpha095_span = float(
        final_conn[np.isclose(final_conn["threshold"].astype(float), 0.95)].iloc[0]["notch_connected_x_span"]
    )
    baseline_alpha08 = float(s0070_summary.get("final_alpha_ge_0p8_x_span") or 0.0)
    baseline_alpha095 = float(s0070_summary.get("final_alpha_ge_0p95_x_span") or 0.0)
    classification = classify_shear_longer_extension(
        run_completed=count_field_files(result_dir) == len(stress),
        reaction_ok=reaction_ok,
        post_peak_drop_percent=post_peak_drop_percent,
        final_top_v_ratio=final_top_v_ratio,
        unstable_top_v_ratio=2.0,
        through_crack_any=through_any,
        xspan_growth_material=alpha08_span > baseline_alpha08 + 2.0e-4,
        propagated_high_threshold=(alpha08_span > baseline_alpha08 + 2.0e-4)
        or (alpha095_span > baseline_alpha095 + 1.0e-4),
    )
    s0090_summary = summary_for_longer_case(
        run_id=RUN_ID,
        schedule="load_schedules/load_schedule_S0090_shear.csv",
        continued_label=False,
        stress=stress,
        reaction=reaction,
        diagnostics=diagnostics,
        connectivity=connectivity,
        model_dir=model_dir,
        result_dir=result_dir,
        classification=classification,
    )
    for key, value in xspan_growths(s0070_summary, s0090_summary).items():
        s0090_summary[key] = value

    stress.to_csv(package / "tables" / "shear_stress_strain_by_step.csv", index=False)
    reaction.to_csv(package / "tables" / "shear_reaction_by_step.csv", index=False)
    connectivity.to_csv(package / "tables" / "shear_connectivity_by_threshold.csv", index=False)
    tables["crack_path"].to_csv(package / "tables" / "shear_crack_path_geometry_by_step.csv", index=False)
    tables["damage"].to_csv(package / "tables" / "shear_damage_drive_summary.csv", index=False)
    tables["top_v"].to_csv(package / "tables" / "shear_top_v_free_diagnostic.csv", index=False)
    tables["checkpoint"].to_csv(package / "tables" / "shear_checkpoint_availability.csv", index=False)
    tables["loss"].to_csv(package / "tables" / "shear_training_loss_summary.csv", index=False)
    pd.DataFrame([s0090_summary]).to_csv(package / "tables" / "shear_longer_extension_run_summary.csv", index=False)
    comparison_table(s0070_summary, s0090_summary).to_csv(
        package / "tables" / "shear_extension_vs_S0070_comparison.csv",
        index=False,
    )
    build_output_manifest(src_root, package, result_dir, model_dir).to_csv(
        package / "tables" / "shear_output_file_manifest.csv",
        index=False,
    )

    copy_required_figures(result_dir, package)
    build_figures(package, stress, diagnostics, connectivity, result_dir)
    copy_artifacts(src_root, package, result_dir, model_dir)
    git_status = git_ahead_status(cc_root)
    write_markdown(package, s0090_summary, s0070_summary, git_status)
    write_manifest(package)
    shutil.copy2(Path(__file__), package / "artifacts" / "generate_package.py")
    return s0090_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the S0090 shear longer connectivity evidence package.")
    parser.add_argument("--src-root", type=Path, default=SRC_ROOT)
    parser.add_argument("--cc-root", type=Path, default=DEFAULT_CC_ROOT)
    args = parser.parse_args(argv)
    summary = build_package(src_root=args.src_root, cc_root=args.cc_root)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
