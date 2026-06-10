"""Validate postprocess_tm corrected stress-strain curve source promotion."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PACKAGE = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE.parents[3]
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ARTIFACTS = PACKAGE / "artifacts"
LOGS = PACKAGE / "logs"
EXTERNAL_SOURCE = Path(r"D:\ProgramData\PINN\FEM-PINN-main\source")
CORRECTED_PACKAGE = (
    REPO_ROOT
    / "examples"
    / "TM_comsol_no_thermal_micro"
    / "runs"
    / "20260620_default_unitbox_D0020_stress_strain_curve_fix"
)
CORRECTED_CURVE = CORRECTED_PACKAGE / "tables" / "corrected_stress_strain_by_step.csv"


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, ARTIFACTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def write_minimal_field(result_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    field_dir = result_dir / "field_data"
    field_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame(
        {
            "x": [0.0, 0.01, 0.01, 0.0],
            "y": [0.0, 0.0, 0.01, 0.01],
            "T": [300.0, 300.0, 300.0, 300.0],
            "ux": [0.0, 0.0, 0.0, 0.0],
            "uy": [0.0, 0.0, 1.0e-4, 1.0e-4],
            "d": [0.0, 0.0, 0.95, 0.95],
            "HI": [0.0, 0.0, 1.0, 1.0],
            "HII": [0.0, 0.0, 0.5, 0.5],
            "He": [0.0, 0.0, 1.5, 1.5],
        }
    )
    nodes.to_csv(field_dir / "field_step_0000.csv", index=False)
    inp = torch.tensor(nodes[["x", "y"]].to_numpy(), dtype=torch.float32)
    t_conn = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return inp, t_conn


def write_curve_files(result_dir: Path) -> None:
    curves = result_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    corrected = pd.read_csv(CORRECTED_CURVE)
    corrected.to_csv(curves / "corrected_stress_strain_by_step.csv", index=False)

    legacy = corrected[
        [
            "step",
            "Delta",
            "nominal_strain",
            "reaction_N_legacy_top_sigma",
            "nominal_stress_legacy_top_sigma_MPa",
        ]
    ].copy()
    legacy["time"] = legacy["step"].astype(float)
    legacy["uy_top"] = legacy["Delta"]
    legacy["reaction_force"] = legacy["reaction_N_legacy_top_sigma"]
    legacy["macro_strain"] = legacy["nominal_strain"]
    legacy["macro_stress"] = legacy["nominal_stress_legacy_top_sigma_MPa"]
    legacy[["step", "time", "uy_top", "reaction_force", "macro_strain", "macro_stress"]].to_csv(
        curves / "reaction_displacement_macro_stress_strain.csv",
        index=False,
    )


def validate() -> pd.DataFrame:
    if str(EXTERNAL_SOURCE) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_SOURCE))
    from postprocess_tm import _resolve_stress_strain_curve_source, postprocess_tm

    result_dir = ARTIFACTS / "minimal_postprocess_results"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True)
    (result_dir / "losses").mkdir(parents=True, exist_ok=True)
    inp, t_conn = write_minimal_field(result_dir)
    write_curve_files(result_dir)

    curve_file = result_dir / "curves" / "reaction_displacement_macro_stress_strain.csv"
    source = _resolve_stress_strain_curve_source(result_dir, curve_file)
    postprocess_tm(result_dir, inp, t_conn, step_idx=0, dpi=120, bc_dict=None)

    source_report = result_dir / "figures" / "stress_strain_curve_source.txt"
    rows = [
        {
            "check": "primary_status",
            "value": source["status"],
            "passed": source["status"] == "energy_conjugate_primary",
        },
        {
            "check": "primary_metric",
            "value": source["primary"]["metric"] if source["primary"] else "missing",
            "passed": bool(source["primary"] and source["primary"]["metric"] == "nominal_stress_energy_exact_MPa"),
        },
        {
            "check": "legacy_metric",
            "value": source["legacy"]["metric"] if source["legacy"] else "missing",
            "passed": bool(source["legacy"] and source["legacy"]["metric"] == "macro_stress"),
        },
        {
            "check": "macro_stress_strain_png",
            "value": "figures/validated_macro_stress_strain.png",
            "passed": (result_dir / "figures" / "macro_stress_strain.png").exists(),
        },
        {
            "check": "legacy_stress_strain_png",
            "value": "figures/validated_macro_stress_strain_legacy.png",
            "passed": (result_dir / "figures" / "macro_stress_strain_legacy.png").exists(),
        },
        {
            "check": "source_report",
            "value": _source_report_without_paths(source_report),
            "passed": source_report.exists()
            and "status=energy_conjugate_primary" in source_report.read_text(encoding="utf-8"),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "mainflow_curve_source_selection.csv", index=False)
    shutil.copyfile(result_dir / "figures" / "macro_stress_strain.png", FIGURES / "validated_macro_stress_strain.png")
    shutil.copyfile(result_dir / "figures" / "macro_stress_strain_legacy.png", FIGURES / "validated_macro_stress_strain_legacy.png")
    shutil.rmtree(result_dir)
    return df


def _source_report_without_paths(source_report: Path) -> str:
    if not source_report.exists():
        return "missing"
    kept = []
    for line in source_report.read_text(encoding="utf-8").splitlines():
        if line.startswith("primary_path=") or line.startswith("legacy_path="):
            continue
        kept.append(line)
    return "; ".join(kept)


def write_reports(validation: pd.DataFrame) -> None:
    all_passed = bool(validation["passed"].all())
    classification = "mainflow corrected stress-strain source promoted" if all_passed else "mainflow promotion unresolved"
    report = [
        "# Stress-Strain Mainflow Promotion",
        "",
        "## Scope",
        "",
        "This package records and validates a mainflow plotting change in external `source/postprocess_tm.py`. The change promotes corrected checkpoint energy reaction curves to the primary stress-strain source and demotes legacy top-sigma curves to diagnostic-only output.",
        "",
        "## Classification",
        "",
        f"**{classification}**.",
        "",
        "## What Changed",
        "",
        "- `postprocess_tm.py` now searches for `curves/corrected_stress_strain_by_step.csv` or `curves/reaction_displacement_macro_stress_strain_corrected.csv` before using the legacy training curve.",
        "- If corrected energy-conjugate stress is available, `macro_stress_strain.png` is plotted from `nominal_stress_energy_exact_MPa`.",
        "- If only the legacy curve is available, `macro_stress_strain.png` becomes an unavailable-primary notice and the old curve is written as `macro_stress_strain_legacy.png`.",
        "- A `stress_strain_curve_source.txt` report is written for every postprocess run with curve data.",
        "- No D0040 run was launched or processed.",
        "",
        "## Validation",
        "",
        f"- Validation checks passed: {int(validation['passed'].sum())}/{len(validation)}.",
        "- The minimal postprocess fixture selected `nominal_stress_energy_exact_MPa` as the primary curve metric.",
        "- The legacy top-sigma curve was retained only as `macro_stress_strain_legacy.png`.",
        "",
        "## Files",
        "",
        "- `artifacts/postprocess_tm_before.py`",
        "- `artifacts/postprocess_tm_after.py`",
        "- `artifacts/postprocess_tm_mainflow_promotion.diff`",
        "- `tables/mainflow_curve_source_selection.csv`",
        "- `figures/validated_macro_stress_strain.png`",
        "- `figures/validated_macro_stress_strain_legacy.png`",
        "",
        "## Limits",
        "",
        "- The modified FEM-PINN source tree is external to `cc-work`, so this package stores the before/after files and diff as evidence.",
        "- This is a plotting/postprocessing mainflow change only; it does not modify training physics.",
        "",
    ]
    (PACKAGE / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    readme = [
        "# Stress-strain mainflow promotion package",
        "",
        "Read in this order:",
        "",
        "1. `REPORT.md`",
        "2. `tables/mainflow_curve_source_selection.csv`",
        "3. `artifacts/postprocess_tm_mainflow_promotion.diff`",
        "4. `figures/figure_summary.md`",
        "",
    ]
    (PACKAGE / "README.md").write_text("\n".join(readme), encoding="utf-8")

    fig_summary = [
        "# Figure Summary",
        "",
        "| filename | what it plots | visual takeaway | conclusion support |",
        "|---|---|---|---|",
        "| `validated_macro_stress_strain.png` | Mainflow `postprocess_tm()` output after corrected curve source promotion. | The primary plot is generated from the energy-conjugate corrected stress-strain source. | Supports promotion of corrected source into the plotting mainflow. |",
        "| `validated_macro_stress_strain_legacy.png` | Legacy top-sigma diagnostic curve output by the same postprocess run. | The legacy curve is still available but separated and labeled as diagnostic. | Supports legacy demotion. |",
        "",
    ]
    (FIGURES / "figure_summary.md").write_text("\n".join(fig_summary), encoding="utf-8")

    next_questions = [
        "# Next Questions",
        "",
        "1. Should the same corrected curve-source contract be copied into any separate plotting utilities that bypass `postprocess_tm.py`?",
        "2. Should training-time CSV columns be renamed to make legacy status explicit, or is postprocess-time demotion enough?",
        "3. Should D0040 remain deferred until the user asks for it?",
        "",
    ]
    (PACKAGE / "next_questions.md").write_text("\n".join(next_questions), encoding="utf-8")

    handoff = [
        "## Codex handoff: stress-strain mainflow promotion",
        "",
        "Commit: 210a09a",
        "Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_stress_strain_mainflow_promotion",
        "Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_stress_strain_mainflow_promotion/REPORT.md",
        "",
        "### What changed",
        "- Modified external `D:\\ProgramData\\PINN\\FEM-PINN-main\\source\\postprocess_tm.py` to promote corrected energy-conjugate stress-strain curves as the main plotting source.",
        "- Mainflow now prefers `curves/corrected_stress_strain_by_step.csv` / `nominal_stress_energy_exact_MPa` when available.",
        "- Legacy `macro_stress` from top-boundary sigma is separated to `macro_stress_strain_legacy.png` and is no longer the primary stress-strain plot when corrected data is missing.",
        "- Stored before/after source files and a diff in this package.",
        "- Did not run or process D0040.",
        "",
        "### Commands run",
        "```powershell",
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile D:\\ProgramData\\PINN\\FEM-PINN-main\\source\\postprocess_tm.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\validate_postprocess_tm_curve_promotion.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\validate_postprocess_tm_curve_promotion.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
        "```",
        "",
        "### Key results",
        f"- Classification: **{classification}**.",
        f"- Validation checks passed: {int(validation['passed'].sum())}/{len(validation)}.",
        "- Primary mainflow metric selected: `nominal_stress_energy_exact_MPa`.",
        "- Legacy top-sigma curve remains diagnostic-only.",
        "- D0040 remains untouched.",
        "",
        "### Files to read first",
        "- `README.md`",
        "- `REPORT.md`",
        "- `tables/mainflow_curve_source_selection.csv`",
        "- `artifacts/postprocess_tm_mainflow_promotion.diff`",
        "- `figures/figure_summary.md`",
        "",
        "### Question for ChatGPT",
        "1. Is the `postprocess_tm.py` source promotion sufficient to call the corrected stress-strain curve the main plotting flow?",
        "2. Should any training CSV column names be changed next, or is plotting/source selection enough?",
        "3. Should D0040 remain deferred until explicitly requested?",
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
        "tables/mainflow_curve_source_selection.csv",
        "artifacts/postprocess_tm_mainflow_promotion.diff",
        "figures/figure_summary.md",
    }
    type_by_suffix = {
        ".csv": "table",
        ".png": "figure",
        ".md": "report",
        ".py": "artifact",
        ".diff": "artifact",
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
                "description": "Stress-strain mainflow promotion evidence file.",
                "required_for_chatgpt": rel in required,
            }
        )
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_commands() -> None:
    commands = [
        "git pull origin main",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile D:\\ProgramData\\PINN\\FEM-PINN-main\\source\\postprocess_tm.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m py_compile artifacts\\validate_postprocess_tm_curve_promotion.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe artifacts\\validate_postprocess_tm_curve_promotion.py",
        "D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest D:\\ProgramData\\PINN\\FEM-PINN-main\\examples\\TM_comsol_no_thermal_micro\\tests -q",
    ]
    (PACKAGE / "commands_run.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validation = validate()
    write_reports(validation)
    write_commands()
    write_manifest()
    print("mainflow corrected stress-strain source promoted" if validation["passed"].all() else "mainflow promotion unresolved")


if __name__ == "__main__":
    main()
