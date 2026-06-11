import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
TABLE_DIR = PACKAGE_DIR / "tables"
FIGURE_DIR = PACKAGE_DIR / "figures"

PACKAGE_REL = "examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1"
REPORT_REL = f"{PACKAGE_REL}/REPORT.md"
HANDOFF_REL = f"{PACKAGE_REL}/HANDOFF_COMMENT.md"

CLASSIFICATION = "constant-k0 heat PDE phase1 implemented and patch tests passed"


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def implementation_summary_rows():
    return [
        {
            "item": "isolated heat PDE module",
            "status": "implemented",
            "details": "`heat_pde.py` provides constant-k0 SI heat residual, flux, divergence, and mm-to-m conversion utilities.",
            "default_behavior": "inactive unless imported directly by focused tests or future approved code",
            "risk_or_note": "Not threaded into training or mechanics loss.",
        },
        {
            "item": "steady residual",
            "status": "implemented",
            "details": "`steady_heat_residual_W_per_m3` computes `-div(k0*grad(T)) - Q` through Fourier flux divergence.",
            "default_behavior": "`Q=0`, `k0=418 W/m/K`, coordinates interpreted as mm unless specified otherwise",
            "risk_or_note": "Uses autograd; input coordinates must require gradients for nonconstant derivative tests.",
        },
        {
            "item": "transient residual",
            "status": "implemented",
            "details": "`transient_heat_residual_W_per_m3` computes `rho*c*dTdt - div(k0*grad(T)) - Q`.",
            "default_behavior": "available as utility only; not used by training",
            "risk_or_note": "No physical time schedule is introduced.",
        },
        {
            "item": "mm-to-m conversion",
            "status": "implemented",
            "details": "`coords_mm_to_m` and derivative chain-rule scaling convert gradients from mm input coordinates to K/m.",
            "default_behavior": "`coordinate_unit='mm'` explicit default for project mesh coordinates",
            "risk_or_note": "Patch test checks against direct meter-coordinate computation.",
        },
        {
            "item": "damage-dependent conductivity",
            "status": "not implemented",
            "details": "No alpha, damage, degradation, or k(d) inputs exist in the Phase 1 heat PDE API.",
            "default_behavior": "constant k0 only",
            "risk_or_note": "Guard test scans public signatures and source tokens.",
        },
        {
            "item": "prescribed-temperature mechanics branch",
            "status": "unchanged",
            "details": "Existing thermal strain relation remains in `thermal_prescribed.py` and `compute_energy_mixed_tm.py`.",
            "default_behavior": "`thermal_mode=off` remains default",
            "risk_or_note": "Existing prescribed thermal strain tests still pass.",
        },
    ]


def unit_conversion_rows():
    return [
        {
            "quantity": "temperature T",
            "implemented_unit": "K",
            "source_or_default": "test fields and future heat PDE utilities",
            "conversion": "none",
            "validation": "constant, linear, quadratic, and transient patch tests",
        },
        {
            "quantity": "coordinate x/y for heat PDE",
            "implemented_unit": "m internally",
            "source_or_default": "project mesh coordinates are mm",
            "conversion": "`x_m = x_mm * 1e-3`; derivatives divide by `1e-3` for first derivatives",
            "validation": "mm-to-m chain-rule patch test against direct meter-coordinate computation",
        },
        {
            "quantity": "rho",
            "implemented_unit": "kg/m^3",
            "source_or_default": "`DEFAULT_THERMAL_RHO_KG_PER_M3 = 1040.0`",
            "conversion": "none inside heat PDE utilities",
            "validation": "transient manufactured residual test",
        },
        {
            "quantity": "specific heat",
            "implemented_unit": "J/kg/K",
            "source_or_default": "`DEFAULT_THERMAL_C_J_PER_KGK = 170.0`",
            "conversion": "none inside heat PDE utilities",
            "validation": "transient manufactured residual test",
        },
        {
            "quantity": "constant conductivity k0",
            "implemented_unit": "W/m/K",
            "source_or_default": "`DEFAULT_THERMAL_K0_W_PER_MK = 418.0`",
            "conversion": "none inside heat PDE utilities",
            "validation": "linear flux and quadratic source patch tests",
        },
        {
            "quantity": "heat source Q",
            "implemented_unit": "W/m^3",
            "source_or_default": "`DEFAULT_HEAT_SOURCE_Q_W_PER_M3 = 0.0`",
            "conversion": "caller must supply SI value",
            "validation": "quadratic steady manufactured source and transient manufactured source tests",
        },
        {
            "quantity": "heat residual",
            "implemented_unit": "W/m^3",
            "source_or_default": "steady and transient residual utilities",
            "conversion": "computed fully in SI heat units",
            "validation": "focused heat PDE patch test suite",
        },
    ]


def patch_test_rows():
    return [
        {
            "test_id": "PT01",
            "test_name": "constant temperature steady residual",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "grad T = 0; steady residual = 0",
            "result": "passed",
            "notes": "Covers zero-gradient baseline and autograd constant-gradient handling.",
        },
        {
            "test_id": "PT02",
            "test_name": "linear 1D steady residual and heat flux",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "steady residual = 0; q_x = -k0*a",
            "result": "passed",
            "notes": "Validates Fourier sign convention.",
        },
        {
            "test_id": "PT03",
            "test_name": "quadratic manufactured steady source",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "`-div(k0 grad T) = -2*k0*b`; matching Q cancels residual",
            "result": "passed",
            "notes": "Documents residual sign convention.",
        },
        {
            "test_id": "PT04",
            "test_name": "boundary normal heat flux",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "`q dot n = -k0 grad(T) dot n`",
            "result": "passed",
            "notes": "Checks zero flux for constant T and sign for linear T.",
        },
        {
            "test_id": "PT05",
            "test_name": "transient uniform no-source",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "`dTdt=0`, `grad T=0`, transient residual = 0",
            "result": "passed",
            "notes": "Transient utility exists but remains inactive outside tests.",
        },
        {
            "test_id": "PT06",
            "test_name": "transient manufactured residual",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "matching Q cancels `rho*c*dTdt - div(k0 grad T)`",
            "result": "passed",
            "notes": "Covers rho*c scaling.",
        },
        {
            "test_id": "PT07",
            "test_name": "mm-to-m chain-rule",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "mm coordinate residual matches direct meter-coordinate residual",
            "result": "passed",
            "notes": "Would fail if 1e3 or 1e6 derivative factors were omitted.",
        },
        {
            "test_id": "PT08",
            "test_name": "no damage-dependent conductivity API",
            "command": "pytest test_constant_k0_heat_pde_patch.py",
            "expected": "no alpha/damage/k(d) API inputs or source tokens",
            "result": "passed",
            "notes": "Phase 1 remains constant-k0 only.",
        },
    ]


def sign_convention_rows():
    return [
        {
            "expression": "heat flux",
            "implemented_form": "`q = -k0 * grad(T)`",
            "unit": "W/m^2",
            "test_evidence": "linear 1D flux and normal flux tests",
        },
        {
            "expression": "steady residual",
            "implemented_form": "`-div(k0 * grad(T)) - Q`, implemented as `div(q) - Q`",
            "unit": "W/m^3",
            "test_evidence": "constant, linear, and quadratic manufactured source tests",
        },
        {
            "expression": "transient residual",
            "implemented_form": "`rho*c*dTdt - div(k0 * grad(T)) - Q`, implemented as `rho*c*dTdt + div(q) - Q`",
            "unit": "W/m^3",
            "test_evidence": "uniform transient and transient manufactured source tests",
        },
        {
            "expression": "quadratic source check",
            "implemented_form": "For `T=T0+b*x_m^2`, `-div(k0 grad T) = -2*k0*b`; choose `Q=-2*k0*b` for zero residual.",
            "unit": "W/m^3",
            "test_evidence": "quadratic manufactured steady source test",
        },
    ]


def source_files_changed_rows():
    return [
        {
            "path": "examples/TM_comsol_thermal_micro/heat_pde.py",
            "change_type": "added",
            "purpose": "Phase 1 constant-k0 SI heat PDE utilities",
            "behavior_scope": "inactive unless imported directly",
            "no_thermal_touched": "false",
        },
        {
            "path": "examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py",
            "change_type": "added",
            "purpose": "Focused analytical patch tests for SI heat PDE utilities",
            "behavior_scope": "tests only",
            "no_thermal_touched": "false",
        },
        {
            "path": "examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md",
            "change_type": "modified",
            "purpose": "Record Phase 1 implementation status and boundaries",
            "behavior_scope": "documentation only",
            "no_thermal_touched": "false",
        },
        {
            "path": PACKAGE_REL,
            "change_type": "added",
            "purpose": "Implementation report package",
            "behavior_scope": "documentation and tables only",
            "no_thermal_touched": "false",
        },
    ]


def guard_check_rows():
    return [
        {
            "guard": "damage-dependent conductivity",
            "expected": "not implemented",
            "result": "passed",
            "evidence": "No alpha/damage/k(d) parameters in public heat_pde API; guard test passed.",
        },
        {
            "guard": "heat PDE training coupling",
            "expected": "not implemented",
            "result": "passed",
            "evidence": "`heat_pde.py` is isolated; training files are unchanged.",
        },
        {
            "guard": "prescribed-temperature fallback",
            "expected": "still available and default-off",
            "result": "passed",
            "evidence": "Existing prescribed thermal strain patch tests passed.",
        },
        {
            "guard": "no-thermal project untouched",
            "expected": "no diff under examples/TM_comsol_no_thermal_micro",
            "result": "passed",
            "evidence": "`git diff --name-only -- examples/TM_comsol_no_thermal_micro` produced no output.",
        },
        {
            "guard": "broad diagnostics",
            "expected": "not run",
            "result": "passed",
            "evidence": "No D0040, seed, shear, S0110, A/B/C, heat-fracture diagnostic, or training run was performed.",
        },
    ]


def validation_results_rows():
    return [
        {
            "command": "git status",
            "result": "passed",
            "details": "Clean and up to date before implementation; final status reported in handoff/final response.",
        },
        {
            "command": "python recursive py_compile under examples/TM_comsol_thermal_micro",
            "result": "passed",
            "details": "All thermal Python files compiled successfully.",
        },
        {
            "command": "pytest examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py -q",
            "result": "passed",
            "details": "8 passed.",
        },
        {
            "command": "pytest examples/TM_comsol_thermal_micro/tests/test_prescribed_thermal_strain_patch.py -q",
            "result": "passed",
            "details": "Existing prescribed-temperature fallback tests passed.",
        },
        {
            "command": "package schema/file existence check",
            "result": "passed",
            "details": "Required files and CSV headers present.",
        },
        {
            "command": "git diff --check",
            "result": "passed",
            "details": "No whitespace errors.",
        },
        {
            "command": "git diff --name-only -- examples/TM_comsol_no_thermal_micro",
            "result": "passed",
            "details": "No output.",
        },
    ]


def next_steps_rows():
    return [
        {
            "step": "review Phase 1 package",
            "recommendation": "Read report, unit conversion table, sign convention table, and tests.",
            "blocked_by": "none",
            "do_not_do": "Do not start fracture diagnostics from this package alone.",
        },
        {
            "step": "decide next heat scope",
            "recommendation": "If approved, choose whether to add solved-temperature representation or additional BC tests next.",
            "blocked_by": "reviewer approval",
            "do_not_do": "Do not implement damage-dependent conductivity yet.",
        },
        {
            "step": "preserve fallback",
            "recommendation": "Keep prescribed-temperature branch and `thermal_mode=off` as regression checks.",
            "blocked_by": "none",
            "do_not_do": "Do not change material constants, l0, history, mechanics BCs, or reaction route.",
        },
    ]


TABLES = {
    "implementation_summary.csv": (
        ["item", "status", "details", "default_behavior", "risk_or_note"],
        implementation_summary_rows,
    ),
    "unit_conversion_summary.csv": (
        ["quantity", "implemented_unit", "source_or_default", "conversion", "validation"],
        unit_conversion_rows,
    ),
    "patch_test_results.csv": (
        ["test_id", "test_name", "command", "expected", "result", "notes"],
        patch_test_rows,
    ),
    "heat_residual_sign_convention.csv": (
        ["expression", "implemented_form", "unit", "test_evidence"],
        sign_convention_rows,
    ),
    "source_files_changed.csv": (
        ["path", "change_type", "purpose", "behavior_scope", "no_thermal_touched"],
        source_files_changed_rows,
    ),
    "guard_checks.csv": (
        ["guard", "expected", "result", "evidence"],
        guard_check_rows,
    ),
    "validation_results.csv": (
        ["command", "result", "details"],
        validation_results_rows,
    ),
    "next_steps.csv": (
        ["step", "recommendation", "blocked_by", "do_not_do"],
        next_steps_rows,
    ),
}


def report_text():
    return """# Constant-k0 Heat PDE Phase 1 Implementation

## 1. Purpose

Implement Phase 1 constant-conductivity heat PDE infrastructure inside `examples/TM_comsol_thermal_micro`, with explicit SI heat units, mm-to-m coordinate conversion, and focused analytical patch tests. This is implementation work, not another plan-only package.

## 2. What was implemented

Added `heat_pde.py`, an isolated utility module for constant-`k0` heat-transfer calculations. It provides:

- `coords_mm_to_m`
- `temperature_gradient_m`
- `heat_flux_W_per_m2`
- `divergence_m`
- `steady_heat_residual_W_per_m3`
- `transient_heat_residual_W_per_m3`
- `normal_heat_flux_W_per_m2`

The module is not threaded into fracture training, mechanics energy, history, checkpointing, or postprocessing.

## 3. Unit convention implemented

Heat PDE utilities use SI units internally:

- `T`: K
- length for heat gradients: m
- time: s
- `rho`: kg/m^3
- `c`: J/kg/K
- `k0`: W/m/K
- `Q`: W/m^3
- residual: W/m^3

Project mesh coordinates are in mm, so `coordinate_unit='mm'` applies `x_m = x_mm * 1e-3` through derivative chain-rule scaling. See `tables/unit_conversion_summary.csv`.

## 4. Heat residual sign convention

Fourier heat flux is:

```text
q = -k0 * grad(T)
```

The steady residual is:

```text
-div(k0 * grad(T)) - Q
```

implemented as:

```text
div(q) - Q
```

The transient residual is:

```text
rho*c*dTdt - div(k0 * grad(T)) - Q
```

implemented as:

```text
rho*c*dTdt + div(q) - Q
```

See `tables/heat_residual_sign_convention.csv`.

## 5. Patch tests added

Added `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`, covering constant temperature, linear 1D conduction, quadratic manufactured source, normal flux, transient uniform no-source, transient manufactured source, mm-to-m chain rule, and no damage-dependent conductivity API guard.

## 6. Patch test results

The focused heat PDE patch suite passed: `8 passed`. Existing prescribed thermal strain patch tests also passed, preserving the default-off prescribed-temperature mechanics branch.

See `tables/patch_test_results.csv` and `tables/validation_results.csv`.

## 7. What remains disabled by default

The new heat PDE utilities are inactive unless imported directly by tests or future approved code. No training loss, heat residual objective, solved temperature field, fracture diagnostic, boundary-condition CLI, checkpoint schema, or postprocess route was enabled.

## 8. Prescribed-temperature fallback status

The prescribed-temperature branch is unchanged. `thermal_mode=off` remains default, and the mechanics relation remains:

```text
delta_T = T - Tref
exx_e = exx - alpha_T * delta_T
eyy_e = eyy - alpha_T * delta_T
exy_e = exy
```

Existing prescribed thermal strain patch tests passed.

## 9. Damage-dependent conductivity guard

Damage-dependent conductivity remains unimplemented. The Phase 1 API has no `alpha`, damage, degradation, or `k(d)` conductivity inputs, and the focused guard test checks public signatures and source tokens.

## 10. Source files changed

Behavior source changed only by adding isolated `heat_pde.py`; no existing mechanics/training/source route was modified. Tests and documentation/package files were added, and `PROJECT_MEMORY.md` was updated.

See `tables/source_files_changed.csv`.

## 11. Validation commands and results

Required validation passed:

1. `git status`
2. recursive `py_compile` for Python files under `examples/TM_comsol_thermal_micro`
3. new focused heat PDE patch tests
4. existing prescribed thermal strain patch tests
5. package schema/file existence check
6. `git diff --check`
7. `git diff --name-only -- examples/TM_comsol_no_thermal_micro`

See `tables/validation_results.csv`.

## 12. Limitations

This phase does not implement damage-dependent conductivity, solved-temperature fracture training, broad thermal-mechanical diagnostics, D0040, seed studies, shear extension, S0110, material changes, `l0` changes, history changes, mechanics boundary-condition changes, or reaction-route changes.

The transient residual utility is analytical infrastructure only; no physical time/loading schedule was introduced.

## 13. Final classification

`constant-k0 heat PDE phase1 implemented and patch tests passed`

Phase 1 constant-conductivity heat PDE infrastructure was implemented with SI heat units, explicit mm-to-m coordinate conversion, and focused analytical patch tests. The new utilities remain inactive by default and do not change the prescribed-temperature mechanics route. Existing prescribed thermal strain tests still pass. Damage-dependent conductivity remains unimplemented and guarded.

## 14. Exact next recommended task

Review this Phase 1 package and tests. If approved, decide the next narrow heat task: either add additional boundary-condition patch tests or design a solved-temperature representation. Do not implement damage-dependent conductivity or run heat-fracture diagnostics until constant-k0 heat utilities and any future solved-T coupling are independently reviewed.
"""


def handoff_text():
    return f"""# Handoff: Constant-k0 Heat PDE Phase 1

## Status

Final classification: `{CLASSIFICATION}`

Commit hash:

- Primary implementation commit: `PENDING_PRIMARY_COMMIT`.
- Handoff sync commit: `PENDING_HANDOFF_SYNC_COMMIT`; do not chase this file's own sync hash indefinitely.

Push status:

- Primary implementation commit push: `PENDING_PUSH_STATUS`.
- Final status at generation time: `PENDING_FINAL_GIT_STATUS`.
- Final HEAD known at handoff-sync edit time: `PENDING_FINAL_HEAD`.

## Package

- Package path: `{PACKAGE_REL}`
- Report: `{REPORT_REL}`
- Manifest: `{PACKAGE_REL}/MANIFEST.json`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Added isolated Phase 1 constant-k0 heat PDE utilities in `examples/TM_comsol_thermal_micro/heat_pde.py`.
- Added focused patch tests in `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`.
- Did not run training, D0040, seed study, shear extension, S0110, broad A/B/C diagnostics, or heat-fracture diagnostics.
- Did not implement damage-dependent conductivity, solved-temperature fracture training, material changes, `l0` changes, history changes, mechanics boundary changes, loss changes, source model behavior changes, or reaction-route changes.

## Key Implementation Details

- Heat PDE utilities use SI heat units internally.
- Mesh coordinates supplied in mm are converted through derivative chain-rule scaling to meters.
- Steady residual sign: `-div(k0 * grad(T)) - Q`.
- Transient residual sign: `rho*c*dTdt - div(k0 * grad(T)) - Q`.
- Defaults: `rho=1040 kg/m^3`, `c=170 J/kg/K`, `k0=418 W/m/K`, `Q=0`.
- Conductivity is constant `k0`; no alpha/damage/k(d) conductivity API exists.
- Existing prescribed-temperature mechanics fallback remains unchanged and default-off.

## Validation To Report

- `git status`
- recursive py_compile under `examples/TM_comsol_thermal_micro`
- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_thermal_micro\\tests\\test_constant_k0_heat_pde_patch.py -q`
- `D:\\anaconda3\\envs\\torch_env\\python.exe -m pytest examples\\TM_comsol_thermal_micro\\tests\\test_prescribed_thermal_strain_patch.py -q`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples\\TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `{REPORT_REL}`
2. `examples/TM_comsol_thermal_micro/heat_pde.py`
3. `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`
4. `{PACKAGE_REL}/tables/unit_conversion_summary.csv`
5. `{PACKAGE_REL}/tables/heat_residual_sign_convention.csv`
6. `{PACKAGE_REL}/tables/patch_test_results.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Review this Phase 1 implementation package. If approved, decide the next narrow heat task: additional boundary-condition patch tests or a solved-temperature representation design. Do not implement damage-dependent conductivity or run heat-fracture diagnostics without separate approval.
"""


def manifest():
    return {
        "package": PACKAGE_REL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": CLASSIFICATION,
        "planning_only": False,
        "heat_pde_phase1_implemented": True,
        "constant_conductivity_only": True,
        "damage_dependent_conductivity_implemented": False,
        "training_run": False,
        "heat_fracture_diagnostic_run": False,
        "no_thermal_project_touched": False,
        "source_files": [
            "examples/TM_comsol_thermal_micro/heat_pde.py",
            "examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py",
            "examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md",
        ],
        "required_outputs": [
            "REPORT.md",
            "HANDOFF_COMMENT.md",
            "MANIFEST.json",
            "figures/figure_summary.md",
            *[f"tables/{name}" for name in sorted(TABLES.keys())],
        ],
        "unit_convention": {
            "temperature": "K",
            "length_for_heat_gradients": "m",
            "input_mesh_coordinates": "mm converted by x_m = x_mm * 1e-3",
            "time": "s",
            "rho": "kg/m^3",
            "c": "J/kg/K",
            "k0": "W/m/K",
            "Q": "W/m^3",
            "residual": "W/m^3",
        },
        "validation_summary": {
            "focused_heat_tests": "8 passed",
            "prescribed_thermal_strain_tests": "passed",
            "diff_check": "passed",
            "no_thermal_guard": "passed",
        },
    }


def figure_summary_text():
    return """# Figure Summary

No PNG figures were generated for this Phase 1 implementation package. The analytical residual behavior is fully documented by the patch-test tables and focused tests.
"""


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, (columns, rows_fn) in TABLES.items():
        write_csv(TABLE_DIR / name, rows_fn(), columns)
    write_text(PACKAGE_DIR / "REPORT.md", report_text())
    write_text(PACKAGE_DIR / "HANDOFF_COMMENT.md", handoff_text())
    write_text(FIGURE_DIR / "figure_summary.md", figure_summary_text())
    with open(PACKAGE_DIR / "MANIFEST.json", "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest(), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
