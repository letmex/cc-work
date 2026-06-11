# Handoff: Constant-k0 Heat PDE Phase 1

## Status

Final classification: `constant-k0 heat PDE phase1 implemented and patch tests passed`

Commit hash:

- Primary implementation commit: `b289127b664ee0e8770fcbe91e9e8b09ca1908c4` (`Implement constant-k0 heat PDE phase1`).
- Handoff sync commit: recorded in final Codex response; this file does not chase its own sync hash.

Push status:

- Primary implementation commit push: pushed to `origin/main`.
- Final status after primary push: `## main...origin/main`, ahead/behind `0 0`.
- Final HEAD known at handoff-sync edit time: `b289127b664ee0e8770fcbe91e9e8b09ca1908c4`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1`
- Report: `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/MANIFEST.json`

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
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_constant_k0_heat_pde_patch.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/REPORT.md`
2. `examples/TM_comsol_thermal_micro/heat_pde.py`
3. `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`
4. `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/tables/unit_conversion_summary.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/tables/heat_residual_sign_convention.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1/tables/patch_test_results.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Review this Phase 1 implementation package. If approved, decide the next narrow heat task: additional boundary-condition patch tests or a solved-temperature representation design. Do not implement damage-dependent conductivity or run heat-fracture diagnostics without separate approval.
