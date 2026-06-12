# Handoff: Constant-k0 Thermal Functional Phase 1

## Status

Final classification: `constant-k0 thermal functional phase1 implemented and tests passed`

Commit hash:

- Primary implementation commit: pending until exact-path commit.
- Handoff sync commit: this file may be updated once after the primary push; do
  not chase the sync commit self-hash.

Push status:

- Pending until primary implementation commit is pushed to `origin/main`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1`
- Report: `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/MANIFEST.json`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Added thermal functional / weak-form pointwise density utilities to
  `examples/TM_comsol_thermal_micro/heat_pde.py`.
- Kept strong-form residual utilities as diagnostics and patch-test sanity
  checks.
- Made the autograd gradient helper stricter so detached nonconstant fields and
  non-gradient coordinates raise clear errors.
- Did not run training, D0040, seed study, shear extension, S0110, heat-fracture
  diagnostics, heat PDE training, or solved-temperature training.
- Did not implement damage-dependent conductivity or `k(d)=g(d)k0`.
- Did not change material parameters, `l0`, mechanics boundary conditions,
  phase-field/history logic, loss route, source model behavior, or reaction
  route.

## Key Implementation Details

- Steady functional density:
  `0.5 * k0 * |grad_m T|^2 - Q * T`.
- Transient incremental functional density:
  `rho*c/(2*dt) * (T - T_prev)^2 + 0.5*k0*|grad_m T|^2 - Q*T`.
- `dt_s <= 0` raises `ValueError`.
- `mean_thermal_functional_density` is a pointwise patch-test average only, not
  mesh quadrature.
- Heat utilities keep SI heat units and explicit mm-to-m derivative scaling.
- Damage-dependent conductivity remains absent from public signatures.

## Validation To Report

- `git status`
- recursive py_compile under `examples/TM_comsol_thermal_micro`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_constant_k0_heat_pde_patch.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/REPORT.md`
2. `examples/TM_comsol_thermal_micro/heat_pde.py`
3. `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`
4. `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/tables/thermal_functional_summary.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/tables/autograd_guard_summary.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1/tables/patch_test_results.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Review this package. If approved, design the solved-temperature representation
and weak-form quadrature strategy without coupling it to fracture training yet.
Do not implement damage-dependent conductivity or run heat-fracture diagnostics
without separate approval.
