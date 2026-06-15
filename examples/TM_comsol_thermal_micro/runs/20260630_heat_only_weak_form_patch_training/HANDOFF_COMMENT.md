# Handoff: Heat-only Weak-form Patch Training

## Status

Final classification: `heat-only weak-form patch trainer implemented and tests passed`

Commit hash:

- Primary implementation commit: pending until exact-path commit.
- Handoff sync commit: this file may be updated once after the primary push; do
  not chase the sync commit self-hash.

Push status:

- Pending until primary implementation commit is pushed to `origin/main`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training`
- Report: `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training/MANIFEST.json`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Added independent solved-temperature helpers, triangle-area quadrature, and a
  small heat-only weak-form patch trainer.
- Did not modify or call `train_mixed_tm.py`.
- Did not couple solved temperature to mechanics, thermal strain mechanics,
  phase field, history, fracture training, or reaction routes.
- Did not implement damage-dependent conductivity or `k(d)=g(d)k0`.
- Did not run D0040, seed study, shear extension, S0110, production
  bottom-cooling, or heat-fracture diagnostics.

## Key Implementation Details

- Temperature ansatz functions hard-satisfy top-bottom and bottom-only
  Dirichlet boundaries.
- Triangle areas are converted from mm geometry to m^2.
- Primary heat-only loss is area-weighted mean thermal functional density.
- Strong residuals are diagnostics only.
- H1 analytical max temperature error: `4.752286448592713e-04 K`.
- H2 analytical max temperature error: `0.0 K`.
- H1 strong residual diagnostic is high:
  `7.240993859241002e10 W/m^3`; review quadrature/regularization before any
  mechanics coupling.

## Validation To Report

- `git status`
- recursive py_compile under `examples/TM_comsol_thermal_micro`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_constant_k0_heat_pde_patch.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_heat_only_weak_form_training.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training/REPORT.md`
2. `examples/TM_comsol_thermal_micro/thermal_field.py`
3. `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
4. `examples/TM_comsol_thermal_micro/train_heat_only.py`
5. `examples/TM_comsol_thermal_micro/tests/test_heat_only_weak_form_training.py`
6. `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training/tables/heat_patch_training_results.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Review this heat-only weak-form package. If approved, do a quadrature and
regularization review for H1 before any solved-temperature mechanics coupling.
Do not implement damage-dependent conductivity or run heat-fracture diagnostics
without separate approval.
