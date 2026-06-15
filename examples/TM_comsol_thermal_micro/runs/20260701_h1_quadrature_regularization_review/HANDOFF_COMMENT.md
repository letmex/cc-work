# Handoff: H1 Quadrature and Regularization Review

## Status

Final classification: `h1 review implemented; trained residual remains high but explained`

Commit hash:

- Primary implementation commit: pending until exact-path commit.
- Handoff sync commit: this file may be updated once after the primary push; do
  not chase the sync commit self-hash.

Push status:

- Pending until primary implementation commit is pushed to `origin/main`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review`
- Report: `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review/MANIFEST.json`

## Key Result

The no-training lift baseline has zero temperature error, zero correction, and
zero strong residual. Therefore the high trained H1 residual does not come from
the exact linear lift, `heat_pde.py`, or mm-to-m gradient scaling.

The high trained residual comes from a small learned nonlinear correction:

- centroid max correction: `4.752286448592713e-04 K`
- centroid max strong residual: `7.240993859241002e10 W/m^3`
- triangle 3-point max correction: `3.703253884168589e-04 K`
- triangle 3-point max strong residual: `4.8776662018449715e10 W/m^3`
- triangle 3-point + correction L2 max strong residual:
  `4.862652370993762e10 W/m^3`

Strong residual was used only to diagnose smoothness/curvature of the learned
temperature field. It was not used as the heat training objective.

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Did not modify `train_mixed_tm.py`.
- Did not implement solved-temperature mechanics coupling, thermal-strain
  coupling, fracture training with heat, damage-dependent conductivity,
  heat-fracture diagnostics, or broad training runs.

## Validation To Report

- `git status`
- recursive py_compile under `examples/TM_comsol_thermal_micro`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_constant_k0_heat_pde_patch.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_heat_only_weak_form_training.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_h1_quadrature_regularization_review.py -q`
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review/REPORT.md`
2. `examples/TM_comsol_thermal_micro/train_heat_only.py`
3. `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
4. `examples/TM_comsol_thermal_micro/tests/test_h1_quadrature_regularization_review.py`
5. `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review/tables/h1_quadrature_comparison.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review/tables/h1_residual_normalization.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Do not start solved-temperature mechanics coupling yet. First decide whether H1
should constrain the correction space more strongly, use richer quadrature, or
use an exact-lift freeze/zero-correction policy for pure linear Dirichlet patch
cases.
