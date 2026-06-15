# Handoff: Heat Correction Policy

## Status

Final classification: `heat correction policy implemented and tests passed`

Primary implementation commit:

- `17e4da0fc9062d8b67a6d87c3254e78ed78ef25a`
  (`Add heat correction policy`)

Push status:

- Primary implementation commit pushed to `origin/main`.
- Final status after primary push: `## main...origin/main`.
- Final HEAD known at handoff-sync edit time:
  `17e4da0fc9062d8b67a6d87c3254e78ed78ef25a`.
- This file does not chase the handoff-sync commit's own hash.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy`
- Report: `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy/MANIFEST.json`

## Key Result

For pure H1 linear top-bottom Dirichlet and H2 uniform bottom-only patch cases,
`run_steady_heat_patch_case` now defaults to `correction_policy="frozen_lift"`.

H1 frozen lift observed:

- max correction: `0.0 K`
- max strong residual: `0.0 W/m^3`
- normalized residual max: `0.0`

H1 trainable correction remains available by explicit override and preserves
the previous diagnostic behavior.

Strong-form residual remains diagnostic-only and is not used as the primary
heat objective.

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Did not modify `train_mixed_tm.py`.
- Did not implement solved-temperature mechanics coupling, thermal-strain
  coupling, fracture training with heat, heat-fracture diagnostics, or
  damage-dependent conductivity.
- Did not run D0040, seed study, shear extension, S0110, transient production,
  or bottom-cooling production training.

## Validation

- `git status`: PASS, clean before work and only expected thermal files changed before commit.
- recursive `py_compile` under `examples\TM_comsol_thermal_micro`: PASS.
- `test_constant_k0_heat_pde_patch.py`: PASS, 18 passed.
- `test_heat_only_weak_form_training.py`: PASS, 11 passed.
- `test_h1_quadrature_regularization_review.py`: PASS, 9 passed.
- `test_heat_correction_policy.py`: PASS, 7 passed.
- `test_prescribed_thermal_strain_patch.py`: PASS, 8 passed.
- package schema/file existence check: PASS.
- `git diff --check`: PASS.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: PASS, empty output.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy/REPORT.md`
2. `examples/TM_comsol_thermal_micro/train_heat_only.py`
3. `examples/TM_comsol_thermal_micro/tests/test_heat_correction_policy.py`
4. `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy/tables/h1_policy_comparison.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy/tables/h2_policy_summary.csv`
6. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Review whether the frozen-lift policy should also be documented in the
user-facing thermal README before any solved-temperature mechanics-coupling
work is approved.
