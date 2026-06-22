# Handoff: Solved-Temperature Mechanics Smoke CLI

## Status

Final classification: `solved-temperature mechanics smoke cli implemented and tests passed`

Primary implementation commit:

- pending until this package is committed

Push status:

- pending until commit/push

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli`
- Report: `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/MANIFEST.json`

## Key Result

Added an explicit opt-in script that runs one fixed solved-temperature mechanics
smoke evaluation and writes `mechanics_smoke_results.csv`.

The script does not implement coupled heat-mechanics training. It does not
train heat and mechanics together.

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Did not modify `train_mixed_tm.py`.
- Did not change material, `l0`, history, reaction, boundary, or loss routes.
- Did not run broad mechanics/fracture training.
- Did not implement heat-fracture diagnostics, fracture heat training, or
  damage-dependent conductivity.

## Validation

- recursive `py_compile` under `examples\TM_comsol_thermal_micro`: PASS.
- `test_constant_k0_heat_pde_patch.py`: PASS, 18 passed.
- `test_heat_only_weak_form_training.py`: PASS, 11 passed.
- `test_h1_quadrature_regularization_review.py`: PASS, 9 passed.
- `test_heat_correction_policy.py`: PASS, 7 passed.
- `test_solved_temperature_mechanics_bridge.py`: PASS, 7 passed.
- `test_solved_temperature_mechanics_smoke_adapter.py`: PASS, 6 passed.
- `test_solved_temperature_mechanics_smoke_cli.py`: PASS, 3 passed.
- `test_prescribed_thermal_strain_patch.py`: PASS, 8 passed.
- explicit smoke CLI run: PASS, wrote `mechanics_smoke_results.csv`.
- package schema/file existence check: PASS.
- `git diff --check`: PASS.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: PASS, empty output.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/REPORT.md`
2. `examples/TM_comsol_thermal_micro/run_solved_temperature_mechanics_smoke.py`
3. `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_cli.py`
4. `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/tables/mechanics_smoke_results.csv`
5. `examples/TM_comsol_thermal_micro/thermal_mechanics_adapter.py`
6. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Document this opt-in smoke runner in the thermal README and keep any guarded
production call-site flag as a separate reviewed task.
