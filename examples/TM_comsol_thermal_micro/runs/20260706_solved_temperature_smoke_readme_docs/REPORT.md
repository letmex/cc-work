# Solved-Temperature Smoke README Documentation

## 1. Purpose

This package documents the explicit solved-temperature mechanics smoke CLI in
the user-facing thermal README and adds a focused documentation guard test.

This task does not implement coupled heat-mechanics training. It does not
modify `train_mixed_tm.py` and does not run broad mechanics or fracture
training.

## 2. What was implemented

Modified:

- `examples/TM_comsol_thermal_micro/README.md`
- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_readme_docs.py`
- `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs`

## 3. README content added

The README now documents:

- the explicit smoke CLI command:
  `D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\run_solved_temperature_mechanics_smoke.py --output-dir examples\TM_comsol_thermal_micro\outputs\solved_temperature_mechanics_smoke`
- the output file: `mechanics_smoke_results.csv`
- the call path through `thermal_mechanics_adapter.build_mechanics_thermal_kwargs_from_bridge`
- source mode `solved_frozen_lift`
- case `H1`
- `element_centroid` evaluation
- observed `DeltaT` range:
  `6.666666666666686 K` to `13.333333333333371 K`
- `mechanics_max_abs_diff = 0.0`
- `network_training_run = false`
- `train_mixed_tm.py remains unmodified`

## 4. Boundary language added

The README now states that the smoke CLI is opt-in and:

- does not implement coupled heat-mechanics training;
- does not add heat-fracture diagnostics;
- does not add damage-dependent conductivity;
- does not run D0040, seed, shear, S0110, transient, or bottom-cooling
  production studies;
- preserves `examples/TM_comsol_no_thermal_micro` as the frozen baseline.

## 5. Documentation guard

The new focused test verifies the README contains the command, output table,
observed values, opt-in status, and boundary statements. It also checks that
`train_mixed_tm.py` does not import the smoke runner or output table token.

## 6. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/README.md`
- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_readme_docs.py`
- `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs`

## 7. Limitations

This package is documentation and guard testing only. It does not add:

- new physics;
- new mechanics training behavior;
- a production call-site flag;
- heat PDE coupling inside mechanics;
- heat-fracture diagnostics;
- damage-dependent conductivity.

## 8. Final classification

`solved-temperature smoke README documentation implemented and tests passed`

## 9. Exact next recommended task

Keep any guarded production call-site flag as a separate reviewed task with its
own opt-in CLI/config tests before touching `train_mixed_tm.py`.
