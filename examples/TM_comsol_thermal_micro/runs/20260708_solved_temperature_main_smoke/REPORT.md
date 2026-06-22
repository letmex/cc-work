# Solved-Temperature Main Entrypoint Smoke

## Purpose

This run moves beyond the adapter/unit-test layer and executes `examples/TM_comsol_thermal_micro/main.py` with the guarded solved-temperature mechanics flag enabled.

## Command

```text
D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\main.py 2 20 7 TrainableReLU 3.0 --solved-temperature-mechanics-smoke --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --run-suffix solved_temperature_main_smoke --output-root runs\20260708_solved_temperature_main_smoke\raw_outputs
```

## Result

Classification: `main-entrypoint solved-temperature mechanics smoke passed`

The run reached the actual `train_mixed_tm` loop for one load step:

- step: `0`
- displacement: `1e-06 mm`
- loss_total: `2.0283415956556805e-07`
- alpha_max: `0.5515962839126587`
- He_max: `1.4632401871494949e-05`
- mechanics_drive_max: `1.4632401871494949e-05`

Solved-temperature diagnostics from the main-entrypoint result directory:

- active: `true`
- source_mode: `solved_frozen_lift`
- case_id: `H1`
- evaluation_location: `element_centroid`
- network_training_run: `false`
- delta_T_min_K: `0.227020263671875`
- delta_T_max_K: `19.773162841796875`

Thermal strain entered the mechanics summary fields:

- notch_tip_delta_T_mean: `9.999632835388184`
- notch_tip_delta_T_max: `10.5921630859375`
- bottom_right_delta_T_mean: `0.4020843505859375`
- bottom_right_delta_T_max: `0.5771484375`

## Code Change

`train_mixed_tm.py` now persists `solved_temperature_mechanics_smoke_diagnostics.csv` under `results_path` whenever the guarded solved-temperature smoke adapter is active. The new regression test first failed because the CSV was missing, then passed after adding the writer.

## Scope

This is still a smoke-scale mechanics run, not broad production training and not heat-fracture coupling. It proves the solved-temperature path reaches the real `main.py -> train_mixed_tm -> commit_mixed_tm_history_from_model` mechanics path and leaves scalar audit evidence in the result directory.

## Local Raw Outputs

`raw_outputs/` contains generated checkpoints, fields, curves, figures, and logs from the smoke execution. The committed archive should keep only scalar tables and reports.

## Next

Promote to a tiny 3-step schedule and compare the solved-temperature adapter against a matched prescribed `linear_y` thermal field before moving to phase-field/heat coupling diagnostics.
