# Handoff: Solved-Temperature Main Entrypoint Smoke

Status: `main-entrypoint solved-temperature mechanics smoke passed`

Primary run command:

```text
D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\main.py 2 20 7 TrainableReLU 3.0 --solved-temperature-mechanics-smoke --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --run-suffix solved_temperature_main_smoke --output-root runs\20260708_solved_temperature_main_smoke\raw_outputs
```

Key evidence:

- `tables/main_smoke_summary.csv`
- `tables/solved_temperature_mechanics_smoke_diagnostics.csv`
- `tables/validation_results.csv`
- raw local results under `raw_outputs/results/solved_temperature_main_smoke/`

Implementation note:

`train_mixed_tm.py` now writes `solved_temperature_mechanics_smoke_diagnostics.csv` from the guarded smoke path so the main entrypoint leaves direct scalar evidence instead of only mutating `training_dict`.

Recommended next task:

Run a 3-step mini schedule with the same flag and archive alpha/He/mechanics-drive changes by step.
