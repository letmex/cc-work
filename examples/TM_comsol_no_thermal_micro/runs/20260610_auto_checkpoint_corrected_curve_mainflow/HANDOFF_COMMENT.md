## Codex handoff: Auto checkpointed corrected stress-strain curve

Commit: 42df2c4
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_auto_checkpoint_corrected_curve_mainflow
Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_auto_checkpoint_corrected_curve_mainflow/REPORT.md

### What changed
- Changed the active local project source at `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`.
- `config.py` now defaults `--save-step-checkpoints` to `true`.
- This makes normal training save per-step checkpoint states without requiring the user to remember an extra CLI flag.
- Since `main.py` already runs `run_corrected_reaction_postprocess(...)` after `train_mixed_tm(...)`, completed training can now automatically write `results/<run>/curves/corrected_stress_strain_by_step.csv` from exact energy-conjugate reaction data.
- No physics/model parameters were changed: `l0`, material parameters, TM split, alpha/history logic, phase-field notch handling, and training loss are unchanged.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests\test_history_mode_controls.py::test_save_step_checkpoints_defaults_to_true -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile config.py main.py train_mixed_tm.py corrected_reaction_postprocess.py plot_clean_tm_results.py tests\test_history_mode_controls.py tests\test_corrected_reaction_postprocess.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --max-steps 2 --delta-max 1e-6 --n-rprop 1 --n-lbfgs 0 --top-u-mode free --coord-normalization unit_box --run-suffix auto_ckpt_default_smoke
```

### Key results
- The new focused test failed before the code change because the default was `False`.
- After the code change, the focused test passed.
- Full local example test suite passed: `23 passed, 8 warnings`.
- `py_compile` passed for modified and related scripts.
- Smoke training was run without passing `--save-step-checkpoints`.
- The smoke run wrote `save_step_checkpoints: True` and `checkpoint_every_step: True` to `model_settings.txt`.
- The smoke run generated `best_models/step_checkpoints/checkpoint_mixedH_TM_step_0000.pt`.
- The smoke run generated `curves/corrected_stress_strain_by_step.csv`.
- `corrected_reaction_availability.csv` reports `energy_exact_primary` and `exact_reaction_computable=True`.
- The corrected stress-strain primary metric is `nominal_stress_energy_exact_MPa`; legacy top-boundary reaction remains diagnostic only.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/default_checkpoint_smoke_summary.csv`
- `tables/smoke_corrected_reaction_availability.csv`
- `tables/smoke_corrected_stress_strain_by_step.csv`
- `artifacts/source_patch.diff`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is this sufficient to make corrected stress-strain generation the default completed-training path?
2. For the next D0020 full run, should Codex add a hard report check that every load step has both a checkpoint and a corrected-curve row?
3. Should plotting/reporting fail loudly when the corrected curve exists but still has `reaction_metric_unavailable`?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this smoke/software-path verification.
- Do not run D0040 for this issue; the current task is correcting the stress-strain source path.
