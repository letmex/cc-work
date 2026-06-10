## Codex handoff: auto corrected reaction mainflow

Commit: e8015d5
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_auto_corrected_reaction_mainflow
Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_auto_corrected_reaction_mainflow/REPORT.md

### What changed
- Added example-local `corrected_reaction_postprocess.py`.
- Updated example-local `main.py` so training calls `run_corrected_reaction_postprocess(...)` after `train_mixed_tm(...)`.
- The postprocessor writes `results/<run>/curves/corrected_stress_strain_by_step.csv`.
- If step checkpoints exist, it computes exact checkpoint `dPi/dDelta` with autograd and promotes `reaction_N_energy_exact`.
- If checkpoints are absent, it writes a corrected CSV with `reaction_metric_unavailable`.
- Updated example-local `plot_clean_tm_results.py` so it auto-discovers `result_dir/curves/corrected_stress_strain_by_step.csv`.
- Added TDD tests for curve building, unavailable fallback, main hook, and `result_dir/curves` auto-discovery.
- Did not modify shared `source/` files.
- Did not run or process D0040.

### Commands run
```powershell
git -C <cc-work> pull origin main
D:/anaconda3/envs/torch_env/python.exe -m pytest D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/tests/test_corrected_reaction_postprocess.py -q
D:/anaconda3/envs/torch_env/python.exe -m py_compile D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/corrected_reaction_postprocess.py D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/main.py D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py
D:/anaconda3/envs/torch_env/python.exe D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/corrected_reaction_postprocess.py --model-dir <TM_comsol_no_thermal_micro>/<checkpointed_D0020_seed42_model_dir> --result-dir <TM_comsol_no_thermal_micro>/results/<checkpointed_D0020_seed42_result_dir> --device cpu
D:/anaconda3/envs/torch_env/python.exe -m pytest D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/tests -q
D:/anaconda3/envs/torch_env/python.exe D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py --result-dir <TM_comsol_no_thermal_micro>/results/<checkpointed_D0020_seed42_result_dir> --out-dir <TM_comsol_no_thermal_micro>/results/clean_figures/result_curves_auto_discovery_seed42 --run-label result_curves_auto_discovery_seed42 --dpi 80
```

### Key results
- Focused tests passed: 4 passed.
- Full example-local tests passed: 22 passed.
- `py_compile` passed for the modified example-local scripts.
- Smoke exact postprocess on checkpointed D0020 seed 42 found 34 checkpoints and wrote corrected reaction/curve tables.
- Smoke status: `energy_exact_primary`, `exact_reaction_computable=True`.
- Plotting without `--corrected-stress-strain-csv` auto-discovered `result_dir/curves/corrected_stress_strain_by_step.csv`.
- Smoke primary metric: `nominal_stress_energy_exact_MPa`.
- Smoke primary final/peak: 0.00453097223744288.
- Legacy top final/peak: 0.945459761653825.
- This is a pipeline fix and smoke verification, not new physical validation.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/mainflow_smoke_summary.csv`
- `tables/corrected_reaction_availability_seed42_smoke.csv`
- `tables/corrected_stress_strain_by_step_seed42_smoke.csv`
- `tables/plot_auto_discovery_stress_strain_data_seed42.csv`
- `figures/figure_summary.md`
- `artifacts/corrected_reaction_postprocess.py`
- `artifacts/main_after.py`
- `artifacts/plot_clean_tm_results_after.py`
- `artifacts/test_corrected_reaction_postprocess.py`

### Question for ChatGPT
1. Is this enough to consider corrected stress-strain generation part of the example-local mainflow?
2. Should full runs now default to `--save-step-checkpoints` so exact corrected reaction is computable rather than unavailable?
3. Should older D0020 packages be regenerated with this mainflow, or should this only apply to future runs?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this pipeline smoke.
- Do not run D0040 unless explicitly requested.

