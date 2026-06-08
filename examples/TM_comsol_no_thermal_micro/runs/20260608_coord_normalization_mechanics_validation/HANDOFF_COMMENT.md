## Codex handoff: coordinate normalization mechanics validation

Commit: 07693252c7c95a6f75011466e6bc297e8c429875
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_coord_normalization_mechanics_validation
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_coord_normalization_mechanics_validation/REPORT.md

### What changed
- Added a controlled `coord_normalization=none|unit_box` option in the normal PINN path.
- Default `none` preserves old raw physical-mm NN inputs.
- `unit_box` maps only NN input coordinates to `[-1,1]`.
- T3 gradients, boundary ansatz, energy/postprocessing coordinates, `l0`, material parameters, TM split, alpha seeding, phase-field notch behavior, thermal terms, and history logic were not changed.
- Added tests and code snapshots inside the package because the runnable project is outside the pushed Git repo.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m pytest tests\test_coord_normalization.py -q
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile field_computation.py config.py main.py train_mixed_tm.py history_field_mixed_tm.py debug_prefit_then_energy_mechanics.py debug_coord_normalization_mechanics_validation.py tests\test_coord_normalization.py
D:\anaconda3\envs\torch_env\python.exe debug_coord_normalization_mechanics_validation.py --out-dir <package_root> --target examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz --seed 2 --hidden-layers 8 --neurons 400 --prefit-epochs 800 --random-energy-epochs 300 --continuation-epochs 10,30,100,300
gh --version
gh auth status
```

### Key results
- Local TM project tests passed: `18 passed`.
- Repository-root tests failed during collection because `ref_files.Chinese_SENT_reproduction` is missing; this is outside `TM_comsol_no_thermal_micro`.
- Mapping diagnostics confirm `unit_box` maps physical `[0,0.01]` mm to `[-1,1]` and record `t3_gradients_use_physical_xy=True`.
- Raw-coordinate random energy-only case remained `broad/background`: displacement rel RMSE 0.362793, strain rel RMSE 0.853290, He_current corr 0.026446.
- Unit-box random energy-only case became `notch-amplified`: displacement rel RMSE 0.077358, strain rel RMSE 0.283503, He_current corr 0.993423, energy ratio 0.978174, reaction ratio 0.764252.
- Unit-box displacement+strain prefit closely matched exact FE: displacement rel RMSE 0.001408, strain rel RMSE 0.011566, He_current corr 0.999973, energy ratio 1.000045, reaction ratio 0.995461.
- Energy continuation still drifts: 10-epoch continuation can become boundary-dominated; 100/300 epochs recover notch-amplified He but strain/reaction remain degraded.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/coord_normalization_case_comparison.csv`
- `tables/energy_continuation_drift.csv`
- `tables/coord_mapping_diagnostics.csv`
- `figures/figure_summary.md`
- `artifacts/code_snapshot/debug_coord_normalization_mechanics_validation.py`

### Question for ChatGPT
1. Is `coord_normalization=unit_box` sufficiently justified as a controlled NN-input ansatz fix for a short coupled alpha-init history smoke run?
2. Should the next run test coupled alpha-init history with `unit_box`, or should mechanics-only energy-continuation drift be diagnosed further first?
3. If coupled smoke is approved, what minimal load schedule and comparison table should Codex use?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not change alpha seeding, phase-field notch behavior, thermal terms, or history update logic.
- Do not add notch/lip loss, notch masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.
- Do not claim physical validation from mechanics-only diagnostics or one seed.
