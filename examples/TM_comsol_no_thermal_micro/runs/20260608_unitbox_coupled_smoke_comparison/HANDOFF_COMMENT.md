## Codex handoff: unit-box coupled smoke and controlled comparison

Commit: 100451003581f81d321853285b903493d6ab5158
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_unitbox_coupled_smoke_comparison
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_unitbox_coupled_smoke_comparison/REPORT.md

### What changed
- Ran a short coupled smoke test with `coord-normalization unit_box`.
- Because smoke passed, ran matched 12-step controlled comparison:
  - `history + alpha-init-intact + coord-normalization none`
  - `history + alpha-init-intact + coord-normalization unit_box`
- Kept `l0`, material parameters, `tm_source` split, alpha seeding logic, phase-field notch behavior, thermal terms, and history update logic unchanged.
- Did not add notch/lip loss, notch masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 2 TrainableReLU 3 --smoke --alpha-init-intact --top-u-mode free --coord-normalization unit_box --run-suffix coord_unitbox_coupled_smoke
D:\anaconda3\envs\torch_env\python.exe main.py 4 100 2 TrainableReLU 3 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --alpha-init-intact --top-u-mode free --coord-normalization none --run-suffix coord_cmp_none_m12
D:\anaconda3\envs\torch_env\python.exe main.py 4 100 2 TrainableReLU 3 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --alpha-init-intact --top-u-mode free --coord-normalization unit_box --run-suffix coord_cmp_unitbox_m12
D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <run_dir> --out <package>/tables/<case>_stepwise_summary.csv --events-out <package>/tables/<case>_broadening_events.csv --summary <package>/<case>_summary.md --case <case>
D:\anaconda3\envs\torch_env\python.exe plot_clean_tm_results.py --result-dir <run_dir> --out-dir <package>/figures/<case> --run-label <case> --dpi 170
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\field_computation.py examples\TM_comsol_no_thermal_micro\config.py examples\TM_comsol_no_thermal_micro\main.py examples\TM_comsol_no_thermal_micro\train_mixed_tm.py examples\TM_comsol_no_thermal_micro\history_field_mixed_tm.py examples\TM_comsol_no_thermal_micro\tests\test_coord_normalization.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
gh --version
gh auth status
```

### Key results
- `unit_box` coupled smoke passed and wrote `fields_mixed_tm_step_0000.npz`.
- Smoke diagnostics: `coord_normalization=unit_box`, `x_hat/y_hat=[-1,1]`, `t3_gradients_use_physical_xy=True`, `top_v_error_max=0`, bottom displacement residuals zero.
- Controlled `none` run final step: `alpha_mean=0.167670`, `alpha_std=0.002104`, `alpha_max=0.174432`, bulk/notch He ratio `1.005457`, bottom/notch He ratio `0.963485`; classified as still uniform/background.
- Controlled `unit_box` run final step: `alpha_mean=0.103954`, `alpha_std=0.155334`, `alpha_max=1.000465`, bulk/notch He ratio `0.000344`, bottom/notch He ratio `0.000279`; classified as notch-localized under this diagnostic.
- Project-local tests passed: `18 passed`.
- Repository-root tests failed during collection because `ref_files.Chinese_SENT_reproduction` is missing; this is outside `TM_comsol_no_thermal_micro`.
- `gh` is installed but unauthenticated, so this is markdown-only handoff.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `tables/broadening_events.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should `coord-normalization unit_box` now be accepted for the next longer alpha-init history run?
2. Should the next run add at least one extra seed before a full-stage run?
3. Do the reaction differences between the `none` and `unit_box` branches require another mechanics/coupled diagnostic before longer training?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not change alpha seeding, phase-field notch behavior, thermal terms, or history update logic.
- Do not add notch/lip loss, notch masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.
- Do not claim physical validation from this controlled comparison or one seed.
