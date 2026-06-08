## Codex handoff: coord-normalization / alpha-init 2x2 near-production comparison

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_coordnorm_alpha_2x2_comparison
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_coordnorm_alpha_2x2_comparison/REPORT.md

### What changed
- Ran four matched 8x400 coupled cases for coordinate normalization and alpha initialization.
- Used the first 12 steps of `load_schedule_D0020_extended.csv` with `RPROP=300`, `LBFGS=0`.
- Generated merged stepwise, event, reaction, coordinate mapping, and final comparison tables.
- Added final alpha, `He_current`, mechanics-drive, and reaction figures with text summaries.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization none --load-schedule-file load_schedule_D0020_extended.csv --run-suffix coordnorm2x2_default_none

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix coordnorm2x2_default_unitbox

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization none --alpha-init-intact --load-schedule-file load_schedule_D0020_extended.csv --run-suffix coordnorm2x2_intact_none

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --alpha-init-intact --load-schedule-file load_schedule_D0020_extended.csv --run-suffix coordnorm2x2_intact_unitbox

D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile analyze_drive_broadening_stepwise.py plot_clean_tm_results.py config.py field_computation.py train_mixed_tm.py
```

### Key results
- This is a near-production diagnostic, not full D0020 validation.
- `default_none`: near-uniform alpha with boundary-side/broad drive; final `alpha_mean=0.186222`, `alpha_std=0.000632`, `alpha_max=0.187203`, `bulk/notch He=1.003120`.
- `default_unitbox`: notch-amplified; final `alpha_max=1.000307`, `alpha_std=0.169213`, `bulk/notch He=0.000221`.
- `intact_none`: notch-amplified but weaker; final `alpha_max=0.507639`, `bulk/notch He=0.045656`.
- `intact_unitbox`: notch-amplified; final `alpha_max=1.000230`, `bulk/notch He=0.000189`.
- Project-local tests passed: `18 passed in 1.50s`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `tables/coord_mapping_diagnostics.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the 8x400 12-step evidence sufficient to choose `coord_normalization=unit_box` as the next full-D0020 production path?
2. Should the full D0020 run include both alpha-init branches, or focus on `alpha-init-intact + unit_box` first?
3. How many additional seeds should be required before treating the localized branch as robust?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from medium/diagnostic runs.

