## Codex handoff: full D0020 unit-box candidate

Commit: db36df8573ff498bd764abbb229f4187ca5130f7
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_full_D0020_unitbox_candidate
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_full_D0020_unitbox_candidate/REPORT.md

### What changed
- Ran the primary full-D0020 production-path candidate:
  `history + alpha-init-intact + top-u-mode free + coord_normalization unit_box`.
- Also ran the optional branch-selection comparison:
  `history + default alpha init + top-u-mode free + coord_normalization unit_box`.
- Both runs used 8x400, seed 2, AT2, full 34-step D0020 schedule, and full default optimizer budget `RPROP=10000`, `LBFGS=1`.
- Generated final comparison, stepwise diagnostics, event diagnostics, reaction/stress-strain table, coordinate mapping diagnostics, and figures.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --alpha-init-intact --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix full_D0020_seed2_history_intact_unitbox

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix full_D0020_seed2_history_default_unitbox

D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile analyze_drive_broadening_stepwise.py plot_clean_tm_results.py config.py field_computation.py train_mixed_tm.py
```

### Key results
- Primary `alpha-init-intact + unit_box` completed all 34 steps and classified as `A. 12-step localized -> full-stage localized`.
- Primary final metrics: `alpha_mean=0.113297`, `alpha_std=0.191751`, `alpha_max=1.001266`, `alpha>0.5 area=0.052116`, `bulk/notch He=1.97e-05`, `reaction_N_tm_eff=0.820360`.
- Optional `default-alpha + unit_box` also completed all 34 steps and classified as `A. 12-step localized -> full-stage localized`.
- Optional final metrics: `alpha_mean=0.112522`, `alpha_std=0.191365`, `alpha_max=1.001675`, `alpha>0.5 area=0.051377`, `bulk/notch He=1.46e-05`, `reaction_N_tm_eff=0.894020`.
- For both runs, final max `He_current`, `He_history`, and mechanics-drive locations stayed near `(x,y)=(0.005046,0.005040)`.
- Project-local tests passed: `18 passed in 1.71s`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `tables/reaction_stress_strain.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is this full-D0020 seed-2 evidence sufficient to promote `unit_box` to the controlled production path for the next seed?
2. Should the next task run another seed only for `alpha-init-intact + unit_box`, or include default-alpha again?
3. What seed count or additional diagnostic would be required before treating the branch as robust?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from a single seed.
