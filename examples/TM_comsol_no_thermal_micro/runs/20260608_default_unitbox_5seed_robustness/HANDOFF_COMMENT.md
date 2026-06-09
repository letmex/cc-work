## Codex handoff: default-alpha unit_box 5-seed robustness

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_default_unitbox_5seed_robustness
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_default_unitbox_5seed_robustness/REPORT.md

### What changed
- Ran five predeclared full-D0020 seeds for `history + default alpha init + top-u-mode free + coord_normalization unit_box`.
- Seeds were `7, 13, 21, 42, 99`; none were replaced or cherry-picked.
- Included existing seed 2 default-alpha `unit_box` full result as `reference_only=yes`.
- Did not use `--alpha-init-intact`.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 <seed> TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix full_D0020_seed<seed>_history_default_unitbox
D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\tables\stepwise_<case>.csv --events-out <package>\tables\broadening_events_<case>.csv --summary <package>\artifacts\summary_<case>.md --case <case>
D:\anaconda3\envs\torch_env\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\figures\<case> --run-label <case> --dpi 180
D:\anaconda3\envs\torch_env\python.exe artifacts\build_seed_robustness_package.py
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260608_default_unitbox_5seed_robustness\artifacts\build_seed_robustness_package.py
```

### Key results
- Completed seeds: 7, 13, 21, 42, 99.
- Failed seeds: none.
- New random seeds classified `A. full localized stable`: 5/5.
- Robustness decision under user rule: **seed-robust trend observed**.
- Verification: `pytest examples\TM_comsol_no_thermal_micro\tests -q` passed with 18 tests; package aggregation script `py_compile` passed.
- This is a seed robustness trend only; it is not physical validation.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/final_seed_comparison.csv`
- `tables/stepwise_seed_summary.csv`
- `tables/reaction_stress_strain_by_seed.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does the 5/5 default-alpha `unit_box` result justify using this as the next controlled production path?
2. Should the next Codex task add more seeds or move to another explicit robustness axis?
3. What acceptance criteria should be required before stronger validation claims are considered?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.
- Do not use `--alpha-init-intact` as the main route unless explicitly requested.
- Do not change TM split/material parameters/thermal terms/history update logic unless a clear bug is found.
- Do not claim physical validation from this seed robustness package.
