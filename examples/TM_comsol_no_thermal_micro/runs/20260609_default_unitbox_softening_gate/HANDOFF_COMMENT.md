## Codex handoff: default unit_box softening gate

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260609_default_unitbox_softening_gate
Main report: examples/TM_comsol_no_thermal_micro/runs/20260609_default_unitbox_softening_gate/REPORT.md

### What changed
- Audited existing D0020 5-seed data for post-peak reaction indicators.
- Created `load_schedule_D0040_softening_gate.csv` because no D0040/D0060 schedule existed.
- Ran required extended D0040 seeds `7, 13, 42` using default alpha init, top-u free, and coord-normalization `unit_box`.
- Computed alpha threshold area fractions, connected high-alpha crack proxy, ligament crossing proxy, reaction peak/final/drop, and reaction consistency.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 <seed> TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0040_softening_gate.csv --run-suffix softgate_D0040_seed<seed>_history_default_unitbox
D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <result_dir> --out <package>\tables\stepwise_<case>.csv --events-out <package>\tables\broadening_events_<case>.csv --summary <package>\artifacts\summary_<case>.md --case <case>
D:\anaconda3\envs\torch_env\python.exe plot_clean_tm_results.py --result-dir <result_dir> --out-dir <package>\figures\<case> --run-label <case> --dpi 180
D:\anaconda3\envs\torch_env\python.exe artifacts\build_softening_gate_package.py
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260609_default_unitbox_softening_gate\artifacts\build_softening_gate_package.py
```

### Key results
- Completed required extended seeds: 3/3.
- Required seeds with >=10% post-peak drop and connected crack-growth proxy: 0/3.
- Reaction consistency audit: degraded stress path confirmed = True.
- Gate decision: **softening gate not passed**.
- Verification: `pytest examples\TM_comsol_no_thermal_micro\tests -q` passed with 18 tests; package script `py_compile` passed.
- No physical validation is claimed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/d0020_softening_audit.csv`
- `tables/extended_softening_summary.csv`
- `tables/reaction_softening_by_case.csv`
- `tables/alpha_connectivity_by_case.csv`
- `tables/reaction_consistency_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this package resolve the post-peak softening blocker for the current route?
2. If the gate passes, what should be the next diagnostic before any physical-validation language is allowed?
3. If the reaction drop is present, is the connected crack proxy sufficient or should a stricter geometric crack-path metric be requested?

### Constraints
- Do not change `l0`.
- Do not change material parameters, TM split, thermal terms, or history update logic unless a clear bug is found.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.
- Do not use `--alpha-init-intact` as the main route.
- Do not claim physical validation.
