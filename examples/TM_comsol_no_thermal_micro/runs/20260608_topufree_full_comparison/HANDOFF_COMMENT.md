## Codex handoff: top-u-free full comparison

Commit: ba469209093210e5d5e4b7fdfc0f02216d49372b
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_topufree_full_comparison`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_topufree_full_comparison/REPORT.md`

### What changed
- Ran a controlled full D0020 seed2 diagnostic for `history + alpha-init-intact + --top-u-mode free`.
- Re-analyzed top-u fixed, top-u free, and old history reference runs with absolute drive validity checks.
- Added `first_step_where_ratio_valid` with threshold `notch_tip_He_current_max > 1e-8`.
- Added fixed/free comparison tables and the report `reports/topufree_full_comparison.md`.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe main.py 8 400 2 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --alpha-init-intact --top-u-mode free --load-schedule-file load_schedule_D0020_extended.csv --run-suffix history_alpha_init_intact_topufree_D0020_seed2

D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <top-u-fixed-run> --out <package>\tables\stepwise_summary_alpha_intact_history_topufixed_full_seed2.csv --events-out <package>\tables\broadening_events_alpha_intact_history_topufixed_full_seed2.csv --summary <package>\reports\drive_broadening_alpha_intact_history_topufixed_full_seed2.md --case alpha_intact_history_topufixed_full_seed2

D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <top-u-free-run> --out <package>\tables\stepwise_summary_alpha_intact_history_topufree_full_seed2.csv --events-out <package>\tables\broadening_events_alpha_intact_history_topufree_full_seed2.csv --summary <package>\reports\drive_broadening_alpha_intact_history_topufree_full_seed2.md --case alpha_intact_history_topufree_full_seed2

D:\anaconda3\envs\torch_env\python.exe analyze_drive_broadening_stepwise.py --run-dir <old-history-reference-run> --out <package>\tables\stepwise_summary_old_history_full_seed2_reference.csv --events-out <package>\tables\broadening_events_old_history_full_seed2_reference.csv --summary <package>\reports\drive_broadening_old_history_full_seed2_reference.md --case old_history_full_seed2_reference

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

D:\anaconda3\envs\torch_env\python.exe -m py_compile analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py plot_clean_tm_results.py debug_recompute_he_current.py main.py

gh --version
gh auth status
& 'C:\Program Files\GitHub CLI\gh.exe' --version
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
```

### Key results
- Top-u-free full run completed with 34 analyzed steps.
- Top-u-free did not remove step-0 broad drive. With `notch_tip_He_current_max > 1e-8`, the first valid ratio step is still step 0.
- Top-u-free final `alpha_mean = 0.48817826120409724`, `alpha_std = 4.0094150429576516e-05`, and `alpha_max = 0.4882843494415283`.
- Top-u-free final bulk/notch He ratio is `1.0008025829954588`; bottom/notch He ratio is `0.9995179868457842`.
- Top-u-free final `reaction_N_tm_eff = -1.7486419677734375`; the sign remains negative.
- New top-u-free full run classification: `B. medium-stage uniform -> full-stage still uniform`.
- This indicates top-boundary ansatz mismatch is not the main cause of the alpha-init uniform/background damage branch.
- GitHub CLI exists at `C:\Program Files\GitHub CLI\gh.exe`, but it is unauthenticated and no token environment variable was present; issue #1 was not auto-commented.

### Files to read first
- `README.md`
- `REPORT.md`
- `reports/topufree_full_comparison.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary_topufixed_vs_topufree.csv`
- `tables/broadening_events_topufixed_vs_topufree.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Given the absolute drive validity checks, should the next diagnosis target why alpha-init history has broad/background drive already at step 0?
2. What is the next minimal diagnostic intervention after ruling out top-u-free as the main cause?
3. Should the next prompt focus on displacement/strain ansatz, optimizer/loss scaling, or field recomputation consistency?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not change thermal field, thermal expansion, or history update logic based on this one-seed diagnostic.
- Do not claim physical validation from this package.

