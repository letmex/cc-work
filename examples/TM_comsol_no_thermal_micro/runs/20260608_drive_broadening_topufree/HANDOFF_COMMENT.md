## Codex handoff: drive broadening and top-u-free smoke

Commit: 2aeae5742a49bc8a6bde49aab0c4e0b29ef283bc
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_drive_broadening_topufree`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_drive_broadening_topufree/REPORT.md`

### What changed
- Added a drive-broadening evidence package for four existing full D0020 seed 2 runs.
- Added stepwise CSVs, broadening event CSVs, and final case comparison.
- Added a top-u-free smoke summary with boundary displacement diagnostics.
- Included source snapshots for the local non-git PINN project under `artifacts/`.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe main.py 2 20 2 TrainableReLU 3.0 --smoke --pff-model AT2 --mixed-mechanics-mode history --alpha-init-intact --top-u-mode free --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix topufree_smoke

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

D:\anaconda3\envs\torch_env\python.exe -m py_compile config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py plot_clean_tm_results.py debug_recompute_he_current.py main.py analyze_drive_broadening_stepwise.py

gh --version
gh auth status

& 'C:\Program Files\GitHub CLI\gh.exe' --version
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
```

### Key results
- `alpha_intact_history_full_seed2`: classified as `B. medium-stage uniform -> full-stage still uniform`; drive ratios are broad from step 0 and final `alpha_mean = 0.48825829612572075`.
- `alpha_intact_current_split_full_seed2`: classified as `B. medium-stage uniform -> full-stage still uniform`; drive ratios are broad from step 0 and final `alpha_mean = 0.4882849213420915`.
- `old_history_full_seed2`: classified as `A. medium-stage uniform -> full-stage localized`; final drive remains notch-localized.
- `old_current_split_full_seed2`: classified as `C. medium-stage uniform -> full-stage boundary/corner damage`; final bottom-right/notch drive ratio is very large.
- Top-u-free smoke completed; `top_u_mode = free`, `top_v_error_max = 0.0`, `bottom_u_abs_max = 0.0`, and `bottom_v_abs_max = 0.0`.
- Tests passed: `13 passed in 0.07s`.
- GitHub CLI exists at `C:\Program Files\GitHub CLI\gh.exe`, but it is unauthenticated and no token environment variable was present; issue #1 was not auto-commented.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `tables/topufree_smoke_summary.csv`
- `reports/drive_broadening_alpha_intact_history_full_seed2.md`
- `reports/drive_broadening_alpha_intact_current_split_full_seed2.md`
- `reports/drive_broadening_old_history_full_seed2.md`
- `reports/drive_broadening_old_current_split_full_seed2.md`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Based on the full stepwise data, should the alpha-init uniform damage be interpreted as drive broadening before alpha growth?
2. Is a controlled full D0020 seed 2 run with `--top-u-mode free` justified next?
3. If not top-u-free full, what is the next minimal diagnostic intervention?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from medium/diagnostic runs or this one-seed package.

