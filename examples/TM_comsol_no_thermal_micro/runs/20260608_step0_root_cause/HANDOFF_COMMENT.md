## Codex handoff: step-0 root-cause diagnostic

Commit: PENDING
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_step0_root_cause`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_step0_root_cause/REPORT.md`

### What changed
- Added a minimal diagnostic package for alpha-init history step-0 broad/background drive.
- Added and ran `debug_step0_root_cause.py` to inspect saved step-0 full-run fields, notch-lip displacement jumps, strain proxies, and optimizer budget sensitivity.
- Recomputed `He_current` from saved PINN strain fields for step 0.
- Ran small FE-DOF alpha-zero baselines at `Delta = 1e-6` for top-u fixed/free comparison.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe debug_step0_root_cause.py --out-dir D:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_step0_root_cause --fixed-run <alpha-init-history-top-u-fixed-full-run> --topufree-run <alpha-init-history-top-u-free-full-run> --old-history-run <old-history-full-run> --rprop-budgets 0 1 10 100

D:\anaconda3\envs\torch_env\python.exe debug_recompute_he_current.py --npz <alpha-init-history-top-u-fixed-step0-npz> --out <package>\tables\recompute_he_step0_alpha_intact_topufixed.csv --alpha-mode saved

D:\anaconda3\envs\torch_env\python.exe debug_recompute_he_current.py --npz <alpha-init-history-top-u-free-step0-npz> --out <package>\tables\recompute_he_step0_alpha_intact_topufree.csv --alpha-mode saved

D:\anaconda3\envs\torch_env\python.exe debug_recompute_he_current.py --npz <old-history-step0-npz> --out <package>\tables\recompute_he_step0_old_history.csv --alpha-mode saved

D:\anaconda3\envs\torch_env\python.exe debug_fedof_energy_baseline.py --delta 1e-6 --epochs 300 --alpha-mode zero --top-u-mode fixed --out <package>\tables\fedof_step0_alpha_zero_fixed.csv --npz <package>\artifacts\fedof_step0_alpha_zero_fixed_fields.npz

D:\anaconda3\envs\torch_env\python.exe debug_fedof_energy_baseline.py --delta 1e-6 --epochs 300 --alpha-mode zero --top-u-mode free --out <package>\tables\fedof_step0_alpha_zero_free.csv --npz <package>\artifacts\fedof_step0_alpha_zero_free_fields.npz

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_step0_root_cause.py debug_recompute_he_current.py debug_fedof_energy_baseline.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py

gh --version
gh auth status
& 'C:\Program Files\GitHub CLI\gh.exe' --version
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
```

### Key results
- Alpha-init history step 0 is broad/background before meaningful alpha growth: top-u fixed `bulk/notch He = 1.0137009507038128`; top-u free `bulk/notch He = 1.022799927576047`.
- Old history default-alpha step 0 is notch-localized: `bulk/notch He = 0.0182547240109948`.
- Step-0 optimizer budget sweep shows alpha-init broad drive exists even at `rprop_budget = 0`: `bulk/notch He = 1.0003164965627505`, `alpha_mean = 0.0`.
- Saved PINN field recomputation is consistent: max absolute recompute errors are about `1e-13` to `2e-12`.
- FE-DOF alpha-zero baselines at `Delta = 1e-6` produce notch-dominated drive: bottom/notch ratios are `0.03377878814045944` fixed and `0.028618440934266305` free.
- Evidence points toward displacement/strain representation or early mechanics optimization path near the narrow explicit notch, not top-u mode, alpha growth, postprocessing, `l0`, material parameters, TM split, phase-field notch seeding, or history update logic.
- Tests passed: `13 passed in 0.05s`.
- GitHub CLI exists at `C:\Program Files\GitHub CLI\gh.exe`, but it is unauthenticated and no token environment variable was present; issue #1 was not auto-commented.

### Files to read first
- `README.md`
- `REPORT.md`
- `reports/step0_root_cause_summary.md`
- `tables/step0_field_summary.csv`
- `tables/optimizer_budget_step0_summary.csv`
- `tables/recompute_he_step0_summary.csv`
- `tables/fedof_step0_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this evidence justify targeting displacement/strain ansatz capacity near the narrow explicit notch next?
2. Should the next minimal diagnostic be a mechanics-only PINN vs FE-DOF comparison at `Delta = 1e-6` with alpha fixed to zero?
3. Should Codex test a local notch-lip enrichment or separate local degrees-of-freedom diagnostic before any physical model change?

### Constraints
- Do not change `l0`.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not change thermal field, thermal expansion, phase-field notch behavior, or history update logic based on this diagnostic.
- Do not claim physical validation from this package.
