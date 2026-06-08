## Codex handoff: PINN prefit to FE-DOF mechanics

Commit: PENDING_COMMIT_SHA
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_pinn_prefit_fedof_mechanics`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_pinn_prefit_fedof_mechanics/REPORT.md`

### What changed
- Added a mechanics prefit diagnostic package using FE-DOF alpha-zero displacement fields as supervised targets.
- Added and ran `debug_pinn_prefit_fedof_mechanics.py` as a diagnostic-only script snapshot in `artifacts/`.
- Trained only current PINN displacement ansatz outputs against FE-DOF `u/v`; alpha stayed fixed to zero.
- Ran `disp_only`, `disp_lip`, and `disp_strain` variants for top-u free and top-u fixed targets.
- Recomputed predicted strain and `He_current` through the existing alpha-zero TM mechanics path.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe debug_pinn_prefit_fedof_mechanics.py --out-dir <package> --target-free <previous_package>/artifacts/fedof_free_log10_energy_e300_fields.npz --target-fixed <previous_package>/artifacts/fedof_fixed_log10_energy_e300_fields.npz --top-u-modes free fixed --variants disp_only disp_lip disp_strain --epochs 1000 --delta 1e-6 --seed 2 --hidden-layers 8 --neurons 400

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_pinn_prefit_fedof_mechanics.py debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py

gh --version
gh auth status
& 'C:\Program Files\GitHub CLI\gh.exe' --version
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
```

### Key results
- Top-u-free `disp_only` prefit reaches displacement relative RMSE `0.00671628` with `u_corr = 0.999943` and `v_corr = 0.999901`.
- Top-u-free `disp_only` matches notch-lip jumps: predicted/target `u` jump ratio `0.999594`, `v` jump ratio `1.00745`.
- Top-u-free `disp_only` reconstructs notch-amplified drive: predicted `bulk/notch He = 0.119223`, `bottom/notch He = 0.0139126`, `He_current_corr = 0.926113`.
- Top-u-free `disp_strain` improves strain and He agreement: strain relative RMSE `0.0766687`, `strain_corr = 0.997029`, `He_current_corr = 0.984394`.
- `disp_lip` can enforce lip jump but degrades global/strain/He correlation; it is diagnostic evidence that local weighting can force the branch but may over-concentrate drive.
- Current evidence points more toward energy optimization not finding the FE-DOF-like localized mechanics branch than a hard expressivity limit of the current PINN ansatz.
- Tests passed: `13 passed`.
- GitHub CLI exists at `C:\Program Files\GitHub CLI\gh.exe`, but it is unauthenticated and no token environment variable was present; issue #1 was not auto-commented.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/prefit_case_comparison.csv`
- `tables/notch_lip_prefit_metrics.csv`
- `tables/strain_he_reconstruction_metrics.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should the next Codex task test mechanics pretraining/curriculum from this supervised prefit before any coupled full run?
2. Should the next diagnostic initialize alpha-zero energy minimization from the `disp_strain` prefit and check whether it stays notch-amplified?
3. Should notch-lip enrichment be deferred unless pretraining/localized loss guidance fails?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Keep phase-field notch behavior, alpha seeding, and history update logic unchanged.
- Do not claim physical validation from this diagnostic package.
