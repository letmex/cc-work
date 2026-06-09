## Codex handoff: D0020 energy-stress conjugacy audit

Commit: COMMIT_PLACEHOLDER
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260619_default_unitbox_D0020_energy_stress_conjugacy_audit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260619_default_unitbox_D0020_energy_stress_conjugacy_audit/REPORT.md

### What changed
- Audited whether postprocessed `sigma_tm_eff` is conjugate to the exact checkpoint mechanics energy.
- Computed elementwise energy-autograd stress and energy-autograd virtual work for existing D0020 seeds 7, 13, and 42.
- Did not run D0040, retrain, extend loading, or change physics.

### Commands run
```powershell
git pull origin main
Read previous D0020 reaction-mode handoff/report/tables.
D:\anaconda3\envs\torch_env\python.exe -m py_compile artifacts\run_d0020_energy_stress_conjugacy_audit.py
D:\anaconda3\envs\torch_env\python.exe artifacts\run_d0020_energy_stress_conjugacy_audit.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
```

### Key results
- Classification: **stress postprocessing bug identified**.
- Energy-autograd virtual-work median pre-through relative error: 8.36581e-08.
- Postprocessed sigma virtual-work median pre-through relative error: 1.08384.
- Energy-autograd virtual work matches exact reaction in 3/3 seeds.
- Energy-autograd reaction collapses after through-crack in 3/3 seeds.
- D0040 remains deferred.
- No production mechanics change is made.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/energy_autograd_virtual_work_identity.csv`
- `tables/corrected_reaction_candidate_summary.csv`
- `tables/energy_autograd_stress_summary.csv`
- `tables/stress_energy_formula_path_audit.csv`
- `tables/history_branch_conjugacy_summary.csv`
- `tables/shear_and_gradient_scaling_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this evidence justify classifying the mismatch as a postprocessed-stress conjugacy bug?
2. Should `reaction_N_tm_eff` be demoted to legacy diagnostic-only status?
3. What minimal postprocessing change should Codex implement or audit next before D0040?

### Constraints
- Do not run D0040 yet.
- Do not extend loading.
- Do not retrain the main model.
- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
