## Codex handoff: global prefit to mechanics-energy continuation

Commit: PENDING_COMMIT_SHA
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_global_prefit_energy_continuation`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_global_prefit_energy_continuation/REPORT.md`

### What changed
- Added a mechanics-only alpha=0 diagnostic package for global FE-DOF prefit followed by mechanics-energy continuation.
- Added a script snapshot at `artifacts/debug_prefit_then_energy_mechanics.py`.
- The run does not use notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or geometry-label guidance in the loss.
- Region metrics are reported only for postprocessing diagnosis.

### Commands run
```powershell
git pull origin main
```

```powershell
cd <local_pinn_project>/examples/TM_comsol_no_thermal_micro
D:\anaconda3\envs\torch_env\python.exe debug_prefit_then_energy_mechanics.py --out-dir <repo_root>\examples\TM_comsol_no_thermal_micro\runs\20260608_global_prefit_energy_continuation --target-free <repo_root>\examples\TM_comsol_no_thermal_micro\runs\20260608_mechanics_only_notch_ansatz\artifacts\fedof_free_log10_energy_e300_fields.npz --target-fixed <repo_root>\examples\TM_comsol_no_thermal_micro\runs\20260608_mechanics_only_notch_ansatz\artifacts\fedof_fixed_log10_energy_e300_fields.npz --top-u-modes free --cases random_init_energy disp_global_prefit_then_energy disp_strain_global_prefit_then_energy global_curriculum --prefit-epochs 1000 --energy-epochs 300 --curriculum-epochs 1000 --delta 1e-6 --seed 2 --hidden-layers 8 --neurons 400
```

Verification results are listed in `commands_run.txt`.

```powershell
D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_prefit_then_energy_mechanics.py debug_pinn_prefit_fedof_mechanics.py debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py
```

Result: passed.

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

Result: failed during collection because `ref_files.Chinese_SENT_reproduction` is missing in the current environment. See `commands_run.txt`.

### Key results
- `disp_global` prefit reached displacement relative RMSE `0.006716`, `He_current_corr=0.926113`, classification `target-like`.
- `disp_strain_global` prefit reached displacement relative RMSE `0.004930`, strain relative RMSE `0.076669`, `He_current_corr=0.984394`, classification `target-like`.
- After energy-only continuation, FE-DOF target agreement collapsed: `disp_global` energy end had `He_current_corr=-0.152065`; `disp_strain_global` energy end had `He_current_corr=-0.173204`.
- The simple global curriculum ramping fully to energy ended as `boundary-dominated` with `He_current_corr=-0.009037`.
- Interpretation: this run supports energy-objective or optimizer-path failure over global ansatz expression failure.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/global_prefit_case_comparison.csv`
- `tables/energy_continuation_metrics.csv`
- `tables/global_strain_reconstruction_metrics.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Given that global prefit can fit the FE-DOF branch but energy continuation moves away from it, should the next diagnostic focus on energy scaling, optimizer path, or a non-geometry global proximal anchor?
2. What non-geometry-specific curriculum should be tested next while keeping notch/lip labels out of the training objective?
3. Which metrics should define successful preservation of the FE-DOF-like branch in the next mechanics-only run?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, or local displacement-jump targets to the training loss unless explicitly requested.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this mechanics-only diagnostic.
