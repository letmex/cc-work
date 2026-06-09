## Codex handoff: actual saved-PINN reaction postprocessing

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260616_default_unitbox_pinn_energy_reaction_postprocess
Main report: examples/TM_comsol_no_thermal_micro/runs/20260616_default_unitbox_pinn_energy_reaction_postprocess/REPORT.md

### What changed
- Added and ran saved-field reaction postprocessing for D0040 seeds 7/13/42 and D0020 seeds 7/13/21/42/99.
- Audited checkpoint/model availability for exact actual-PINN autograd `dPi/dDelta`.
- Computed saved-field energy finite-difference proxies, saved-field virtual-work proxies, boundary reactions, internal cut forces, and through-crack reaction summaries.
- Regenerated reaction metric comparison figures without changing physics or training.

### Commands run
```powershell
git pull origin main
Read 20260615_default_unitbox_fedof_reaction_reference handoff/report/tables/figure summary.
D:\anaconda3\envs\torch_env\python.exe artifacts\run_pinn_energy_reaction_postprocess.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260616_default_unitbox_pinn_energy_reaction_postprocess\artifacts\run_pinn_energy_reaction_postprocess.py
```

### Key results
- Identified cause/status: **reaction postprocessing unresolved: exact actual-PINN dPi/dDelta unavailable because checkpoints are absent**.
- Exact autograd availability: 0/8 target runs, D0040 exact availability 0/3.
- All numerical alternative reaction metrics in this package are saved-field proxies, not exact actual-PINN `dPi/dDelta`.
- No production mechanics change is justified from this package.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/saved_artifact_availability.csv`
- `tables/pinn_energy_conjugate_reaction_by_step.csv`
- `tables/saved_field_energy_proxy_reaction.csv`
- `tables/pinn_virtual_work_reaction.csv`
- `tables/pinn_reaction_boundary_cut_consistency.csv`
- `tables/post_peak_drop_by_metric.csv`
- `tables/through_crack_reaction_by_metric.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should `reaction_N_tm_eff` be demoted now, or only relabeled until exact checkpointed `dPi/dDelta` is available?
2. Should the next Codex task add checkpoint/model-settings saving plus exact reaction hooks to future runs?
3. Is a short checkpointed D0040 rerun the minimal intervention needed to resolve reaction postprocessing?

### Constraints
- Do not extend loading.
- Do not retrain the main model unless explicitly requested.
- Do not evolve alpha in this postprocessing task.
- Do not change `l0`, material parameters, thermal terms, TM split, history update logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance.
- Do not claim physical validation.
