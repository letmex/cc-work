## Codex handoff: FE-DOF energy-conjugate reaction reference audit

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260615_default_unitbox_fedof_reaction_reference
Main report: examples/TM_comsol_no_thermal_micro/runs/20260615_default_unitbox_fedof_reaction_reference/REPORT.md

### What changed
- Added and ran a diagnostic FE-DOF frozen-alpha mechanics reference on the existing final_D0040 mesh.
- Tested continuous and piecewise upper/lower topology variants under original top/bottom BC and minimal rigid-body BC.
- Computed top/bottom stress-integral reactions, constrained DOF reactions, energy-conjugate reactions, internal cuts, and free-body residuals.
- No loading, alpha, `l0`, material constants, TM split, thermal terms, or history logic was changed.

### Commands run
```powershell
git pull origin main
Read 20260614_default_unitbox_boundary_reaction_audit handoff/report/tables/figure summary.
D:\anaconda3\envs\torch_env\python.exe artifacts\run_fedof_reaction_reference.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260615_default_unitbox_fedof_reaction_reference\artifacts\run_fedof_reaction_reference.py
```

### Key results
- Identified cause/status: **FE-DOF reference unresolved: energy-relaxed crack-band-void reaction collapses and does not reproduce persistent PINN reaction**.
- Continuous current-split/original-BC top sigma reactions [N]: [0.007959751516285682, 0.007461224731703281, 0.007362949225679203].
- Continuous void/original-BC top sigma reactions [N]: [-2.498043896949281e-14, 2.270166053847394e-15, -2.1801969067599804e-13].
- Continuous void/original-BC energy-conjugate reactions [N]: [-1.713039432527097e-14, 2.6129272356900657e-14, -2.2714035513571318e-13].
- Piecewise void/minimal-BC energy-conjugate reactions [N]: [8.847089727481716e-14, -2.0079424234431542e-13, 1.43982048506075e-13].
- The FE-DOF reference does not reproduce the previous persistent PINN/replay post-crack top reaction after crack-band voiding.
- The diagnostic supports adding an energy-conjugate or constrained-DOF reaction metric to saved-run postprocessing before changing the physical model.
- No production physics change is justified directly from this diagnostic.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/fedof_reference_solve_summary.csv`
- `tables/energy_conjugate_reaction_audit.csv`
- `tables/reaction_metric_comparison.csv`
- `tables/boundary_condition_sensitivity.csv`
- `tables/fedof_free_body_consistency.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the FE-DOF reference strong enough to demote `reaction_N_tm_eff` as the primary post-crack load metric?
2. Should the next Codex task add energy-conjugate/constrained-DOF reaction postprocessing to saved-run analysis without changing the physical model?
3. Is any production model change still unjustified until reaction postprocessing is corrected?

### Constraints
- Do not extend loading.
- Do not evolve alpha.
- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.
- Do not claim physical validation.
