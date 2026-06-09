## Codex handoff: boundary/reaction/free-body audit

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260614_default_unitbox_boundary_reaction_audit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260614_default_unitbox_boundary_reaction_audit/REPORT.md

### What changed
- Added and ran a diagnostic-only boundary/reaction/free-body audit.
- Audited final_D0040 split-domain replay fields for seeds 7, 13, 42 and variants continuous_baseline, split_domain_current_split, split_domain_minus_degraded_crack_band, and split_domain_crack_band_void.
- Reconstructed total/effective/variant stress fields from saved u/v, alpha, and strain fields.
- Computed top/bottom/left/right boundary force integrals, upper/lower subdomain free-body terms, internal cut consistency, boundary-condition audit, and synthetic rigid-body sanity tests.
- No loading, alpha, material, `l0`, TM split, or history logic was changed.

### Commands run
```powershell
git pull origin main
Read 20260613_default_unitbox_discontinuous_kinematic_replay handoff/report/tables/figure summary.
D:\anaconda3\envs\torch_env\python.exe artifacts\run_boundary_reaction_audit.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260614_default_unitbox_boundary_reaction_audit\artifacts\run_boundary_reaction_audit.py
```

### Key results
- Identified cause/status: **reaction/boundary cause identified: boundary ansatz overconstrains separated subdomains and top reaction is a local boundary stress metric**.
- `reaction_N_tm_eff` is a top-boundary `sigma_yy_tm_eff` integral; it is locally conjugate to top vertical displacement but is not by itself a global cracked-ligament load metric.
- Crack-band-void replay still has nonzero top reactions for seeds 7 and 13, while synthetic piecewise-rigid upper/lower fields nearly remove top reaction under crack-band-void treatment.
- Boundary-condition and prefit effects remain plausible contributors; no production model change is justified directly.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/reaction_definition_audit.csv`
- `tables/all_boundary_force_balance.csv`
- `tables/subdomain_free_body_audit.csv`
- `tables/reaction_vs_internal_cut_consistency.csv`
- `tables/boundary_condition_overconstraint_audit.csv`
- `tables/rigid_body_sanity_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the dominant reaction/boundary cause identified across at least 2/3 seeds?
2. Should the next minimal diagnostic be a FE-DOF free-body/reaction comparison rather than another PINN split replay?
3. Is any production model change justified yet?

### Constraints
- Do not extend loading.
- Do not evolve alpha.
- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.
- Do not claim physical validation.
