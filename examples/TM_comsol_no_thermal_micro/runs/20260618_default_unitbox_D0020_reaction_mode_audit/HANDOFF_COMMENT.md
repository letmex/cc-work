## Codex handoff: D0020 reaction-mode audit

Commit: UNSET
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260618_default_unitbox_D0020_reaction_mode_audit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260618_default_unitbox_D0020_reaction_mode_audit/REPORT.md

### What changed
- Audited the pre-through exact/legacy reaction scaling mismatch using existing checkpointed D0020 seeds 7, 13, and 42.
- Did not run D0040, retrain, extend loading, or change physics.
- Added reaction-mode, virtual-work, boundary-work, unit-scaling, and linearity tables.

### Commands run
```powershell
git pull origin main
Read previous D0020 exact-reaction handoff/report/tables.
D:\anaconda3\envs\torch_env\python.exe -m py_compile artifacts\run_d0020_reaction_mode_audit.py
D:\anaconda3\envs\torch_env\python.exe artifacts\run_d0020_reaction_mode_audit.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
```

### Key results
- Classification: **reaction-mode audit unresolved**.
- Global Delta pre-through ratio: 2.17453.
- Pure top-vertical pre-through ratio: 2.17276.
- No-horizontal pre-through ratio: 2.17478.
- Pure top-vertical collapse count: 3/3.
- Current-stress virtual-work relative error before through: 1.08384.
- Legacy reaction metric is not demoted because pure top-vertical reaction does not agree with legacy before through-crack onset.
- D0040 remains deferred.
- No production mechanics change is justified by this package alone.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/reaction_by_loading_mode_summary.csv`
- `tables/reaction_by_loading_mode.csv`
- `tables/delta_loading_mode_decomposition.csv`
- `tables/virtual_work_identity_check.csv`
- `tables/boundary_work_decomposition.csv`
- `tables/reaction_unit_scaling_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this audit indicate the remaining mismatch is a stress-postprocessing / actual-energy-conjugacy issue rather than a loading-mode issue?
2. What is the next minimal diagnostic to reconcile exact energy derivative with a stress-based virtual-work reaction?
3. Should Codex defer D0040 until the D0020 reaction metric is accepted?

### Constraints
- Do not run D0040 yet.
- Do not extend loading.
- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
