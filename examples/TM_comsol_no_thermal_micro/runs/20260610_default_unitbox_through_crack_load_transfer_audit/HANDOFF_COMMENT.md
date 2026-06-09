## Codex handoff: through-crack load-transfer audit

Commit: 89bb0ce10affff6f71ed9a7e2572ca7fcc0ae14d
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_default_unitbox_through_crack_load_transfer_audit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_default_unitbox_through_crack_load_transfer_audit/REPORT.md

### What changed
- Postprocessed existing D0040 seed 7/13/42 fields only; no new training or load extension was run.
- Confirmed through-crack onset for alpha thresholds 0.5, 0.8, and 0.95.
- Audited cut-line load transfer, reaction decomposition, mechanics training path, and stress split sanity.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe artifacts\build_through_crack_load_transfer_audit.py
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260610_default_unitbox_through_crack_load_transfer_audit\artifacts\build_through_crack_load_transfer_audit.py
```

### Key results
- Identified status: **through-crack load-transfer cause identified**.
- Cause evidence: high-alpha crack band still transmits effective traction dominated by non-degraded negative/compressive stress; positive tensile stress is degraded correctly in the crack band and residual stiffness is negligible.
- Final cut-line mean |sigma_yy_tm_eff| in alpha>=0.8 band averaged over cases/cuts: 37.3096.
- Final cut-line mean |minus|/|effective| in alpha>=0.8 band: 1.00048.
- Final cut-line max residual-stiffness positive contribution: 1.07547e-06.
- Final mean v-jump proxy across cut lines: 0.000178971 mm.
- Mechanics path audit: alpha degradation enters training energy, not only postprocessing.
- Reaction path audit: `reaction_N_tm_eff` integrates degraded `sigma_yy_tm_eff`.
- Verification: `pytest examples\TM_comsol_no_thermal_micro\tests -q` passed with 18 tests; package audit script `py_compile` passed.
- No physical validation is claimed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/through_crack_geometry_audit.csv`
- `tables/crack_section_load_transfer_audit.csv`
- `tables/reaction_decomposition_audit.csv`
- `tables/mechanics_training_path_audit.csv`
- `tables/stress_split_sanity_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this evidence identify the dominant reason for continued post-crack load transfer?
2. Is the next minimal intervention a fixed-alpha kinematic/enrichment diagnostic, or a deeper audit of the non-degraded negative branch?
3. What exact Codex prompt should run next without changing physical parameters?

### Constraints
- Do not extend loading as the main action.
- Do not change `l0`, material parameters, TM split, thermal terms, or history update logic unless a clear bug is found.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance in this diagnostic unless explicitly requested.
- Do not use `--alpha-init-intact` as the main route.
- Do not claim physical validation.
