## Codex handoff: D0020 corrected reaction pipeline

Commit: COMMIT_PLACEHOLDER
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_corrected_reaction_pipeline
Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_corrected_reaction_pipeline/REPORT.md

### What changed
- Implemented corrected energy-consistent D0020 reaction postprocessing in a package script.
- Standardized corrected and legacy reaction metric names.
- Added corrected softening gate summary and legacy metric policy table.
- Regenerated D0020 reaction figures with explicit primary/legacy/diagnostic labels.
- Did not run D0040, extend loading, retrain, or change physics.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m py_compile artifacts\run_d0020_corrected_reaction_pipeline.py
D:\anaconda3\envs\torch_env\python.exe artifacts\run_d0020_corrected_reaction_pipeline.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
```

### Key results
- Classification: **corrected D0020 reaction pipeline validated; legacy reaction demoted**.
- Corrected metric names: `reaction_N_energy_exact`, `reaction_N_energy_virtual_work`, `reaction_N_legacy_top_sigma`, `reaction_N_bottom_sigma_legacy`, `reaction_N_internal_cut_above`, `reaction_N_internal_cut_below`.
- Processed seeds: [7, 13, 42].
- Exact reaction reproduction: 3/3 seeds.
- Energy virtual-work agreement: 3/3 seeds.
- Corrected softening gate pass: 3/3 seeds.
- Legacy `reaction_N_tm_eff` is demoted to diagnostic-only status.
- Old non-checkpointed no-softening conclusions should be relabeled as legacy-metric-only / `reaction_metric_unavailable` for primary classification.
- D0040 remains deferred.
- No production mechanics change is justified by this package.

### Files to read first
- `README.md`
- `REPORT.md`
- `REACTION_METRIC_POLICY.md`
- `tables/corrected_softening_gate_summary.csv`
- `tables/corrected_reaction_by_step.csv`
- `tables/legacy_reaction_metric_policy.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this package correctly promote `reaction_N_energy_exact` as the primary checkpoint reaction metric?
2. Is the legacy demotion policy for `reaction_N_tm_eff` strict enough for future D0020/D0040 reports?
3. What is the next minimal Codex task before D0040 reprocessing?

### Constraints
- Do not run D0040 yet.
- Do not extend loading.
- Do not retrain seed 7/13/42 unless existing checkpoints are missing or corrupt.
- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
