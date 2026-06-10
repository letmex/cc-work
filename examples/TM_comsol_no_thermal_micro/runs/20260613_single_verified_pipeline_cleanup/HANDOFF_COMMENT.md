## Codex handoff: single verified pipeline cleanup

Commit: 00e52bcdb0750e924b8359d8b3db6c300a17907d
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260613_single_verified_pipeline_cleanup
Main report: examples/TM_comsol_no_thermal_micro/runs/20260613_single_verified_pipeline_cleanup/REPORT.md

### What changed
- Simplified the local TM project to one normal route: `mixedH_TM + tm_source + history`.
- Removed the alpha-init-intact branch and obsolete root diagnostic reports.
- Removed user-facing alternate split/mechanics/solver options from normal config/training.
- Reworked normal postprocessing to output energy-conjugate reaction columns only: `reaction_N_energy` and `nominal_stress_energy_MPa`.
- Rewrote docs and tests to describe and guard the single verified workflow.
- Cleared root/generated artifacts and left only managed output directory skeletons.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m pytest -p no:cacheprovider tests\test_single_verified_pipeline.py -q
D:\anaconda3\envs\torch_env\python.exe -m pytest -p no:cacheprovider tests\test_project_cleanup_interface.py tests\test_postprocess_results.py -q
D:\anaconda3\envs\torch_env\python.exe -m pytest -p no:cacheprovider tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile config.py main.py mixed_mode_tm.py compute_energy_mixed_tm.py train_mixed_tm.py history_field_mixed_tm.py postprocess_results.py plot_results.py tests\test_single_verified_pipeline.py tests\test_project_cleanup_interface.py tests\test_postprocess_results.py tests\test_history_mode_controls.py tests\test_project_directory_hygiene.py tests\test_coord_normalization.py
rg forbidden old-route terms in normal source/docs
PowerShell root artifact scan
```

### Key results
- Full tests: `38 passed, 8 warnings`.
- `py_compile`: exit 0.
- Forbidden-term check on normal source/docs: pass.
- Root cleanliness check: pass.
- No training run, no seed study, and no D0040 run were performed.
- No physics formula was intentionally changed; unused alternate branches and old output metrics were removed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/forbidden_term_check.csv`
- `tables/reaction_metric_cleanup_audit.csv`
- `tables/cli_simplification_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the normal route now sufficiently strict, or should the retained advanced `--top-u-mode` and `--coord-normalization` overrides be moved out of the normal CLI too?
2. Should the next Codex task run only a short smoke training of this simplified route, or first review the source snapshots for accidental loss of useful diagnostics?
3. Should training summaries receive energy-conjugate reaction values only after postprocess succeeds, or stay reaction-free during training?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from cleanup, medium, or diagnostic runs.
