## Codex handoff: TM default unit-box project cleanup

Commit: 0601e4d
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_project_cleanup
Main report: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_project_cleanup/REPORT.md

### What changed
- Normal TM route defaults were moved into code: `pff-model=AT2`, `mixed-mechanics-mode=history`, `top-u-mode=free`, and `coord-normalization=unit_box`.
- `corrected_reaction_postprocess.py` was replaced by the functional entry point `postprocess_results.py`.
- `plot_clean_tm_results.py` was replaced by the functional helper `plot_results.py`.
- Old filenames remain only as deprecated compatibility wrappers with `DeprecationWarning`.
- Training completion now imports and calls `run_results_postprocess(...)` and writes curves plus figures through the renamed path.
- New postprocess outputs use functional names: `stress_strain_by_step.csv`, `reaction_by_step.csv`, and `reaction_metric_availability.csv`.
- Figure filenames use short labels such as `stress_strain_seed23_D0020.png`, not full model directory names.
- README and `POSTPROCESS_WORKFLOW.md` document the automatic postprocess route.
- No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, or training losses were changed.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m pytest tests\test_project_cleanup_interface.py tests\test_postprocess_results.py -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile postprocess_results.py plot_results.py corrected_reaction_postprocess.py plot_clean_tm_results.py main.py config.py tests\test_project_cleanup_interface.py tests\test_postprocess_results.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

### Key results
- Focused cleanup/postprocess tests: `12 passed`.
- Full local example tests: `31 passed`.
- `py_compile` passed for changed scripts and focused tests.
- No new seed study was run.
- D0040 was not run.
- Report classification: `project cleanup completed`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/default_cli_cleanup_audit.csv`
- `tables/program_file_rename_audit.csv`
- `tables/output_filename_cleanup_audit.csv`
- `tables/reaction_metric_naming_policy.csv`
- `tables/postprocess_cli_cleanup_audit.csv`
- `tables/training_completion_postprocess_audit.csv`
- `tables/test_cleanup_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the new default D0020 command clean enough for normal users?
2. Should the deprecated hidden `--corrected-*` compatibility aliases be removed in the next cleanup pass, or kept for one transition period?
3. Should old historical result folders be migrated to new filenames, or left unchanged to preserve prior evidence?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this cleanup package.
- Do not run additional seeds or D0040 for this cleanup.
