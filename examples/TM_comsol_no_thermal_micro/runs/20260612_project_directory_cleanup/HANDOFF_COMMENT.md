## Codex handoff: Project directory cleanup

Commit: 68332f8
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260612_project_directory_cleanup
Main report: examples/TM_comsol_no_thermal_micro/runs/20260612_project_directory_cleanup/REPORT.md

### What changed
- Cleaned the real project root at `D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro` by deleting 140 root-level generated/temporary items.
- Removed root training result folders, legacy `results/`, run log folders, debug CSV/NPZ outputs, one-off `debug_*.py` scripts, and obsolete wrappers `corrected_reaction_postprocess.py` / `plot_clean_tm_results.py`.
- Enforced managed output directories: `outputs/checkpoints`, `outputs/results`, `outputs/figures`, `outputs/curves`, `outputs/logs`, `outputs/debug`, plus `runs/` for audit packages.
- Updated path defaults in `config.py`, `main.py`, `postprocess_results.py`, and `plot_results.py` so normal outputs do not write to the example root.
- Updated docs and tests; source/test snapshots are included because the real project folder is not itself a git repo.

### Commands run
```powershell
git pull origin main
D:/anaconda3/envs/torch_env/python.exe -m pytest tests/test_project_directory_hygiene.py -q
D:/anaconda3/envs/torch_env/python.exe -m pytest tests/test_project_cleanup_interface.py tests/test_postprocess_results.py -q
Remove-Item for 140 candidates from tables/deleted_artifacts.csv with resolved-path root guard
New-Item outputs/checkpoints outputs/results outputs/figures outputs/curves outputs/logs outputs/debug runs
D:/anaconda3/envs/torch_env/python.exe -m pytest -p no:cacheprovider tests -q
py_compile changed source/tests
D:/anaconda3/envs/torch_env/python.exe -m pytest -p no:cacheprovider tests/test_project_directory_hygiene.py tests/test_project_cleanup_interface.py tests/test_postprocess_results.py -q
D:/anaconda3/envs/torch_env/python.exe -m pytest -p no:cacheprovider tests -q
```

### Key results
- Deleted artifact count: 140.
- Root cleanliness after final tests: pass; no `debug_*`, root training folders, root logs, `.pytest_cache`, or `__pycache__` remain.
- Focused tests: 15 passed, 8 warnings.
- Full local example tests: 34 passed, 8 warnings.
- `py_compile`: 8 changed source/test files passed.
- No training, seed study, D0040 run, or physical/model change was performed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/deleted_artifacts.csv`
- `tables/root_cleanliness_check.csv`
- `tables/output_path_default_audit.csv`
- `tables/script_cleanup_audit.csv`
- `tables/test_cleanup_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is this directory hygiene package sufficient to treat the example root as source-only going forward?
2. Should the next task proceed with the managed `outputs/` workflow, or should any additional cleanup policy be added first?
3. Are the removed old wrapper names acceptable, with `postprocess_results.py` and `plot_results.py` as the stable user-facing entry points?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from cleanup, medium, or diagnostic runs.
