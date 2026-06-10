# Project Directory Cleanup Report

Classification: project directory cleanup completed.

## 1. Deleted generated files and folders

The cleanup removed 140 root-level generated or obsolete items from the real example folder. Deletion categories:

- cache folder to delete: 2
- generated debug CSV/NPZ/output file to delete: 49
- generated log folder to delete: 4
- generated training result folder to delete: 64
- obsolete compatibility wrapper to delete or deprecate: 2
- one-off debug script to delete: 19

The detailed deletion list is in `tables/deleted_artifacts.csv`. The initial inventory is in `tables/root_directory_inventory.csv`.

## 2. Deleted one-off debug scripts

Root-level `debug_*.py` scripts and one-off diagnostic scripts such as `analyze_drive_broadening_stepwise.py`, `mesh_l0_diagnostics.py`, and `validate_mechanics_target.py` were deleted. Old user-facing wrappers `corrected_reaction_postprocess.py` and `plot_clean_tm_results.py` were also removed from the root. See `tables/script_cleanup_audit.csv`.

## 3. Remaining root contents

The root now contains only source files, documentation, mesh/load-schedule inputs, `source/`, `tests/`, `outputs/`, and `runs/`. The final inventory is in `tables/root_directory_inventory_after_cleanup.csv`, and an allowlist view is in `tables/remaining_root_files.csv`.

## 4. Future training outputs

Training defaults are redirected to managed directories:

- checkpoints: `outputs/checkpoints/<run_id>/`
- results: `outputs/results/<run_id>/`
- logs: `outputs/logs/<run_id>/`

Advanced path overrides remain available through `--output-root`, `--model-dir`, `--result-dir`, `--log-dir`, `--figure-dir`, and `--curve-dir`. See `tables/output_path_default_audit.csv`.

## 5. Future postprocessing outputs

`postprocess_results.py` writes curves to `<result_dir>/curves` and figures to `<result_dir>/figures` by default. `plot_results.py` defaults direct plots to `outputs/figures/<label>` and searches `outputs/results` before legacy `results`.

## 6. .gitignore update

The example-level `.gitignore` ignores `outputs/`, legacy `results/`, root training folders, root logs, debug CSV/NPZ outputs, pycache, and pytest cache. See `tables/gitignore_policy_audit.csv`.

## 7. Script naming cleanup

The functional user-facing entry points are now `postprocess_results.py` and `plot_results.py`. Normal generated table/figure names use `stress_strain_by_step.csv`, `reaction_by_step.csv`, `stress_strain_<label>.png`, and `reaction_strain_<label>.png`; normal filenames no longer use temporary `clean` or `corrected` labels.

## 8. Tests updated

Tests now check root cleanliness, managed output defaults, functional postprocess names, and temporary-directory based postprocess behavior. See `tables/test_cleanup_summary.csv`.

## 9. Root cleanliness check

`tables/root_cleanliness_check.csv` reports pass for forbidden root patterns and the final root allowlist.

## 10. Physics changes

No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, or training losses were changed.

## 11. Additional seed or D0040 runs

No new seed study was run. D0040 was not run. The verification was limited to tests, py_compile, file deletion, path policy, and documentation/evidence generation.

## Notes

Importing `config.py` during tests can create managed output subfolders under `outputs/`. That is expected and is the intended replacement for root-folder pollution.
