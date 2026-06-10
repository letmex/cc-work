# Project Directory Cleanup Evidence Package

Package: `examples/TM_comsol_no_thermal_micro/runs/20260612_project_directory_cleanup`

This package records the cleanup of the real project folder:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro`

The task removed generated root artifacts and moved the normal workflow policy to managed output directories under `outputs/`. No training, seed study, D0040 run, or physics/model change was performed.

Read first:

- `REPORT.md`
- `tables/deleted_artifacts.csv`
- `tables/root_cleanliness_check.csv`
- `tables/output_path_default_audit.csv`
- `tables/script_cleanup_audit.csv`
- `tables/test_cleanup_summary.csv`

Source and test snapshots are under `artifacts/` because the real project folder is not itself a git repository.
