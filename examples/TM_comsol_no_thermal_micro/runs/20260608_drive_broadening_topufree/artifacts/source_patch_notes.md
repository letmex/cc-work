# Source Patch Notes

The local PINN project at `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro` is not a git repository, so this package includes snapshots of the changed/added files under `artifacts/`.

Included snapshots:
- `artifacts/analyze_drive_broadening_stepwise.py`
- `artifacts/history_field_mixed_tm.py`
- `artifacts/train_mixed_tm.py`
- `artifacts/test_history_mode_controls.py`

Code-level changes represented by these snapshots:
- Added `analyze_drive_broadening_stepwise.py` to compute stepwise alpha, drive, region, ratio, event-ordering, and final comparison summaries from existing full-run NPZ/CSV outputs.
- Added boundary displacement diagnostics in `history_field_mixed_tm.py`.
- Passed `u`, `v`, and `top_u_mode` into the mixed TM summary writer from `train_mixed_tm.py`.
- Added regression tests for top-u-free diagnostics and the drive-broadening script.

No physical model parameter was intentionally changed by this diagnostic package.
