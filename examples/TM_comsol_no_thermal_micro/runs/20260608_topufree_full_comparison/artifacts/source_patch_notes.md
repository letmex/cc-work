# Artifact Notes

The local PINN project is not a git repository. This package includes a snapshot of the analysis script used to generate the CSV tables:

- `artifacts/analyze_drive_broadening_stepwise.py`

Analysis-script changes for this package:

- Added absolute drive threshold events:
  - `first_notch_He_gt_1e-8`
  - `first_notch_He_gt_1e-6`
  - `first_bulk_He_p95_gt_1e-8`
  - `first_bulk_He_p95_gt_1e-6`
- Added `first_step_where_ratio_valid` with threshold `notch_tip_He_current_max > 1e-8`.
- Added valid ratio fields that are set only after the denominator threshold is satisfied.
- Added boundary displacement diagnostics into stepwise CSVs, including `top_u_mode`, `top_u_abs_max`, `top_v_error_max`, `bottom_u_abs_max`, and `bottom_v_abs_max`.

No physical model parameter was changed by this analysis-script update.
