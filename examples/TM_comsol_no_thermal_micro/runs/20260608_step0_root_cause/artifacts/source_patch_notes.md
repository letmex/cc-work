# Source Patch Notes

The local PINN project is not a git repository. This evidence package includes a snapshot of the diagnostic script:

- `artifacts/debug_step0_root_cause.py`

Purpose of the script:
- summarize saved full-run step-0 fields;
- compute notch-lip displacement jump diagnostics;
- run small step-0 optimizer budget sweeps;
- write CSV and Markdown summaries.

The script is diagnostic-only. It does not change `l0`, material parameters, TM split, phase-field notch seeding, thermal field, or history update logic.
