# PINN Prefit to FE-DOF Mechanics Package

This package performs a mechanics prefit diagnostic for `TM_comsol_no_thermal_micro`.

Package path:

`examples/TM_comsol_no_thermal_micro/runs/20260608_pinn_prefit_fedof_mechanics`

## Read First

1. `REPORT.md`
2. `tables/prefit_case_comparison.csv`
3. `tables/notch_lip_prefit_metrics.csv`
4. `tables/strain_he_reconstruction_metrics.csv`
5. `figures/figure_summary.md`
6. `HANDOFF_COMMENT.md`

## Purpose

The diagnostic uses FE-DOF alpha-zero mechanics fields from the previous package as supervised targets and trains only the current PINN displacement ansatz to fit `u/v`.

It separates two hypotheses:

- The current PINN displacement ansatz cannot express the narrow-notch localized mechanics field.
- The ansatz can express the field, but energy-only optimization does not naturally find that branch.

## Scope and Constraints

- No physical model changes.
- No `l0` change.
- No material parameter change.
- No `tm_source` split change.
- No phase-field notch addition.
- No imposed `alpha=1` on the real geometric notch.
- No history update change.
- No coupled phase-field full run.

Targets come from:

- `../20260608_mechanics_only_notch_ansatz/artifacts/fedof_free_log10_energy_e300_fields.npz`
- `../20260608_mechanics_only_notch_ansatz/artifacts/fedof_fixed_log10_energy_e300_fields.npz`
