# Step-0 Root-Cause Diagnostic Package

This package investigates why `alpha-init history` full runs show broad/background drive at step 0.

Scope:
- Inspect saved full-run step-0 fields for top-u fixed, top-u free, and old history reference runs.
- Check displacement/strain ansatz behavior near the narrow explicit notch lips.
- Check whether early optimizer budget changes the step-0 broad-drive pattern.
- Recompute `He_current` from saved PINN strain fields.
- Compare against a small FE-DOF alpha-zero baseline at `Delta = 1e-6`.

This is a diagnostic package only. It is not physical validation and it does not change the physical model.

Unchanged:
- `l0`
- material parameters
- `tm_source` split
- phase-field notch behavior
- `alpha=1` geometric notch seeding
- thermal field / thermal expansion
- history update logic
- `Gc / GcII`

Read first:
- `REPORT.md`
- `reports/step0_root_cause_summary.md`
- `tables/step0_field_summary.csv`
- `tables/optimizer_budget_step0_summary.csv`
- `tables/recompute_he_step0_summary.csv`
- `tables/fedof_step0_summary.csv`
- `figures/figure_summary.md`
