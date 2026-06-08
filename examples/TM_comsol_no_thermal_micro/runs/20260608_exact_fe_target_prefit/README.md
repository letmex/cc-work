# Exact FE Target Prefit Diagnostic

This package replaces the rejected FE-DOF RPROP supervision target with a direct sparse FE alpha=0 top-u-free mechanics target, then tests global-only PINN mechanics prefit and short mechanics-only energy continuation.

No coupled phase-field full training was run. No `l0`, material parameter, `tm_source` split, alpha seeding, phase-field notch behavior, thermal term, or history update logic was changed. No notch-lip loss, notch masks, local weights, displacement-jump target, enrichment, or geometry-label guidance was added.

## Read First

- `REPORT.md`
- `tables/target_guard_check_summary.csv`
- `tables/fedof_rprop_audit.csv`
- `tables/exact_fe_target_summary.csv`
- `tables/exact_target_prefit_metrics.csv`
- `tables/exact_target_energy_continuation.csv`
- `figures/figure_summary.md`

## Main Finding

The old FE-DOF RPROP field is rejected as a mechanics supervision target. The direct sparse FE target passes all guard checks. The PINN can fit the exact target displacement globally with low displacement error, but global displacement fitting does not reconstruct the exact strain and `He_current` field. Short energy continuation from the exact-target strain prefit moves toward a boundary-dominated field with low `He_current` correlation.

