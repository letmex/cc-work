# Exact FE Mechanics Audit

This package audits the alpha=0 mechanics objective for `TM_comsol_no_thermal_micro`.

The diagnostic assembles and solves a direct sparse P1/T3 plane-stress FE elasticity problem on the same mesh with `Delta = 1e-6`, then compares it against existing FE-DOF RPROP, supervised PINN prefit, and collapsed PINN energy-continuation fields.

No coupled phase-field full training was run. No physical model parameters, `l0`, material parameters, `tm_source` split, phase-field notch behavior, alpha seeding, thermal terms, or history update logic were changed.

## Read First

- `REPORT.md`
- `tables/exact_fe_summary.csv`
- `tables/mechanics_field_comparison.csv`
- `tables/energy_decomposition_comparison.csv`
- `tables/residual_comparison.csv`
- `figures/figure_summary.md`

## Main Finding

The direct alpha=0 FE solve is notch-amplified at the explicit notch and has tiny free-DOF residuals. The previous FE-DOF RPROP target and supervised PINN prefit fields are not close to this exact FE solution; they sit on a high-energy, high-residual branch. Collapsed PINN energy-continuation fields are much closer to the exact FE mechanics energy than the FE-DOF target.

