# Figure Summary

No PNG figures are included in this package.

The primary evidence is tabular:

- `tables/exact_fe_summary.csv` records exact FE free/fixed baseline quantities.
- `tables/mechanics_field_comparison.csv` compares exact FE, FE-DOF RPROP, supervised PINN prefit, and collapsed PINN fields.
- `tables/energy_decomposition_comparison.csv` compares standard FE internal energy and current PINN mechanics energy under identical postprocessing assumptions.
- `tables/residual_comparison.csv` and `tables/boundary_residuals.csv` summarize residual and boundary/reaction checks.

These tables support a diagnostic conclusion only: the previous FE-DOF RPROP target is not a trustworthy alpha=0 mechanics supervision target relative to the direct sparse FE solve. They do not support a physical validation claim.

