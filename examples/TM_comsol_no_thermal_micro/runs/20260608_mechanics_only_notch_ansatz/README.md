# Mechanics-Only Notch Ansatz Diagnostic Package

This package tests whether the current PINN displacement/strain ansatz can express localized strain on the narrow explicit notch when alpha is frozen to zero.

Package path:

`examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz`

## Read First

1. `REPORT.md`
2. `tables/key_metrics_summary.csv`
3. `tables/mechanics_only_comparison.csv`
4. `tables/notch_lip_comparison.csv`
5. `figures/figure_summary.md`
6. `HANDOFF_COMMENT.md`

## Scope

- Mechanics-only diagnostic at `Delta = 1e-6`.
- Alpha is frozen to zero, with zero history fields.
- PINN uses the existing displacement ansatz and `8 x 400 TrainableReLU` seed 2 setup.
- FE-DOF uses independent nodal displacement degrees of freedom on the same T3 mesh.
- Top-u fixed and top-u free are both included.
- `log10_energy` and `raw_energy` optimizer scaling are both included.

## Constraints Held Fixed

- `l0 = 1.5e-4 mm`
- material parameters
- `tm_source` split
- geometric/phase-field notch behavior
- alpha initialization and history update logic in the main model

This is a diagnostic package only. It does not claim physical validation.
