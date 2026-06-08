# Coordinate Normalization / Alpha Initialization 2x2 Comparison

This package contains a near-production diagnostic comparison for
`TM_comsol_no_thermal_micro`.

Run level:

- Network: `8 x 400`, `TrainableReLU`, seed `2`
- Coupling: coupled, `mixed_mechanics_mode=history`
- Boundary ansatz: `top_u_mode=free`
- Load schedule: first 12 steps from `load_schedule_D0020_extended.csv`
- Final analyzed step: step `11`, `Delta = 4.9e-5`
- Optimizer budget: `RPROP=300`, `LBFGS=0`

This is not a full D0020 production validation. It is a controlled
near-production diagnostic to test whether coordinate normalization changes
the branch selected by the same physical/model settings.

Cases:

- `default_none`: default alpha initialization, `coord_normalization=none`
- `default_unitbox`: default alpha initialization, `coord_normalization=unit_box`
- `intact_none`: `--alpha-init-intact`, `coord_normalization=none`
- `intact_unitbox`: `--alpha-init-intact`, `coord_normalization=unit_box`

Read first:

- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `figures/figure_summary.md`
- `HANDOFF_COMMENT.md`

