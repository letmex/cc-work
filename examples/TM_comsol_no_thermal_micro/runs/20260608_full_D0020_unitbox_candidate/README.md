# Full D0020 Unit-Box Candidate

This package contains the first full-D0020 production-path validation candidate
for `TM_comsol_no_thermal_micro` using the controlled coordinate normalization
fix:

```text
coord_normalization = unit_box
```

Two full runs were executed:

- Primary route: `history + alpha-init-intact + top-u-mode free + unit_box`
- Optional comparison: `history + default alpha init + top-u-mode free + unit_box`

Both runs used:

- `hidden_layers = 8`
- `neurons = 400`
- `seed = 2`
- `TrainableReLU`
- `init_coeff = 3.0`
- `AT2`
- `mixed_mechanics_mode = history`
- `l0 = 1.5e-4`
- full `load_schedule_D0020_extended.csv`, 34 steps
- full default optimizer budget: `RPROP=10000`, `LBFGS=1`

This is a full-schedule single-seed candidate, not physical validation by
itself.

Read first:

- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `figures/figure_summary.md`
- `HANDOFF_COMMENT.md`

