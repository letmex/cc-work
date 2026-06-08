# Unit-Box Coupled Smoke and Controlled Comparison

This package checks whether `coord-normalization unit_box` works in the normal
coupled alpha-init history training path, then compares the same controlled
12-step setup with and without coordinate normalization.

Package root:

```text
examples/TM_comsol_no_thermal_micro/runs/20260608_unitbox_coupled_smoke_comparison
```

Scope:

- Coupled `u/v/alpha` training path.
- `mixed-mechanics-mode history`.
- `alpha-init-intact`.
- `top-u-mode free`.
- `l0 = 1.5e-4`.
- Same material parameters and `tm_source` split.
- No phase-field notch initialization.
- No imposed `alpha=1` at the geometric notch.
- No notch/lip loss, local weights, displacement-jump targets, enrichment, or
  geometry-label guidance.

Runs:

1. `smoke_unitbox_coupled`: 2x20 network, one load step, `coord-normalization unit_box`.
2. `none_alpha_init_history_m12`: 4x100 network, 12 load steps, `coord-normalization none`.
3. `unitbox_alpha_init_history_m12`: 4x100 network, 12 load steps, `coord-normalization unit_box`.

Read first:

1. `REPORT.md`
2. `tables/final_case_comparison.csv`
3. `tables/stepwise_summary.csv`
4. `tables/broadening_events.csv`
5. `figures/figure_summary.md`

