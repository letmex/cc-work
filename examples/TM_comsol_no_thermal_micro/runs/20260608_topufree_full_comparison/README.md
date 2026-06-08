# Top-U-Free Full Comparison Package

This package contains a controlled full D0020 seed2 boundary-equivalence diagnostic for `TM_comsol_no_thermal_micro`.

Main comparison:
- old `history + alpha-init-intact` full run with top-u fixed;
- new `history + alpha-init-intact + --top-u-mode free` full run;
- old history default-alpha full run as a reference only.

This package is not physical validation.

Unchanged constraints:
- no change to `l0`;
- no change to `Gc / GcII`;
- no change to `E / nu` or material parameters;
- no change to `tm_source split`;
- no phase-field notch was added;
- no `alpha=1` geometric notch seed was imposed;
- no thermal field or thermal expansion setting was changed;
- no history update logic was changed.

Read first:
- `REPORT.md`
- `reports/topufree_full_comparison.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary_topufixed_vs_topufree.csv`
- `tables/broadening_events_topufixed_vs_topufree.csv`
- `figures/figure_summary.md`
