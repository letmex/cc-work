# Global Anchor Energy Continuation Diagnostic

This package tests whether an FE-DOF-like alpha=0 mechanics branch can be preserved during mechanics-only energy continuation using only global, non-geometry-specific controls.

## Scope

- Project: `examples/TM_comsol_no_thermal_micro`
- Mode: mechanics-only
- Alpha: fixed to zero
- Delta: `1e-6`
- Boundary mode: `top-u free`
- Network: current 8x400 `TrainableReLU` displacement ansatz
- Target: FE-DOF alpha=0 mechanics field from `20260608_mechanics_only_notch_ansatz`

No coupled phase-field full training was run.

## Constraints Kept

- No physical model change
- No `l0` change
- No material parameter change
- No `tm_source` split change
- No thermal term change
- No history update change
- No phase-field notch or alpha seeding change
- No notch-lip loss, notch-tip/lip mask, local notch weight, displacement-jump target, or geometry-label guidance in the training loss

Region metrics such as notch-tip, bulk, and bottom-right values are postprocessing diagnostics only.

## Continuation Modes

The package includes:

1. `pure_energy_baseline`: start from `disp_strain_global` prefit and continue with pure log10 mechanics energy.
2. `global_displacement_anchor`: normalized mechanics energy plus global displacement anchor with `lambda_u` sweep.
3. `global_strain_anchor`: normalized mechanics energy plus global strain anchor with `lambda_eps` sweep.
4. `global_displacement_plus_strain_anchor`: combined global displacement and global strain anchors.
5. `trust_region_to_previous_step`: chunked continuation with a global change penalty to the previous accepted displacement field.
6. `energy_normalization_variants`: raw energy and normalized energy comparisons.

## Read First

- `REPORT.md`
- `tables/anchor_sweep_metrics.csv`
- `tables/success_threshold_summary.csv`
- `tables/continuation_checkpoint_metrics.csv`
- `figures/figure_summary.md`

