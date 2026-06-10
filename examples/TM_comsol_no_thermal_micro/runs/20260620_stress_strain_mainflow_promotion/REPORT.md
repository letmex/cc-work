# Stress-Strain Mainflow Promotion

## Scope

This package records and validates a mainflow plotting change in external `source/postprocess_tm.py`. The change promotes corrected checkpoint energy reaction curves to the primary stress-strain source and demotes legacy top-sigma curves to diagnostic-only output.

## Classification

**mainflow corrected stress-strain source promoted**.

## What Changed

- `postprocess_tm.py` now searches for `curves/corrected_stress_strain_by_step.csv` or `curves/reaction_displacement_macro_stress_strain_corrected.csv` before using the legacy training curve.
- If corrected energy-conjugate stress is available, `macro_stress_strain.png` is plotted from `nominal_stress_energy_exact_MPa`.
- If only the legacy curve is available, `macro_stress_strain.png` becomes an unavailable-primary notice and the old curve is written as `macro_stress_strain_legacy.png`.
- A `stress_strain_curve_source.txt` report is written for every postprocess run with curve data.
- No D0040 run was launched or processed.

## Validation

- Validation checks passed: 6/6.
- The minimal postprocess fixture selected `nominal_stress_energy_exact_MPa` as the primary curve metric.
- The legacy top-sigma curve was retained only as `macro_stress_strain_legacy.png`.

## Files

- `artifacts/postprocess_tm_before.py`
- `artifacts/postprocess_tm_after.py`
- `artifacts/postprocess_tm_mainflow_promotion.diff`
- `tables/mainflow_curve_source_selection.csv`
- `figures/validated_macro_stress_strain.png`
- `figures/validated_macro_stress_strain_legacy.png`

## Limits

- The modified FEM-PINN source tree is external to `cc-work`, so this package stores the before/after files and diff as evidence.
- This is a plotting/postprocessing mainflow change only; it does not modify training physics.
