# D0020 Stress-Strain Curve Fix

## Scope

This package fixes the D0020 stress-strain curve output by using the corrected energy-conjugate reaction as the primary stress source. It does not run or process D0040.

## Classification

**D0020 stress-strain curve softening fixed**.

## What Changed

- Primary curve source is `reaction_N_energy_exact`, not legacy `reaction_N_tm_eff`.
- Nominal stress is computed as `reaction_N_energy_exact / 0.01 mm^2` using the same 0.01 mm specimen width and unit-thickness convention used by the reaction scripts.
- Legacy top sigma is retained only as `nominal_stress_legacy_top_sigma_MPa` for comparison.
- D0040 is intentionally not run or reprocessed.

## Results

- Corrected primary stress-strain curve softens in 3/3 D0020 seeds.
- Legacy top-sigma diagnostic softens by the same 50% gate in 0/3 seeds.
- Corrected and legacy curve conclusions disagree in 3/3 seeds.

## Files

- `tables/corrected_stress_strain_by_step.csv`
- `tables/stress_strain_softening_summary.csv`
- `tables/stress_strain_curve_source_policy.csv`
- `figures/D0020_corrected_nominal_stress_strain.png`
- `figures/D0020_corrected_vs_legacy_stress_strain.png`
- `figures/D0020_stress_strain_softening_gate.png`

## Limits

- This fixes the curve output for checkpointed D0020 using already validated corrected reactions.
- This is not a D0040 validation and not a physical validation claim.
- Non-checkpointed old curves remain `reaction_metric_unavailable` for corrected primary stress-strain classification.
