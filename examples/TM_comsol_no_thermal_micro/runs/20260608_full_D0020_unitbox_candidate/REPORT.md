# Full D0020 Unit-Box Candidate Report

## Purpose

The previous 8x400, 12-step 2x2 diagnostic supported `coord_normalization =
unit_box` as the next controlled full-D0020 candidate. This package runs the
full 34-step D0020 schedule with the same physical/model settings and the full
default optimizer budget.

No changes were made to `l0`, material parameters, `tm_source`, phase-field
notch behavior, thermal terms, or history update logic. No `alpha=1` condition
was imposed on the geometric notch. No notch/lip loss, masks, local weights,
displacement-jump targets, enrichment, or geometry-label guidance were added.

## Runs

Primary run:

```text
history + alpha-init-intact + top-u-mode free + coord_normalization unit_box
```

Optional comparison run:

```text
history + default alpha init + top-u-mode free + coord_normalization unit_box
```

Both runs used `8 x 400`, seed `2`, `TrainableReLU`, AT2, full
`load_schedule_D0020_extended.csv`, and default full optimizer settings
`RPROP=10000`, `LBFGS=1`.

## Final-Step Summary

Final step: `33`, `Delta = 1.0e-4`.

| case | alpha init | alpha_mean | alpha_std | alpha_max | alpha>0.5 area | bulk He / notch He | bottom He / notch He | reaction_N_tm_eff | classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| intact_unitbox_full | alpha-init-intact | 0.113297 | 0.191751 | 1.001266 | 0.052116 | 0.0000197 | 0.0000182 | 0.820360 | A. 12-step localized -> full-stage localized |
| default_unitbox_full | default | 0.112522 | 0.191365 | 1.001675 | 0.051377 | 0.0000146 | 0.0000180 | 0.894020 | A. 12-step localized -> full-stage localized |

The detailed table is `tables/final_case_comparison.csv`.

## Required Questions

1. Did the full alpha-init-intact + unit_box run complete?

Yes. The primary run completed all 34 D0020 steps with full default optimizer
budget. The final field file is `fields_mixed_tm_step_0033.npz`.

2. Does the notch-localized branch survive all 34 D0020 steps?

Yes for the primary run. At step 11, the case was notch-localized
(`alpha_max = 1.000779`, `bulk/notch He = 0.000250`). At final step 33, it
remained notch-localized (`alpha_max = 1.001266`, `bulk/notch He =
0.0000197`). The maximum `He_current`, `He_history`, and mechanics-drive
locations remained at approximately `(x, y) = (0.005046, 0.005040)`.

3. Does alpha remain localized, or does background damage grow later?

Alpha remains dominated by the notch-localized branch. The mean alpha increased
from `0.105958` at step 11 to `0.113297` at final step 33 in the primary run,
and the final `alpha>0.5` area fraction was `0.052116`. There is low-level
background alpha, but it does not become the dominant drive pattern in this
full run. The bulk/notch and bottom/notch He ratios decrease further by the
final step.

4. Do `He_current`, `He_history`, and mechanics_drive stay notch-centered?

Yes. For both the primary and optional default-alpha runs, final maximum
locations for `He_current`, `He_history`, and mechanics_drive remain at the
notch neighborhood near `(0.005046, 0.005040)`. Final bulk/notch He ratios are
below `2e-5` for both runs.

5. What happens to reaction/stress-strain?

The reaction remains positive over the full schedule. Final
`reaction_N_tm_eff` is `0.820360` for the primary alpha-init-intact run and
`0.894020` for the optional default-alpha run. The stress/reaction curves are
included in `tables/reaction_stress_strain.csv` and the comparison figures.
This reaction behavior should be treated as a diagnostic output of this
single-seed branch, not physical stiffness validation.

6. Was default-alpha + unit_box also run? If yes, how does it compare?

Yes. The optional default-alpha + unit_box full run also completed all 34 steps
and classified as `A. 12-step localized -> full-stage localized`. Its final
alpha and drive metrics are very close to the primary run, with slightly higher
final reaction (`0.894020` versus `0.820360`). This suggests that under
`unit_box`, alpha initialization is not the dominant factor for this seed.

7. Is another seed justified next?

Yes. This package is a full-schedule candidate for seed 2 only. At least one
additional seed is justified before treating the branch as robust. A matched
seed sweep should preserve `unit_box`, `l0`, material parameters, `tm_source`,
history update logic, and boundary/geometry assumptions.

8. What cannot be concluded?

This package does not prove physical validation, seed robustness, material
parameter correctness, or platform equivalence. It does not justify changing
`l0`, material parameters, `tm_source`, phase-field notch behavior, thermal
terms, or history update logic. It only shows that the previously observed
12-step `unit_box` notch-localized branch survives the full D0020 schedule for
seed 2.

## Verification

Project-local tests:

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
```

Result:

```text
18 passed in 1.71s
```

`py_compile` passed for:

```text
analyze_drive_broadening_stepwise.py
plot_clean_tm_results.py
config.py
field_computation.py
train_mixed_tm.py
```

