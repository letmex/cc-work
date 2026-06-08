# Coordinate Normalization / Alpha Initialization 2x2 Comparison Report

## Purpose

The previous `unit_box` coupled smoke and 4x100 controlled comparison showed a
possible coordinate-normalization effect, but that result was too small to use
for model-effect decisions. This package reruns the comparison with a
production-size network (`8 x 400`) while keeping the optimizer budget bounded.

All cases keep the same `l0`, material parameters, `tm_source` split, AT2
phase-field route, thermal-free terms, history update logic, explicit COMSOL
micro-notch geometry, and `top_u_mode=free`. No `alpha=1` condition was imposed
on the geometric notch.

## Run Level

All four cases were run with `8 x 400`, seed `2`, `TrainableReLU`, coupled
training, `mixed_mechanics_mode=history`, `top_u_mode=free`, and the first 12
steps of `load_schedule_D0020_extended.csv`. The optimizer budget was
`RPROP=300`, `LBFGS=0` for every case.

This is a near-production diagnostic, not validation. It uses the production
network size, but not the full 34-step D0020 schedule or full optimizer budget.

## Final-Step Summary

Final analyzed step: step `11`, `Delta = 4.9e-5`.

| case | alpha init | coord | alpha_mean | alpha_std | alpha_max | bulk He / notch He | bottom He / notch He | reaction_N_tm_eff | classification |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| default_none | default | none | 0.186222 | 0.000632 | 0.187203 | 1.003120 | 0.975683 | 2.116092 | boundary/corner dominated with near-uniform alpha |
| default_unitbox | default | unit_box | 0.105908 | 0.169213 | 1.000307 | 0.000221 | 0.000148 | 0.735997 | localized/notch-amplified |
| intact_none | alpha-init-intact | none | 0.204560 | 0.063965 | 0.507639 | 0.045656 | 0.035568 | 1.728470 | localized/notch-amplified, weaker than unit_box |
| intact_unitbox | alpha-init-intact | unit_box | 0.106060 | 0.170958 | 1.000230 | 0.000189 | 0.000133 | 0.726342 | localized/notch-amplified |

The detailed numerical table is `tables/final_case_comparison.csv`.

## Questions Answered

1. Were all four 2x2 cases run with 8x400 or another clearly labeled near-production setting?

Yes. All four cases used `8 x 400`, seed `2`, the same first 12 D0020 steps,
and the same `RPROP=300`, `LBFGS=0` budget. The run is clearly labeled as a
near-production diagnostic, not validation.

2. Does `unit_box` improve the default-alpha branch?

Yes in this diagnostic. `default_none` ended with near-uniform alpha
(`alpha_std = 0.000632`, `alpha_max = 0.187203`) and drive ratios near one
relative to the notch. `default_unitbox` ended with notch-amplified damage
(`alpha_max = 1.000307`, `alpha_std = 0.169213`) and very low background drive
ratios (`bulk/notch He = 0.000221`, `bottom/notch He = 0.000148`).

3. Does `unit_box` improve the alpha-init-intact branch?

Yes. `intact_none` already showed a localized/notch-amplified branch, but it
was weaker and more diffuse (`alpha_max = 0.507639`, `bulk/notch He =
0.045656`). `intact_unitbox` reached a stronger notch-localized branch
(`alpha_max = 1.000230`, `bulk/notch He = 0.000189`).

4. Does alpha initialization dominate branch selection?

Not by itself in this diagnostic. Alpha initialization interacts with
coordinate normalization: with raw coordinates, `alpha-init-intact` improves
localization relative to default alpha initialization; with `unit_box`, both
alpha-initialization branches reach similar notch-amplified final states.
Coordinate normalization is the stronger controlled factor in this 12-step,
single-seed comparison.

5. Does any case still show background/uniform damage?

Yes. `default_none` remains near-uniform in alpha and has drive ratios near one
relative to the notch. Its maximum `He_current` location is at `x =
4.3103908e-05`, `y = 0.0038745392`, i.e. near the left boundary side rather
than at the notch tip. This is best read as a raw-coordinate background/boundary
branch, not as a physical crack result.

6. What does the reaction curve/sign show?

All four reaction curves remain positive over the analyzed 12 steps. The two
`unit_box` cases have lower final `reaction_N_tm_eff` (`0.735997` and
`0.726342`) than the raw-coordinate cases (`2.116092` and `1.728470`). This
reaction difference tracks the selected damage branch, but it should not be
used as physical stiffness validation from this single-seed diagnostic.

7. Is full D0020 production validation justified next, or is another diagnostic needed?

A full D0020 production run is justified before model changes, because the
8x400 controlled diagnostic supports `unit_box` as a controlled fix candidate
without changing the physical model. The full run should still be treated as
validation only if it includes enough steps, comparable optimizer budget, and
preferably more than one seed.

8. What cannot be concluded?

This package does not prove physical validation, seed robustness, material
parameter correctness, or final full-D0020 behavior. It also does not justify
changing `l0`, material parameters, `tm_source`, history update logic, thermal
terms, or phase-field notch behavior. The result only supports the diagnostic
claim that raw physical-mm coordinate inputs are still a branch-selection risk
at 8x400, while `unit_box` stabilizes notch-amplified branches in this
controlled subset.

## Files

- `tables/final_case_comparison.csv`: final-step metrics and classification.
- `tables/stepwise_summary.csv`: merged stepwise metrics for all four cases.
- `tables/broadening_events.csv`: merged event-crossing diagnostics.
- `tables/reaction_comparison.csv`: stepwise reaction/energy/loss curves.
- `tables/coord_mapping_diagnostics.csv`: coordinate-map and boundary residual checks.
- `figures/final_alpha_2x2.png`: final alpha fields.
- `figures/final_He_current_2x2.png`: final current drive fields.
- `figures/final_mechanics_drive_2x2.png`: final mechanics-drive fields.
- `figures/reaction_comparison.png`: reaction curves.

## Verification

Project-local tests:

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
```

Result:

```text
18 passed in 1.50s
```

`py_compile` passed for:

```text
analyze_drive_broadening_stepwise.py
plot_clean_tm_results.py
config.py
field_computation.py
train_mixed_tm.py
```

