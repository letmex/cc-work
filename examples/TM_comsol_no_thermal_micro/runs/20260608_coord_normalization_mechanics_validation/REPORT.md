# Coordinate Normalization Mechanics Validation Report

## Purpose

The diagnostic tests whether coordinate input normalization should be promoted
into the normal PINN training path as a controlled mechanics ansatz
implementation fix. The test is mechanics-only with alpha fixed to zero and
uses the accepted exact-FE target:

```text
examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz
```

## Implementation Check

Coordinate normalization was implemented only at the NN input level. In
`FieldComputation.fieldCalculation(inp)`, the network now receives
`self.network_input(inp)`. The boundary ansatz still uses the original physical
`inp` coordinates, and T3 gradients still receive the physical mesh
coordinates.

The default option is:

```text
coord_normalization = none
```

The controlled non-default option is:

```text
coord_normalization = unit_box
```

For `unit_box`, physical `x,y` in mm are mapped to `[-1,1]` before feeding the
network. No material parameter, `l0`, TM split, alpha seeding, phase-field
notch behavior, thermal term, or history update was changed.

## Tests

Local project tests passed:

```text
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
18 passed
```

`py_compile` also passed for modified scripts and the new diagnostic script.

Repository-root tests were attempted:

```text
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

They failed during collection because
`ref_files.Chinese_SENT_reproduction` is not available. The failure is outside
`TM_comsol_no_thermal_micro`.

## Key Results

Main table:

```text
tables/coord_normalization_case_comparison.csv
```

Coordinate mapping table:

```text
tables/coord_mapping_diagnostics.csv
```

Mapping diagnostics confirm:

- `none`: network input spans physical `[0, 0.01]` in both x and y.
- `unit_box`: network input spans `[-1, 1]` in both x and y.
- `t3_gradients_use_physical_xy = True`.

Representative mechanics-only results:

| case | displacement rel RMSE | strain rel RMSE | He_current corr | energy ratio | reaction ratio | classification |
|---|---:|---:|---:|---:|---:|---|
| none random energy e300 | 0.362793 | 0.853290 | 0.026446 | 1.633100 | 1.612183 | broad/background |
| unit_box random energy e300 | 0.077358 | 0.283503 | 0.993423 | 0.978174 | 0.764252 | notch-amplified |
| unit_box disp+strain prefit | 0.001408 | 0.011566 | 0.999973 | 1.000045 | 0.995461 | notch-amplified |

Interpretation:

- Raw physical-mm NN inputs reproduce the previous broad/background mechanics
  branch in random energy-only training.
- Unit-box coordinate normalization substantially improves the random energy
  mechanics solution: the field becomes notch-amplified and He correlation is
  high.
- Unit-box global displacement+strain prefit can closely match the exact-FE
  target without local notch/lip losses or geometry labels.

## Energy Continuation Drift

Main table:

```text
tables/energy_continuation_drift.csv
```

From the unit-box prefit state:

- 10-epoch raw/log10/normalized continuation can drift into a
  boundary-dominated branch.
- 30-epoch continuation improves but still has poor strain/reaction accuracy.
- 100 and 300 epochs recover notch-amplified He localization with high He
  correlation, but strain relative RMSE remains about 0.27-0.28 and reaction
  ratio remains below the target range.

This means coordinate normalization improves the ansatz and optimization
conditioning, but it does not fully remove energy-continuation drift.

## Answers Required by Prompt

1. Was coordinate normalization implemented only at NN input level?

Yes. The code snapshot shows `self.net(self.network_input(inp))`. Physical
boundary ansatz and T3 energy/postprocessing inputs still use `inp`.

2. Do T3 gradients and energy still use physical coordinates?

Yes. `compute_energy.field_grads(inp, ...)` is unchanged. The added test
`test_t3_field_grads_use_physical_coordinates` verifies a physical triangle
gradient. The mapping diagnostics also record `t3_gradients_use_physical_xy =
True`.

3. Does unit_box coordinate normalization improve exact-FE mechanics prefit?

Yes. The unit-box displacement+strain prefit reaches displacement relative RMSE
0.001408, strain relative RMSE 0.011566, and He_current correlation 0.999973.

4. Does unit_box improve random-init mechanics energy training?

Yes. Compared with `none`, unit-box random energy-only training changes the
classification from `broad/background` to `notch-amplified`, improves
He_current correlation from 0.026446 to 0.993423, and lowers the mechanics
energy ratio from 1.633100 to 0.978174. Reaction ratio is 0.764252, slightly
below the requested 0.8-1.2 target.

5. Does energy continuation still drift after coordinate normalization?

Yes. Short 10-epoch continuation from the prefit state can become
boundary-dominated. Longer 100/300 epoch continuation returns to
notch-amplified He fields, but strain and reaction drift remain.

6. Is coordinate normalization ready for a short coupled alpha-init history
smoke run, or are more mechanics-only fixes needed?

It is ready for a short coupled alpha-init history smoke run as a controlled
ansatz implementation fix, provided the run is interpreted as smoke evidence
only. More mechanics-only work may still be needed for energy-continuation
drift, especially reaction accuracy.

7. What cannot be concluded?

This package cannot claim physical validation, cannot prove coupled
phase-field behavior, cannot prove history alpha evolution is fixed, and cannot
justify changing `l0`, material parameters, TM split, alpha seeding, phase-field
notch behavior, thermal terms, or history logic.

## Suggested Next Step

Run a short coupled alpha-init history smoke test with `--coord-normalization
unit_box`, keeping every physical/model setting unchanged, and compare against
the previous raw-coordinate smoke/full evidence.
