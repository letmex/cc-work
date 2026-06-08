# Coordinate Normalization Mechanics Validation

This package validates a controlled coordinate input normalization option for
the `TM_comsol_no_thermal_micro` PINN mechanics ansatz.

Package root:

```text
examples/TM_comsol_no_thermal_micro/runs/20260608_coord_normalization_mechanics_validation
```

What changed:

- Added `coord_normalization=none|unit_box` to `FieldComputation`.
- Default `none` preserves old raw physical-mm NN input behavior.
- `unit_box` maps only NN input coordinates to `[-1, 1]`.
- Physical outputs, boundary ansatz, T3 gradients, mechanics energy, material
  parameters, `l0`, TM split, alpha seeding, phase-field notch behavior,
  thermal terms, and history logic were not changed.
- Added local project tests for coordinate normalization and physical-coordinate
  T3 gradient preservation.

Diagnostic scope:

- alpha fixed to zero.
- top-u mode free.
- Delta = 1e-6.
- seed = 2.
- 8 hidden layers, 400 neurons, TrainableReLU.
- No coupled phase-field full training.
- No notch/lip loss, local weights, displacement-jump target, enrichment, or
  geometry-label guidance.

Read first:

1. `REPORT.md`
2. `tables/coord_normalization_case_comparison.csv`
3. `tables/energy_continuation_drift.csv`
4. `tables/coord_mapping_diagnostics.csv`
5. `artifacts/code_snapshot/debug_coord_normalization_mechanics_validation.py`

