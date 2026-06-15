# Heat Correction Policy

## 1. Purpose

This package implements an explicit correction-policy gate for the independent
heat-only solved-temperature weak-form patch trainer.

The policy prevents pure linear Dirichlet H1 and uniform H2 patch cases from
training an unnecessary correction network when the exact solution is already
represented by the hard lift.

## 2. Relationship to H1 quadrature/regularization review

The previous H1 review package showed that the no-training linear lift baseline
has zero temperature error, zero correction, and zero strong residual. It also
showed that the high trained H1 strong-form residual diagnostic comes from a
tiny learned nonlinear correction admitted by weak-form training.

This package turns that review conclusion into the default policy for the
heat-only patch trainer.

## 3. What was implemented

Implemented in `examples/TM_comsol_thermal_micro/train_heat_only.py`:

- explicit `correction_policy` handling;
- default `frozen_lift` policy for H1 and H2 in `run_steady_heat_patch_case`;
- explicit `trainable_correction` override preserving the previous behavior;
- explicit `regularized_correction` option with correction L2 regularization;
- policy-aware diagnostics for correction, normalized residual, primary loss,
  and trainability;
- `run_correction_policy_comparison` for small H1/H2 policy summaries.

Added focused tests in
`examples/TM_comsol_thermal_micro/tests/test_heat_correction_policy.py`.

## 4. Correction policy design

The supported policies are:

- `frozen_lift`
- `trainable_correction`
- `regularized_correction`

The public patch trainer defaults to `frozen_lift` when `correction_policy` is
not supplied. The old trainable behavior remains available only through an
explicit override.

## 5. Frozen-lift policy

For H1, the policy evaluates:

```text
T = T_lift
correction = 0
```

For H2, the policy evaluates:

```text
T = T_bottom
correction = 0
```

No optimizer step is run for frozen cases. The thermal functional density and
strong-form residual diagnostics are still evaluated for reporting.

For pure linear Dirichlet H1, the exact solution is already represented by the
lift. The default policy therefore freezes the correction and uses T=T_lift.
Trainable correction remains available only as an explicit diagnostic or future
nontrivial heat-solve option.

## 6. Trainable correction policy

The `trainable_correction` policy preserves the previous heat-only patch
behavior:

```text
H1: T = T_lift + eta(1-eta) * N_T
H2: T = T_bottom + (y-y_min) * N_T
```

The training objective remains the thermal functional / weak-form density. This
policy is now explicit so it can be used as a diagnostic or future nontrivial
heat-solve option without being the default for exact-lift patch cases.

## 7. Regularized correction policy

The `regularized_correction` policy is opt-in. It uses the trainable correction
route plus:

```text
lambda_corr * mean((T - T_lift)^2)
```

It is not the H1 or H2 default.

## 8. H1 policy comparison

Observed with `nx=4`, `ny=4`, `num_epochs=8`, and `float64`:

- H1 frozen lift: max correction `0.0 K`, max strong residual
  `0.0 W/m^3`, normalized residual max `0.0`.
- H1 trainable correction: max correction `4.752286448592713e-04 K`,
  max strong residual `7.240993859241002e10 W/m^3`, normalized residual max
  `8.660610491253389e-04`.
- H1 regularized correction: max correction `4.7525331416409244e-04 K`,
  max strong residual `7.241048018147003e10 W/m^3`, normalized residual max
  `8.660675224405754e-04`.

The frozen case represents the pure linear Dirichlet patch exactly. The
trainable and regularized cases are retained as explicit diagnostics.

## 9. H2 policy summary

Observed with `nx=4`, `ny=4`, `num_epochs=8`, and `float64`:

- H2 frozen lift: `T=300 K`, max correction `0.0 K`, max strong residual
  `0.0 W/m^3`, normalized residual max `0.0`, and top/side flux `0.0 W/m^2`.

## 10. Strong residual diagnostic status

Strong-form residual remains diagnostic-only and is not used as the primary
heat objective.

The primary heat objective remains:

```text
thermal_functional_area_weighted_mean
```

No strong residual loss was added.

## 11. Prescribed-temperature fallback status

The prescribed-temperature fallback route was not changed. The focused
prescribed thermal strain fallback test suite still passes.

## 12. Damage-dependent conductivity guard

No damage-dependent conductivity, `k(d)=g(d)k0`, mechanics coupling,
phase-field coupling, heat-fracture diagnostic, or fracture heat training was
implemented.

No `train_mixed_tm.py` changes were made.

## 13. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/train_heat_only.py`
- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/tests/test_heat_correction_policy.py`
- `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy`

## 14. Limitations

This is a heat-only patch-trainer policy gate. It does not implement:

- solved-temperature mechanics coupling;
- solved-temperature thermal-strain coupling;
- fracture training with heat;
- heat-fracture diagnostics;
- damage-dependent conductivity;
- transient production training;
- bottom-cooling production runs;
- D0040, seed study, shear extension, or S0110.

## 15. Final classification

`heat correction policy implemented and tests passed`

## 16. Exact next recommended task

Review whether the frozen-lift policy should also be documented in the
user-facing thermal README before any solved-temperature mechanics-coupling
work is approved.
