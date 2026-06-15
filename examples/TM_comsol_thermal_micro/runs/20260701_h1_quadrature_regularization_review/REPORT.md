# H1 Quadrature and Regularization Review

## 1. Purpose

Review and control the H1 strong-form residual diagnostic discrepancy observed
after the heat-only weak-form patch trainer.

The review answers why H1 can have a small analytical temperature error while
recording a large strong-form residual diagnostic, before any solved-temperature
mechanics coupling is considered.

## 2. Relationship to heat-only weak-form patch trainer

The previous package implemented an independent constant-`k0` solved-temperature
weak-form patch trainer. It used the thermal functional density as the primary
loss and reported strong residuals only as diagnostics.

This review keeps that policy. Strong-form residual was used only to diagnose
smoothness/curvature of the learned temperature field. It was not used as the
heat training objective.

## 3. What was implemented

Added focused diagnostics:

- H1 no-training lift baseline
- triangle centroid and triangle 3-point quadrature comparison
- correction diagnostics for trained H1
- normalized strong residual metrics
- optional `correction_l2` regularization experiment

Added focused tests in:

- `examples/TM_comsol_thermal_micro/tests/test_h1_quadrature_regularization_review.py`

## 4. H1 issue being reviewed

The H1 ansatz is:

```text
T = T_lift + eta(1-eta) * N_T
```

The exact linear conduction solution is already present when `N_T = 0`.
Training can still introduce a very small nonzero correction. That correction is
small in temperature norm, but it can be large in second derivatives relative to
the strong-form residual diagnostic.

## 5. No-training lift baseline

The no-training baseline initializes the final network layer to zero, so
`N_T=0` and `T=T_lift` exactly.

Observed baseline:

- max temperature error: `0.0 K`
- L2 temperature error: `0.0 K`
- bottom boundary error: `0.0 K`
- top boundary error: `0.0 K`
- max correction: `0.0 K`
- max correction gradient: `0.0 K/m`
- max strong residual: `0.0 W/m^3`
- normalized residual max: `0.0`

This shows the exact linear lift, heat residual implementation, and mm-to-m
gradient scaling are not the cause of the high trained H1 residual.

## 6. Correction diagnostics

For trained H1, the correction is:

```text
correction = T - T_lift
```

Centroid training produced:

- max correction: `4.752286448592713e-04 K`
- L2 correction: `2.846484229769935e-04 K`
- max correction gradient: `262.4596641292851 K/m`
- mean correction gradient: `150.57294510042425 K/m`
- max strong residual: `7.240993859241002e10 W/m^3`

The trained residual comes from this learned nonlinear correction, not from the
linear lift.

## 7. Quadrature comparison

The review added `triangle_quadrature_points_mm(..., rule=...)` with:

- `centroid`
- `triangle_3point`

The 3-point rule uses barycentric points:

```text
(1/6, 1/6, 2/3)
(1/6, 2/3, 1/6)
(2/3, 1/6, 1/6)
```

Each sample carries `area/3`.

Observed comparison:

- no-training lift baseline: residual `0.0 W/m^3`
- centroid: max correction `4.752286448592713e-04 K`, residual `7.240993859241002e10 W/m^3`
- triangle 3-point: max correction `3.703253884168589e-04 K`, residual `4.8776662018449715e10 W/m^3`
- triangle 3-point + correction L2: max correction `3.693907176511857e-04 K`, residual `4.862652370993762e10 W/m^3`

The 3-point rule reduces the learned correction and residual diagnostic but does
not eliminate them in this lightweight smoke setting.

## 8. Residual normalization metric

The normalized residual metric is:

```text
normalized_residual = abs(residual) / max(k0 * |grad_T| / L_m, eps)
```

where `L_m = y_height_m`.

Observed normalized max values:

- no-training lift baseline: `0.0`
- centroid: `8.660610491253389e-04`
- triangle 3-point: `5.835101016116359e-04`
- triangle 3-point + correction L2: `5.817138620095239e-04`

This confirms the raw residual is large in SI units, while its scale relative to
`k0*|grad_T|/L_m` remains below `1e-3` for these smoke runs.

## 9. Optional regularization experiment

The review implemented an optional H1-only `correction_l2` term:

```text
lambda_corr * mean((T - T_lift)^2)
```

It is disabled by default and is not a production default.

With triangle 3-point quadrature and `lambda_corr=1.0e12`, the residual and
correction decrease slightly relative to unregularized 3-point training:

- max correction: `3.703253884168589e-04 K` to `3.693907176511857e-04 K`
- max residual: `4.8776662018449715e10 W/m^3` to `4.862652370993762e10 W/m^3`

It does not fully remove the trained residual diagnostic.

## 10. Strong residual diagnostic status

Strong residual remains diagnostic-only. No residual loss was added, and strong
residual was not minimized directly.

## 11. Prescribed-temperature fallback status

The prescribed-temperature fallback behavior is unchanged. Existing prescribed
thermal strain patch tests still pass.

## 12. Damage-dependent conductivity guard

No damage-dependent conductivity, `k(d)=g(d)k0`, phase-field coupling, or
mechanics coupling was added.

## 13. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
- `examples/TM_comsol_thermal_micro/train_heat_only.py`
- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/tests/test_h1_quadrature_regularization_review.py`
- this report package

## 14. Limitations

This remains a focused diagnostic review. It does not implement:

- solved-temperature mechanics coupling
- thermal-strain mechanics coupling
- fracture training with heat
- heat-fracture diagnostics
- damage-dependent conductivity
- transient production training
- D0040, seed study, shear extension, or S0110

The regularization experiment is intentionally small and should not be treated
as a production default.

## 15. Final classification

`h1 review implemented; trained residual remains high but explained`

The no-training lift baseline has zero residual, so the heat PDE residual
implementation and mm-to-m scaling are not the cause. The high trained residual
comes from a small learned nonlinear correction admitted by the weak-form
training. This correction is small in temperature norm but large in the
second-derivative diagnostic. Triangle 3-point quadrature and correction L2
regularization reduce the diagnostic but do not eliminate it.

## 16. Exact next recommended task

Do not start solved-temperature mechanics coupling yet. First decide whether H1
should constrain the correction space more strongly, use richer quadrature, or
use an exact-lift freeze/zero-correction policy for pure linear Dirichlet patch
cases. Strong residual should remain diagnostic-only unless a separate review
approves a different objective.
