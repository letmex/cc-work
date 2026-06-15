# Heat-only Weak-form Patch Training Implementation

## 1. Purpose

Implement the next narrow heat-only step after constant-`k0` thermal functional
Phase 1 inside `examples/TM_comsol_thermal_micro`.

This package adds an independent solved-temperature representation, triangle-area
quadrature, and small heat-only patch training cases. It is implementation work,
not a planning-only package.

## 2. Relationship to constant-k0 thermal functional Phase 1

The previous Phase 1 package added pointwise thermal functional density utilities
in `heat_pde.py`. This package uses those utilities as the primary heat-only
loss route and adds the missing solved-temperature and area quadrature layer.

The heat-only trainer uses the thermal functional / weak-form density as the
primary loss. Strong-form residuals are used only as post-training diagnostics.

## 3. What was implemented

Added:

- `thermal_field.py`
- `thermal_quadrature.py`
- `train_heat_only.py`
- `tests/test_heat_only_weak_form_training.py`

The implementation is independent of the existing mechanics/fracture trainer and
does not modify `train_mixed_tm.py`.

## 4. Solved temperature representation

`thermal_field.py` provides rectangular coordinate normalization and temperature
ansatz helpers. Coordinates remain in mm for the ansatz and network input. Heat
gradients are still converted to SI meter units downstream by `heat_pde.py`.

Network inputs are normalized to `[-1, 1]` from rectangular mm bounds.

## 5. Dirichlet ansatz details

Top-bottom Dirichlet case:

```text
eta = (y - y_min)/(y_max - y_min)
T_lift = (1 - eta) * T_bottom + eta * T_top
B_T = eta * (1 - eta)
T = T_lift + B_T * N_T(x_hat, y_hat)
```

This hard-satisfies `T(y_min)=T_bottom` and `T(y_max)=T_top`.

Bottom-only Dirichlet case:

```text
T = T_bottom + (y - y_min) * N_T(x_hat, y_hat)
```

This hard-satisfies `T(y_min)=T_bottom`.

## 6. Weak-form quadrature strategy

`thermal_quadrature.py` implements simple triangle-area quadrature:

- triangle centroids in mm
- triangle areas converted from mm geometry to m^2
- area-weighted mean density
- 2D density integral without thickness

This is area-weighted patch quadrature, not a full production integration route.

## 7. Heat-only training loss

The trainer evaluates temperature at triangle centroids, computes:

```text
0.5*k0*|grad_m T|^2 - Q*T
```

through `heat_pde.steady_thermal_energy_density_J_per_m3`, and uses:

```text
sum(area * density) / sum(area)
```

as the primary loss.

Strong-form residuals are computed only after training as diagnostics.

## 8. Patch case H1: top-bottom Dirichlet linear conduction

H1 uses a small rectangular triangular mesh with:

- `T_bottom = 300 K`
- `T_top = 320 K`
- `Q = 0`
- constant `k0`

Expected analytical solution:

```text
T_exact(y) = T_bottom + (T_top - T_bottom) * (y-y_min)/(y_max-y_min)
```

Observed default smoke metrics:

- max absolute temperature error: `4.752286448592713e-04 K`
- L2 temperature error: `2.846484229769935e-04 K`
- bottom boundary error: `0.0 K`
- top boundary error: `0.0 K`
- final area-weighted functional loss: `8.359952464370274e14`
- strong residual diagnostic max: `7.240993859241002e10 W/m^3`

The high strong residual diagnostic reflects a small curved correction admitted
by the lightweight centroid weak-form smoke. It is recorded as a diagnostic and
future quadrature/regularization warning, not used as the primary loss.

## 9. Patch case H2: bottom-only Dirichlet with natural insulation

H2 uses:

- `T_bottom = 300 K`
- top/left/right natural
- `Q = 0`
- constant `k0`

Expected analytical solution:

```text
T = 300 K
```

Observed default smoke metrics:

- max absolute temperature error: `0.0 K`
- L2 temperature error: `0.0 K`
- bottom boundary error: `0.0 K`
- final area-weighted functional loss: `0.0`
- strong residual diagnostic max: `0.0 W/m^3`
- top/side normal flux diagnostic max: `0.0 W/m^2`

## 10. Quadrature sanity checks

For a rectangle with area `2.0000000000000003e-10 m^2` and constant density
`12.5`, the triangle quadrature produced:

- total triangle area: `2.0000000000000003e-10 m^2`
- integrated density: `2.5000000000000005e-09`
- area-weighted mean density: `12.5`

## 11. Transient storage sanity checks

No transient training was implemented. The existing transient incremental
functional was exercised on a uniform field:

- `T == T_prev`: mean incremental density `0.0`
- `T - T_prev = 4 K`, `dt = 2 s`: mean incremental density `707200.0`

This matches `rho*c/(2*dt)*(DeltaT)^2`.

## 12. Strong residual diagnostic status

Strong residuals are not used as the heat-only training objective. They remain
diagnostic only. H1 records a large residual diagnostic despite passing the
analytical temperature tolerance, which should be considered when designing the
next quadrature strategy.

## 13. Prescribed-temperature fallback status

The prescribed-temperature mechanics fallback remains unchanged. Existing
prescribed thermal strain patch tests still pass.

## 14. Damage-dependent conductivity guard

Damage-dependent conductivity remains unimplemented. No `k(d)=g(d)k0` route,
conductivity degradation input, phase-field coupling, or mechanics coupling was
added.

## 15. Source files changed

Behavior source added:

- `examples/TM_comsol_thermal_micro/thermal_field.py`
- `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
- `examples/TM_comsol_thermal_micro/train_heat_only.py`

Focused tests added:

- `examples/TM_comsol_thermal_micro/tests/test_heat_only_weak_form_training.py`

Documentation/package files were added under this run folder, and
`PROJECT_MEMORY.md` was updated.

## 16. Limitations

This package does not implement:

- solved-temperature coupling to mechanics
- solved-temperature coupling to thermal strain mechanics
- fracture training with heat
- damage-dependent conductivity
- `k(d)=g(d)k0`
- transient training
- production bottom-cooling runs
- heat-fracture diagnostics
- D0040, seed study, shear extension, or S0110
- material, `l0`, history, mechanics boundary, loss route, or reaction changes

The H1 strong residual diagnostic indicates that centroid weak-form smoke tests
are not a substitute for a reviewed production quadrature strategy.

## 17. Final classification

`heat-only weak-form patch trainer implemented and tests passed`

An independent heat-only solved-temperature branch was implemented for
constant-`k0` weak-form patch training. The temperature ansatz hard-satisfies
Dirichlet boundaries, triangle-area quadrature provides area-weighted functional
loss, and H1/H2 analytical heat patches pass focused validation. The
implementation does not couple solved temperature to mechanics or fracture
training, and damage-dependent conductivity remains unimplemented.

## 18. Exact next recommended task

Review this heat-only weak-form package. If approved, the next task should be a
quadrature and regularization review for H1 before any solved-temperature
mechanics coupling. Do not implement damage-dependent conductivity or run
heat-fracture diagnostics without separate approval.
