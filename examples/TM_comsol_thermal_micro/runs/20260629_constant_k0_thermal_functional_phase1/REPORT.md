# Constant-k0 Thermal Functional Phase 1 Implementation

## 1. Purpose

Complete the Phase 1 constant-`k0` heat infrastructure by adding a thermal
functional / weak-form utility layer and stricter autograd guards in
`examples/TM_comsol_thermal_micro`.

This is an implementation follow-up to the previous residual/diagnostic package.
It is not a plan-only package and it does not enable heat training.

## 2. Relationship to previous Phase 1 residual utilities

The previous Phase 1 package added SI-unit strong-form heat residual, flux,
divergence, and mm-to-m derivative conversion utilities. This package keeps
those utilities, but clarifies their intended role.

The main future heat loss route is the thermal functional / weak-form route.
Strong-form residual utilities remain available only for patch-test diagnostics
and sign/unit sanity checks.

## 3. What was implemented

Updated `examples/TM_comsol_thermal_micro/heat_pde.py` with:

- `steady_thermal_energy_density_J_per_m3`
- `transient_thermal_incremental_energy_density_J_per_m3`
- `mean_thermal_functional_density`
- stricter `_grad` validation for detached outputs and non-gradient coordinates
- docstring updates that classify strong-form residuals as diagnostic utilities

Updated focused tests in
`examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`.

## 4. Thermal functional / weak-form route

The steady utility evaluates the pointwise constant-`k0` functional density:

```text
0.5 * k0 * |grad_m T|^2 - Q * T
```

The transient incremental utility evaluates:

```text
rho*c/(2*dt) * (T - T_prev)^2
+ 0.5*k0*|grad_m T|^2
- Q*T
```

`mean_thermal_functional_density` is only a pointwise mean helper for analytical
patch tests. It is not mesh quadrature and not a domain integral.

## 5. Strong residual diagnostic status

The strong-form residual utilities remain present:

- `steady_heat_residual_W_per_m3`
- `transient_heat_residual_W_per_m3`

They are diagnostic-only utilities for manufactured patch tests and sign/unit
checks. They were not threaded into training or promoted as the future primary
heat objective.

## 6. Unit and sign convention

Heat utilities use SI heat units internally:

- `T`: K
- heat-gradient length: m
- time: s
- `rho`: kg/m^3
- `c`: J/kg/K
- `k0`: W/m/K
- `Q`: W/m^3

Project coordinates supplied in mm are converted through derivative chain-rule
scaling using `x_m = x_mm * 1e-3`.

The steady functional source term uses `-Q*T`, consistent with the residual sign
convention `-div(k0*grad(T)) - Q`.

The steady function name retains `J_per_m3` for continuity with loss-scaling
language, but the returned steady density is documented as a variational
thermal-power density in `W/m^3`, or a `J/m^3`-equivalent only after a chosen
time/load scaling.

## 7. Autograd guard changes

The internal gradient helper now raises clear `ValueError` exceptions when:

- coordinates do not have `requires_grad=True` for nonconstant derivative calls
- outputs do not require gradients and are not explicitly accepted as constant
- autograd cannot connect nonconstant outputs to inputs

Known constant fields are allowed through an explicit constant path used by the
public heat utilities. Constant temperature fields still return zero gradients,
zero residuals, and zero zero-source steady functional density.

## 8. Tests added

The focused heat PDE patch test file now includes coverage for:

- constant zero-source steady functional density
- linear steady functional density `0.5*k0*a^2`
- quadratic/source sign consistency with residual convention
- transient uniform `T == T_prev` storage density
- transient uniform `T != T_prev` storage density
- rejection of nonpositive `dt_s`
- mm-to-m chain rule for energy density
- no damage-dependent conductivity inputs on functional APIs
- strict gradient guard rejection of detached nonconstant fields
- strict gradient guard rejection of non-gradient coordinates
- allowed constant-field zero-gradient fallback

## 9. Test results

Focused validation passed:

- `test_constant_k0_heat_pde_patch.py`: `18 passed`
- `test_prescribed_thermal_strain_patch.py`: `8 passed`

See `tables/patch_test_results.csv` and `tables/validation_results.csv`.

## 10. Prescribed-temperature fallback status

The prescribed-temperature mechanics branch is unchanged. `thermal_mode=off`
remains the default fallback route. Existing prescribed thermal strain patch
tests still pass.

## 11. Damage-dependent conductivity guard

Damage-dependent conductivity remains unimplemented and guarded. The functional
APIs do not accept `alpha`, `damage`, `d`, `g_d`, or `k_d`, and focused tests
check both public signatures and forbidden source tokens.

No `k(d)=g(d)k0` route was implemented.

## 12. Source files changed

Behavior source changed only in:

- `examples/TM_comsol_thermal_micro/heat_pde.py`

Focused tests changed in:

- `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`

Documentation/package files were added under this run folder, and
`examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md` was updated.

## 13. Limitations

This package does not implement:

- solved-temperature training
- heat PDE training loss
- heat-fracture diagnostics
- damage-dependent conductivity
- `k(d)=g(d)k0`
- mesh quadrature
- material changes
- `l0` changes
- history or phase-field logic changes
- mechanics boundary-condition changes
- reaction-route changes
- D0040, seed study, shear extension, or S0110 runs

## 14. Final classification

`constant-k0 thermal functional phase1 implemented and tests passed`

Phase 1 now includes both constant-`k0` strong-form diagnostic utilities and
thermal functional / weak-form utilities. The future main heat loss route is the
thermal functional route, while residuals remain diagnostic. SI units and
mm-to-m coordinate conversion are preserved. Strict autograd guards reduce the
risk of silently treating detached fields as valid zero gradients. Existing
prescribed thermal strain tests still pass. Damage-dependent conductivity remains
unimplemented and guarded.

## 15. Exact next recommended task

Review this implementation package. If approved, the next narrow task should be
designing the solved-temperature representation and weak-form quadrature strategy
without coupling it to fracture training yet. Do not implement damage-dependent
conductivity or run heat-fracture diagnostics until the constant-`k0` functional
route is reviewed.
