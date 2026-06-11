# Constant-k0 Heat PDE Phase 1 Implementation

## 1. Purpose

Implement Phase 1 constant-conductivity heat PDE infrastructure inside `examples/TM_comsol_thermal_micro`, with explicit SI heat units, mm-to-m coordinate conversion, and focused analytical patch tests. This is implementation work, not another plan-only package.

## 2. What was implemented

Added `heat_pde.py`, an isolated utility module for constant-`k0` heat-transfer calculations. It provides:

- `coords_mm_to_m`
- `temperature_gradient_m`
- `heat_flux_W_per_m2`
- `divergence_m`
- `steady_heat_residual_W_per_m3`
- `transient_heat_residual_W_per_m3`
- `normal_heat_flux_W_per_m2`

The module is not threaded into fracture training, mechanics energy, history, checkpointing, or postprocessing.

## 3. Unit convention implemented

Heat PDE utilities use SI units internally:

- `T`: K
- length for heat gradients: m
- time: s
- `rho`: kg/m^3
- `c`: J/kg/K
- `k0`: W/m/K
- `Q`: W/m^3
- residual: W/m^3

Project mesh coordinates are in mm, so `coordinate_unit='mm'` applies `x_m = x_mm * 1e-3` through derivative chain-rule scaling. See `tables/unit_conversion_summary.csv`.

## 4. Heat residual sign convention

Fourier heat flux is:

```text
q = -k0 * grad(T)
```

The steady residual is:

```text
-div(k0 * grad(T)) - Q
```

implemented as:

```text
div(q) - Q
```

The transient residual is:

```text
rho*c*dTdt - div(k0 * grad(T)) - Q
```

implemented as:

```text
rho*c*dTdt + div(q) - Q
```

See `tables/heat_residual_sign_convention.csv`.

## 5. Patch tests added

Added `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`, covering constant temperature, linear 1D conduction, quadratic manufactured source, normal flux, transient uniform no-source, transient manufactured source, mm-to-m chain rule, and no damage-dependent conductivity API guard.

## 6. Patch test results

The focused heat PDE patch suite passed: `8 passed`. Existing prescribed thermal strain patch tests also passed, preserving the default-off prescribed-temperature mechanics branch.

See `tables/patch_test_results.csv` and `tables/validation_results.csv`.

## 7. What remains disabled by default

The new heat PDE utilities are inactive unless imported directly by tests or future approved code. No training loss, heat residual objective, solved temperature field, fracture diagnostic, boundary-condition CLI, checkpoint schema, or postprocess route was enabled.

## 8. Prescribed-temperature fallback status

The prescribed-temperature branch is unchanged. `thermal_mode=off` remains default, and the mechanics relation remains:

```text
delta_T = T - Tref
exx_e = exx - alpha_T * delta_T
eyy_e = eyy - alpha_T * delta_T
exy_e = exy
```

Existing prescribed thermal strain patch tests passed.

## 9. Damage-dependent conductivity guard

Damage-dependent conductivity remains unimplemented. The Phase 1 API has no `alpha`, damage, degradation, or `k(d)` conductivity inputs, and the focused guard test checks public signatures and source tokens.

## 10. Source files changed

Behavior source changed only by adding isolated `heat_pde.py`; no existing mechanics/training/source route was modified. Tests and documentation/package files were added, and `PROJECT_MEMORY.md` was updated.

See `tables/source_files_changed.csv`.

## 11. Validation commands and results

Required validation passed:

1. `git status`
2. recursive `py_compile` for Python files under `examples/TM_comsol_thermal_micro`
3. new focused heat PDE patch tests
4. existing prescribed thermal strain patch tests
5. package schema/file existence check
6. `git diff --check`
7. `git diff --name-only -- examples/TM_comsol_no_thermal_micro`

See `tables/validation_results.csv`.

## 12. Limitations

This phase does not implement damage-dependent conductivity, solved-temperature fracture training, broad thermal-mechanical diagnostics, D0040, seed studies, shear extension, S0110, material changes, `l0` changes, history changes, mechanics boundary-condition changes, or reaction-route changes.

The transient residual utility is analytical infrastructure only; no physical time/loading schedule was introduced.

## 13. Final classification

`constant-k0 heat PDE phase1 implemented and patch tests passed`

Phase 1 constant-conductivity heat PDE infrastructure was implemented with SI heat units, explicit mm-to-m coordinate conversion, and focused analytical patch tests. The new utilities remain inactive by default and do not change the prescribed-temperature mechanics route. Existing prescribed thermal strain tests still pass. Damage-dependent conductivity remains unimplemented and guarded.

## 14. Exact next recommended task

Review this Phase 1 package and tests. If approved, decide the next narrow heat task: either add additional boundary-condition patch tests or design a solved-temperature representation. Do not implement damage-dependent conductivity or run heat-fracture diagnostics until constant-k0 heat utilities and any future solved-T coupling are independently reviewed.
