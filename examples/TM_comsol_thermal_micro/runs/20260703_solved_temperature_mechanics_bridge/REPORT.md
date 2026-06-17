# Solved-Temperature Mechanics Bridge

## 1. Purpose

This package implements the first minimal bridge from solved-temperature
representations to the existing mechanics thermal-strain route.

This task does not implement coupled heat-mechanics training. It only adds a
bridge that evaluates solved-temperature fields and converts them into the same
thermal-strain input form used by the prescribed-temperature route.

## 2. Relationship to heat correction policy

The previous heat correction policy package made pure H1 and H2 heat-only patch
cases default to `correction_policy="frozen_lift"`. This bridge reuses that
policy concept: the solved-temperature source mode evaluates frozen lift fields
directly and does not train a correction network.

## 3. What was implemented

Added:

- `examples/TM_comsol_thermal_micro/thermal_solution_bridge.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_bridge.py`

The bridge exposes:

- `evaluate_temperature_source_at_coords_mm`
- `temperature_increment_from_reference`
- `build_thermal_strain_input_from_temperature`

Supported source modes:

- `prescribed_uniform`
- `solved_frozen_lift`

No default mechanics training path was switched to solved temperature.

## 4. Existing prescribed-temperature path discovered

The existing mechanics thermal-strain path is:

```text
thermal_prescribed.delta_T_from_temperature
thermal_prescribed.apply_thermal_strain
compute_energy_mixed_tm.compute_mixed_tm_fields(..., thermal_delta_T=...)
```

Inside `compute_mixed_tm_fields`, `thermal_delta_T` is resolved on elements and
then passed to `apply_thermal_strain`:

```text
eps_xx_elastic = exx - alpha_T * DeltaT
eps_yy_elastic = eyy - alpha_T * DeltaT
eps_xy_elastic = exy
```

Those elastic strains feed the existing TM split, history, stress, and energy
route. The bridge does not duplicate that mechanics logic.

## 5. Solved-temperature bridge design

The bridge evaluates a temperature source at mechanics/material coordinates,
then computes:

```text
DeltaT = T - T_ref
```

The returned dictionary includes `thermal_delta_T`, `thermal_temperature`,
thermal strain components, correction metadata, and `network_training_run`.

For `solved_frozen_lift`:

- H1 evaluates the top-bottom linear lift.
- H2 evaluates the bottom uniform lift.
- `correction_policy` is `frozen_lift`.
- `network_training_run` is `False`.
- `correction_K` is zero.

## 6. Uniform temperature equivalence

The bridge was tested with:

```text
T_ref = 300 K
T = 320 K
DeltaT = 20 K
```

Observed:

- minimum bridge `DeltaT`: `20.0 K`
- maximum bridge `DeltaT`: `20.0 K`
- prescribed uniform `DeltaT`: `20.0 K`
- thermal strain components match the existing prescribed path.

## 7. Zero temperature increment equivalence

The bridge was tested with:

```text
T_ref = 300 K
T = 300 K
DeltaT = 0 K
```

Observed:

- maximum absolute bridge `DeltaT`: `0.0 K`
- zero bridge input matches the default mechanics route for selected stress and
  energy fields.

## 8. H1 frozen-lift DeltaT check

For H1:

```text
T_bottom = 300 K
T_top = 320 K
T_ref = 300 K
DeltaT(y) = 20 * eta
```

Observed at `y = [0.0, 0.0025, 0.005, 0.01] mm`:

- `T = [300.0, 305.0, 310.0, 320.0] K`
- `DeltaT = [0.0, 5.0, 10.0, 20.0] K`
- maximum correction: `0.0 K`
- network training run: `False`

## 9. Mechanics-level equivalence check

A focused mechanics-level check reuses `compute_mixed_tm_fields`.

The prescribed uniform `DeltaT=20 K` route was compared against the bridge
route that produces the same uniform `thermal_delta_T`. Maximum absolute
differences were:

- `thermal_delta_T`: `0.0`
- `thermal_eps_xx`: `0.0`
- `thermal_eps_yy`: `0.0`
- `eps_xx_elastic`: `0.0`
- `eps_yy_elastic`: `0.0`
- `psi_total`: `0.0`
- `sigma_xx_tm_total`: `0.0`
- `sigma_yy_tm_total`: `0.0`

## 10. Strong residual diagnostic status

Strong-form heat residual remains diagnostic-only and is not used as a training
objective.

This bridge does not evaluate or minimize a heat residual.

## 11. Prescribed-temperature fallback status

The prescribed-temperature fallback route is unchanged. Existing
`thermal_mode="off"` defaults and prescribed thermal strain tests still pass.

## 12. Damage-dependent conductivity guard

No damage-dependent conductivity, `k(d)=g(d)k0`, heat-fracture diagnostic, or
fracture heat training was implemented.

## 13. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/thermal_solution_bridge.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_bridge.py`
- `examples/TM_comsol_thermal_micro/runs/20260703_solved_temperature_mechanics_bridge`

## 14. Limitations

This package does not implement:

- coupled heat-mechanics training;
- solved-temperature phase-field fracture coupling;
- heat PDE loss inside mechanics/fracture training;
- heat-fracture diagnostics;
- damage-dependent conductivity;
- D0040, seed study, shear extension, S0110;
- transient production or bottom-cooling production runs.

The bridge currently supports the minimal uniform and frozen-lift temperature
sources needed for equivalence testing.

## 15. Final classification

`solved-temperature mechanics bridge implemented and tests passed`

## 16. Exact next recommended task

Add one narrow call-site adapter that can pass bridge-produced
`thermal_delta_T` into a cheap smoke mechanics evaluation without changing
production defaults or running broad mechanics/fracture training.
