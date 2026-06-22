# Solved-Temperature Mechanics Smoke Adapter

## 1. Purpose

This package implements the next narrow step after the solved-temperature
mechanics bridge: a smoke adapter that samples supported temperature sources at
mechanics element centroids and passes an element-sized `thermal_delta_T` into
the existing mechanics route.

This task does not implement coupled heat-mechanics training. It does not
modify `train_mixed_tm.py` and does not run broad mechanics or fracture
training.

## 2. What was implemented

Added:

- `examples/TM_comsol_thermal_micro/thermal_mechanics_adapter.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`

The adapter exposes:

- `element_centroid_coords_mm`
- `build_element_thermal_delta_T_from_bridge`
- `build_mechanics_thermal_kwargs_from_bridge`

The adapter reuses `thermal_solution_bridge.build_thermal_strain_input_from_temperature`.
It does not duplicate the thermal-source formulas or the mechanics thermal
strain logic.

## 3. Mechanics call path

The smoke adapter follows this limited call path:

```text
mechanics nodes + triangle connectivity
-> element_centroid_coords_mm
-> thermal_solution_bridge.build_thermal_strain_input_from_temperature
-> element-sized thermal_delta_T
-> compute_mixed_tm_fields(..., thermal_delta_T=...)
```

The existing mechanics default remains unchanged:

```text
thermal_mode="off"
thermal_temperature=None
thermal_delta_T=None
```

## 4. Element centroid check

For two triangles:

```text
nodes = [(0,0), (3,0), (0,6), (3,6)]
elements = [(0,1,2), (1,3,2)]
```

Observed centroids:

```text
[(1,2), (2,4)]
```

The adapter returns one centroid per triangle and preserves the input tensor
dtype/device through tensor indexing and mean operations.

## 5. Uniform temperature adapter check

For a two-element mechanics patch and:

```text
source_mode = prescribed_uniform
uniform_delta_T_K = 20
T_ref = 300 K
```

Observed:

- `evaluation_location`: `element_centroid`
- `thermal_delta_T` shape: `(2,)`
- minimum `DeltaT`: `20.0 K`
- maximum `DeltaT`: `20.0 K`

## 6. H1 frozen-lift centroid check

For the H1 frozen-lift patch:

```text
T_bottom = 300 K
T_top = 320 K
T_ref = 300 K
DeltaT(y) = 20 * eta
```

The adapter evaluates this field at element centroids. For the two-element
unit patch with y-centroids at `0.003333... mm` and `0.006666... mm`, expected
centroid increments are:

```text
[6.6666666667 K, 13.3333333333 K]
```

The test verifies this through `thermal_delta_T == 20 * centroid_y / 0.01`.
The correction remains zero and `network_training_run` remains `False`.

## 7. Mechanics smoke equivalence

The adapter-produced `thermal_kwargs` were passed directly into
`compute_mixed_tm_fields` and compared against a direct mechanics call using the
same element-sized `thermal_delta_T`.

Maximum absolute differences for selected fields:

- `thermal_delta_T`: `0.0`
- `thermal_eps_xx`: `0.0`
- `thermal_eps_yy`: `0.0`
- `eps_xx_elastic`: `0.0`
- `eps_yy_elastic`: `0.0`
- `psi_total`: `0.0`
- `sigma_xx_tm_total`: `0.0`
- `sigma_yy_tm_total`: `0.0`

This proves the adapter is only a call-site bridge and does not alter mechanics
behavior.

## 8. Training entrypoint status

`train_mixed_tm.py` was not modified. The adapter is not imported from the
training entrypoint, and no production default now activates solved-temperature
mechanics.

## 9. Guard status

No damage-dependent conductivity, `k(d)=g(d)k0`, heat-fracture diagnostic, or
fracture heat training was implemented.

No D0040, seed study, shear extension, S0110, transient production, or
bottom-cooling production run was added.

## 10. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/thermal_mechanics_adapter.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`
- `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter`

## 11. Limitations

This package does not implement:

- coupled heat-mechanics training;
- solved-temperature phase-field fracture coupling;
- heat PDE loss inside mechanics/fracture training;
- heat-fracture diagnostics;
- damage-dependent conductivity;
- D0040, seed study, shear extension, S0110;
- transient production or bottom-cooling production runs.

The adapter is intentionally limited to element-centroid smoke mechanics
evaluation.

## 12. Final classification

`solved-temperature mechanics smoke adapter implemented and tests passed`

## 13. Exact next recommended task

Add an explicit opt-in smoke script or small CLI wrapper that calls this adapter
for one fixture mechanics patch and writes a compact result table, still without
changing `train_mixed_tm.py` defaults or running broad training.
