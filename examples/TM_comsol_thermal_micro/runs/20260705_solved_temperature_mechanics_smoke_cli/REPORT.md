# Solved-Temperature Mechanics Smoke CLI

## 1. Purpose

This package adds an explicit opt-in smoke runner for the solved-temperature
mechanics adapter. The runner executes one fixed two-element mechanics patch,
samples the H1 frozen-lift solved-temperature source at element centroids, and
writes a compact CSV result table.

This task does not implement coupled heat-mechanics training. It does not
modify `train_mixed_tm.py` and does not run broad mechanics or fracture
training.

## 2. What was implemented

Added:

- `examples/TM_comsol_thermal_micro/run_solved_temperature_mechanics_smoke.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_cli.py`

The script can be run explicitly:

```text
D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\run_solved_temperature_mechanics_smoke.py --output-dir <output_dir>
```

It writes:

```text
mechanics_smoke_results.csv
```

## 3. Smoke call path

The smoke runner follows this path:

```text
fixed two-element mechanics patch
-> thermal_mechanics_adapter.build_mechanics_thermal_kwargs_from_bridge
-> H1 solved_frozen_lift at element centroids
-> compute_mixed_tm_fields(..., thermal_delta_T=...)
-> mechanics_smoke_results.csv
```

The existing training entrypoint is not changed and does not import this script.

## 4. Actual smoke output

The generated table is:

- `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/tables/mechanics_smoke_results.csv`

Observed values:

- classification: `solved-temperature mechanics smoke cli passed`
- source mode: `solved_frozen_lift`
- case id: `H1`
- evaluation location: `element_centroid`
- network training run: `false`
- correction policy: `frozen_lift`
- nodes: `4`
- elements: `2`
- minimum `DeltaT`: `6.666666666666686 K`
- maximum `DeltaT`: `13.333333333333371 K`
- minimum centroid y: `0.0033333333333333335 mm`
- maximum centroid y: `0.006666666666666667 mm`
- mechanics maximum absolute difference: `0.0`
- `train_mixed_tm.py` modified: `false`

## 5. Mechanics equivalence check

The script computes two mechanics field sets:

1. direct `compute_mixed_tm_fields(..., thermal_delta_T=adapter diagnostics)`
2. bridged `compute_mixed_tm_fields(..., **adapter["thermal_kwargs"])`

It compares selected thermal strain, elastic strain, stress, and energy fields.
The maximum absolute difference is:

```text
0.0
```

## 6. Opt-in status

This smoke runner is not a production default. It is only executed when the
script is called directly or when the focused pytest file invokes it.

`train_mixed_tm.py` remains unmodified.

## 7. Guard status

No damage-dependent conductivity, heat-fracture diagnostic, fracture heat
training, D0040, seed study, shear extension, S0110, transient production, or
bottom-cooling production was added.

## 8. Source files changed

Modified:

- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

Added:

- `examples/TM_comsol_thermal_micro/run_solved_temperature_mechanics_smoke.py`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_cli.py`
- `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli`

## 9. Limitations

This package does not implement:

- coupled heat-mechanics training;
- solved-temperature phase-field fracture coupling;
- heat PDE loss inside mechanics/fracture training;
- heat-fracture diagnostics;
- damage-dependent conductivity;
- broad mechanics/fracture runs.

It is a single explicit smoke runner for adapter-level review.

## 10. Final classification

`solved-temperature mechanics smoke cli implemented and tests passed`

## 11. Exact next recommended task

Document this opt-in smoke runner in the thermal README and keep any guarded
production call-site flag as a separate reviewed task.
