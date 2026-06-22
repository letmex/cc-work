# Solved-Temperature Train Guard

## 1. Purpose

This package adds a guarded, default-off training-entry flag for the already
reviewed solved-temperature mechanics smoke path.

The flag is intentionally narrow. It only lets `train_mixed_tm.py` opt into the
H1 frozen-lift element-centroid adapter after fine-mesh connectivity exists.
It does not implement coupled heat-mechanics training.

## 2. What Was Implemented

Modified:

- `examples/TM_comsol_thermal_micro/config.py`
- `examples/TM_comsol_thermal_micro/train_mixed_tm.py`
- `examples/TM_comsol_thermal_micro/README.md`
- `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`
- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`

Added:

- `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_train_guard.py`
- `examples/TM_comsol_thermal_micro/runs/20260707_solved_temperature_train_guard`

## 3. New Flag

The new explicit flag is:

```text
--solved-temperature-mechanics-smoke
```

It defaults to `False`.

When enabled, `training_dict` records:

- `solved_temperature_mechanics_smoke = True`
- `solved_temperature_source_mode = solved_frozen_lift`
- `solved_temperature_case_id = H1`
- `solved_temperature_evaluation_location = element_centroid`
- `solved_temperature_bounds = [[0.0, 0.01], [0.0, 0.01]]`
- `solved_temperature_T_bottom = 300.0`
- `solved_temperature_T_top = 320.0`
- `solved_temperature_T_ref = 300.0`

## 4. Train Hook Placement

The hook is placed after fine mesh data are available:

```text
inp, T_conn, area_T, _ = prep_input_data(...)
```

and before:

```text
training_set = _make_training_set(inp, device)
```

This is the first point in `train_mixed_tm.py` where both nodal coordinates and
triangle connectivity are available for the element-centroid adapter.

## 5. Guard Behavior

Default path:

- returns existing thermal kwargs unchanged;
- does not import `thermal_mechanics_adapter`;
- keeps `_thermal_energy_kwargs({})` defaults as `thermal_mode="off"`,
  `thermal_temperature=None`, and `thermal_delta_T=None`.

Opt-in path:

- locally imports `build_mechanics_thermal_kwargs_from_bridge`;
- validates fixed H1 smoke settings before import/use;
- rejects prescribed thermal input combinations;
- rejects malformed or non-fixed bounds;
- produces element-sized `thermal_delta_T`;
- records scalar diagnostics in `training_dict`.

Rejected combinations:

- `--solved-temperature-mechanics-smoke` with `--thermal-temperature-K`;
- `--solved-temperature-mechanics-smoke` with `--thermal-delta-T`;
- `--solved-temperature-mechanics-smoke` with non-`off` `--thermal-mode`.

## 6. Focused Test Results

The new guard tests verify:

- config flag default is off;
- opt-in flag sets the training boolean;
- prescribed thermal combinations are rejected;
- no-flag helper does not import the adapter;
- opt-in helper returns `thermal_delta_T = [20/3, 40/3] K` for the two-element
  fixture;
- non-fixed H1 smoke settings are rejected before adapter import;
- malformed broadcastable bounds are rejected;
- adapter import is local, not module-level;
- `_thermal_energy_kwargs` defaults remain off/None.

## 7. Subagent Review

Subagent-driven development was used for implementation and review.

Spec review findings:

- Initial behavior was compliant except `solved_temperature_evaluation_location`
  needed fixed-value validation.
- After adding validation and tests, spec review found no behavior gaps except
  the new test file needing exact-path staging.

Code quality review findings:

- Initial important issue: prescribed thermal options could conflict with the
  smoke adapter `thermal_delta_T`.
- Fix: config and helper-level guards reject those combinations.
- Final important issue: broadcastable malformed bounds could pass validation.
- Fix: bounds shape must match `(2, 2)` before value comparison.
- Final quality review approved.

## 8. Boundaries

This package does not implement:

- coupled heat-mechanics training;
- solved-temperature phase-field fracture coupling;
- heat PDE loss inside mechanics/fracture training;
- heat-fracture diagnostics;
- damage-dependent conductivity;
- D0040, seed study, shear extension, S0110;
- transient production or bottom-cooling production runs.

`examples/TM_comsol_no_thermal_micro` remains untouched.

## 9. Final Classification

`solved-temperature train guard implemented and tests passed`

## 10. Exact Next Recommended Task

Run one tiny explicit smoke invocation of `main.py` with
`--solved-temperature-mechanics-smoke --smoke --n-rprop 1 --n-lbfgs 0
--max-steps 1`, archive only scalar diagnostics, and do not run broad training.
