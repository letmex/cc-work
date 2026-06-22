# Handoff: Solved-Temperature Mechanics Smoke Adapter

## Status

Final classification: `solved-temperature mechanics smoke adapter implemented and tests passed`

Primary implementation commit:

- pending until this package is committed

Push status:

- pending until commit/push

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter`
- Report: `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter/MANIFEST.json`

## Key Result

Added a narrow element-centroid adapter that converts supported bridge
temperature sources into an element-sized `thermal_delta_T` for
`compute_mixed_tm_fields`.

The adapter does not implement coupled heat-mechanics training. It does not
train heat and mechanics together.

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Did not modify `train_mixed_tm.py`.
- Did not change material, `l0`, history, reaction, boundary, or loss routes.
- Did not run broad mechanics/fracture training.
- Did not implement heat-fracture diagnostics, fracture heat training, or
  damage-dependent conductivity.

## Validation

- `git status`: PASS, clean before work and only expected thermal files changed before commit.
- recursive `py_compile` under `examples\TM_comsol_thermal_micro`: PASS.
- `test_constant_k0_heat_pde_patch.py`: PASS, 18 passed.
- `test_heat_only_weak_form_training.py`: PASS, 11 passed.
- `test_h1_quadrature_regularization_review.py`: PASS, 9 passed.
- `test_heat_correction_policy.py`: PASS, 7 passed.
- `test_solved_temperature_mechanics_bridge.py`: PASS, 7 passed.
- `test_solved_temperature_mechanics_smoke_adapter.py`: PASS, 6 passed.
- `test_prescribed_thermal_strain_patch.py`: PASS, 8 passed.
- package schema/file existence check: PASS.
- `git diff --check`: PASS.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: PASS, empty output.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter/REPORT.md`
2. `examples/TM_comsol_thermal_micro/thermal_mechanics_adapter.py`
3. `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`
4. `examples/TM_comsol_thermal_micro/thermal_solution_bridge.py`
5. `examples/TM_comsol_thermal_micro/compute_energy_mixed_tm.py`
6. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Add an explicit opt-in smoke script or small CLI wrapper that calls this adapter
for one fixture mechanics patch and writes a compact result table, still without
changing `train_mixed_tm.py` defaults or running broad training.
