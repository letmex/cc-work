# Handoff: Solved-Temperature Smoke README Documentation

## Status

Final classification: `solved-temperature smoke README documentation implemented and tests passed`

Primary implementation commit:

- pending until this package is committed

Push status:

- pending until commit/push

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs`
- Report: `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs/MANIFEST.json`

## Key Result

Documented the solved-temperature mechanics smoke CLI in the thermal README and
added a focused README guard test.

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Did not modify `train_mixed_tm.py`.
- Did not change material, `l0`, history, reaction, boundary, or loss routes.
- Did not run broad mechanics/fracture training.
- Did not implement heat-fracture diagnostics, fracture heat training, or
  damage-dependent conductivity.

## Validation

- recursive `py_compile` under `examples\TM_comsol_thermal_micro`: PASS.
- `test_solved_temperature_mechanics_readme_docs.py`: PASS, 2 passed.
- `test_solved_temperature_mechanics_smoke_cli.py`: PASS, 3 passed.
- `test_solved_temperature_mechanics_smoke_adapter.py`: PASS, 6 passed.
- `test_solved_temperature_mechanics_bridge.py`: PASS, 7 passed.
- `test_prescribed_thermal_strain_patch.py`: PASS, 8 passed.
- package schema/file existence check: PASS.
- `git diff --check`: PASS.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: PASS, empty output.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/README.md`
2. `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_readme_docs.py`
3. `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs/REPORT.md`
4. `examples/TM_comsol_thermal_micro/run_solved_temperature_mechanics_smoke.py`
5. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Keep any guarded production call-site flag as a separate reviewed task with its
own opt-in CLI/config tests before touching `train_mixed_tm.py`.
