# Project Memory Handoff Report

## Scope

This task created a self-contained project memory for a new Codex conversation window. It summarizes the current verified `TM_comsol_no_thermal_micro` route, reaction policy, shear run history, hard constraints, and the next recommended task.

## Files Written

- `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`
- `examples/TM_comsol_no_thermal_micro/PROJECT_MEMORY.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/README.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/REPORT.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/commands_run.txt`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/HANDOFF_COMMENT.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/MANIFEST.json`

## What Was Not Done

- No training was run.
- No postprocessing was run.
- No D0040 run was started.
- No seed study was run.
- No physics, model, or source-code changes were made.

## Memory Contents

The memory file includes:

- repository and working paths
- current verified normal route
- cleanup package and reaction policy
- existing shear ansatz and top-v monitor policy
- S0020, S0030, and S0050 shear run history
- current S0050 key numbers
- next recommended same-path shear connectivity extension
- required next diagnostics and tables/figures
- persistent constraints for future Codex tasks

## Next Recommended Task

Run one slightly longer same-path shear extension with detailed crack-connectivity diagnostics:

`examples/TM_comsol_no_thermal_micro/runs/20260617_existing_geometry_shear_connectivity_extension`

Suggested schedule:

`load_schedules/load_schedule_S0070_shear.csv`

Use seed 23 only, existing geometry, same shear ansatz, same top-v-free boundary, same physical route, checkpointed energy reaction, and normal postprocessing. If clean continuation from S0050 is not safe, rerun from step 0 and document `continued_from_S0050=False`.
