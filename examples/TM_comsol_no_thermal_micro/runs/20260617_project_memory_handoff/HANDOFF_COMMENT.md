## Codex handoff: Project memory for next window

Commit: cbc83e6cfc9c732d1e92fabd253a46130e18b4a6
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/REPORT.md`

### What changed
- Created `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`.
- Created `examples/TM_comsol_no_thermal_micro/PROJECT_MEMORY.md` as a project-level pointer to the canonical memory file.
- Created this lightweight handoff package for the memory-only task.
- No source code or model behavior was changed.

### Commands run
```powershell
git pull origin main
Test-Path examples\TM_comsol_no_thermal_micro\runs\CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md
Test-Path examples\TM_comsol_no_thermal_micro\PROJECT_MEMORY.md
rg -n "20260616_existing_geometry_shear_load_extension|29\.9647|1\.00034|S0070|shear_connectivity_extension|Do not run D0040" examples\TM_comsol_no_thermal_micro\runs\CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md
```

### Key results
- Memory file exists and contains the current verified route, reaction policy, shear ansatz, shear run history, S0050 key results, persistent constraints, and next recommended task.
- S0050 is recorded as the current main existing-geometry shear diagnostic result.
- Next recommended task is a slightly longer same-path shear extension with detailed crack-connectivity diagnostics.
- No training was run.
- No postprocessing was run.
- No D0040 run was started.
- No seed study was run.
- No code was changed.

### Files to read first
- `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/REPORT.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/HANDOFF_COMMENT.md`
- `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/REPORT.md`

### Question for ChatGPT
1. Is this memory file sufficient as the first-read document for a new Codex conversation window?
2. Should the next prompt now proceed with the S0070 same-path shear connectivity extension, or should it first add a reusable connectivity analysis helper without running training?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from single-seed diagnostic runs.
- Do not run D0040 or a seed study unless explicitly requested.
