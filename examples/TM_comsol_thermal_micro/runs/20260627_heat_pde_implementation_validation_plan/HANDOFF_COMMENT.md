# Handoff: Heat PDE Implementation and Validation Plan

## Status

Final classification: `heat PDE implementation plan complete`

Commit hash:

- Primary plan commit: `218b84bdcf019e591a8dcdbdb4fbd437fe3b044d` (`Plan heat PDE implementation`).
- Handoff sync commit: recorded in final Codex response; this file does not chase its own sync hash.

Push status:

- Primary plan commit push: pushed to `origin/main`.
- Final status after primary push: `## main...origin/main`, ahead/behind `0 0`.
- Final HEAD known at handoff-sync edit time: `218b84bdcf019e591a8dcdbdb4fbd437fe3b044d`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan`
- Report: `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/REPORT.md`
- Manifest: `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/MANIFEST.json`

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- This task did not run training, rerun A/B/C, run D0040, run a seed study, run shear extension, or run S0110.
- This task did not implement heat PDE, damage-dependent conductivity, or a trainable/PDE temperature field.
- This task did not change material parameters, `l0`, history logic, training losses, boundary conditions, source model behavior, or reaction route.
- Energy-conjugate `reaction_N_energy` remains the primary reaction.

## Key Planning Decisions

- First heat PDE implementation should be constant-conductivity heat transfer: `rho * c * dT/dt - div(k0 * grad T) = Q` with `Q=0` initially.
- Use constant `k0`; do not implement `k(d)=g(d)k0` until constant-k0 heat PDE and solved-T-to-mechanics coupling pass independently.
- Preserve `thermal_mode=off` default and the prescribed-temperature fallback branch.
- Preserve thermal strain mechanics: `delta_T = T - Tref`; `exx_e = exx - alpha_T * delta_T`; `eyy_e = eyy - alpha_T * delta_T`; `exy_e = exy`.
- Treat SI-to-project unit conversion as an implementation blocker, not a solved detail.

## Tables Generated

- `tables/heat_pde_scope_summary.csv`
- `tables/thermal_variables_units.csv`
- `tables/implementation_phases.csv`
- `tables/validation_matrix.csv`
- `tables/patch_test_plan.csv`
- `tables/boundary_initial_condition_plan.csv`
- `tables/coupling_dependency_plan.csv`
- `tables/deferred_features.csv`
- `tables/risk_register.csv`
- `tables/comsol_alignment_notes.csv`
- `tables/source_touch_plan.csv`
- `tables/next_decision_gate.csv`
- `tables/changed_files_summary.csv`

## Validation To Report

- `git status`
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/build_heat_pde_plan_package.py`
- package schema/file existence check
- `git diff --check`
- `git diff --name-only -- examples/TM_comsol_no_thermal_micro`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/REPORT.md`
2. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/tables/thermal_variables_units.csv`
3. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/tables/implementation_phases.csv`
4. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/tables/validation_matrix.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/tables/risk_register.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan/tables/next_decision_gate.csv`
7. `examples/TM_comsol_thermal_micro/PROJECT_MEMORY.md`

## Exact Next Recommended Task

Hold the reviewer decision-gate review. If approved, implement only Phase 1 constant-conductivity heat PDE/unit-conversion infrastructure and patch tests. Do not implement damage-dependent conductivity, do not run training, do not run D0040 or seed/shear studies, and do not touch `examples/TM_comsol_no_thermal_micro`.
