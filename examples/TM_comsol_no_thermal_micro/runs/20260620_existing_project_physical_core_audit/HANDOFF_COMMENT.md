# Handoff Comment: Existing Project Physical-Core Audit

Package folder: `examples/TM_comsol_no_thermal_micro/runs/20260620_existing_project_physical_core_audit`
Package commit hash: `df09311e13ffac2a7c0bde116c1f5a09cb97a7a1`
Commit pushed: `yes`

## Source Packages and Files Reviewed

- Project memory: `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`
- Project memory pointer: `examples/TM_comsol_no_thermal_micro/PROJECT_MEMORY.md`
- Current execution tree: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`
- Current source files reviewed: `config.py`, `compute_energy_mixed_tm.py`, `mixed_mode_tm.py`, `history_field_mixed_tm.py`, `field_computation.py`, `postprocess_results.py`, `source/material_properties.py`, `source/pff_model.py`
- Current docs reviewed: `README.md`, `POSTPROCESS_WORKFLOW.md`, `PROJECT_STRUCTURE.md`
- Tests reviewed by source inspection: `test_single_verified_pipeline.py`, `test_history_mode_controls.py`, `test_postprocess_results.py`, `test_shear_load_case.py`, `test_shear_connectivity.py`, `test_shear_package_builder.py`, `test_coord_normalization.py`, `test_project_cleanup_interface.py`, `test_project_directory_hygiene.py`
- Prior packages reviewed: `20260618_existing_geometry_shear_longer_connectivity_extension`, `20260619_shear_stress_magnitude_sanity_audit`

## Execution Status

- Source code modified: no
- Training run: no
- Postprocessing run: no
- D0040 run: no
- Seed study run: no
- COMSOL runtime used: no
- Physics/boundary/shear ansatz/material/l0/history/alpha/loss changes: no

## Final Physical-Core Classification

`no-thermal physical core acceptable as thermal baseline with documented platform differences`

## Biggest Acceptable Differences

- COMSOL can degrade positive stress through an external-stress construction; PINN can degrade crack-driving energy directly.
- COMSOL solves FEM PDEs; PINN optimizes a neural ansatz with energy/residual losses.
- COMSOL heat transfer is active in comp3; the current baseline intentionally omits heat transfer.
- COMSOL boundary constraints and PINN ansatz constraints differ in implementation but preserve boundary meaning.
- COMSOL reaction extraction and PINN checkpoint energy derivative differ, with `reaction_N_energy` remaining the trusted current route.

## Biggest Unacceptable Risks To Avoid Later

- Wrong material or `l0` units.
- Silent plane-stress versus plane-strain mismatch in thermal strain tests.
- Thermal strain using `T` instead of `T-Tref`.
- Degrading compressive stress without justification.
- Reintroducing legacy top-sigma reaction as the primary metric.
- Losing HI/HII max-history irreversibility.
- Mixing comp4 or `TFinal` into the single-notch branch.

## Recommended Next Step

Create a prescribed-temperature thermal-strain branch with patch tests first. Do not start with full heat-equation or damage-dependent-conductivity coupling.

## Evidence Status

- repo_memory: present
- project_memory: present
- current_config: present
- current_energy: present
- current_split: present
- current_history: present
- current_fields: present
- current_postprocess: present
- current_material: present
- current_pff: present
- current_readme: present
- postprocess_workflow: present
- project_structure: present
- s0090_handoff: present
- stress_audit_handoff: present
