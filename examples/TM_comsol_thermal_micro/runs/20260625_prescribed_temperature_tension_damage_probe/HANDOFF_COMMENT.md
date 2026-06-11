# Handoff: Moderate Prescribed-Temperature Tension Damage Probe

## Status

Final classification: `moderate prescribed-temperature damage probe passed`

Commit hash:

- Primary diagnostic commit: `e58ee646cd75186c954dcac09cc75fc97b569045` (`Run prescribed temperature tension damage probe`).
- Handoff sync commit: recorded in final Codex response. This file does not chase its own commit hash.

Push status:

- Primary diagnostic commit pushed to `origin/main`.
- Final status after primary push: `## main...origin/main`, ahead/behind `0 0`.
- Final HEAD known at handoff-sync edit time: `e58ee646cd75186c954dcac09cc75fc97b569045`.

## Package

- Package path: `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe`
- Report: `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/REPORT.md`
- Schedule: `examples/TM_comsol_thermal_micro/load_schedules/load_schedule_D0020_tension_thermal_damage_probe.csv`

## Runs

- Case A: `20260625_damage_probe_A_off_seed23`, thermal mode `off`, delta_T `0.0 K`, seed 23
- Case B: `20260625_damage_probe_B_deltaT0_seed23`, thermal mode `uniform`, delta_T `0.0 K`, seed 23
- Case C: `20260625_damage_probe_C_deltaT20_seed23`, thermal mode `uniform`, delta_T `20.0 K`, seed 23

## Scope

- Worked only under `examples/TM_comsol_thermal_micro`.
- Did not modify `examples/TM_comsol_no_thermal_micro`.
- Ran training for A/B/C D0020 tension damage probe only.
- Did not implement heat PDE, damage-dependent conductivity, trainable/PDE temperature, D0040, seed study, shear extension, or S0110.
- Did not change material parameters, l0, history logic, training losses, boundary conditions, source model behavior, or reaction route.
- Energy-conjugate `reaction_N_energy` is the primary reaction.

## Tables Generated

- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_probe_case_summary.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_probe_case_comparison.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/reaction_stress_by_step.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/alpha_notch_metrics_by_step.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/alpha_threshold_connectivity_by_step.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_delay_summary.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/HI_HII_drive_by_step.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/energy_terms_by_step.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/no_heat_pde_guard_summary.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/training_diagnostics_summary.csv`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/changed_files_summary.csv`

## Figures Generated

- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/reaction_vs_displacement.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/nominal_stress_vs_strain.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/reaction_shift_C_minus_A.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/alpha_max_vs_step.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/notch_alpha_vs_displacement.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/HI_HII_peaks_vs_step.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/energy_terms_vs_step.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/alpha_threshold_area_vs_step.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_global_scale.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_low_range_scale.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_high_threshold_masks.png`
- `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/figure_summary.md`

## Main Conclusion

The zero-temperature thermal branch matches the no-thermal branch. The +20 K branch keeps the expected downward reaction/stress shift and lower notch-tip/high-threshold alpha growth. Low-level diffuse alpha background is reported separately and is not used as fracture evidence.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/REPORT.md`
2. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_probe_case_summary.csv`
3. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_probe_case_comparison.csv`
4. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/damage_delay_summary.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/tables/alpha_threshold_connectivity_by_step.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_global_scale.png`
7. `examples/TM_comsol_thermal_micro/runs/20260625_prescribed_temperature_tension_damage_probe/figures/final_alpha_high_threshold_masks.png`

## Exact Next Recommended Task

Review this package and the high-threshold/notch metrics before deciding on any further validation. Do not begin heat PDE or damage-dependent conductivity work from this diagnostic alone.
