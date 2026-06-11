# Handoff: Strong Prescribed-Temperature Tension Diagnostic

## Status

Classification: `strong prescribed-temperature tension diagnostic passed`

Commit status at package creation: pending commit. After the primary commit is created, update this file with the commit hash in a follow-up handoff sync commit if needed. Final pushed HEAD and push status are also reported in the final Codex response.

## Package

Package path:

- `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic`

Primary report:

- `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/REPORT.md`

## Scope Boundaries

Worked only under:

- `examples/TM_comsol_thermal_micro`

Did not modify:

- `examples/TM_comsol_no_thermal_micro`

Did not implement or run:

- full heat PDE
- damage-dependent conductivity
- trainable/PDE temperature field
- D0040
- seed study
- shear extension
- S0110
- material parameter changes
- `l0` changes
- history logic changes
- training loss changes
- boundary condition changes
- legacy top-sigma as primary reaction

## Diagnostic Runs

All cases used seed 23, tension loading, full mode, checkpointed training, and the same compensation schedule:

- `examples/TM_comsol_thermal_micro/load_schedules/load_schedule_D0015_tension_thermal_compensation.csv`

Schedule steps in mm:

- `1.0e-6`, `2.0e-6`, `3.0e-6`, `3.8e-6`, `5.0e-6`, `7.5e-6`, `1.0e-5`, `1.25e-5`, `1.5e-5`

Training settings:

- `hidden_layers=8`
- `neurons=400`
- `seed=23`
- `activation=TrainableReLU`
- `init_coeff=3.0`
- `--full`
- `--n-rprop 300`
- `--n-lbfgs 1`
- `--load-case tension`

Cases:

- Case A: `20260623_strong_A_off_seed23`, `thermal_mode=off`, `delta_T=0 K`
- Case B: `20260623_strong_B_deltaT0_seed23`, `thermal_mode=uniform`, `delta_T=0 K`
- Case C: `20260623_strong_C_deltaT20_seed23`, `thermal_mode=uniform`, `delta_T=+20 K`

Successful run times:

- Case A: `00:02:52.0600479`
- Case B: `00:02:56.2386905`
- Case C: `00:10:32.3439919`

## Key Results

- A and B reaction, stress, selected energy terms, alpha, HI/HII, and checkpoint availability are identical within table precision.
- Case C shifted reaction/stress downward under positive uniform thermal expansion.
- Case C crossed energy-conjugate reaction from compressive to tensile between `3.0e-6` and `3.8e-6 mm`.
- Case C interpolated zero crossing: `3.450374303916948e-6 mm`.
- Ideal compensation estimate: `3.78e-6 mm`.
- Final nominal stress: A/B `124.04142180457713 MPa`, C `94.41127767786384 MPa`.
- Final alpha max: A/B `0.1582225263118744`, C `0.036508958786726`.
- HI/HII remained finite.
- Final mechanics-drive location classification: `notch_tip_region` for A/B/C.
- Energy-conjugate reaction was available for all 9 steps in all 3 cases.
- No heat PDE or damage-dependent conductivity was active.

## Output Tables And Figures

Required tables:

- `tables/strong_thermal_case_summary.csv`
- `tables/strong_thermal_case_comparison.csv`
- `tables/reaction_stress_by_step.csv`
- `tables/thermal_compensation_analysis.csv`
- `tables/thermal_effect_summary.csv`
- `tables/checkpoint_availability_summary.csv`
- `tables/damage_drive_summary.csv`
- `tables/energy_terms_by_step.csv`
- `tables/training_diagnostics_summary.csv`
- `tables/no_heat_pde_guard_summary.csv`
- `tables/changed_files_summary.csv`

Figures:

- `figures/reaction_vs_displacement.png`
- `figures/nominal_stress_vs_strain.png`
- `figures/reaction_shift_C_minus_A.png`
- `figures/alpha_max_vs_step.png`
- `figures/HI_HII_peaks_vs_step.png`
- `figures/energy_terms_vs_step.png`
- `figures/final_alpha_comparison.png`
- `figures/figure_summary.md`

## Validation Run Before Commit

```powershell
git status --short --branch
D:\anaconda3\envs\torch_env\python.exe -m compileall -q examples\TM_comsol_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q
<package schema/file existence check>
git diff --check
git diff --name-only -- examples/TM_comsol_no_thermal_micro
```

Results:

- `git status --short --branch`: `## main...origin/main`, with only the new thermal schedule and package untracked before staging.
- `compileall`: exit code 0.
- focused patch test: `8 passed in 1.65s`.
- package schema/file existence check: `package_check=passed; classification=strong prescribed-temperature tension diagnostic passed`.
- `git diff --check`: no output.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: no output.

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/REPORT.md`
2. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/tables/strong_thermal_case_summary.csv`
3. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/tables/strong_thermal_case_comparison.csv`
4. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/tables/thermal_compensation_analysis.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/tables/thermal_effect_summary.csv`
6. `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic/tables/no_heat_pde_guard_summary.csv`
7. `examples/TM_comsol_thermal_micro/thermal_prescribed.py`
8. `examples/TM_comsol_thermal_micro/compute_energy_mixed_tm.py`

## Next Recommended Task

Review this package, then run one moderate non-smoke prescribed-temperature tension diagnostic with a slightly denser schedule around `3.0e-6` to `4.5e-6 mm`, still limited to A/B/C, still using checkpointed energy-conjugate reaction, and still without heat PDE or damage-dependent conductivity.
