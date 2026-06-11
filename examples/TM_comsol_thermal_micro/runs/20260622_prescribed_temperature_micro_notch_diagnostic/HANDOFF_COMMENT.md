# Handoff: Prescribed-Temperature Micro-Notch Diagnostic

## Status

Classification: `prescribed-temperature micro-notch diagnostic passed`

Primary diagnostic commit:

- `29e76d7` (`Run prescribed temperature micro notch diagnostic`)

This handoff update is the follow-up sync change requested by the diagnostic prompt. Final pushed HEAD and push status are reported in the final Codex response after `git push`; the self-hash of this sync commit cannot be known until the commit is created.

## Package

Package path:

- `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic`

Primary report:

- `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/REPORT.md`

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
- shear continuation
- S0110
- material parameter changes
- `l0` changes
- history logic changes
- training loss changes
- legacy top-sigma as primary reaction

## Diagnostic Runs

All runs used seed 23, tension loading, smoke mesh, checkpointed training, and the same three-step diagnostic schedule:

- `examples/TM_comsol_thermal_micro/load_schedules/load_schedule_D0003_tension_thermal_micro_notch.csv`

Training settings:

- `hidden_layers=2`
- `neurons=20`
- `seed=23`
- `activation=TrainableReLU`
- `init_coeff=3.0`
- `--smoke`
- `--n-rprop 3`
- `--n-lbfgs 0`
- `--load-case tension`

Cases:

- Case A: `20260622_diag_A_off_seed23`, `thermal_mode=off`, `delta_T=0 K`
- Case B: `20260622_diag_B_deltaT0_seed23`, `thermal_mode=uniform`, `delta_T=0 K`
- Case C: `20260622_diag_C_deltaT20_seed23`, `thermal_mode=uniform`, `delta_T=+20 K`

## Key Results

- A and B reaction, nominal stress, selected energy terms, final alpha max, and checkpoint availability are identical within table precision.
- Case C shifts the response downward under positive uniform thermal expansion.
- Case C step-0 reaction is compressive: `-0.1116742569138296 N`.
- Case C final nominal stress is `5.304048681864515 MPa`, versus Case A final nominal stress `10.95041079679504 MPa`.
- Final alpha max is stable and equal across A/B/C: `0.5875001549720764`.
- Final HI/HII remain finite.
- Energy-conjugate reaction is available for all three steps in all three cases.
- No heat PDE or damage-dependent conductivity was active.

## Output Tables And Figures

Required tables:

- `tables/micro_notch_thermal_case_summary.csv`
- `tables/micro_notch_thermal_case_comparison.csv`
- `tables/reaction_stress_by_step.csv`
- `tables/thermal_effect_summary.csv`
- `tables/checkpoint_availability_summary.csv`
- `tables/damage_drive_summary.csv`
- `tables/no_heat_pde_guard_summary.csv`
- `tables/changed_files_summary.csv`

Additional table:

- `tables/thermal_reaction_shift_by_step.csv`

Figures:

- `figures/reaction_vs_displacement.png`
- `figures/nominal_stress_vs_strain.png`
- `figures/alpha_max_vs_step.png`
- `figures/final_alpha_comparison.png`
- `figures/final_alpha_comparison_spatial.png`
- `figures/thermal_effect_reaction_shift.png`
- `figures/figure_summary.md`

## Validation Already Run

Fresh validation before staging:

```powershell
git status --short --branch
git diff --name-only -- examples/TM_comsol_no_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m compileall -q examples\TM_comsol_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q
```

Results:

- `git status --short --branch`: `## main...origin/main`, with only the new thermal schedule and package untracked before staging.
- `git diff --name-only -- examples/TM_comsol_no_thermal_micro`: no output.
- `compileall`: exit code 0.
- `pytest`: `8 passed in 1.60s`.

Additional validation still required after writing this file and `MANIFEST.json`:

- package schema/file existence check
- `git diff --check`
- final `git status`
- commit
- push to `origin/main`
- final clean/up-to-date `git status`

## Reviewer Should Read Next

1. `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/REPORT.md`
2. `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/tables/micro_notch_thermal_case_summary.csv`
3. `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/tables/micro_notch_thermal_case_comparison.csv`
4. `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/tables/thermal_effect_summary.csv`
5. `examples/TM_comsol_thermal_micro/runs/20260622_prescribed_temperature_micro_notch_diagnostic/tables/no_heat_pde_guard_summary.csv`
6. `examples/TM_comsol_thermal_micro/THERMAL_STRAIN_PATCH_TESTS.md`
7. `examples/TM_comsol_thermal_micro/thermal_prescribed.py`
8. `examples/TM_comsol_thermal_micro/compute_energy_mixed_tm.py`

## Next Recommended Task

Run a slightly less smoke-like prescribed-temperature tension diagnostic in `examples/TM_comsol_thermal_micro` with the same A/B/C comparison and checkpointed energy reaction. Keep it prescribed-temperature only and continue to avoid heat PDE, damage-dependent conductivity, and long fracture-extension schedules until this route is reviewed.
