# Prescribed-Temperature Micro-Notch Diagnostic Report

## Classification

`prescribed-temperature micro-notch diagnostic passed`

This diagnostic confirms that the thermal subproject can run the existing micro-notch mechanics route with prescribed thermal strain. The `delta_T=0` branch agrees with the no-thermal branch within documented deterministic tolerance, energy-conjugate reaction remains available, and the positive uniform `delta_T` case produces a physically interpretable thermoelastic shift without activating a heat PDE or damage-dependent conductivity. This is a small diagnostic only, not physical validation.

## 1. Purpose

The purpose was to exercise the already implemented prescribed-temperature thermal-strain branch in `examples/TM_comsol_thermal_micro` on a small micro-notch mechanics diagnostic. The diagnostic answered whether:

- `thermal_mode=off` still works as the thermal-subproject no-thermal route.
- `thermal_mode=uniform` with `delta_T=0 K` reproduces the no-thermal branch.
- `thermal_mode=uniform` with `delta_T=+20 K` creates the expected thermoelastic shift.
- Damage/history/reaction outputs remain finite and usable.
- The run remains strictly prescribed-temperature only.

## 2. Cases Run

| Case | Run id | Thermal mode | delta_T_K | Purpose |
|---|---|---:|---:|---|
| A | `20260622_diag_A_off_seed23` | `off` | 0 | Local no-thermal baseline inside the thermal subproject. |
| B | `20260622_diag_B_deltaT0_seed23` | `uniform` | 0 | Zero thermal-strain branch equivalence check. |
| C | `20260622_diag_C_deltaT20_seed23` | `uniform` | 20 | Positive prescribed uniform heating response check. |

No Case D was run. No seed study, D0040, S0110, shear continuation, high-load fracture extension, heat PDE, or damage-dependent conductivity run was performed.

## 3. Schedule And Training Settings

Schedule:

- `load_schedules/load_schedule_D0003_tension_thermal_micro_notch.csv`
- 3 displacement steps: `1.0e-6`, `2.0e-6`, `3.0e-6` mm
- Final displacement: `3.0e-6 mm`
- This is a conservative micro-notch diagnostic schedule, not a fracture-extension schedule and not physical validation.

Training settings used for all three cases:

- `hidden_layers=2`
- `neurons=20`
- `seed=23`
- `activation=TrainableReLU`
- `init_coeff=3.0`
- `--smoke`
- `--n-rprop 3`
- `--n-lbfgs 0`
- `--load-case tension`
- checkpointed training, with three step checkpoints available per case

Commands used:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 23 TrainableReLU 3.0 --smoke --n-rprop 3 --n-lbfgs 0 --load-schedule-file load_schedules/load_schedule_D0003_tension_thermal_micro_notch.csv --load-case tension --run-suffix 20260622_diag_A_off_seed23
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 23 TrainableReLU 3.0 --smoke --n-rprop 3 --n-lbfgs 0 --load-schedule-file load_schedules/load_schedule_D0003_tension_thermal_micro_notch.csv --load-case tension --thermal-mode uniform --thermal-delta-T 0 --run-suffix 20260622_diag_B_deltaT0_seed23
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 23 TrainableReLU 3.0 --smoke --n-rprop 3 --n-lbfgs 0 --load-schedule-file load_schedules/load_schedule_D0003_tension_thermal_micro_notch.csv --load-case tension --thermal-mode uniform --thermal-delta-T 20 --run-suffix 20260622_diag_C_deltaT20_seed23
```

## 4. No-Thermal Project Scope

The original no-thermal project `examples/TM_comsol_no_thermal_micro` was not touched. All committed artifacts for this diagnostic are under `examples/TM_comsol_thermal_micro`.

## 5. Heat PDE Status

No full heat PDE was implemented or activated. The diagnostic used only prescribed thermal strain through the existing thermal mode CLI.

## 6. Damage-Dependent Conductivity Status

No damage-dependent conductivity was implemented or activated. No `k(d)=g(d)k0` behavior was added or used.

## 7. delta_T=0 Equivalence

Case B reproduced Case A exactly within the CSV output precision used for this deterministic diagnostic:

- Maximum reaction difference: `0`
- Maximum nominal stress difference: `0`
- Selected energy term differences: `0`
- Final alpha max difference: `0`
- Checkpoint count: `3` in both cases
- Energy-conjugate reaction availability: true in both cases

This indicates no `delta_T=0` thermal-branch regression was detected.

## 8. delta_T=+20 K Response

Case C produced a downward thermoelastic response shift relative to Case A:

| Step | Delta_mm | A stress MPa | C stress MPa | C - A stress MPa | A reaction N | C reaction N | C - A reaction N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1.0e-6` | 3.525284 | -11.167426 | -14.692710 | 0.035253 | -0.111674 | -0.146927 |
| 1 | `2.0e-6` | 7.233906 | 1.421877 | -5.812029 | 0.072339 | 0.014219 | -0.058120 |
| 2 | `3.0e-6` | 10.950411 | 5.304049 | -5.646362 | 0.109504 | 0.053040 | -0.056464 |

The first positive-heating step is compressive under the energy-conjugate reaction metric, then the response becomes tensile at larger imposed displacement.

## 9. Thermal Reaction Sign And Trend

The sign and trend are physically interpretable for the current prescribed displacement ansatz. Positive uniform thermal expansion subtracts from the normal elastic strains through:

```text
delta_T = T - Tref
exx_e = exx - alpha_T * delta_T
eyy_e = eyy - alpha_T * delta_T
exy_e = exy
```

At fixed small tensile displacement, this reduces the effective tensile elastic strain and can shift the energy-conjugate reaction downward. In Case C this produces a compressive reaction at the first step and a lower tensile reaction at later steps.

## 10. Alpha/Damage Stability

Alpha did not run away in the prescribed thermal case. Final alpha max values:

- Case A: `0.5875001549720764`
- Case B: `0.5875001549720764`
- Case C: `0.5875001549720764`

The final alpha max location was the same across cases:

- `(0.000288248237, 0.000118981952)`

This short diagnostic did not validate fracture growth; it only checked that prescribed thermal strain wiring did not destabilize the damage/history route.

## 11. HI/HII/History

HI and HII remained finite and interpretable:

- Case A final HI peak: `6.2915160015109e-06`
- Case A final HII peak: `3.749779352801852e-06`
- Case B final HI peak: `6.2915160015109e-06`
- Case B final HII peak: `3.749779352801852e-06`
- Case C final HI peak: `8.32687146612443e-06`
- Case C final HII peak: `5.25907762494171e-06`

The final mechanics-drive location classification was `other_domain` for all three cases.

## 12. Energy-Conjugate Reaction Availability

Energy-conjugate reaction was available at all steps for all three cases:

- Case A: 3 checkpoints, 3 field files, exact reaction computable true
- Case B: 3 checkpoints, 3 field files, exact reaction computable true
- Case C: 3 checkpoints, 3 field files, exact reaction computable true

## 13. Legacy Reaction Metrics

Legacy top-sigma reaction was not used as the primary reaction. The reported primary reaction is the energy-conjugate reaction from checkpointed training outputs.

## 14. Physical Validation Status

This is not physical validation. It is a small software/physics-routing diagnostic for prescribed thermal strain in the thermal subproject.

## 15. Next Safe Task

The next safe task is to run a slightly less smoke-like prescribed-temperature tension diagnostic in `examples/TM_comsol_thermal_micro` with the same A/B/C thermal comparison and checkpointed energy reaction, still without heat PDE or damage-dependent conductivity, before considering longer fracture-extension schedules.

## Evidence Files

- `tables/micro_notch_thermal_case_summary.csv`
- `tables/micro_notch_thermal_case_comparison.csv`
- `tables/reaction_stress_by_step.csv`
- `tables/thermal_effect_summary.csv`
- `tables/checkpoint_availability_summary.csv`
- `tables/damage_drive_summary.csv`
- `tables/no_heat_pde_guard_summary.csv`
- `tables/changed_files_summary.csv`
- `figures/figure_summary.md`
- `figures/reaction_vs_displacement.png`
- `figures/nominal_stress_vs_strain.png`
- `figures/alpha_max_vs_step.png`
- `figures/final_alpha_comparison.png`
- `figures/final_alpha_comparison_spatial.png`
- `figures/thermal_effect_reaction_shift.png`

## Validation Run Before Commit

Fresh validation executed from repository root:

```powershell
git status --short --branch
git diff --name-only -- examples/TM_comsol_no_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m compileall -q examples\TM_comsol_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q
```

Observed results:

- `git status --short --branch`: branch `main...origin/main`; only the new thermal schedule and diagnostic package were untracked before staging.
- `git diff --name-only -- examples/TM_comsol_no_thermal_micro`: no output.
- `compileall`: exit code 0.
- `pytest`: `8 passed in 1.60s`.
