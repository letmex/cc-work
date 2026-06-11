# Strong Prescribed-Temperature Tension Diagnostic Report

## Classification

`strong prescribed-temperature tension diagnostic passed`

The stronger prescribed-temperature tension diagnostic confirms that the prescribed thermal-strain branch remains consistent under non-smoke training. The `delta_T=0 K` branch agrees with the no-thermal branch, and the `delta_T=+20 K` branch shows a physically interpretable downward reaction/stress shift across the thermal-expansion compensation region while maintaining finite HI/HII, stable alpha, and checkpointed energy-conjugate reaction. This is still a diagnostic, not physical validation, and heat PDE/damage-dependent conductivity remain future work.

## 1. Purpose

The purpose was to repeat the previous A/B/C prescribed-temperature comparison with stronger training, a displacement schedule that crosses the expected `delta_T=20 K` thermal-expansion compensation point, and richer diagnostics for reaction, stress, alpha, HI/HII, selected energy terms, checkpoint availability, and guard status.

## 2. Difference From Previous Smoke Diagnostic

The previous diagnostic used `--smoke`, `hidden_layers=2`, `neurons=20`, `--n-rprop 3`, `--n-lbfgs 0`, and only three displacement steps up to `3.0e-6 mm`. This stronger run used full mode, `hidden_layers=8`, `neurons=400`, `--n-rprop 300`, `--n-lbfgs 1`, and nine displacement steps up to `1.5e-5 mm`.

## 3. Cases Run

| Case | Run id | Thermal mode | delta_T_K | Purpose |
|---|---|---:|---:|---|
| A | `20260623_strong_A_off_seed23` | `off` | 0 | Strong no-thermal baseline inside the thermal subproject. |
| B | `20260623_strong_B_deltaT0_seed23` | `uniform` | 0 | Active thermal branch with zero thermal strain. |
| C | `20260623_strong_C_deltaT20_seed23` | `uniform` | 20 | Positive prescribed uniform heating across compensation region. |

No negative `delta_T`, D0040, seed study, shear extension, S0110, heat PDE, or damage-dependent conductivity run was performed.

## 4. Stronger Training Settings

All cases used:

- `hidden_layers=8`
- `neurons=400`
- `seed=23`
- `activation=TrainableReLU`
- `init_coeff=3.0`
- `--full`
- `--n-rprop 300`
- `--n-lbfgs 1`
- `--load-case tension`
- checkpointed training with every step checkpointed

Observed successful run times:

- Case A: `00:02:52.0600479`
- Case B: `00:02:56.2386905`
- Case C: `00:10:32.3439919`

The initial Case A launch used a PowerShell redirection wrapper that treated native stderr/tqdm output as an error after writing only settings/displacement files. The same strong configuration was rerun with `Start-Process -Wait` and completed successfully with 9 checkpoints and 9 field files. No training settings were weakened.

## 5. Schedule And Compensation Coverage

Schedule:

- `load_schedules/load_schedule_D0015_tension_thermal_compensation.csv`
- Steps in mm: `1.0e-6`, `2.0e-6`, `3.0e-6`, `3.8e-6`, `5.0e-6`, `7.5e-6`, `1.0e-5`, `1.25e-5`, `1.5e-5`
- Nominal engineering strain range: about `1.0e-4` to `1.5e-3`

For `delta_T=20 K`:

```text
epsilon_th = alpha_T * delta_T = 18.9e-6 * 20 = 3.78e-4
Delta_cross_estimate = epsilon_th * 0.01 mm = 3.78e-6 mm
```

The schedule brackets this estimate with the `3.0e-6` and `3.8e-6 mm` steps. This is a stronger thermal-mechanical diagnostic schedule, not a fracture-extension schedule.

## 6. Case B Versus Case A

Case B reproduced Case A exactly within the CSV output precision:

- max absolute reaction difference: `0`
- max absolute nominal stress difference: `0`
- max absolute `Pi_total_kNmm` difference: `0`
- max absolute elastic and fracture energy differences: `0`
- max absolute alpha, HI, and HII peak differences: `0`
- checkpoints: `9` for A and `9` for B
- exact energy-conjugate reaction computable: true for both

No `delta_T=0` regression was detected under stronger training.

## 7. Estimated Thermal Compensation Displacement

The ideal estimate using the project nominal-strain convention is:

- `alpha_T = 1.89e-05 1/K`
- `delta_T = 20 K`
- `epsilon_th = 0.000378`
- `effective_height = 0.01 mm`
- `estimated_Delta_cross = 3.78e-6 mm`

See `tables/thermal_compensation_analysis.csv`.

## 8. Case C Reaction/Stress Shift

Case C showed a downward reaction/stress shift relative to Case A at fixed displacement:

| Metric | Case A | Case C | Shift |
|---|---:|---:|---:|
| Step-0 nominal stress MPa | `7.884819206` | `-14.755522716` | `-22.640341922` |
| Final nominal stress MPa | `124.041421805` | `94.411277678` | `-29.630144127` |
| Final reaction N | `1.240414218` | `0.944112777` | `-0.296301441` |
| Final alpha max | `0.158222526` | `0.036508959` | `-0.121713568` |

The stronger run therefore preserved the expected downward thermoelastic shift while still trending tensile at larger imposed displacement.

## 9. Case C Zero Crossing

Case C crossed from compressive to tensile energy-conjugate reaction:

- at `3.0e-6 mm`: `reaction_N_energy = -0.0019457493181107 N`
- at `3.8e-6 mm`: `reaction_N_energy = 0.0015104857311598 N`
- interpolated zero crossing: `3.450374303916948e-6 mm`
- ideal compensation estimate: `3.78e-6 mm`

The observed zero crossing is below the ideal estimate but inside the scheduled compensation region. The offset is interpreted through the current boundary ansatz, damage/history state, nonlinearity, and training quality. It is not treated as a physical validation discrepancy.

## 10. Alpha/Damage Stability

Alpha remained finite and interpretable:

- Case A final alpha max: `0.1582225263118744`
- Case B final alpha max: `0.1582225263118744`
- Case C final alpha max: `0.036508958786726`

Final alpha max locations:

- Case A/B: `(0.00502108456567, 0.00499864388257)`
- Case C: `(0.00503098359331, 0.00490684481338)`

The lower Case C alpha is consistent with reduced effective tensile drive in this diagnostic. No runaway alpha was observed.

## 11. HI/HII/History

HI and HII remained finite:

- Case A final HI peak: `0.0096437549218535`
- Case A final HII peak: `0.0060907872393727`
- Case C final HI peak: `0.0041668526828289`
- Case C final HII peak: `0.0026316905859857`

Final HII/HI peak ratio was about `0.63158` in both A/B and C at final displacement. The final mechanics-drive location classification was `notch_tip_region` for all cases.

## 12. Energy-Conjugate Reaction Availability

Energy-conjugate reaction was available at every step:

- Case A: 9 checkpoints, 9 field files, exact reaction computable true
- Case B: 9 checkpoints, 9 field files, exact reaction computable true
- Case C: 9 checkpoints, 9 field files, exact reaction computable true

The primary reaction metric remained `reaction_N_energy`.

## 13. Heat PDE Status

No heat PDE was implemented or activated. This was strictly a prescribed-temperature thermal-strain diagnostic using scalar `--thermal-delta-T` for Case C.

## 14. Damage-Dependent Conductivity Status

No damage-dependent conductivity was implemented or activated. No `k(d)=g(d)k0` coupling was added or used.

## 15. Legacy Reaction Metrics

Legacy top-sigma reaction was not used as the primary reaction. The reported reaction/stress curves use checkpointed energy-conjugate reaction.

## 16. Physical Validation Status

This is not physical validation. It is a stronger software/physics-route diagnostic for prescribed thermal strain in the thermal subproject.

## 17. Next Safe Task

The next safe task is to review this package and then run one moderate non-smoke prescribed-temperature tension diagnostic with a slightly denser schedule around `3.0e-6` to `4.5e-6 mm`, still limited to A/B/C, still using checkpointed energy-conjugate reaction, and still without heat PDE or damage-dependent conductivity.

## Evidence Files

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
- `figures/figure_summary.md`
- `figures/reaction_vs_displacement.png`
- `figures/nominal_stress_vs_strain.png`
- `figures/reaction_shift_C_minus_A.png`
- `figures/alpha_max_vs_step.png`
- `figures/HI_HII_peaks_vs_step.png`
- `figures/energy_terms_vs_step.png`
- `figures/final_alpha_comparison.png`

## Validation Run Before Commit

Fresh validation executed from repository root:

```powershell
git status --short --branch
D:\anaconda3\envs\torch_env\python.exe -m compileall -q examples\TM_comsol_thermal_micro
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q
<package schema/file existence check>
git diff --check
git diff --name-only -- examples/TM_comsol_no_thermal_micro
```

Observed results:

- `git status --short --branch`: branch `main...origin/main`; only the new thermal schedule and diagnostic package were untracked before staging.
- `compileall`: exit code 0.
- focused patch test: `8 passed in 1.65s`.
- package schema/file existence check: `package_check=passed; classification=strong prescribed-temperature tension diagnostic passed`.
- `git diff --check`: no output.
- `git diff --name-only -- examples\TM_comsol_no_thermal_micro`: no output.
