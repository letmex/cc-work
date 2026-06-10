# Existing-Geometry Shear Smoke Report

## Scope

This package reports one minimal shear smoke test on the existing `TM_comsol_no_thermal_micro` notch geometry. It uses the cleaned single route: `mixedH_TM + tm_source + history`, AT2, default alpha initialization, top-u-free style mechanics, unit-box coordinate normalization, existing material parameters, existing `l0`, existing TM split, and checkpointed energy-conjugate reaction. No D0040 run, seed study, alpha-init-intact route, staggered route, imposed alpha notch, local/lip/jump loss, or legacy top-sigma output was added.

## Implementation Summary

The new normal load-case option supports `tension` and `shear`, with `tension` remaining the default. The shear ansatz uses

```text
eta = (y - y_min) / H
bubble = eta * (1 - eta)
free_top_shape = eta + bubble
u = Delta_s * (eta + bubble * raw_u)
v = Delta_s * free_top_shape * raw_v
```

This fixes bottom `u=v=0`, imposes top `u=Delta_s`, and leaves top `v` free through the network output. The shear schedule is `load_schedules/load_schedule_S0020_shear.csv` with five steps: `1e-6, 5e-6, 1e-5, 1.5e-5, 2e-5` mm.

Postprocessing reads `load_case=shear` from settings and writes shear labels: `Delta_s`, `engineering_shear_strain`, `reaction_N_energy`, and `nominal_shear_stress_energy_MPa`. The primary reaction is checkpointed energy-conjugate `dPi/dDelta_s`. The normal output does not contain `legacy_top_sigma`, `reaction_N_legacy`, or tensile `nominal_stress_energy_MPa` fields for this shear run.

## Run

Training command:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 20 --n-lbfgs 0 --load-case shear --load-schedule-file load_schedules/load_schedule_S0020_shear.csv --run-suffix seed23_S0020_shear
```

Postprocess command:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0020_shear --result-dir outputs\results\seed23_S0020_shear --device cpu
```

Local source output:

```text
D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\outputs\results\seed23_S0020_shear
```

## Required Qualitative Checks

1. Did the shear load case run to completion?  
Yes. Seed 23 completed all five S0020 shear steps.

2. Did checkpointed energy-conjugate shear reaction compute successfully?  
Yes. `reaction_metric_availability.csv` reports `status=energy_conjugate`, `checkpoint_count=5`, and `exact_reaction_computable=True`.

3. Does the final `u` field show horizontal shear displacement?  
Yes. Final `u` ranges from 0 at the bottom to `Delta_s=2e-5` mm at the top and shows the intended horizontal shear gradient.

4. Is top-boundary `v` free, and what are its mean/min/max values?  
Yes. The final top-boundary values are mean `-1.621801e-06` mm, min `-1.022874e-05` mm, and max `7.009092e-06` mm. The top `v` field is not identically zero.

5. Does the final `v` field show reasonable relaxation rather than uncontrolled drift?  
The final `v` field is smooth and finite. Its absolute top-boundary maximum is `1.022874e-05` mm, about `0.511` times `Delta_s`, classified here as moderate rather than runaway.

6. Does alpha initiate or grow from the existing notch region?  
Only weakly. Final `alpha_max=1.420977e-02` occurs near `(x,y)=(4.935277e-03, 5.471391e-03)` mm, close to the explicit notch-tip region, but the magnitude is very small.

7. Is HII active under shear loading?  
Yes. Final `HII_max=9.437907e-05` and all steps have nonzero HII.

8. What is the HII/HI peak ratio by step and at final step?  
The peak ratio is recorded in `tables/shear_damage_drive_summary.csv`. It is approximately 0.62 to 0.63 across the smoke, and final `HII_max/HI_max=0.632`.

9. Does mechanics drive localize near the notch or crack path?  
Not convincingly. There is notch-region signal, but the global mechanics-drive maximum at the final step is near `(x,y)=(1.128096e-04, 2.940627e-04)` mm, i.e. close to the lower boundary/corner. This is the main reason the smoke is not classified as convincing.

10. Does the shear stress-strain curve show a peak or post-peak drop?  
No. The five-step S0020 smoke is monotonic, rising from `1.834144` MPa to `29.297048` MPa.

11. Does alpha>=0.8 through-crack form?  
No. `alpha0p8_through_crack=False` for all five steps.

12. Is the result qualitatively reasonable enough to continue, or should the next task diagnose shear ansatz/boundary/drive issues?  
Classification: `shear smoke not convincing`. The implementation path is working, but the drive localization is boundary/corner dominated and the short smoke shows no peak/drop or through-crack. The next minimal task should diagnose shear ansatz/boundary/drive localization before treating the shear response as a useful physical result.

## Classification

`shear smoke not convincing`

Reason: run completion, checkpointed reaction, shear labels, and top-v freedom are all verified; however, the mechanics-drive maximum is not notch-path dominated, the shear curve is monotonic, and no alpha>=0.8 through-crack appears in this smoke-level loading.

## What This Does Not Claim

This package does not claim physical validation, shear fracture calibration, seed robustness, or model correctness. It only verifies that the single cleaned pipeline can execute one shear load case and identifies the next diagnostic issue.
