# Existing-Geometry Shear Load Extension Report

## Scope

This package runs one controlled shear load extension on the existing COMSOL micro-notch geometry using seed 23. The shear ansatz, top-v-free boundary condition, mixed-drive formula, material parameters, `l0`, TM split, history logic, alpha initialization, and training losses were not changed.

The extension uses `load_schedules/load_schedule_S0050_shear.csv`, a 33-step monotonic schedule from `Delta_s=1e-6` mm to `8e-5` mm. Clean continuation from S0030 was not implemented/was ambiguous, so this is a full rerun from step 0: `continued_from_S0030=False`.

## Commands

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0050_shear.csv --run-suffix seed23_S0050_shear
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0050_shear --result-dir outputs\results\seed23_S0050_shear --device cpu
```

## Required Questions

1. Did the extended shear run complete?  
Yes. Seed 23 completed all 33 S0050 shear steps.

2. Was this a continuation from S0030 or a full rerun?  
It was a full rerun from step 0. `continued_from_S0030=False`.

3. What load schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0050_shear.csv`, 33 steps ending at `Delta_s=8e-5` mm. Training: `RPROP=300, LBFGS=1`, matching the prior stronger S0030 run.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. All 33 checkpoints exist and the normal postprocess reports `exact_reaction_computable=True`; `reaction_by_step.csv` and `stress_strain_by_step.csv` use the energy-conjugate metric.

5. Does the shear stress-strain curve remain monotonic or show peak/post-peak behavior?  
It shows peak/post-peak behavior. The peak nominal shear stress is 29.9647 MPa at step 24, engineering shear strain 0.006; the final stress is 28.4379 MPa.

6. Does alpha continue growing beyond the S0030 final `alpha_max=0.358412`?  
Yes. Final `alpha_max=1.00034`, above the S0030 final value.

7. Does alpha remain notch-localized?  
Yes. The final alpha maximum is near `(x,y)=(0.00508378, 0.00496023)` mm, in the explicit notch-tip region.

8. Does `alpha>=0.5` connected damage form?  
Yes. A notch-connected `alpha>=0.5` component first appears at step 22 and has final connected count 51.

9. Does `alpha>=0.8` through-crack form?  
No full alpha>=0.8 through-crack to the right boundary is detected. A notch-connected `alpha>=0.8` component first appears at step 25 and reaches final x-span 0.000384137 mm, but the through-to-right flag remains false.

10. Is HII still active and notch-localized?  
Yes. The final HII/HI peak ratio is 0.597398, and HII remains localized near the notch region.

11. Does mechanics drive remain notch-dominated?  
Yes. The final mechanics-drive maximum is at `(x,y)=(0.0050445, 0.00495711)` mm and is classified as `notch-dominated`.

12. Does top `v` remain finite, or does `top_v_absmax/Delta_s` become excessive?  
Top `v` remains finite. Final `top_v_absmax/Delta_s=1.08336` and maximum over the run is 1.08336, below the 1.5 warning threshold and below the 2.0 unstable threshold.

13. Compared with S0030, is the extension more convincing?  
Yes as a diagnostic result. Compared with S0030, alpha grows from 0.358412 to 1.00034, connected damage appears near the notch, and the energy-conjugate shear stress curve develops post-peak behavior.

14. If no through-crack or post-peak drop appears, should the next step extend loading again, adjust boundary stabilization, or diagnose the shear damage drive?  
A post-peak drop does appear, so the immediate need is not boundary stabilization. Because no full through-crack reaches the right boundary, the next minimal step should be either a slightly longer same-path shear extension or a connectivity/mesh-resolution audit of the notch-connected crack path before changing physics.

15. Was any physics changed?  
No. No physical formulas, material parameters, `l0`, TM split, history logic, alpha initialization, shear ansatz, boundary condition, or training losses were changed.

16. Was any seed study or D0040 run performed?  
No. Only seed 23 was run. D0040 was not run.

## Classification

`shear extension successful with crack growth`

Reason: the run completed with checkpointed energy-conjugate reaction at every step; alpha grew beyond S0030 and remained notch-localized; notch-connected `alpha>=0.8` damage appeared; the shear stress-strain curve showed post-peak behavior; top v stayed finite. A full alpha>=0.8 right-boundary through-crack was not detected, so this remains a diagnostic result rather than physical validation.

## Not Claimed

This package does not claim physical validation. It is a single-seed controlled diagnostic of the existing shear path.
