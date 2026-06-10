# Existing-Geometry Shear Stronger Training Report

## Scope

This package reruns the existing-geometry shear case with a fuller S0030 schedule and stronger training before drawing conclusions about shear drive localization. The shear ansatz, top-v-free boundary condition, physical formulas, material parameters, `l0`, TM split, history logic, and default alpha initialization were not changed.

## Run Setup

Training command:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0030_shear.csv --run-suffix seed23_S0030_shear
```

Postprocess command:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0030_shear --result-dir outputs\results\seed23_S0030_shear --device cpu
```

The schedule has 21 monotonic steps from `1e-6` to `5e-5` mm. Training was materially stronger than smoke: `RPROP=300, LBFGS=1` versus smoke `RPROP=20, LBFGS=0`.

## Required Questions

1. Did the stronger shear run complete?  
Yes. Seed 23 completed all 21 S0030 shear steps.

2. What training settings were used, and how do they differ from the smoke?  
The stronger run used `RPROP=300, LBFGS=1`; the smoke used `RPROP=20, LBFGS=0`. The schedule also increased from 5 steps ending at `2e-5` mm to 21 steps ending at `5e-5` mm.

3. Did checkpointed energy-conjugate shear reaction compute at all steps?  
Yes. `reaction_metric_availability.csv` reports `checkpoint_count=21` and `exact_reaction_computable=True`.

4. Does the shear stress-strain curve remain monotonic, or show peak/post-peak behavior?  
It remains monotonic. The peak stress is the final stress, `27.285604` MPa at step 20.

5. Does alpha grow materially beyond the smoke result?  
Yes. Final alpha max increases from smoke `0.014210` to stronger `0.358412`.

6. Does alpha initiate or concentrate near the existing notch region?  
Yes. Final alpha max is located near `(x,y)=(5.044497e-03, 4.957105e-03)` mm, at the explicit notch tip region.

7. Does alpha>=0.8 through-crack form?  
No. `alpha0p8_through_crack=False` for all 21 steps.

8. Is HII active, and how does HII/HI evolve with load?  
Yes. HII is active at all steps. The HII/HI peak ratio stays near 0.63 and is `0.632` at the final step.

9. Does mechanics drive become notch-path dominated after stronger training/load, or remain boundary/corner dominated?  
It becomes notch-dominated in this run. The final global mechanics-drive maximum is at `(x,y)=(5.021085e-03, 4.998644e-03)` mm, which is the notch-tip region.

10. Is top `v` still finite and non-runaway?  
Yes. Final top `v` mean/min/max are `1.220332e-05`, `-2.455418e-05`, and `5.065076e-05` mm. Final `top_v_absmax/Delta_s=1.013`; this is finite but high enough to monitor in later runs.

11. Compared with the smoke test, is the stronger result qualitatively more convincing?  
Yes, as a diagnostic result. The stronger run moves alpha and mechanics-drive localization to the notch tip and grows alpha materially. It still does not show post-peak softening or through-crack formation.

12. If still not convincing, is the next step likely boundary/ansatz diagnosis, load schedule extension, or mixed-drive diagnosis?  
Because localization improved without changing the ansatz or physics, the next minimal step should not immediately change the boundary or mixed-drive model. A controlled schedule/training extension or checkpointed continuation is the most direct next test; top-v drift should be monitored.

13. Was any physics changed?  
No. No physical formulas, material parameters, `l0`, TM split, history update, alpha initialization, shear ansatz, or boundary condition were changed.

14. Was any seed study or D0040 run performed?  
No. This package uses only seed 23 and does not run D0040.

## Classification

`stronger shear run qualitatively improved`

Reason: stronger training/load moves alpha and mechanics-drive localization to the explicit notch tip, while energy reaction and top-v-free diagnostics remain computable and finite; the curve is still monotonic and no through-crack forms, so this is not physical validation.

## Not Claimed

This is not physical validation. It is a single-seed diagnostic showing that the previous smoke result was training/schedule-limited and should not be used alone to reject the shear path.
