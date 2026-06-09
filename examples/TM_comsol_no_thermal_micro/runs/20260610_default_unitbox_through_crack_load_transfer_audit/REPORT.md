# Through-crack load-transfer audit

## Scope

This package diagnoses why the default-alpha `unit_box` route keeps transmitting load after a through-going high-alpha crack has formed. It uses existing D0040 fields only; no new training or extended loading was run.

## Through-crack onset

| seed | alpha threshold | first through step | reaction at onset [N] | reaction drop after onset [%] |
|---:|---:|---:|---:|---:|
| 7 | 0.5 | 13 | 0.71557 | -30.3 |
| 7 | 0.8 | 14 | 0.707263 | -31.9 |
| 7 | 0.95 | 18 | 0.68083 | -37 |
| 13 | 0.5 | 12 | 0.777396 | -21.3 |
| 13 | 0.8 | 14 | 0.781371 | -20.7 |
| 13 | 0.95 | 18 | 0.775515 | -21.6 |
| 42 | 0.5 | 12 | 0.727392 | -26.4 |
| 42 | 0.8 | 14 | 0.753032 | -22.1 |
| 42 | 0.95 | 17 | 0.764111 | -20.3 |

## Reaction decomposition at final step

| seed | effective reaction | undegraded total | positive undegraded | degraded positive | negative/non-degraded | negative/effective |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.932704 | 1.41829 | 2.61362 | 2.12803 | -1.19533 | -1.28 |
| 13 | 0.942999 | 1.41332 | 2.53626 | 2.06593 | -1.12294 | -1.19 |
| 42 | 0.919255 | 1.38007 | 2.47122 | 2.01041 | -1.09115 | -1.19 |

## Answers

1. Through-crack first forms at the steps listed above for alpha thresholds 0.5, 0.8, and 0.95.
2. Reaction at through-crack onset is listed in `tables/through_crack_geometry_audit.csv` and summarized above.
3. Reaction does not strongly collapse after through-crack onset; the previous softening gate reported only sub-10% final post-peak drops.
4. Effective traction remains inside the high-alpha crack band; final cut-line mean |sigma_yy_tm_eff| averaged across cases/cuts is 37.3096.
5. The final top-boundary reaction is a net result of positive and negative split contributions. The crack-section audit is more diagnostic for the through-crack load path: inside the alpha>=0.8 band, the effective traction is dominated by the non-degraded negative/compressive component.
6. Alpha degradation enters mechanics training through the variational energy loss, not only postprocessing. The path is `train_mixed_tm.py -> compute_mixed_tm_energy -> compute_mixed_tm_fields`, with `history_elastic_energy_density = g_alpha * He_trial + psi_minus`.
7. Stress split sanity audit flags opening tensile stress misclassification as False; the primary evidence points to non-degraded branch/continuous displacement bridging rather than a simple missing sigma_plus degradation.
8. Cause status: **through-crack load-transfer cause identified**. Identified evidence: high-alpha crack band still transmits effective traction dominated by non-degraded negative/compressive stress; positive tensile stress is degraded correctly in the crack band and residual stiffness is negligible. Mean |minus|/|effective| inside final alpha>=0.8 cut bands is 1; mean |degraded positive|/|effective| is 0.000484; max residual-stiffness positive contribution in the cut bands is 1.08e-06.
9. Next minimal intervention: do not change physics parameters first; run a focused kinematic/weak-form audit that compares a discontinuous or crack-face separated displacement representation against the current continuous PINN field on the same saved alpha field, and quantify reaction collapse when the cracked band is allowed to separate.

## Interpretation

The audit does not support `reaction integration uses undegraded stress` as the dominant cause: `reaction_N_tm_eff` integrates `sigma_yy_tm_eff`. It also does not support `alpha degradation is only postprocessing`: degradation enters the training energy. It also does not show tensile opening stress being excluded from `sigma_plus`. The stronger finding is that the high-alpha band still carries a non-degraded negative/compressive stress component, while the continuous PINN displacement field provides no displacement discontinuity that would remove that contact-like load path.

No physical validation is claimed.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed in 1.65 s.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260610_default_unitbox_through_crack_load_transfer_audit\artifacts\build_through_crack_load_transfer_audit.py`: passed.
