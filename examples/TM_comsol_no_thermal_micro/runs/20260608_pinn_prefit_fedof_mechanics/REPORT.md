# PINN Prefit to FE-DOF Mechanics Diagnostic Report
## 1. Question
This diagnostic asks whether the current PINN displacement ansatz can represent the FE-DOF alpha-zero localized mechanics solution, or whether the previous mechanics-only energy optimization failed because it could not find that solution branch. The diagnostic trains only the PINN displacement outputs against FE-DOF `u/v` targets and keeps alpha fixed to zero.
## 2. Method
The FE-DOF targets are the alpha-zero mechanics baselines from the previous package at `Delta = 1e-6`, using the same mesh, material constants, `l0 = 1.5e-4 mm`, and `tm_source` split. Both top-u free and top-u fixed targets are included, with top-u free treated as the priority case.
Three supervised prefit variants were run for 1000 RPROP epochs with the same `8 x 400 TrainableReLU`, seed 2 PINN ansatz:
- `disp_only`: global node-wise displacement MSE only.
- `disp_lip`: displacement MSE plus notch-lip jump weighted loss.
- `disp_strain`: displacement MSE plus element strain MSE.

After each prefit, predicted fields were passed through the existing alpha-zero TM mechanics recomputation path to obtain predicted `eps_xx`, `eps_yy`, `eps_xy`, `psiI`, `psiII`, and `He_current`.
## 3. Main Results
| case | disp rel RMSE | lip v pred/target | strain rel RMSE | He corr | pred bulk/notch | pred bottom/notch | classification |
|---|---:|---:|---:|---:|---:|---:|---|
| prefit_free_disp_only_e1000 | 0.00671628 | 1.00745 | 0.192409 | 0.926113 | 0.119223 | 0.0139126 | notch-amplified |
| prefit_free_disp_lip_e1000 | 0.841803 | 1.01353 | 20.9836 | 0.00461491 | 0.0437286 | 3.891166e-05 | notch-amplified with boundary max |
| prefit_free_disp_strain_e1000 | 0.00492976 | 0.99206 | 0.0766687 | 0.984394 | 0.184108 | 0.0229406 | notch-amplified with boundary max |
| prefit_fixed_disp_only_e1000 | 0.00544183 | 1.02111 | 0.173948 | 0.95138 | 0.144719 | 0.0188067 | notch-amplified |
| prefit_fixed_disp_lip_e1000 | 0.108557 | 0.995575 | 2.46642 | 0.0348253 | 0.0108549 | 1.435549e-04 | notch-amplified with boundary max |
| prefit_fixed_disp_strain_e1000 | 0.00563022 | 0.99336 | 0.0783211 | 0.987427 | 0.203295 | 0.0263692 | notch-amplified with boundary max |

## 4. Answers to Required Questions
### 4.1 Can PINN supervised-fit FE-DOF u/v?
Yes for the tested target. In the priority top-u-free `disp_only` run, displacement relative RMSE is `0.00671628`, with `u_corr = 0.999943` and `v_corr = 0.999901`. The fixed case is similar: relative RMSE `0.00544183`.
### 4.2 Is global displacement MSE small enough?
For `disp_only`, yes. Top-u-free final displacement MSE is `5.661882e-11` and relative RMSE is below 1%. Adding the strain term further reduces the top-u-free relative RMSE to `0.00492976`.
### 4.3 Can notch-lip jump be fitted?
Yes. In top-u-free `disp_only`, predicted/target notch-lip `v` jump ratio is `1.00745` and `u` jump ratio is `0.999594`. The `disp_strain` variant also keeps ratios near one. The `disp_lip` variant forces lip jump strongly but worsens global/strain consistency.
### 4.4 Does fitted He_current move from broad/background to notch-amplified?
Yes for the successful supervised fits. Top-u-free `disp_only` gives predicted `bulk/notch He = 0.119223` and `bottom/notch He = 0.0139126`, classified as `notch-amplified`. Target ratios are `bulk/notch = 0.234402` and `bottom/notch = 0.0286184`.
### 4.5 Is the issue ansatz expressivity or energy optimization?
This evidence points more toward the energy optimization path not finding the FE-DOF-like localized mechanics branch under the current energy-only training, rather than a hard expressivity limit of the current network ansatz. The ansatz can supervised-fit FE-DOF `u/v`, notch-lip jumps, strains, and notch-amplified `He_current` when given the target field.
### 4.6 Is notch-lip enrichment still needed?
A permanent enrichment is not yet justified as the first fix, because the current ansatz can represent the FE-DOF-like field under supervision. A smaller next diagnostic is mechanics pretraining/curriculum or a loss-localization strategy that guides the energy optimization toward the localized mechanics branch. Notch-lip enrichment remains a useful fallback if pretraining cannot be integrated stably.
### 4.7 What cannot be concluded?
- This does not validate a physical crack path.
- This does not prove the FE-DOF field is the correct physical solution.
- This does not justify changing `l0`, material parameters, TM split, phase-field notch behavior, alpha seeding, or history update logic.
- This does not prove a coupled phase-field full run will succeed after pretraining; that remains a later controlled test.
## 5. Figure Interpretation
The figure set compares target FE-DOF and prefit PINN `log10(He_current)` for the top-u-free target. `disp_only` and `disp_strain` visually reconstruct the notch-amplified FE-DOF pattern. `disp_lip` enforces the lip jump but produces over-concentrated high drive and poorer global reconstruction, consistent with its low `He_current_corr`.
## 6. Verification
Verification passed:
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q` -> `13 passed in 0.07s`.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_pinn_prefit_fedof_mechanics.py debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py` -> passed.

GitHub CLI status: bare `gh` is not in PATH. `C:\Program Files\GitHub CLI\gh.exe` exists and reports version `2.93.0`, but `gh auth status` reports no logged-in hosts and no `GH_TOKEN`/`GITHUB_TOKEN` environment variable was present. This package therefore uses markdown-only handoff.
