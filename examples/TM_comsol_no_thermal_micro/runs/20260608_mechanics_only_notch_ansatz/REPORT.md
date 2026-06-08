# Mechanics-Only Notch Ansatz Diagnostic Report
## 1. Question
This package tests whether the current PINN displacement/strain ansatz can express the localized strain field expected around the narrow explicit notch. The diagnostic freezes alpha to zero and compares PINN mechanics against an independent nodal FE-DOF mechanics baseline on the same mesh, load step, material constants, `l0`, and `tm_source` split.
Top-u-free has already been checked in the previous package and is not treated here as the main cause. It is still included as a controlled boundary-condition variant.
## 2. Diagnostic Design
- Load step: `Delta = 1e-6`.
- Alpha: fixed to zero for all nodes; fracture energy remains zero.
- History fields: `HI = 0`, `HII = 0`.
- PINN: current global displacement ansatz, `8` hidden layers, `400` neurons, `TrainableReLU`, seed `2`.
- PINN optimizer budgets: `0`, `100`, and `300` RPROP epochs.
- Loss scaling: `log10_energy` and `raw_energy`.
- FE-DOF: independent nodal displacement DOFs, `300` RPROP epochs.
- Boundary variants: `top_u_mode = fixed` and `top_u_mode = free`.

The main comparison metrics are notch-tip `He_current`, bulk/notch and bottom-right/notch ratios, notch-lip displacement jumps, strain norm localization, and max-drive location.
## 3. Representative Results
| case | method | top-u | epochs | notch He | bulk/notch He | bottom/notch He | notch lip v jump max | classification |
|---|---|---|---:|---:|---:|---:|---:|---|
| pinn_fixed_log10_energy_e0 | PINN | fixed | 0 | 7.127430e-07 | 0.999821 | 0.999449 | 1.000285e-08 | broad/background |
| pinn_fixed_log10_energy_e300 | PINN | fixed | 300 | 6.753918e-07 | 1.01504 | 1.68897 | 1.055679e-08 | boundary/background dominated |
| fedof_fixed_log10_energy_e300 | FE_DOF | fixed | 300 | 31.2768 | 0.260135 | 0.0337788 | 4.115091e-04 | notch-amplified with boundary max |
| pinn_free_log10_energy_e0 | PINN | free | 0 | 7.131488e-07 | 0.999621 | 0.998886 | 1.000285e-08 | broad/background |
| pinn_free_log10_energy_e300 | PINN | free | 300 | 8.579536e-07 | 0.787217 | 1.41868 | 1.326197e-08 | broad/background |
| fedof_free_log10_energy_e300 | FE_DOF | free | 300 | 34.6503 | 0.234402 | 0.0286184 | 4.299072e-04 | notch-amplified with boundary max |

## 4. PINN vs FE-DOF Key Ratios
| top-u | FE-DOF / PINN notch He | PINN bulk/notch He | FE-DOF bulk/notch He | PINN bottom/notch He | FE-DOF bottom/notch He | FE-DOF / PINN notch-lip v jump |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 4.630918e+07 | 1.01504 | 0.260135 | 1.68897 | 0.0337788 | 3.898051e+04 |
| free | 4.038712e+07 | 0.787217 | 0.234402 | 1.41868 | 0.0286184 | 3.241655e+04 |

## 5. Interpretation
The PINN mechanics-only runs remain broad/background or boundary/background over the tested optimizer budgets and both loss scalings. At `300` epochs with `log10_energy`, PINN notch-tip `He_current` is about `6.75e-7` for top-u fixed and `8.58e-7` for top-u free. The corresponding bulk/notch ratios are near order one, and bottom-right/notch ratios remain greater than one.
The FE-DOF baseline, using the same mesh and alpha-zero mechanics energy, can create much larger notch-region drive: about `31.28` for top-u fixed and `34.65` for top-u free. Bulk/notch ratios drop to about `0.26` and `0.23`, and bottom-right/notch ratios drop to about `0.034` and `0.029`. FE-DOF still has boundary/corner maxima, so the result should be read as notch-amplified rather than purely notch-dominated.
The notch-lip displacement jump difference is also large. For representative `log10_energy` 300-epoch runs, FE-DOF notch-lip v jump is about `3.2e4` to `3.9e4` times larger than PINN. This supports the interpretation that the same mesh can represent a localized notch-lip mechanics response, but the current PINN ansatz/optimization path does not express it under this alpha-zero mechanics-only setup.
Raw-energy loss scaling does not remove the PINN broad/background behavior in this diagnostic. The broad drive is therefore not explained solely by using `log10(E)` instead of raw energy.
## 6. What This Does Not Prove
- It does not prove a final model fix.
- It does not validate any physical crack path.
- It does not justify changing `l0`, material parameters, `tm_source` split, phase-field notch behavior, alpha seeding, or history update logic.
- It does not distinguish whether the PINN issue is primarily ansatz expressivity, initialization, optimizer path, or loss weighting in the mechanics subproblem.
## 7. Recommended Next Diagnostic
A minimal next step is to test a diagnostic-only displacement representation near the notch: either a local notch-lip enrichment, independent local nodal DOFs around the explicit notch blended into the PINN field, or a mechanics prefit target from the FE-DOF displacement field. This should remain a mechanics-only diagnostic before any physical model change.
## 8. Verification
Verification passed:
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q` -> `13 passed in 0.06s`.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py` -> passed.

GitHub CLI status: bare `gh` is not in PATH. `C:\Program Files\GitHub CLI\gh.exe` exists and reports version `2.93.0`, but `gh auth status` reports no logged-in hosts and no `GH_TOKEN`/`GITHUB_TOKEN` environment variable was present. This package therefore uses markdown-only handoff.
