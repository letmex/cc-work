# True Staggered Diagnostic Report

## Local AT2 alpha equilibrium

The existing local predictor

```text
alpha_eq = 2H / (2H + Gc/l0)
```

does predict the previously observed `alpha ~= 0.488` background state when the saved history/mechanics drive is used.

| case | drive | alpha_eq_mean | saved alpha_mean | mean abs diff |
|---|---|---:|---:|---:|
| history + alpha-init-intact | mechanics_drive | 0.489055 | 0.488258 | 0.001030 |
| current_split + alpha-init-intact | mechanics_drive | 0.488767 | 0.488285 | 0.000640 |
| elastic-only alpha=0 | He_current | 0.326682 | 0.0 | 0.326682 |

The elastic-only field is not uniform in the same way: its predicted `alpha_eq` is high at the notch tip (`notch_tip_alpha_eq_max = 0.985993`) and lower at bottom-right (`bottom_right_alpha_eq_max = 0.477181`). The uniform `0.488` branch appears after the trained/coupled history-drive field has become broad.

## Fixed-u/v alpha-only result

Fixed displacement/strain diagnostics separate the phase subproblem from the neural-network displacement optimization.

| case | drive | final alpha_mean | final alpha_max | notch_tip_alpha_max | bottom_right_alpha_max |
|---|---|---:|---:|---:|---:|
| elastic-only fixed u/v | He_current | 0.332044 | 0.956127 | 0.956127 | 0.482429 |
| trained history fixed u/v | mechanics_drive | 0.489052 | 0.489971 | 0.489556 | 0.488651 |

Interpretation: with the elastic-only localized drive, alpha still prefers the notch most strongly. With the trained history/mechanics drive, the alpha-only problem itself reproduces the nearly uniform `alpha ~= 0.489` state. That means the phase subproblem cannot be treated as innocent once the drive field is broad.

## FE-DOF staggered result

Added `debug_fedof_staggered_baseline.py`.

Run:

```powershell
D:\anaconda3\envs\torch_env\python.exe debug_fedof_staggered_baseline.py --max-steps 34 --mechanics-epochs 50 --phase-epochs 50 --out results\true_staggered_diagnostic\debug_fedof_staggered_D0020_seed2.csv --npz results\true_staggered_diagnostic\debug_fedof_staggered_D0020_seed2_fields.npz
```

Final step:

| quantity | value |
|---|---:|
| alpha_mean | 0.999998 |
| alpha_max | 1.000006 |
| alpha > 0.5 area fraction | 1.0 |
| max He_current location | `(0.004025, 0.004157)` |
| max He_history/mechanics_drive location | `(0.006588, 0.006837)` |
| bottom_right / notch_tip He_current | 0.0397 |
| bottom_right / notch_tip alpha | 0.99998 |
| reaction_N_tm_eff | -1500.37 |

This is not a physical validation. It shows that an independent nodal-DOF staggered discretization can also enter a global damage branch under the current AT2/history-drive setup. Therefore the root cause is not only the shared PINN representation or simultaneous u-v-alpha optimization.

## PINN staggered result

Added experimental CLI:

```text
--solve-scheme coupled|staggered
--stagger-iters N
```

Default remains `coupled`. The `staggered` branch writes `diagnostics_staggered_substeps.csv`.

Medium diagnostic run:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 4 100 2 TrainableReLU 3.0 --n-rprop 20 --n-lbfgs 0 --mixed-mechanics-mode history --alpha-init-intact --solve-scheme staggered --stagger-iters 1 --load-schedule-file load_schedule_D0020_extended.csv --run-suffix stgD0020s2
```

Final step:

| quantity | value |
|---|---:|
| alpha_mean | 0.500405 |
| alpha_min / alpha_max | 0.495638 / 0.505114 |
| n_alpha_lt_0 / n_alpha_gt_1 | 0 / 0 |
| max He_current location | `(0.009711, 0.009880)` |
| max He_history/mechanics_drive location | `(0.009711, 0.009880)` |
| bottom_right / notch_tip He_current | 0.987393 |
| bottom_right / notch_tip mechanics_drive | 0.987393 |
| bottom_right / notch_tip alpha | 1.001504 |
| reaction_N_tm_eff | -1.959883 |

The final substep diagnostics show the phase substep keeps increasing alpha globally:

| step | substep | alpha_mean | alpha_max |
|---:|---|---:|---:|
| 32 | mechanics | 0.469635 | 0.474097 |
| 32 | phase | 0.484511 | 0.489096 |
| 33 | mechanics | 0.484511 | 0.489096 |
| 33 | phase | 0.500405 | 0.505114 |

Implementation limitation: this PINN staggered route uses cached `u,v` during the phase substep and fixed previous alpha during the mechanics substep, but the network still has shared hidden layers. The phase substep can change shared parameters that affect future `u,v`; the next mechanics substep then re-corrects them. This is a diagnostic staggered PINN, not a clean three-head segregated architecture.

## Coupled vs staggered comparison

| case | training level | alpha_mean | alpha_max | max He_current location | bottom_right/notch He_current | bottom_right/notch alpha |
|---|---|---:|---:|---|---:|---:|
| coupled history + alpha-init-intact | full 8x400, RPROP 10000 | 0.488258 | 0.488434 | `(0.009962, 0.003302)` | 0.996167 | 0.999623 |
| coupled current_split + alpha-init-intact | full 8x400, RPROP 10000 | 0.488285 | 0.488571 | `(0.000049, 0.006318)` | 0.999606 | 0.999331 |
| PINN staggered history + alpha-init-intact | medium 4x100, RPROP 20 | 0.500405 | 0.505114 | `(0.009711, 0.009880)` | 0.987393 | 1.001504 |
| FE-DOF staggered | nodal DOF, 50/50 epochs | 0.999998 | 1.000006 | `(0.004025, 0.004157)` | 0.0397 | 0.99998 |

The exact locations differ, but the important common feature is broad/background damage rather than notch-tip crack localization. The medium PINN staggered run is not directly comparable to the full 8x400 runs in optimizer budget, but it is enough to show that simply separating mechanics and phase substeps does not automatically remove the background-damage branch.

## Debug recomputation consistency

For the final PINN staggered NPZ:

```text
debug_recompute_pinn_staggered_D0020_seed2.csv
max_abs_diff = 0.0 at reported precision
max_He_current = 0.007740704 at (0.009711, 0.009880)
```

The saved fields are internally consistent. The artifact is not caused by the plotting script or saved `He_current` mismatch.

Clean figures were generated in:

```text
results/true_staggered_diagnostic/pinn_staggered_clean_figures
```

## Seed stability

Only seed 2 was run for the new true-staggered diagnostics. No seed-stability claim is made. A good-looking seed would not validate the implementation; it would only show that another local branch exists.

## Interpretation

The earlier hypothesis, "fully coupled alpha-u-v Deep-Ritz optimization is the main cause", is incomplete.

Evidence now points to this chain:

1. Pure elastic alpha=0 mechanics can localize `He_current` near the notch, so pure displacement + tm_source is not by itself the root cause.
2. Once the saved history/mechanics drive becomes broad, the local AT2 balance predicts `alpha ~= 0.49`.
3. Fixed-u/v alpha-only optimization with the trained history drive reproduces the same uniform value.
4. PINN staggered still evolves to `alpha ~= 0.5`.
5. FE-DOF staggered can evolve to nearly full-domain `alpha ~= 1`.

Most likely cause: the current AT2 phase subproblem with the present broad dual-history drive admits a diffuse/global damage branch. Coupled PINN training and shared network representation may accelerate or bias branch selection, but they are not sufficient as the sole explanation.

This does not validate the physical model. It instead says the next diagnosis must focus on the phase-field equation/drive magnitude and its exact equivalence to the COMSOL weak form, not on seed selection.

## Recommended next fix

Do not change `l0`, material parameters, tm_source split, or impose a phase-field notch as the next move.

Recommended next code experiment:

1. Implement an exact FE phase subproblem solve for fixed `He_history` on the same T3 mesh, using the AT2 Euler-Lagrange linear system rather than RPROP/log-energy optimization.
2. Feed it three fixed drives: elastic-only `He_current`, coupled-history final `He_history`, and PINN-staggered final `He_history`.
3. Compare alpha localization, alpha_mean, and energy against the current RPROP alpha-only result.

If the exact FE phase solve also gives uniform/background damage, the issue is in drive magnitude/coefficient/source-model equivalence. If it localizes while RPROP/log-energy does not, the optimizer/loss scaling is a primary cause.
