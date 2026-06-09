# Actual saved-PINN reaction postprocessing audit

## Scope

This package postprocesses existing saved PINN field outputs for D0040 seeds 7/13/42 and D0020 seeds 7/13/21/42/99. It does not extend loading, retrain models, evolve alpha, or modify `l0`, materials, TM split, history logic, alpha initialization, or losses.

## Checkpoint availability

- Target runs processed: 8/8.
- Runs with exact autograd `dPi/dDelta` availability: 0/8.
- D0040 runs with exact autograd availability: 0/3.
- All target result directories contain saved fields and diagnostics CSVs but no `.pt`, `.pth`, `checkpoint`, `ckpt`, or `model_settings.txt` files. Exact actual-PINN autograd reconstruction is therefore not possible from the saved artifacts.

## D0040 metric drop summary

| case | metric | peak abs reaction [N] | final abs reaction [N] | post-peak drop [%] |
|---|---|---:|---:|---:|
| D0040_seed7_default_unitbox | legacy-top | 0.968423 | 0.932704 | 3.69 |
| D0040_seed7_default_unitbox | energy-fd-proxy | 1.33384 | 0.0925181 | 93.1 |
| D0040_seed7_default_unitbox | virtual-proxy | 346.662 | 346.662 | 0 |
| D0040_seed13_default_unitbox | legacy-top | 1.01312 | 0.942999 | 6.92 |
| D0040_seed13_default_unitbox | energy-fd-proxy | 1.33177 | 0.0199706 | 98.5 |
| D0040_seed13_default_unitbox | virtual-proxy | 332.397 | 332.397 | 0 |
| D0040_seed42_default_unitbox | legacy-top | 0.98987 | 0.919255 | 7.13 |
| D0040_seed42_default_unitbox | energy-fd-proxy | 1.33634 | 0.0379681 | 97.2 |
| D0040_seed42_default_unitbox | virtual-proxy | 343.542 | 343.542 | 0 |

## Answers

1. None of the inspected saved runs has enough checkpoint/model information for exact actual-PINN `dPi/dDelta`.
2. Exact actual-PINN energy-conjugate reaction cannot be compared with legacy top sigma reaction before through-crack formation.
3. Exact actual-PINN energy-conjugate reaction cannot be tested after through-crack formation.
4. Saved-field energy finite-difference proxies can show different post-peak behavior from legacy top sigma, but they are not exact actual-PINN autograd reactions.
5. Bottom reaction and internal cut force are reported in `tables/pinn_reaction_boundary_cut_consistency.csv`; they should be treated as consistency diagnostics, not exact generalized loads.
6. The previous no-softening conclusion is not resolved by exact reaction metrics because exact metrics are unavailable.
7. Future stress-strain curves should not rely on `reaction_N_tm_eff` alone, but exact energy-conjugate or constrained-DOF reaction requires future runs to save checkpoints/model settings.
8. No production mechanics change is justified from this postprocessing package.
9. Next minimal intervention: add checkpoint/model-settings saving and exact reaction postprocessing hooks to future runs, or rerun a short D0040 checkpointed replay for exact `dPi/dDelta` without changing physics.

## Classification

**reaction postprocessing unresolved: exact actual-PINN dPi/dDelta unavailable because checkpoints are absent**.

## Limitations

- `saved_field_energy_fd_proxy_N` is a finite difference of saved optimized branch energies over the discrete load schedule. It is not an autograd derivative at fixed network state.
- `saved_field_virtual_work_proxy_N` uses saved effective stress and saved strain scaled by Delta. It is not an exact top-mode virtual work unless the unknown PINN top-mode derivative equals the saved displacement scaling.
- Proxy results must not be used to justify production physics changes.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q`: passed, 18 tests in 4.09 s.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260616_default_unitbox_pinn_energy_reaction_postprocess\artifacts\run_pinn_energy_reaction_postprocess.py`: passed.
