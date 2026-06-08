# Exact FE Prefit Strain-Resolution Diagnostic

## Scope

This diagnostic continues the accepted exact-FE mechanics-target line. It uses only:

`examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz`

The old FE-DOF RPROP target is included only as a rejected negative control in `tables/target_guard_check_summary.csv`.

No coupled phase-field training was run. `alpha` was fixed to zero for all mechanics prefit and continuation checks. The diagnostic did not add notch/lip loss, notch-tip masks, local weights, displacement-jump targets, enrichment, geometry-label features, or model-physics changes.

## Target Guard

`tables/target_guard_check_summary.csv` explicitly marks:

- `accepted_direct_sparse_FE_topufree_alpha0_Delta1e-6`: accepted.
- `rejected_old_FE_DOF_RPROP_free_log10_e300`: rejected.

The accepted exact FE target has zero self-error in the guard check. The old FE-DOF RPROP target is rejected by displacement error, strain error, energy ratio, reaction sign, and free residual.

## Implementation Consistency Checks

`tables/coordinate_and_connectivity_checks.csv` shows:

- Recomputing target strain from the stored target displacement using the same T3 formula matches stored `eps_xx`, `eps_yy`, and `eps_xy` to machine precision.
- The reused baseline artifact has the same node coordinates and element connectivity/order as the exact target.
- Current `FieldComputation` passes physical millimeter coordinates directly to `NeuralNet`; it does not normalize `x,y` before the MLP.

This rules out element order mismatch and target T3 strain reconstruction mismatch as the primary cause.

## Prefit Variant Results

Main comparison table: `tables/prefit_variant_comparison.csv`.

Key rows:

| case | disp rel RMSE | strain rel RMSE | strain corr | He corr | std energy ratio | PINN energy ratio | classification |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_global_disp | 0.01977 | 1.15907 | 0.63474 | 0.07393 | 2.04562 | 2.09252 | displacement-only-good / strain-bad |
| normalized_global_disp | 0.01977 | 1.15907 | 0.63474 | 0.07393 | 2.04562 | 2.09252 | displacement-only-good / strain-bad |
| normalized_global_strain_w10 | 0.47004 | 0.56860 | 0.83142 | 0.22070 | 2.33246 | 2.32349 | boundary-dominated |
| capacity_10x500_w1 | 0.44011 | 0.59301 | 0.81745 | 0.17751 | 2.22125 | 2.20782 | boundary-dominated |
| activation_TrainableTanh_w1 | 0.44242 | 0.58146 | 0.82632 | 0.22378 | 2.24075 | 2.23735 | boundary-dominated |
| coordinate_scaled_network_w1 | 0.000621 | 0.009459 | 0.999952 | 0.999977 | 1.000081 | 1.000156 | exact-target-like |

The successful case uses the same 8x400 TrainableReLU width/depth as the baseline, but wraps the network input as a generic coordinate normalization to roughly `[-1, 1]`. This is not a notch-specific feature and does not use any geometry-region label in the training loss.

## Strain and He Error Audit

For the baseline displacement-only artifact:

- nodal displacement error is small: mean `5.24e-9`, 95th percentile `1.18e-8`.
- element strain error remains large: mean `3.57e-5`, 95th percentile `8.89e-5`, max `2.03e-3`.
- strain error correlates with boundary distance (`0.427`) and has larger means in small/high-aspect elements.

For `coordinate_scaled_network_w1`:

- nodal displacement error mean drops to `3.35e-10`.
- element strain error mean drops to `6.27e-7`.
- strain correlation reaches `0.999952`.
- `He_current` correlation reaches `0.999977`.
- max `He_current` location matches the exact FE notch location: `(0.0050210844, 0.0049986435)`.

This indicates the earlier mismatch is not caused by a different derivative formula. It is best explained by coordinate-scale conditioning in the MLP, with derivative amplification turning small global displacement errors into large element strain and drive errors.

## Coordinate Normalization and Capacity

Normalized global displacement alone is equivalent to a positive constant loss rescaling under RPROP sign updates, so it reproduces the baseline artifact. Normalized global strain loss without input scaling improves some strain correlations at high weights but does not reach exact-target-like behavior and tends toward boundary/background branches.

Increasing the network to 10x500 without coordinate scaling does not solve the problem. Switching to TrainableTanh without coordinate scaling also does not solve it. Therefore current 8x400 capacity appears sufficient for the exact FE alpha=0 mechanics target once coordinate input scaling is fixed.

## Short Energy Continuation

Because `coordinate_scaled_network_w1` passes the suggested strain/He thresholds, a short mechanics-only energy continuation was run from that prefit for raw, log10, and normalized energy losses.

All three continuation variants keep the max `He_current` at the exact notch location and retain high `He_current` correlation:

| mode | disp rel RMSE | strain rel RMSE | strain corr | He corr | std energy ratio | PINN energy ratio | reaction ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.07698 | 0.27820 | 0.96284 | 0.99360 | 1.12372 | 0.97869 | 0.76505 |
| log10 | 0.07739 | 0.27632 | 0.96324 | 0.99364 | 1.12349 | 0.97871 | 0.75945 |
| normalized | 0.07556 | 0.27369 | 0.96359 | 0.99373 | 1.11860 | 0.97884 | 0.76847 |

These continuations are not exact-target-like after 100 energy steps because strain relative RMSE rises above `0.2` and reaction drops to about `0.76` of exact FE. They are also not boundary-dominated by max location; the main drive remains notch-located. This is a partial stability result, not a coupled phase-field conclusion.

## Answers

1. The strain/He mismatch is not caused by element connectivity/order or an implementation mismatch in T3 strain recomputation. The strongest diagnosed cause is coordinate-scale conditioning of the current PINN input, with derivative amplification in the unscaled network. Element quality modulates error but is not the primary root cause.
2. `coordinate_scaled_network_w1` is the best non-geometry-specific prefit variant. It passes all suggested thresholds.
3. Coordinate normalization helps decisively. Loss normalization alone does not help under RPROP. Global strain-loss normalization helps only partially without coordinate scaling.
4. Current 8x400 capacity is sufficient once coordinates are normalized. Larger 10x500 capacity without normalization does not fix the mismatch.
5. The improved prefit is partially stable under short energy continuation: `He_current` remains notch-located and highly correlated, but displacement/strain/reaction drift enough that the field is no longer exact-target-like.
6. Before coupled phase-field, the next controlled step should test coordinate normalization as a first-class mechanics ansatz option, then rerun exact-target prefit plus short and medium mechanics-only energy continuation.
7. This package cannot conclude physical crack-path validity, coupled phase-field behavior, or experimental validation. It also cannot justify changing material parameters, `l0`, TM split, thermal terms, history update logic, or phase-field notch behavior.

## Verification

Commands are listed in `commands_run.txt`.

- `py_compile` on the new diagnostic script passed.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q` passed in `D:\ProgramData\PINN\FEM-PINN-main`: `13 passed in 0.07s`.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q` failed during collection because `ref_files.Chinese_SENT_reproduction` is missing in six Chinese_SENT test modules.
