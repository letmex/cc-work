# Global Anchor Energy Continuation Diagnostic

## Purpose

The previous package showed that global FE-DOF prefit can fit an FE-DOF-like alpha=0 mechanics branch, but pure mechanics-energy continuation moves away from that branch. This package tests whether global, non-geometry-specific continuation controls can preserve the branch.

The training losses do not use notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or geometry-label guidance. Notch-tip, bulk, and bottom-right metrics are used only after training for diagnosis.

## Setup

- Target: FE-DOF alpha=0 mechanics field, top-u-free version.
- Delta: `1e-6`.
- Network: 8 hidden layers, 400 neurons, `TrainableReLU`.
- Prefit: `disp_global` and `disp_strain_global`, each 1000 epochs.
- Continuation: 300 epochs for anchor and energy-normalization cases.
- Trust-region continuation: 10 chunks x 30 epochs.
- Energy scale: raw prefit mechanics energy of the `disp_strain_global` prefit state unless otherwise noted.

Diagnostic success thresholds:

- `He_current_corr >= 0.8`
- `strain_corr >= 0.9`
- `bulk/notch He <= 0.35`
- `bottom/notch He <= 0.1`
- `displacement relative RMSE < 0.1`

These thresholds are diagnostic only and are not physical validation.

## Prefit Baseline

Both global prefit states satisfy the diagnostic thresholds before energy continuation:

| case | displacement rel. RMSE | strain corr | He corr | bulk/notch | bottom/notch | classification |
|---|---:|---:|---:|---:|---:|---|
| `prefit_disp_global` | 0.005637 | 0.984313 | 0.893306 | 0.093042 | 0.010113 | target-like |
| `prefit_disp_strain_global` | 0.006095 | 0.984722 | 0.941575 | 0.138732 | 0.016378 | target-like |

This confirms that the current ansatz can enter a target-like FE-DOF branch under global supervision.

## Continuation Results

No continuation case satisfied all diagnostic thresholds.

The pure log10-energy baseline collapsed away from the FE-DOF branch:

- `pure_energy_baseline_log10`
- normalized mechanics energy: `1.066e-7`
- displacement relative RMSE: `0.999704`
- strain corr: `-0.016976`
- He corr: `-0.014971`
- classification: `collapsed/non-target`

The best displacement-anchor case by `He_current_corr` was:

- `global_displacement_anchor_lamU_1e-01`
- `lambda_u=0.1`
- normalized mechanics energy: `0.009815`
- displacement relative RMSE: `0.882046`
- strain corr: `0.881689`
- He corr: `0.869763`
- bulk/notch: `0.180367`
- bottom/notch: `0.003042`
- classification: `boundary-dominated`

This case preserved some He-field correlation and notch/bulk ratios, but it still failed the displacement and strain thresholds and was classified as boundary-dominated.

The best combined displacement+strain anchor by `He_current_corr` was:

- `global_disp_strain_anchor_lamU_1e-02_lamEps_1e-04`
- normalized mechanics energy: `0.0001435`
- displacement relative RMSE: `0.984947`
- strain corr: `0.702404`
- He corr: `0.808617`
- classification: `boundary-dominated`

The strongest tested strain-only anchor did not preserve the branch:

- `global_strain_anchor_lamEps_1e-03`
- displacement relative RMSE: `0.998760`
- strain corr: `0.850342`
- He corr: `0.480261`
- classification: `notch-amplified`

Trust-region continuation did not preserve the branch. For `lambda_trust_u=0.1`, `He_current_corr` dropped from `0.701447` after chunk 1 to `-0.004806` after chunk 10, while displacement relative RMSE rose from `0.912605` to `0.999646`.

Energy normalization alone did not change the branch behavior:

| case | energy mode | displacement rel. RMSE | strain corr | He corr | classification |
|---|---|---:|---:|---:|---|
| `pure_energy_baseline_log10` | log10 | 0.999704 | -0.016976 | -0.014971 | collapsed/non-target |
| `energy_normalization_raw` | raw | 0.999668 | 0.030116 | 0.108680 | notch-amplified |
| `energy_normalization_normalized` | normalized | 0.999704 | -0.018333 | -0.053193 | collapsed/non-target |

## Required Answers

1. Can any global non-geometry anchor preserve the FE-DOF-like branch during energy continuation?

Not in this sweep. The prefit states are target-like, but every continuation case fails at least one required threshold, and most fail several.

2. How weak can the anchor be before the solution collapses?

Weak anchors collapse. For displacement anchors, `lambda_u <= 1e-3` leaves displacement relative RMSE near 1.0 and strain correlation below 0.61. Stronger displacement anchors improve He correlation, but even `lambda_u=0.1` leaves displacement relative RMSE at `0.882046`, which is far above the `<0.1` threshold.

3. Does trust-region continuation help?

Not in the tested form. Chunked global trust-region penalties slow the departure early but do not preserve the branch through 10 chunks.

4. Does energy normalization change the branch behavior?

Not enough in this run. Raw, log10, and normalized energy all end far from the target-like branch without an anchor.

5. Is mechanics pretraining/curriculum justified before coupled full training?

Mechanics pretraining is justified as an initialization tool because it reaches target-like states. However, this package does not support switching fully to the current energy-only objective after pretraining. A future curriculum would need a better global continuation rule or objective treatment.

6. Or does the evidence point to mechanics energy formulation / optimizer path as the main unresolved issue?

The evidence points to the mechanics energy formulation, scaling, or optimizer path as the unresolved issue. The ansatz can fit the FE-DOF branch, but the tested global continuation controls do not keep it there.

7. What cannot be concluded?

This package cannot claim physical validation, coupled phase-field success, seed robustness, mesh-independent localization, or that local notch/lip enrichment is required. It also cannot prove that the FE-DOF target is the correct physical branch; it only shows that the current PINN energy continuation does not preserve that branch under the tested global controls.

## Suggested Next Diagnostic

Before coupled full training, inspect the alpha=0 mechanics energy objective more directly. Useful next checks would compare the energy of the FE-DOF target field, the prefit PINN field, and the collapsed PINN field under exactly the same quadrature and boundary assumptions, then trace which energy terms reward the collapsed branch. This should remain non-geometry-specific unless the user explicitly requests otherwise.

