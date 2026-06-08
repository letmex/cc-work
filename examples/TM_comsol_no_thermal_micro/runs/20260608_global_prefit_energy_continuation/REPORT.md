# Global Prefit Energy Continuation Diagnostic

## Purpose

This diagnostic checks whether the current PINN displacement ansatz can represent an FE-DOF-like alpha=0 mechanics solution under global supervision, and whether that branch is preserved when the optimization objective transitions to alpha-zero mechanics energy.

The run deliberately excludes geometry-specific training guidance. The loss functions do not contain notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or explicit geometry labels. Notch-tip, bulk, and bottom-right quantities are used only as postprocessing metrics.

## Setup

- Target field: FE-DOF alpha=0 mechanics baseline from `20260608_mechanics_only_notch_ansatz`.
- Preferred target: `fedof_free_log10_energy_e300_fields.npz`.
- Delta: `1e-6`.
- Top boundary mode: `top-u free`.
- Alpha treatment: alpha fixed to zero.
- Network: current PINN displacement ansatz, 8 hidden layers, 400 neurons, `TrainableReLU`.
- Physical settings retained from the project defaults: `l0=0.00015`, `tm_source split`, same material parameters, same mesh, same thermal-mechanical source split.
- No coupled phase-field full run was performed.

## Training Modes

The diagnostic compared:

1. `random_init_energy`: random initialization followed by mechanics energy optimization.
2. `disp_global_prefit_then_energy`: global nodal displacement prefit, then mechanics energy optimization.
3. `disp_strain_global_prefit_then_energy`: global nodal displacement plus global element strain prefit, then mechanics energy optimization.
4. `global_curriculum`: global displacement/strain prefit loss ramped to mechanics energy loss with the weight moving from 0 to 1.

The global prefit losses were:

```text
L_disp = mean((u_pinn - u_target)^2 + (v_pinn - v_target)^2)
```

```text
L_prefit = L_disp + strain_weight * L_strain
```

where `L_strain` is computed over all elements, not over a notch-local subset.

## Main Results

The global displacement prefit reached:

- displacement relative RMSE: `0.006716`
- strain relative RMSE: `0.192409`
- `u_corr`: `0.999943`
- `v_corr`: `0.999901`
- `strain_corr`: `0.981630`
- `He_current_corr`: `0.926113`
- classification: `target-like`

The global displacement plus global strain prefit reached:

- displacement relative RMSE: `0.004930`
- strain relative RMSE: `0.076669`
- `u_corr`: `0.999987`
- `v_corr`: `0.999933`
- `strain_corr`: `0.997029`
- `He_current_corr`: `0.984394`
- classification: `target-like`

After energy-only continuation, target agreement was not preserved:

- `disp_global_prefit_then_energy` after energy continuation:
  - displacement relative RMSE: `0.999712`
  - strain relative RMSE: `1.000038`
  - `He_current_corr`: `-0.152065`
  - classification: `notch-amplified`
  - This case retained a weak notch-amplified postprocessing ratio but no longer matched the FE-DOF target field.

- `disp_strain_global_prefit_then_energy` after energy continuation:
  - displacement relative RMSE: `0.999688`
  - strain relative RMSE: `1.000011`
  - `He_current_corr`: `-0.173204`
  - classification: `broad/background`

- `global_curriculum` after ramping fully to energy:
  - displacement relative RMSE: `0.999661`
  - strain relative RMSE: `1.000003`
  - `He_current_corr`: `-0.009037`
  - classification: `boundary-dominated`

The random-initialized energy-only case also stayed non-target-like:

- displacement relative RMSE: `0.999668`
- strain relative RMSE: `1.000006`
- `He_current_corr`: `-0.072716`
- classification: `broad/background`

## Answers to Required Questions

1. Can global prefit fit the FE-DOF mechanics field without geometry injection?

Yes, within this diagnostic. Both global displacement prefit and global displacement plus global strain prefit reproduced the FE-DOF target closely. The stronger displacement+strain prefit achieved `He_current_corr=0.984394` and `strain_corr=0.997029`.

2. Does energy continuation preserve the localized or notch-amplified branch?

Not reliably in this run. The `disp_global` continuation kept a postprocessing classification of `notch-amplified`, but target-field agreement collapsed. The `disp_strain_global` continuation moved to `broad/background`, and the simple global curriculum ended as `boundary-dominated`.

3. If it fails, is this a prefit expression failure or an energy-continuation failure?

The evidence points to energy-continuation failure rather than global ansatz expression failure. The PINN can fit the FE-DOF-like branch under global supervision, but the energy objective/optimizer path moves away from that branch when the global prefit anchor is removed or ramped to zero.

4. Is a mechanics curriculum needed?

A more careful mechanics curriculum is likely needed if the goal is to keep the FE-DOF-like branch while reducing reliance on supervision. The simple ramp used here, where the prefit weight goes to zero and the energy weight goes to one, did not preserve the target-like branch.

5. Is more general network expression enhancement needed?

It is not the first supported conclusion from this run, because the current ansatz did fit the global FE-DOF target under supervision. Broader expression tests may still be useful later, but this package does not show that the existing ansatz is unable to represent the target branch.

6. What conclusions cannot be drawn?

This package cannot claim physical validation, coupled phase-field success, mesh-independent localization, or robustness across seeds. It also does not prove that any particular postprocessing classification is physically correct, because the run is mechanics-only and uses one seed.

## Interpretation

The current training mode is mechanics-only alpha=0. Under global FE-DOF supervision, the PINN displacement ansatz can represent a target-like mechanics field without local notch/lip guidance. However, once the objective becomes pure mechanics energy, the optimizer moves toward low-energy broad/background or boundary-dominated fields. This suggests the next diagnostic should focus on global, non-geometry-specific energy scaling or optimization path control, such as maintaining a global prefit/proximal anchor, testing trust-region continuation, or checking energy normalization and optimizer scheduling.

Any next step should keep the constraint that notch/lip region information is not inserted into the training objective unless the user explicitly requests that experiment.

