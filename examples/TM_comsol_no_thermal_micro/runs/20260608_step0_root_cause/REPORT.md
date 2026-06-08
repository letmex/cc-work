# Step-0 Root-Cause Diagnostic Report

## 1. Question

The previous full top-u-free comparison showed that top-u-free does not remove the alpha-init history branch:

- broad/background drive is present from the first valid step;
- final alpha remains near-uniform around `0.488`;
- reaction remains negative.

This package investigates the narrower root-cause question:

Why does `alpha-init history` have broad/background `He_current` at step 0?

## 2. Saved Step-0 Field Comparison

Three saved full-run step-0 fields were analyzed:

| case | alpha_mean | notch_He | bulk_He_p95 | bulk/notch | bottom/notch | max He location |
|---|---:|---:|---:|---:|---:|---|
| alpha-init history top-u fixed | 2.9959829027589432e-05 | 6.402885901479749e-07 | 6.490611525578061e-07 | 1.0137009507038128 | 1.2000986143914512 | `(0.000296, 0.009887)` |
| alpha-init history top-u free | 7.835008940174137e-05 | 6.344892540255387e-07 | 6.489555630651012e-07 | 1.022799927576047 | 1.6301958071691651 | `(0.009880, 0.000289)` |
| old history default-alpha | -4.113128338827247e-05 | 4.8979756684275344e-05 | 8.941119403971241e-07 | 0.0182547240109948 | 0.029960636769779638 | `(0.005021, 0.004999)` |

The difference is already present at step 0. With alpha-init-intact, the notch-tip drive is only about `6.4e-7`, similar to bulk/background values. In the old history run, the notch-tip drive is about `4.9e-5`, roughly two orders of magnitude larger than the alpha-init notch drive.

## 3. Displacement / Strain Ansatz Evidence

The current PINN displacement ansatz is globally parameterized by the same neural network over the explicit-notch mesh. The saved step-0 fields show that the narrow notch lips have very small displacement jumps in alpha-init runs:

| case | notch lip u jump abs max | notch lip v jump abs max | strain gradient proxy max |
|---|---:|---:|---:|
| alpha-init history top-u fixed | 8.15454370695079e-10 | 1.0181850029766792e-08 | 1.7459189481395167e-05 |
| alpha-init history top-u free | 1.535724436507735e-09 | 1.005508920570719e-08 | 1.9460922679037144e-05 |
| old history default-alpha | 1.3546888055770978e-08 | 1.0718955074651149e-06 | 0.0003331811792943899 |

The old history default-alpha run has much larger notch-lip displacement jump and strain-gradient proxy at step 0. This supports the diagnostic interpretation that the alpha-init branch is not developing a strong localized notch-lip strain field at the first loaded step.

This does not prove the ansatz alone is the cause, but it is consistent with a displacement/strain representation or optimization path that smooths the narrow notch response.

## 4. Optimizer / Loss Scaling Evidence

A small step-0 budget sweep was run with the same physical settings and `alpha-init-intact`, changing only early optimizer budget:

| rprop budget | status | alpha_mean | notch_He | bulk_He_p95 | bulk/notch | bottom/notch | max He location |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | ok | 0.0 | 7.122027909645112e-07 | 7.124282006998328e-07 | 1.0003164965627505 | 0.9998962424109304 | `(0.005258, 0.009714)` |
| 1 | ok | 5.646749808805443e-06 | 7.116518645489123e-07 | 7.121185831238108e-07 | 1.0006558242845247 | 1.000696273145471 | `(0.004998, 0.009858)` |
| 10 | ok | 9.395900120211226e-05 | 6.939629884072929e-07 | 7.047683141081507e-07 | 1.0155704639604153 | 1.0223404912352438 | `(0.009711, 0.009880)` |
| 100 | ok | 7.938321278363354e-05 | 6.630842790400493e-07 | 6.631712466287353e-07 | 1.0001311561613433 | 1.1858190352729021 | `(0.000113, 0.000294)` |

The broad/background drive appears even with `rprop_budget = 0`, where alpha remains exactly zero. This indicates the step-0 broad drive is not produced by later alpha growth. It is already present in the early displacement/strain field before meaningful alpha evolution.

The budget sweep also shows the max-drive location moves among boundary/corner regions, while the bulk/notch ratio remains near 1. This points toward an early mechanics/ansatz/optimization representation issue rather than a late phase-field update issue.

## 5. FE-DOF / PINN Recompute Consistency

Saved PINN `He_current` was recomputed from saved strain fields:

| case | max_abs_diff | relative_max_diff | max He location |
|---|---:|---:|---|
| alpha-init top-u fixed | 1.6918017373241073e-13 | 2.179283565811448e-07 | `(0.000296, 0.009887)` |
| alpha-init top-u free | 1.9712960353180975e-13 | 1.9058460024765225e-07 | `(0.009880, 0.000289)` |
| old history | 2.1310650455668573e-12 | 4.350909824447991e-08 | `(0.005021, 0.004999)` |

The saved PINN fields are internally consistent. The broad/background drive is not a plotting artifact or saved-field recomputation error.

A small FE-DOF alpha-zero baseline at `Delta = 1e-6` gives notch-dominated drive:

| FE-DOF case | notch_He | bottom_right_He | bottom/notch |
|---|---:|---:|---:|
| alpha zero, top-u fixed | 31.276840209960938 | 1.0564937591552734 | 0.03377878814045944 |
| alpha zero, top-u free | 34.6502799987793 | 0.9916369915008545 | 0.028618440934266305 |

The FE-DOF result is not directly comparable in absolute energy scale to the PINN field because it is an independent nodal optimization diagnostic. However, it shows that the same mesh and boundary setup can represent a notch-dominated drive field when the displacement field is represented with nodal degrees of freedom.

## 6. Diagnostic Conclusion

The current evidence supports this chain:

1. Step-0 alpha-init broad/background drive exists before meaningful alpha evolution.
2. It persists under top-u fixed and top-u free.
3. It persists under very small early optimizer budgets, including `rprop_budget = 0`.
4. Saved PINN fields recompute consistently, so this is not a postprocessing artifact.
5. A nodal FE-DOF baseline can represent notch-dominated drive on the same mesh at `Delta = 1e-6`.

The most likely next diagnostic target is the displacement/strain representation and early mechanics optimization path near the narrow explicit notch, not `l0`, material parameters, TM split, phase-field notch seeding, top-u boundary mode, or history update logic.

This package does not prove a final fix. It narrows the next step to a controlled mechanics-only or ansatz-capacity diagnostic.

## 7. Verification and Handoff

Verification passed:

- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q`
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_step0_root_cause.py debug_recompute_he_current.py debug_fedof_energy_baseline.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py`

Test result:

`13 passed in 0.05s`

GitHub CLI status:

- Bare `gh` was not available in the current PowerShell PATH.
- `C:\Program Files\GitHub CLI\gh.exe` exists and reports version `2.93.0`.
- `gh auth status` reports no logged-in GitHub hosts.
- `GH_TOKEN` and `GITHUB_TOKEN` were not present.
- Therefore this package uses markdown-only handoff and was not automatically posted to issue #1.
