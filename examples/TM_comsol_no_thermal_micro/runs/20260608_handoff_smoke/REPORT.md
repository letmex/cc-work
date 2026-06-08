# Handoff Smoke Test Report

## Purpose

This run verifies that Codex can execute a minimal `TM_comsol_no_thermal_micro` training command, collect outputs, create a compact evidence package, generate `HANDOFF_COMMENT.md`, and push the package to GitHub.

This report does not make a physical interpretation of the crack path or model validity.

## Workflow Status

| Item | Status | Notes |
|---|---|---|
| `git pull origin main` | passed | Handoff repository fast-forwarded to the latest remote state before work began. |
| Workflow files read | passed | `AGENT_HANDOFF_WORKFLOW.md` and `CODEX_NO_GH_HANDOFF.md` were read. |
| Short training | passed | One-step smoke command completed with exit code 0. |
| `pytest tests -q` | passed | `11 passed in 0.06s`. |
| `py_compile` | passed | Requested project Python files compiled successfully. |
| diagnostics CSV | found | Copied to `tables/diagnostics_mixed_tm_summary.csv`. |
| fields NPZ | found | Copied to `artifacts/fields_mixed_tm_step_0000.npz`. |
| plots | not generated | No PNG/SVG/JPG figures were produced by the smoke command. |
| handoff markdown | generated | `HANDOFF_COMMENT.md` is included in this package. |

## Smoke Training Output

Local results directory:

```text
D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\smoke_hl_2_Neurons_20_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_rprop_1_lbfgs_0_gradient_numerical_handoff_smoke
```

Primary generated files:

```text
diagnostics_mixed_tm_summary.csv
displacement_list.csv
fields_mixed_tm_step_0000.npz
```

## Key Diagnostics From Step 0

These values only show that the smoke output was generated and parseable.

| Quantity | Value |
|---|---:|
| step | 0 |
| alpha_min | 5.0067901611328125e-06 |
| alpha_max | 5.116065494803479e-06 |
| alpha_mean | 5.064237939222949e-06 |
| n_alpha_lt_0 | 0 |
| n_alpha_gt_1 | 0 |
| max_He_current_x | 0.009910179302096367 |
| max_He_current_y | 0.005197328515350819 |
| max_He_history_x | 0.009910179302096367 |
| max_He_history_y | 0.005197328515350819 |
| max_mechanics_drive_x | 0.009910179302096367 |
| max_mechanics_drive_y | 0.005197328515350819 |
| reaction_N_tm_eff | 0.09520948678255081 |
| elastic_energy | 7.085461384281899e-11 |
| fracture_energy | 2.0420568077427192e-17 |

## Test Results

```text
pytest: 11 passed in 0.06s
py_compile: passed
short training: passed
```

## Interpretation Boundary

This was a one-step smoke run with a minimal optimizer budget. It is not suitable for physical judgment, crack-path interpretation, seed comparison, or model validation.
