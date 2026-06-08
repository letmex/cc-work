# Handoff Smoke Test Package

This package verifies the Codex -> GitHub -> ChatGPT handoff workflow for `examples/TM_comsol_no_thermal_micro`.

It is not a physical validation package and should not be used to judge crack path, damage localization, or model fidelity.

## Source Project

Local training project:

```text
D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro
```

GitHub handoff package:

```text
examples/TM_comsol_no_thermal_micro/runs/20260608_handoff_smoke
```

## Smoke Training Command

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 2 TrainableReLU 3.0 --smoke --pff-model AT2 --mixed-mechanics-mode history --alpha-init-intact --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix handoff_smoke
```

## Local Output Directory

```text
D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\smoke_hl_2_Neurons_20_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_rprop_1_lbfgs_0_gradient_numerical_handoff_smoke
```

## Main Files

```text
README.md
REPORT.md
commands_run.txt
next_questions.md
HANDOFF_COMMENT.md
MANIFEST.json
tables/diagnostics_mixed_tm_summary.csv
tables/displacement_list.csv
figures/figure_summary.md
artifacts/fields_mixed_tm_step_0000.npz
```

## Scope

The run uses history mode and alpha-init-intact with the default project `l0`, material parameters, and TM split. It was intentionally configured as a one-step low-cost workflow smoke.
