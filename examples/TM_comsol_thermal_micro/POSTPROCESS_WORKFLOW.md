# Postprocess Workflow

The inherited no-thermal entry point is:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir <model_dir> --result-dir <result_dir>
```

`main.py` calls the same entry point after training. The postprocessor reads
saved step checkpoints, reconstructs the PINN energy for each saved load step,
and computes the energy-conjugate reaction by differentiating the mechanics
energy with respect to the applied displacement.

## Outputs

The normal postprocess path writes under the result directory:

```text
curves/
    reaction_by_step.csv
    stress_strain_by_step.csv
    reaction_metric_availability.csv
figures/
    reaction_strain_<run>.png
    stress_strain_<run>.png
    final_fields_panel_<run>.png
    final_<field>_<run>.png
```

The stress-strain table uses:

```text
reaction_N_energy
nominal_stress_energy_MPa
reaction_metric_status
is_energy_conjugate
```

For `load_case = shear`, the table uses:

```text
Delta_s
engineering_shear_strain
reaction_N_energy
nominal_shear_stress_energy_MPa
reaction_metric_status
is_energy_conjugate
```

If checkpoints are not available or the energy derivative cannot be computed,
the table is still written with `reaction_metric_unavailable` status and `NaN`
reaction/stress values.

## Thermal Status

The prescribed-temperature thermal-strain branch is part of the mechanics energy
route when enabled by model settings. The postprocessor reuses the same
`delta_T = T - Tref` correction before recomputing checkpoint energies, so
energy-conjugate reactions remain consistent with training for prescribed
thermal-strain runs.

Heat equation postprocessing and damage-dependent conductivity outputs are not
implemented. Future heat-transfer work should keep this energy-conjugate
reaction route unless a separate validated replacement is introduced.
