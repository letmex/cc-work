# Postprocess Workflow

The normal entry point is:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir <model_dir> --result-dir <result_dir>
```

`main.py` calls the same entry point after training. The postprocessor reads
saved step checkpoints, reconstructs the actual PINN energy for each saved load
step, and computes the energy-conjugate reaction by differentiating the
mechanics energy with respect to the applied displacement.

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

If checkpoints are not available or the energy derivative cannot be computed,
the table is still written with `reaction_metric_unavailable` status and `NaN`
reaction/stress values.

## Normal Commands

Training:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-schedule-file load_schedule_D0020_extended.csv --run-suffix seed23_D0020
```

Postprocess:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_D0020 --result-dir outputs\results\seed23_D0020
```

Plotting can also be called directly when a result folder and stress-strain table
already exist:

```powershell
D:\anaconda3\envs\torch_env\python.exe plot_results.py --result-dir outputs\results\seed23_D0020 --stress-strain-csv outputs\results\seed23_D0020\curves\stress_strain_by_step.csv
```

This workflow reports computed diagnostics only; it does not by itself validate
the physical model.
