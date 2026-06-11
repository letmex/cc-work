# TM COMSOL No-Thermal Micro

Self-contained no-thermal TM example for the COMSOL micro-notch geometry.

The neural network predicts:

```text
u(x,y), v(x,y), alpha(x,y)
```

The retained route is:

```text
history_mode = mixedH_TM
mixed_split_mode = tm_source
mixed_mechanics_mode = history
geometry_mode = comsol_micro_gap
top_u_mode = free
coord_normalization = unit_box
PFF_model = AT2
```

The example root is source-only. Generated checkpoints, results, figures,
curves, logs, and temporary diagnostics are written under `outputs/`. Audit and
handoff packages belong under `runs/`.

The notch is an explicit geometric free boundary in `geo_coarse_with_groups_mm.msh`.
Alpha is not used to create a pre-existing phase-field crack in the normal route.

## Model Setup

- Mesh: `geo_coarse_with_groups_mm.msh`, scaled to mm.
- Domain: `0..0.01 mm` by `0..0.01 mm`.
- Notch tip: approximately `(0.005, 0.005) mm`.
- Material: `E = 81.5 kN/mm^2`, `nu = 0.38`.
- Fracture parameters: `Gf0 = 2.4e-6 kN/mm`, `l0 = 1.5e-4 mm`, `GcII = 2*(1+nu)*(0.60)^2*Gf0`.
- No temperature field, heat equation, thermal expansion strain, or thermal transport coupling.
- Source-model history drive: `He_history = HI + (Gc/GcII)*HII`, with `HI/HII` committed after each load step.

This route does not claim physical validation by itself. Stress-strain behavior
must be interpreted from the energy-conjugate checkpoint reaction and the saved
field diagnostics.

## Smoke Check

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix smoke_check
```

## Example Run

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-schedule-file load_schedule_D0020_extended.csv --run-suffix seed23_D0020
```

Shear smoke run on the same geometry:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-case shear --load-schedule-file load_schedules/load_schedule_S0020_shear.csv --run-suffix seed23_S0020_shear
```

By default this command writes:

- checkpoints and model settings to `outputs/checkpoints/seed23_D0020/`
- field data and diagnostics to `outputs/results/seed23_D0020/`
- TensorBoard logs to `outputs/logs/seed23_D0020/`
- curves and figures under the result folder

## Postprocess Results

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir <model_dir> --result-dir <result_dir>
```

Training completion invokes the same `postprocess_results.py` path automatically.
It writes:

- `curves/reaction_by_step.csv`
- `curves/stress_strain_by_step.csv`
- `curves/reaction_metric_availability.csv`
- `figures/reaction_strain_<run>.png`
- `figures/stress_strain_<run>.png`
- final field figures when field NPZ files are available

The normal reaction metric is `reaction_N_energy`, obtained from the saved
checkpoint mechanics energy derivative `dPi/dDelta`. The normal stress column is
`nominal_stress_energy_MPa`.

For `--load-case shear`, the same energy-conjugate derivative is interpreted as
`dPi/dDelta_s`, and the curve uses `engineering_shear_strain` and
`nominal_shear_stress_energy_MPa`.

## Project Hygiene

Do not leave debug scripts, generated CSV/NPZ files, checkpoint folders, or run
logs in the example root. Temporary diagnostics belong under `outputs/debug/` or
inside a named `runs/` audit package. Generated CSV/NPZ/checkpoint data should
not be placed directly in this root folder.
