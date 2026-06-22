# TM COMSOL Thermal Micro

This subproject was copied from `examples/TM_comsol_no_thermal_micro`.
It is the isolated workspace for prescribed-temperature thermoelastic coupling
and patch tests.

The original `TM_comsol_no_thermal_micro` project is the frozen verified
baseline. Thermal experiments should be made here, not in the original
no-thermal project.

## Current Scope

The default route remains the verified no-thermal mechanics baseline:

```text
history_mode = mixedH_TM
mixed_split_mode = tm_source
mixed_mechanics_mode = history
geometry_mode = comsol_micro_gap
top_u_mode = free
coord_normalization = unit_box
PFF_model = AT2
```

A minimal prescribed-temperature thermal-strain branch is available but defaults
to off. A solved-temperature bridge, element-centroid adapter, and explicit
smoke CLI are available for narrow review checks only. No production training
default is switched to solved temperature.

No coupled heat-mechanics training, heat equation residual inside mechanics,
thermal transport production solve, heat-fracture diagnostics, or
damage-dependent conductivity is implemented.

The current reaction policy remains `reaction_N_energy`, obtained from the saved
checkpoint mechanics energy derivative `dPi/dDelta` or `dPi/dDelta_s`. This
route does not claim physical validation by itself.

## Preserved Baseline

- Mesh: `geo_coarse_with_groups_mm.msh`, scaled to mm.
- Domain: `0..0.01 mm` by `0..0.01 mm`.
- Material: `E = 81.5 kN/mm^2`, `nu = 0.38`.
- Fracture parameters: `Gf0 = 2.4e-6 kN/mm`, `l0 = 1.5e-4 mm`, `GcII = 2*(1+nu)*(0.60)^2*Gf0`.
- Source-model history drive: `He_history = HI + (Gc/GcII)*HII`, with `HI/HII` committed after each load step.
- Energy-conjugate checkpoint reaction remains the primary reaction route.

## Prescribed Thermal Strain

When enabled with a prescribed temperature or temperature increment, the branch
uses the current kN-mm mechanics units and applies the thermoelastic correction
before the existing TM split/history/energy route:

```text
delta_T = T - Tref
exx_e = exx - alpha_T*delta_T
eyy_e = eyy - alpha_T*delta_T
exy_e = exy
```

The default `--thermal-mode off` and `delta_T = 0` path preserves the copied
no-thermal behavior within numerical precision. The shear strain component is
not directly altered by isotropic thermal expansion.

Thermal constants used by this prescribed branch:

- `alpha_T = 18.9e-6 1/K`
- `Tref = 273.15 K`
- `T0 = 0 degC`

Reserved transport constants remain future-work references only:

- `rho = 1040 kg/m^3`
- `k0 = 418 W/m/K`
- `c = 170 J/kg/K`

## Solved-Temperature Mechanics Smoke CLI

The solved-temperature mechanics smoke path is opt-in. It is documented so the
bridge and element-centroid adapter can be checked without changing the
training entrypoint.

Run it explicitly from the repository root:

```powershell
D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\run_solved_temperature_mechanics_smoke.py --output-dir examples\TM_comsol_thermal_micro\outputs\solved_temperature_mechanics_smoke
```

The command writes:

```text
mechanics_smoke_results.csv
```

The smoke runner uses one fixed two-element mechanics patch and this call path:

```text
thermal_mechanics_adapter.build_mechanics_thermal_kwargs_from_bridge
-> source_mode = solved_frozen_lift
-> case_id = H1
-> evaluation_location = element_centroid
-> compute_mixed_tm_fields(..., thermal_delta_T=...)
```

The reference package output is stored at:

```text
examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/tables/mechanics_smoke_results.csv
```

Observed values for that fixture:

- `DeltaT` range: `6.666666666666686` K to `13.333333333333371` K
- `mechanics_max_abs_diff = 0.0`
- `network_training_run = false`
- `train_mixed_tm.py remains unmodified`

This smoke CLI does not implement coupled heat-mechanics training.
It does not add heat-fracture diagnostics.
It does not add damage-dependent conductivity.
It does not run D0040, seed, shear, S0110, transient, or bottom-cooling production studies.

The frozen baseline remains `examples/TM_comsol_no_thermal_micro`.

## COMSOL Reference Scope

The theoretical COMSOL reference branch for future thermal work is:

```text
comp3 / solid3 / ht3 / c / state3 / std1
```

`comp4` and `TFinal` are ignored. Exact COMSOL line-by-line matching is not
required; physical-core invariants and documented platform differences are the
target.

## Next Task

The next safe task is to keep any guarded production call-site flag as a
separate reviewed task. Do not implement coupled heat-mechanics training,
heat-fracture diagnostics, full heat PDE coupling, or damage-dependent
conductivity until separate thermal-mechanics diagnostics are reviewed.

## Smoke Check

This command is retained only as a lightweight inherited no-thermal smoke path.
Do not use it as thermal validation or as a replacement for patch tests.

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix smoke_check
```

## Project Hygiene

Generated checkpoints, results, figures, curves, logs, and temporary diagnostics
belong under `outputs/`. Scaffold and audit handoff packages belong under
`runs/`. The example root should remain source-only.
