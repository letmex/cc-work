# TM COMSOL Thermal Micro

This subproject is a copied scaffold from `examples/TM_comsol_no_thermal_micro`.
It is the isolated future workspace for prescribed-temperature thermoelastic
coupling and patch tests.

The original `TM_comsol_no_thermal_micro` project is the frozen verified
baseline. Thermal experiments should be made here, not in the original
no-thermal project.

## Current Scope

The initial code is intentionally no-thermal and should match the verified
baseline route:

```text
history_mode = mixedH_TM
mixed_split_mode = tm_source
mixed_mechanics_mode = history
geometry_mode = comsol_micro_gap
top_u_mode = free
coord_normalization = unit_box
PFF_model = AT2
```

No temperature field, heat equation, thermal expansion strain, or thermal
transport coupling is implemented yet. Damage-dependent conductivity is not
implemented yet.

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

## Future Thermal Reference

The theoretical COMSOL reference branch for future thermal work is:

```text
comp3 / solid3 / ht3 / c / state3 / std1
```

`comp4` and `TFinal` are ignored. Exact COMSOL line-by-line matching is not
required; physical invariants and documented platform differences are the target.

Reserved thermal constants for later work:

- `alpha_T = 18.9 ppm/K`
- `rho = 1040 kg/m^3`
- `k0 = 418 W/m/K`
- `c = 170 J/kg/K`
- `Tref = 273.15 K`
- `T0 = 0 degC`

## Next Task

The next task is prescribed-temperature thermal-strain reintroduction with patch
tests. Do not implement the full heat PDE or damage-dependent conductivity until
those patch tests pass.

## Smoke Check

This command is retained only as a lightweight inherited no-thermal smoke path.
Do not use it as thermal validation.

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix smoke_check
```

## Project Hygiene

Generated checkpoints, results, figures, curves, logs, and temporary diagnostics
belong under `outputs/`. Scaffold and audit handoff packages belong under
`runs/`. The example root should remain source-only.
