# TM COMSOL No-Thermal Micro

Self-contained TM example for the COMSOL micro-notch geometry with the thermal field removed.

This project implements the mechanical phase-field subset of the original COMSOL thermal-elastic phase-field model. The PINN training path intentionally removes:

- temperature field T
- heat equation
- thermal expansion strain
- thermal conductivity, heat capacity, and density thermal coupling

The neural network predicts only:

```text
u(x,y), v(x,y), alpha(x,y)
```

The retained source-model route is:

```text
history_mode = mixedH_TM
mixed_split_mode = tm_source
mixed_mechanics_mode = history
geometry_mode = comsol_micro_gap
```

This project keeps only the files needed to train the current TM route and generate field/reaction plots. It does not include the older audit, diagnosis, scalar-H, single-H, hybrid-history, or rescue scripts.

The notch is an explicit geometric free boundary in `geo_coarse_with_groups_mm.msh`, not a phase-field notch. Alpha is not used to create the pre-existing notch and should start near zero in the solid material domain. Do not set `alpha=1` along the geometric notch unless running a separate, explicitly named phase-field-notch alternative route.

history / dual-history is the closest paper-COMSOL source-model route in this no-thermal project. It uses the dual history fields `HI` and `HII` and the phase-field drive `He_history = HI + (Gc/GcII)*HII`.

current_split is a diagnostic/ablation route. It keeps the same `tm_source` split but drives the energy directly with the current strain field, `He_current = psiI + (Gc/GcII)*psiII`. It is useful for isolating the current strain -> split -> drive chain, but it is not the source-model route.

`history_phase_current_mechanics` is an experimental split-gradient diagnostic. It does not implement a full COMSOL-style segregated solve; it only separates gradient channels in one optimizer closure so that u/v receive current-positive-energy gradients while alpha receives a detached history-drive phase-field contribution.

Known unresolved issue: in `current_split` mode, the `He_current`/`mechanics_drive` hotspot may migrate to the bottom boundary instead of staying near the notch tip. This bottom-boundary He_current/mechanics_drive artifact must be diagnosed before claiming physical correctness.

Known modeling risk: the explicit notch is very narrow. A single continuous global neural network displacement ansatz may be insufficient because the upper and lower notch lips are geometrically close but physically separated. This can contaminate local strain and TM drive near the notch.

`alpha` is constrained by a soft `NonsmoothSigmoid`, not by a strict projection. Slight `alpha < 0` or `alpha > 1` values can occur and should be monitored in the per-step summary diagnostics.

By default, the raw alpha channel has the same neural-network initialization as other outputs; use `--alpha-init-intact` for diagnostics or training runs that must start the material domain near `alpha=0` without imposing a phase-field notch.

## Model Setup

- Mesh: `geo_coarse_with_groups_mm.msh`, converted from COMSOL `geo_coarse.mphtxt` and scaled to mm.
- Domain: `0..0.01 mm` by `0..0.01 mm`.
- Notch tip: approximately `(0.005, 0.005) mm`.
- Material: `E = 81.5 kN/mm^2`, `nu = 0.38`.
- Fracture parameters: `Gf0 = 2.4e-6 kN/mm`, `l0 = 1.5e-4 mm`, `GcII = 2*(1+nu)*(0.60)^2*Gf0`.
- No thermal strain, no temperature field, no COMSOL viscosity term.
- Source-model history drive: `He_history = HI + (Gc/GcII)*HII`, with `HI/HII` committed after each load step.
- Diagnostic current split drive: `He_current = psiI + (Gc/GcII)*psiII`.

## Smoke Check

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 2 20 7 TrainableReLU 3.0 --smoke --n-rprop 1 --n-lbfgs 0 --max-steps 1 --delta-max 1e-6 --run-suffix smoke_check
```

## Example Run

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-schedule-file load_schedule_D0020_extended.csv --run-suffix seed23_D0020
```

The packaged schedule spans `1e-6..1e-4 mm`, matching the COMSOL micro displacement scale.
The normal route defaults are `pff-model=AT2`, `mixed-mechanics-mode=history`,
`top-u-mode=free`, and `coord-normalization=unit_box`, so those flags are not
needed in the standard command. They remain available as advanced overrides for
diagnostic comparisons.

## Postprocess Results

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir <model_dir> --result-dir <result_dir>
```

Training completion invokes the same `postprocess_results.py` path automatically.
It writes `curves/stress_strain_by_step.csv`, `curves/reaction_by_step.csv` when
checkpoint reactions are computable, `curves/reaction_metric_availability.csv`,
and result figures under `figures/`. The optional `plot_results.py` module is a
direct plotting helper, not a required second step.

When checkpoint exact reaction is available, `reaction_N_energy_exact` is the
primary energy-conjugate reaction for stress-strain plotting.
`reaction_N_legacy_top_sigma` is retained as a diagnostic legacy boundary-stress
integral only and must not be used alone to claim post-crack softening behavior.
This interface cleanup does not claim physical validation.

## Debug Recompute

Use `debug_recompute_he_current.py` to check whether a saved displacement/strain field alone produces a bottom-boundary `He_current` hotspot:

```powershell
D:\anaconda3\envs\torch_env\python.exe debug_recompute_he_current.py --npz results\<run>\fields_mixed_tm_step_XXXX.npz --out debug_recompute.csv
```

For a fixed saved displacement/strain field, `He_current` is independent of `alpha`. The script's `--alpha-mode zero` option only changes `g_alpha` and degraded elastic energy diagnostics; it does not change the recomputed `He_current`.
