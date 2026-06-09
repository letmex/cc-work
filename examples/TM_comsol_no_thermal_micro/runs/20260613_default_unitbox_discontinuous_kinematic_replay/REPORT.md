# Discontinuous/split-domain frozen-alpha kinematic replay

## Scope

This diagnostic freezes alpha and saved HI/HII history from D0040 default unit_box states and re-optimizes only displacement fields. Seeds: 7, 13, 42. States: final_D0040. The split-domain variants use independent upper/lower displacement networks selected by a connected alpha>=0.8 crack-path label. The original trial-history `max(old,current)` mechanics logic, material constants, TM split, top-u-free/unit_box ansatz, displacement level, and `l0` are preserved.

The split-domain representation is diagnostic-only and is not a production formulation.

## Domain split construction

The connected alpha>=0.8 crack band is detected from the notch-tip window by element adjacency. A median crack path y(x) is interpolated from the connected band. Nodes above this path use the upper displacement field and nodes below use the lower displacement field. Crack-band elements are audited separately in `tables/domain_split_geometry_audit.csv`.

## Final D0040 summary

| variant | mean reaction removal | seeds with >=30% drop | mean v-jump change | mean crack-band traction removal |
|---|---:|---:|---:|---:|
| split_domain_current_split | -1.39% | 0/3 | 1.88652e-07 | 0.092% |
| split_domain_minus_degraded_crack_band | -8.92% | 0/3 | 4.19386e-08 | 100% |
| split_domain_crack_band_void | 15.8% | 1/3 | 5.8192e-06 | 100% |

## Answers

1. Split-domain/discontinuous replay reached finite losses for all requested rows: True. Detailed statuses are in `tables/discontinuous_convergence.csv`.
2. Reaction changes relative to the continuous baseline are in `tables/discontinuous_reaction_comparison.csv`.
3. Crack opening/jump proxies are in `tables/discontinuous_displacement_jump.csv`.
4. Crack-band traction proxies are in `tables/discontinuous_crack_band_traction.csv`.
5. Energy comparisons are in `tables/discontinuous_energy_comparison.csv`.
6. Continuous baseline rows within 5.0% of the previous frozen-alpha baseline: 3/3.
7. Diagnostic classification: **continuous-field bridging not confirmed**.
8. No production model change is justified directly from this diagnostic package.
9. Next minimal intervention: have ChatGPT review whether the split-domain result identifies continuous-field/boundary bridging or instead points to boundary-condition/reaction definition auditing.

## Cannot conclude

- This package does not validate a physical crack model.
- This package does not justify changing material parameters, `l0`, TM split, or history update logic.
- The split-domain representation is a diagnostic replay, not a production route.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q`: failed because this repo checkout has no `examples\TM_comsol_no_thermal_micro\tests` directory.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260613_default_unitbox_discontinuous_kinematic_replay\artifacts\run_discontinuous_kinematic_replay.py`: passed.
