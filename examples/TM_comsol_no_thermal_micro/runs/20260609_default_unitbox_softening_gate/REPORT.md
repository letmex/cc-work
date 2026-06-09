# Default unit_box softening gate diagnostic

## Scope

This package diagnoses the current blocker: the default-alpha `unit_box` route is seed-robust for notch localization, but D0020 reaction-strain curves did not show clear post-peak softening. The diagnostic extends the load schedule without changing the physics route.

Main route used for extended runs:

`history + default alpha init + top-u-mode free + coord_normalization unit_box`

The runs did not use `--alpha-init-intact` and did not add notch-specific guidance.

## Schedule

No D0040/D0060 schedule existed in the project. A conservative new file `load_schedule_D0040_softening_gate.csv` was created by preserving the D0020 schedule through `1.0e-4` and extending to `2.0e-4` with smaller increments immediately after D0020.

## D0020 post-peak audit

| seed | peak reaction | final reaction | drop % | drop >10% |
|---:|---:|---:|---:|---|
| 7 | 0.79588 | 0.789918 | 0.749 | False |
| 13 | 0.958831 | 0.958831 | 0 | False |
| 21 | 0.864232 | 0.864232 | 0 | False |
| 42 | 0.916372 | 0.866393 | 5.45 | False |
| 99 | 0.965839 | 0.965839 | 0 | False |

## Extended D0040 results

| seed | status | peak reaction | final reaction | drop % | peak step | final step | crack x-span | extends beyond tiny blob |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 7 | completed | 0.968423 | 0.932704 | 3.69 | 53 | 54 | 0.00503633 | True |
| 13 | completed | 1.01312 | 0.942999 | 6.92 | 51 | 54 | 0.0050228 | True |
| 42 | completed | 0.98987 | 0.919255 | 7.13 | 51 | 54 | 0.0050228 | True |

## Reaction consistency audit

- `reaction_N_tm_eff` is computed as top-boundary integration of saved `sigma_yy_tm_eff`.
- `sigma_yy_tm_eff = sigma_yy_tm_total + (g_alpha - 1) * sigma_yy_tm_plus` with `g_alpha=(1-alpha)^2+eta_residual`.
- The same `g_alpha` expression is used in `compute_energy_mixed_tm.py` for mechanics/history elastic energy degradation.
- Degraded and undegraded post hoc reactions are both written in `tables/reaction_consistency_audit.csv`.

## Gate decision

- Required completed seeds: 3/3.
- Required seeds with >=10% post-peak drop plus connected crack growth proxy: 0/3.
- Reaction consistency confirmed: True.
- Decision: **softening gate not passed**.

This decision is a softening diagnostic gate only. It is not physical validation.

## What cannot be concluded

- This does not validate material parameters, `l0`, or mesh independence.
- This does not prove physical fracture behavior against experiments.
- If the gate passes, it only says the current route can produce a post-peak reaction drop under the extended schedule and the reaction computation is internally consistent.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed in 1.82 s.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260609_default_unitbox_softening_gate\artifacts\build_softening_gate_package.py`: passed.
