# Prescribed Thermal Strain Patch Test Report

## 1. What was implemented?

A minimal prescribed-temperature thermal-strain branch was implemented in
`examples/TM_comsol_thermal_micro` only. The branch supports a prescribed
absolute temperature, prescribed `delta_T`, and simple off/uniform/linear-y
prescribed `delta_T` modes. Defaults keep thermal mode off.

## 2. What was not implemented?

No heat PDE, heat equation residual, thermal transport solve, trainable
temperature field, damage-dependent conductivity, or damage-thermal feedback was
implemented. No material parameters or `l0` were changed.

## 3. Where does thermal strain enter the mechanics route?

Total small strain is computed first. The prescribed thermal strain is then
subtracted from the normal strain components before calling the existing
`mixed_mode_energy_split` and TM stress helper:

```text
delta_T = T - Tref
exx_e = exx - alpha_T*delta_T
eyy_e = eyy - alpha_T*delta_T
exy_e = exy
```

The existing split/history/AT2 route is otherwise preserved.

## 4. Does default no-thermal behavior remain unchanged?

Yes. `thermal_mode` defaults to `off`; with no thermal input or
`thermal_delta_T=0`, the split/energy/stress fields match the default route.

## 5. Is `delta_T = T - Tref` used?

Yes. `thermal_prescribed.delta_T_from_temperature` subtracts `Tref`, with
`Tref = 273.15 K` by default.

## 6. Is shear strain left unmodified by thermal expansion?

Yes. The helper and patch tests verify that `exy_e = exy`.

## 7. Does free uniform expansion give near-zero mechanical strain/stress/energy?

Yes. The patch test with `exx=eyy=alpha_T*40 K`, `exy=0`, and `delta_T=40 K`
produced near-zero elastic strain, TM stress, and total strain energy density.

## 8. Does constrained uniform heating give compressive thermal stress?

Yes. With `exx=eyy=exy=0` and `delta_T=50 K`, the elastic normal strains are
`-alpha_T*delta_T`. The project stress convention gives
`sigma_xx = sigma_yy = -0.124221774 kN/mm^2` and `sigma_xy = 0`.

## 9. What constitutive convention was used?

The constrained-heating magnitude uses the current project convention:
`lambda = E*nu/((1+nu)*(1-2nu))`, `mu = E/(2*(1+nu))`, and
`eps_zz = -nu/(1-nu)*(eps_xx + eps_yy)` inside the TM source split/stress
helper. The test does not force a separate plane-stress formula.

## 10. Was any heat PDE implemented?

No.

## 11. Was any damage-dependent conductivity implemented?

No.

## 12. Were any training runs performed?

No.

## 13. What tests were run?

- `D:\anaconda3\envs\torch_env\python.exe` recursive `py_compile` for Python
  files under `examples/TM_comsol_thermal_micro`.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests\test_prescribed_thermal_strain_patch.py -q`
  produced `8 passed`.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_thermal_micro\tests -q --ignore=examples\TM_comsol_thermal_micro\tests\test_project_directory_hygiene.py`
  produced `55 passed, 8 warnings`.

## 14. What is the next safe task?

Run a small prescribed-temperature micro-notch diagnostic inside the thermal
subproject only, comparing against the default no-thermal route. Do not start
heat PDE or damage-dependent conductivity work next.

## Final Classification

`prescribed thermal strain branch implemented and patch tests passed`
