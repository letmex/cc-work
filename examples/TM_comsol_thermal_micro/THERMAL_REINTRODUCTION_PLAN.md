# Thermal Reintroduction Plan

## Purpose

Create an isolated path for prescribed-temperature thermoelastic reintroduction
without disturbing the verified no-thermal baseline.

## Baseline copied from no-thermal project

This scaffold starts from the verified `TM_comsol_no_thermal_micro` route:
`mixedH_TM + tm_source + history`, AT2 phase field, default alpha initialization,
top-u/free tension ansatz, top-v-free shear ansatz, unit-box coordinate
normalization, and checkpointed energy-conjugate reaction.

## COMSOL comp3 reference scope

The future thermal reference branch is `comp3 / solid3 / ht3 / c / state3 /
std1`. `comp4` and `TFinal` are excluded from this single-notch branch.

## Physical invariants to preserve

- AT2 phase-field class.
- TM source/history split.
- HI/HII max-history irreversibility.
- Degradation logic.
- Current material units.
- Current `l0`.
- Current energy-conjugate reaction policy.
- Current postprocess route.
- Current directory hygiene.
- No previous top-sigma route as the primary reaction.

## Platform differences allowed

FEM and PINN implementation details may differ when the physical meaning is
preserved and documented. Exact COMSOL line-by-line matching is not required.

## Thermal constants reserved for future work

- `alpha_T = 18.9 ppm/K`
- `rho = 1040 kg/m^3`
- `k0 = 418 W/m/K`
- `c = 170 J/kg/K`
- `Tref = 273.15 K`
- `T0 = 0 degC`

## Step 1: scaffold only

Create this sibling subproject and document the copied files, excluded artifacts,
and validation checks. Do not implement thermal physics in this step.

## Step 2: prescribed-temperature thermal-strain branch

Add a controlled temperature input and apply the thermoelastic correction before
the TM split:

```text
exx_e = exx - alpha_T*(T - Tref)
eyy_e = eyy - alpha_T*(T - Tref)
exy_e = exy
```

## Step 3: patch tests

Add patch tests before micro-notch diagnostics:

- zero-DeltaT equivalence to the copied no-thermal route;
- uniform free expansion with near-zero stress;
- constrained uniform temperature change with expected stress sign and scale;
- shear strain unchanged by isotropic thermal expansion;
- history unchanged when thermal strain produces no crack-driving increment.

## Step 4: micro-notch thermal-mechanical diagnostic

After patch tests pass, run a small prescribed-temperature micro-notch diagnostic
that keeps the no-thermal route available for comparison.

## Step 5: heat PDE, only after patch tests

Add heat equation coupling only after prescribed-temperature mechanics is
verified. Validate units, thermal boundary conditions, conservation, and output
diagnostics separately.

## Step 6: damage-dependent conductivity, only after heat PDE is stable

Introduce `k_d = g(d)*k0` only after the heat PDE branch is stable and tested.

## Explicit non-goals

- no S0110 or further shear extension;
- no D0040;
- no seed study;
- no full heat PDE in this scaffold task;
- no damage-dependent conductivity in this scaffold task;
- no COMSOL exact matching;
- no source change to the original no-thermal project.
