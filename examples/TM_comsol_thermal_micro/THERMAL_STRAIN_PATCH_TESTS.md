# Prescribed Thermal Strain Patch Tests

The implemented branch is prescribed-temperature mechanics only. It does not
solve a heat PDE and does not introduce damage-dependent conductivity.

Patch-test scope:

- zero `delta_T` equivalence to the no-thermal split route;
- `delta_T = T - Tref`, with `Tref = 273.15 K`;
- free uniform expansion gives near-zero elastic strain, stress, and energy;
- constrained uniform heating gives compressive stress under the project TM
  source convention;
- isotropic thermal strain does not directly modify `exy`;
- guard scan confirms no active heat PDE, no trainable/PDE temperature field,
  and no active `k_d = g(d)*k0` coupling.

The constrained-heating stress check uses the project's current convention:
`eps_zz = -nu/(1-nu)*(eps_xx + eps_yy)` inside the TM source split/stress
helper. It is not forced to match a separate plane-stress formula.
