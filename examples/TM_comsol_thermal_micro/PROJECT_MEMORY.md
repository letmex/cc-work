# Project Memory

This subproject was scaffolded as `examples/TM_comsol_thermal_micro` by copying
the current stable no-thermal source/config/docs/test structure from the
verified `TM_comsol_no_thermal_micro` execution tree.

Current state:

- copied baseline route: `mixedH_TM + tm_source + history`
- phase-field model: AT2
- alpha initialization: default baseline behavior
- tension ansatz: top-u/free
- shear ansatz: top-v-free inherited route
- coordinate normalization: unit-box input
- reaction policy: checkpointed energy-conjugate `reaction_N_energy`
- thermal strain: not implemented
- heat PDE: not implemented
- damage-dependent conductivity: not implemented

Reference scope for future thermal work:

- use COMSOL `comp3 / solid3 / ht3 / c / state3 / std1`
- ignore `comp4`
- do not use `TFinal`
- do not require line-by-line COMSOL matching

The original `examples/TM_comsol_no_thermal_micro` project is the frozen
baseline and should not be modified by thermal experiments.

Next task: add a prescribed-temperature thermal-strain branch with patch tests.
