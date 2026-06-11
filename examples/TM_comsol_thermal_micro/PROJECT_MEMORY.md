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
- prescribed thermal strain: implemented and default-off
- heat PDE: not implemented
- damage-dependent conductivity: not implemented

Reference scope for future thermal work:

- use COMSOL `comp3 / solid3 / ht3 / c / state3 / std1`
- ignore `comp4`
- do not use `TFinal`
- do not require line-by-line COMSOL matching

The original `examples/TM_comsol_no_thermal_micro` project is the frozen
baseline and should not be modified by thermal experiments.

Current thermal branch:

- `delta_T = T - Tref`
- `exx_e = exx - alpha_T*delta_T`
- `eyy_e = eyy - alpha_T*delta_T`
- `exy_e = exy`
- thermal strain enters before the existing TM split/history/energy route
- no-thermal defaults remain the baseline

Current prescribed-temperature stage status:

- stage summary package:
  `examples/TM_comsol_thermal_micro/runs/20260626_prescribed_thermal_strain_stage_summary`
- final classification:
  `prescribed thermal strain stage summary complete`
- patch tests and multiple checkpointed tension diagnostics support preserving
  the prescribed-temperature mechanics branch as a reviewed baseline candidate
- `thermal_mode=uniform` with `delta_T=0` reproduces the no-thermal route in
  completed diagnostics
- prescribed uniform `delta_T=+20 K` consistently shifts
  displacement-controlled tension reaction/stress downward
- the moderate damage probe shows lower notch-tip/high-threshold alpha growth
  for Case C, within diagnostic scope
- broad low-level Case C alpha background remains diagnostic-only and is not
  physical fracture evidence
- this stage is not physical validation against COMSOL or experiment
- safest next task: hold a decision-gate review before any heat PDE planning;
  damage-dependent conductivity remains deferred until heat PDE is stable

Current heat PDE planning status:

- heat PDE implementation/validation planning package:
  `examples/TM_comsol_thermal_micro/runs/20260627_heat_pde_implementation_validation_plan`
- final classification:
  `heat PDE implementation plan complete`
- this package is planning-only: it does not implement a heat PDE, trainable
  temperature field, heat residual loss, boundary-condition code, or
  damage-dependent conductivity
- first approved implementation should start with constant-conductivity heat
  transfer only:
  `rho*c*dT/dt - div(k0*grad(T)) = Q`
- initial heat PDE phase should use `Q=0` and constant `k0`; validate constant
  T, linear steady conduction, insulated flux, unit conversion, and solved
  uniform-T mechanics patch tests before any notch diagnostic
- exact SI-to-project unit conversion is an implementation gate, not a solved
  detail: COMSOL `rho`, `c`, and `k0` are SI constants, while this project uses
  mm geometry and kN/mm-style mechanics quantities
- preserve the prescribed-temperature fallback and `thermal_mode=off` default
  throughout heat PDE work
- keep material parameters, `l0`, history logic, loss route, mechanical
  boundary conditions, source model behavior, and checkpointed
  energy-conjugate reaction unchanged in the first heat PDE phases
- damage-dependent conductivity `k(d)=g(d)k0` remains explicitly deferred until
  constant-conductivity heat PDE and solved-temperature-to-thermal-strain
  coupling are independently validated and separately approved

Standing simplified finalization protocol for all future Codex tasks in this
thermal subproject:

- do not use `git add .`
- do not perform full-repo staging
- use exact-path staging only
- stage exact thermal schedule/package/project-memory paths only
- force-add package PNG figures only when project ignore rules hide required figures
- always check `examples/TM_comsol_no_thermal_micro` has no unstaged or staged changes
- run package schema/compile/focused validation once after package generation
- do not perform repeated full validation after package validation has already
  passed
- do not chase self-referential handoff commit hashes indefinitely
- use at most one handoff-sync commit if the generated handoff still contains pending commit or push status
