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
- heat PDE: Phase 1 constant-`k0` utilities implemented and default-inactive
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

Current constant-`k0` heat PDE Phase 1 status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260628_constant_k0_heat_pde_phase1`
- final classification:
  `constant-k0 heat PDE phase1 implemented and patch tests passed`
- added isolated utility module:
  `examples/TM_comsol_thermal_micro/heat_pde.py`
- added focused analytical patch tests:
  `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`
- heat PDE utilities use SI heat units internally:
  temperature in K, heat-gradient length in m, time in s, `rho` in kg/m^3,
  `c` in J/kg/K, `k0` in W/m/K, `Q` in W/m^3, residual in W/m^3
- project mesh coordinates supplied in mm are converted explicitly for heat
  derivatives by `x_m = x_mm * 1e-3`; the mm-to-m chain-rule path is covered by
  a focused patch test against direct meter-coordinate computation
- steady residual sign convention:
  `-div(k0*grad(T)) - Q`, implemented through `q = -k0*grad(T)` and
  `div(q) - Q`
- transient residual utility is available for analytical tests:
  `rho*c*dTdt - div(k0*grad(T)) - Q`
- default heat source is `Q=0`; conductivity is constant `k0=418 W/m/K`
- this phase remains default-inactive: no heat residual loss, no solved
  temperature field, no fracture-training coupling, no checkpoint schema change,
  no postprocess route change, and no broad thermal-fracture diagnostic
- existing prescribed thermal strain patch tests still pass; `thermal_mode=off`
  remains the default fallback route
- damage-dependent conductivity remains unimplemented and guarded: no alpha,
  damage, degradation, or `k(d)` conductivity input is present in the Phase 1
  heat PDE API

Current constant-`k0` thermal functional Phase 1 status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260629_constant_k0_thermal_functional_phase1`
- final classification:
  `constant-k0 thermal functional phase1 implemented and tests passed`
- extended isolated utility module:
  `examples/TM_comsol_thermal_micro/heat_pde.py`
- extended focused analytical patch tests:
  `examples/TM_comsol_thermal_micro/tests/test_constant_k0_heat_pde_patch.py`
- added steady thermal functional density:
  `0.5*k0*|grad_m T|^2 - Q*T`
- added backward-Euler-style transient incremental thermal functional density:
  `rho*c/(2*dt)*(T-T_prev)^2 + 0.5*k0*|grad_m T|^2 - Q*T`
- added pointwise mean helper for patch tests only; it is not mesh quadrature
  or a domain integral
- the main future heat loss route is the thermal functional / weak-form route;
  strong-form residual utilities remain available only for patch-test
  diagnostics and sign/unit sanity checks
- stricter autograd guards now raise clear errors for detached nonconstant
  temperatures and non-gradient coordinates, while explicitly allowing known
  constant fields to return zero gradients
- focused heat PDE patch tests pass with 18 tests; prescribed thermal strain
  fallback patch tests pass with 8 tests
- this phase remains default-inactive: no heat PDE training loss, no solved
  temperature field, no fracture-training coupling, no checkpoint schema change,
  no postprocess route change, and no broad thermal-fracture diagnostic
- damage-dependent conductivity remains unimplemented and guarded: no `alpha`,
  `damage`, `d`, `g_d`, `k_d`, degradation, or `k(d)` conductivity input is
  present in the functional API

Current heat-only weak-form patch training status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260630_heat_only_weak_form_patch_training`
- final classification:
  `heat-only weak-form patch trainer implemented and tests passed`
- added independent heat-only modules:
  `examples/TM_comsol_thermal_micro/thermal_field.py`
  `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
  `examples/TM_comsol_thermal_micro/train_heat_only.py`
- added focused tests:
  `examples/TM_comsol_thermal_micro/tests/test_heat_only_weak_form_training.py`
- top-bottom and bottom-only temperature ansatz functions hard-satisfy their
  Dirichlet boundaries using mm coordinates for the ansatz and normalized
  `[-1, 1]` network inputs
- triangle-area quadrature computes centroids in mm, converts triangle areas to
  m^2, and uses area-weighted thermal functional density as the primary
  heat-only loss
- the heat-only trainer uses `heat_pde.steady_thermal_energy_density_J_per_m3`
  as the loss density; strong-form residuals are post-training diagnostics only
- H1 top-bottom Dirichlet linear conduction passes focused analytical
  temperature validation with max error about `4.75e-4 K` and hard boundary
  errors of `0 K`
- H2 bottom-only Dirichlet natural-insulation patch passes focused analytical
  temperature validation with zero reported temperature and flux errors
- H3 quadrature sanity and H4 transient storage sanity checks pass
- H1 strong residual diagnostic is high after the lightweight centroid
  weak-form smoke training; review quadrature/regularization before any
  solved-temperature mechanics coupling
- this phase remains independent and default-inactive: no mechanics coupling, no
  thermal-strain mechanics coupling, no fracture-training coupling, no
  `train_mixed_tm.py` change, no checkpoint schema change, and no production
  bottom-cooling run
- damage-dependent conductivity remains unimplemented and guarded

Current H1 quadrature and regularization review status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260701_h1_quadrature_regularization_review`
- final classification:
  `h1 review implemented; trained residual remains high but explained`
- added triangle quadrature point utilities in:
  `examples/TM_comsol_thermal_micro/thermal_quadrature.py`
- added H1 diagnostic functions in:
  `examples/TM_comsol_thermal_micro/train_heat_only.py`
- added focused review tests:
  `examples/TM_comsol_thermal_micro/tests/test_h1_quadrature_regularization_review.py`
- no-training H1 lift baseline has zero temperature error, zero correction, and
  zero strong residual, proving the exact linear lift, `heat_pde.py`, and
  mm-to-m scaling are not the cause of the trained residual diagnostic
- trained H1 residual comes from a small learned nonlinear correction admitted
  by weak-form training; the correction is small in temperature norm but large
  in the second-derivative strong residual diagnostic
- triangle 3-point quadrature and optional `correction_l2` regularization reduce
  but do not eliminate the H1 strong residual diagnostic in the lightweight
  smoke setup
- normalized residual metric is recorded as
  `abs(residual) / max(k0*|grad_T|/L_m, eps)`, with `L_m=y_height_m`
- strong residual remains diagnostic-only; no residual loss was added
- this review does not implement solved-temperature mechanics coupling,
  thermal-strain mechanics coupling, fracture training with heat, damage-
  dependent conductivity, or any broad thermal-fracture diagnostics
- next safest task: decide H1 correction policy before any solved-temperature
  mechanics coupling, such as exact-lift freeze for pure linear Dirichlet
  patches, stronger correction regularization, or richer quadrature

Current heat correction policy status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260702_heat_correction_policy`
- final classification:
  `heat correction policy implemented and tests passed`
- added explicit heat-only correction policies in:
  `examples/TM_comsol_thermal_micro/train_heat_only.py`
- supported policies are:
  `frozen_lift`, `trainable_correction`, and `regularized_correction`
- `run_steady_heat_patch_case` now defaults to `frozen_lift` for the pure H1
  linear top-bottom Dirichlet patch and the H2 uniform bottom-only patch
- for H1 frozen lift, the trainer evaluates `T=T_lift`, does not run optimizer
  steps, reports zero correction, and preserves the thermal functional and
  strong-form residual diagnostics
- for H2 frozen lift, the trainer evaluates `T=T_bottom`, does not run
  optimizer steps, reports zero correction, zero residual, and zero top/side
  flux
- `trainable_correction` remains available only through an explicit override
  for diagnostics or future nontrivial heat solves
- `regularized_correction` remains opt-in and adds correction L2
  regularization; it is not the default H1 or H2 policy
- added focused correction policy tests in:
  `examples/TM_comsol_thermal_micro/tests/test_heat_correction_policy.py`
- strong residual remains diagnostic-only; no residual loss was added and the
  primary heat objective remains `thermal_functional_area_weighted_mean`
- prescribed-temperature fallback tests still pass
- this policy stage does not implement solved-temperature mechanics coupling,
  thermal-strain mechanics coupling, fracture heat training, heat-fracture
  diagnostics, damage-dependent conductivity, D0040, seed study, shear
  extension, S0110, transient production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: document the frozen-lift default in the user-facing thermal
  README before any solved-temperature mechanics-coupling work is approved

Current solved-temperature mechanics bridge status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260703_solved_temperature_mechanics_bridge`
- final classification:
  `solved-temperature mechanics bridge implemented and tests passed`
- added a minimal bridge module:
  `examples/TM_comsol_thermal_micro/thermal_solution_bridge.py`
- added focused bridge tests:
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_bridge.py`
- bridge responsibilities are limited to evaluating supported temperature
  sources at mechanics/material coordinates, computing `DeltaT = T - Tref`,
  and returning the same `thermal_delta_T` input consumed by the existing
  prescribed-temperature mechanics route
- supported source modes are `prescribed_uniform` and `solved_frozen_lift`
- H1 `solved_frozen_lift` evaluates the linear top-bottom lift, returns
  spatial `DeltaT(y) = 20*eta` for the 300 K to 320 K patch, reports zero
  correction, and does not run network training
- H2 `solved_frozen_lift` evaluates the uniform bottom lift, returns zero
  `DeltaT` when `T_bottom=Tref`, reports zero correction, and does not run
  network training
- mechanics-level equivalence was checked by passing bridge-produced
  `thermal_delta_T` to `compute_mixed_tm_fields` and comparing against the
  prescribed uniform `DeltaT` route; selected thermal strain, elastic strain,
  stress, and energy fields matched with zero max difference
- no default mechanics/training mode was switched to solved temperature;
  `thermal_mode=off` and prescribed-temperature fallback behavior remain
  unchanged
- this bridge stage does not implement coupled heat-mechanics training,
  fracture heat training, heat-fracture diagnostics, damage-dependent
  conductivity, D0040, seed study, shear extension, S0110, transient
  production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: add one narrow call-site adapter that passes
  bridge-produced `thermal_delta_T` into a cheap smoke mechanics evaluation
  without changing production defaults

Current solved-temperature mechanics smoke adapter status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260704_solved_temperature_mechanics_smoke_adapter`
- final classification:
  `solved-temperature mechanics smoke adapter implemented and tests passed`
- added a narrow element-centroid adapter:
  `examples/TM_comsol_thermal_micro/thermal_mechanics_adapter.py`
- added focused smoke adapter tests:
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`
- adapter responsibilities are limited to computing triangle element centroid
  coordinates, evaluating the existing solved-temperature bridge at those
  centroids, and returning element-sized `thermal_delta_T` through
  `thermal_kwargs` for `compute_mixed_tm_fields`
- adapter API:
  `element_centroid_coords_mm`,
  `build_element_thermal_delta_T_from_bridge`, and
  `build_mechanics_thermal_kwargs_from_bridge`
- uniform source smoke check returns element-sized `DeltaT=20 K` for the
  two-element fixture patch
- H1 `solved_frozen_lift` smoke check samples
  `DeltaT(y)=20*eta` at element centroids, reports zero correction, and does
  not run network training
- mechanics smoke equivalence was checked by comparing direct element-sized
  `thermal_delta_T` against adapter-produced `thermal_kwargs`; selected thermal
  strain, elastic strain, stress, and energy fields matched with zero max
  difference
- no default mechanics/training mode was switched to solved temperature;
  `train_mixed_tm.py` remains unmodified
- this smoke adapter stage does not implement coupled heat-mechanics training,
  solved-temperature phase-field fracture coupling, heat PDE loss inside
  mechanics/fracture training, heat-fracture diagnostics, damage-dependent
  conductivity, D0040, seed study, shear extension, S0110, transient
  production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: add an explicit opt-in smoke script or small CLI wrapper
  that calls this adapter for one fixture mechanics patch and writes a compact
  result table, still without changing `train_mixed_tm.py` defaults or running
  broad training

Current solved-temperature mechanics smoke CLI status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli`
- final classification:
  `solved-temperature mechanics smoke cli implemented and tests passed`
- added an explicit opt-in smoke runner:
  `examples/TM_comsol_thermal_micro/run_solved_temperature_mechanics_smoke.py`
- added focused smoke CLI tests:
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_cli.py`
- smoke runner responsibilities are limited to constructing one fixed
  two-element mechanics patch, calling
  `thermal_mechanics_adapter.build_mechanics_thermal_kwargs_from_bridge` for
  H1 `solved_frozen_lift`, evaluating `compute_mixed_tm_fields`, and writing
  `mechanics_smoke_results.csv`
- generated smoke result table:
  `examples/TM_comsol_thermal_micro/runs/20260705_solved_temperature_mechanics_smoke_cli/tables/mechanics_smoke_results.csv`
- actual smoke result:
  source mode `solved_frozen_lift`, case `H1`, element-centroid sampling,
  `DeltaT` range `[6.666666666666686, 13.333333333333371] K`, selected-field
  mechanics max absolute difference `0.0`, and `network_training_run=false`
- the CLI is opt-in only; it is not imported by `train_mixed_tm.py` and no
  mechanics/training default was switched to solved temperature
- this smoke CLI stage does not implement coupled heat-mechanics training,
  solved-temperature phase-field fracture coupling, heat PDE loss inside
  mechanics/fracture training, heat-fracture diagnostics, damage-dependent
  conductivity, D0040, seed study, shear extension, S0110, transient
  production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: document the opt-in smoke runner in the thermal README and
  keep any guarded production call-site flag as a separate reviewed task

Current solved-temperature smoke README documentation status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260706_solved_temperature_smoke_readme_docs`
- final classification:
  `solved-temperature smoke README documentation implemented and tests passed`
- updated user-facing README:
  `examples/TM_comsol_thermal_micro/README.md`
- added focused README guard tests:
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_readme_docs.py`
- README now documents the explicit opt-in smoke command:
  `D:\anaconda3\envs\torch_env\python.exe examples\TM_comsol_thermal_micro\run_solved_temperature_mechanics_smoke.py --output-dir examples\TM_comsol_thermal_micro\outputs\solved_temperature_mechanics_smoke`
- README now documents `mechanics_smoke_results.csv`, `solved_frozen_lift`,
  `H1`, `element_centroid`, observed `DeltaT` range
  `[6.666666666666686, 13.333333333333371] K`,
  `mechanics_max_abs_diff = 0.0`, and `network_training_run = false`
- README now explicitly states that `train_mixed_tm.py remains unmodified` and
  that the smoke CLI is opt-in only
- documentation guard also verifies `train_mixed_tm.py` does not import
  `run_solved_temperature_mechanics_smoke` or reference
  `mechanics_smoke_results.csv`
- this README documentation stage does not implement coupled heat-mechanics
  training, solved-temperature phase-field fracture coupling, heat PDE loss
  inside mechanics/fracture training, heat-fracture diagnostics,
  damage-dependent conductivity, D0040, seed study, shear extension, S0110,
  transient production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: keep any guarded production call-site flag as a separate
  reviewed task with its own opt-in CLI/config tests before touching
  `train_mixed_tm.py`

Current solved-temperature train guard status:

- implementation package:
  `examples/TM_comsol_thermal_micro/runs/20260707_solved_temperature_train_guard`
- final classification:
  `solved-temperature train guard implemented and tests passed`
- added guarded default-off CLI/config flag:
  `--solved-temperature-mechanics-smoke`
- modified files:
  `examples/TM_comsol_thermal_micro/config.py`,
  `examples/TM_comsol_thermal_micro/train_mixed_tm.py`,
  `examples/TM_comsol_thermal_micro/README.md`,
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_train_guard.py`,
  and a narrow update to
  `examples/TM_comsol_thermal_micro/tests/test_solved_temperature_mechanics_smoke_adapter.py`
- `training_dict` now records the default-off boolean plus the fixed reviewed
  H1 smoke settings: `solved_frozen_lift`, `H1`, `element_centroid`, bounds
  `[[0.0, 0.01], [0.0, 0.01]]`, and temperatures `300/320/300 K`
- `train_mixed_tm.py` has no module-level import of
  `thermal_mechanics_adapter`; adapter import remains local inside the guarded
  helper
- the train hook runs only after fine-mesh `prep_input_data(...)` provides
  `inp` and `T_conn`, and before `training_set` creation
- when disabled, the helper returns the existing thermal kwargs unchanged and
  does not import the adapter
- when enabled, the helper validates fixed H1 settings, rejects prescribed
  thermal inputs (`thermal_temperature`, `thermal_delta_T`, or non-off
  `thermal_mode`), rejects malformed or non-fixed bounds, and records scalar
  diagnostics including `network_training_run=false`,
  `evaluation_location=element_centroid`, and `DeltaT` min/max
- focused train guard tests cover default config, opt-in config, conflict
  rejection, local import behavior, fixed-setting rejection, malformed bounds
  rejection, H1 centroid `DeltaT` range `[20/3, 40/3] K`, and unchanged
  `_thermal_energy_kwargs` defaults
- subagent spec review and code-quality review were used; final quality review
  approved after adding explicit bounds-shape validation
- this train guard stage does not implement coupled heat-mechanics training,
  solved-temperature phase-field fracture coupling, heat PDE loss inside
  mechanics/fracture training, heat-fracture diagnostics, damage-dependent
  conductivity, D0040, seed study, shear extension, S0110, transient
  production, or bottom-cooling production
- the original `examples/TM_comsol_no_thermal_micro` baseline remains
  untouched
- next safest task: run one tiny smoke invocation of `main.py` with
  `--solved-temperature-mechanics-smoke --smoke --n-rprop 1 --n-lbfgs 0
  --max-steps 1` and archive only scalar diagnostics, still without broad
  training

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
