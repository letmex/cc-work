# Heat PDE Implementation and Validation Plan

## 1. Purpose

This planning package defines the next safe implementation path for adding a heat PDE branch to `examples/TM_comsol_thermal_micro` after the prescribed-temperature thermal-strain stage. It is documentation and validation planning only. It does not implement a heat equation, a trainable or PDE temperature field, damage-dependent conductivity, new boundary-condition code, new losses, or any training run.

## 2. Scope boundaries

Work is scoped to the thermal subproject. The original `examples/TM_comsol_no_thermal_micro` baseline remains frozen and untouched. Existing mechanics behavior, material constants, `l0`, TM split/history logic, phase-field route, reaction route, boundary conditions, and training losses are not changed by this package.

Allowed future work is staged heat PDE implementation after reviewer approval. Disallowed in the first implementation phase are `k(d)=g(d)k0`, thermomechanical heat generation, crack-surface heat transfer, COMSOL line-by-line matching, D0040, seed studies, shear thermal studies, S0110, and multi-pore or `comp4` work.

## 3. Current prescribed-temperature baseline status

The current reviewed thermal branch is prescribed-temperature mechanics only. It keeps `thermal_mode=off` as the default, supports prescribed absolute temperature or prescribed `delta_T`, and applies thermal strain before the existing TM split/history/energy route.

The trusted mechanics relation to preserve is:

```text
delta_T = T - Tref
exx_e = exx - alpha_T * delta_T
eyy_e = eyy - alpha_T * delta_T
exy_e = exy
```

Patch tests and diagnostics show that `thermal_mode=uniform` with `delta_T=0` reproduces the no-thermal thermal-subproject route in completed checks. Prescribed `+20 K` shifts displacement-controlled tension reaction/stress downward and reduces high-threshold/notch alpha in the moderate diagnostic. The broad low-level Case C alpha background remains diagnostic-only and is not physical fracture evidence.

## 4. Heat PDE target equation

The planned heat-transfer equation is:

```text
rho * c * dT/dt - div(k0 * grad T) = Q
```

The recommended first heat PDE implementation uses constant conductivity `k0` and starts with `Q=0`. It should not use damage-dependent conductivity. A steady-state patch-test route may use the corresponding constant-k0 steady residual before any transient term is enabled.

## 5. Unit system and conversion blockers

The main implementation blocker is a precise SI-to-project unit conversion convention. The current project geometry and mechanics use mm and kN/mm style quantities, and neural-network coordinates may be normalized to a unit box. The COMSOL transport constants are SI values: `rho = 1040 kg/m^3`, `c = 170 J/kg/K`, and `k0 = 418 W/m/K`.

This plan does not hand-wave that conversion. Before heat PDE implementation, a reviewer-approved table must define internal units for length, time, thermal energy or power, `rho*c`, `k0`, `Q`, heat flux, and the residual scale. Without that decision, the heat PDE implementation should remain blocked.

See `tables/thermal_variables_units.csv` for the unit and conversion inventory.

## 6. Temperature variable and representation options

The future temperature variable can be represented as a separate trainable output, a separate network, a deterministic analytical field for patch tests, or a staged hybrid. This package does not choose a trainable representation. It recommends using analytical or fixed fields for compatibility checks first, then adding a solved temperature representation only after unit conversion and heat residual patch tests are approved.

Any solved T representation must preserve the existing prescribed-temperature fallback and must be checkpointed and postprocessed without breaking `reaction_N_energy`.

## 7. Boundary and initial condition plan

First supported thermal boundary conditions should be minimal: prescribed Dirichlet temperature and insulated or constant Neumann heat flux, with explicit sign and unit conventions. First initial conditions should be uniform T, with linear initial T reserved for controlled patch tests. Time-dependent boundary temperature and nonzero `Q` are later-phase items.

See `tables/boundary_initial_condition_plan.csv` for the BC/IC staging plan.

## 8. First implementation phase recommendation

The next approved coding task should implement only constant-conductivity heat-transfer patch-test infrastructure, not coupled fracture diagnostics. The safest sequence is:

1. approve unit conversion convention;
2. create isolated heat PDE utilities for constant `k0`, `Q=0`;
3. validate constant T, linear 1D conduction, and insulated boundary checks;
4. only then route solved uniform T into the existing thermal-strain mechanics relation;
5. preserve `thermal_mode=off` default and prescribed-temperature fallback throughout.

## 9. Validation ladder

Validation must start with regressions: default-off no-thermal behavior and prescribed-temperature fallback. Heat residual tests should then proceed from constant T to linear steady conduction, insulated flux, transient uniform no-source, and manufactured transient checks if feasible. Mechanics coupling follows only after thermal residuals pass.

The full ladder is encoded in `tables/validation_matrix.csv`.

## 10. Patch test plan

The patch-test plan prioritizes closed-form checks with simple expected outputs: constant T residual zero, 1D linear conduction, insulated zero flux, transient uniform no-source, free expansion under solved uniform T, and constrained heating under solved uniform T.

See `tables/patch_test_plan.csv`.

## 11. Coupling strategy

First coupling should be one-way: solved T to the already reviewed thermal strain relation. Mechanics-to-heat feedback, heat-to-history direct coupling, and damage-to-conductivity coupling remain deferred. Checkpointing and postprocessing of solved T are separate dependencies and must not break the checkpointed energy-conjugate reaction route.

See `tables/coupling_dependency_plan.csv`.

## 12. Deferred features

Damage-dependent conductivity `k(d)=g(d)k0` is explicitly deferred. The same is true for thermomechanical heat generation, crack-surface heat transfer, temperature-dependent material properties, full COMSOL line-by-line matching, multi-pore `comp4`, D0040, seed studies, and shear thermal studies.

See `tables/deferred_features.csv`.

## 13. COMSOL comp3 alignment and non-alignment

The relevant COMSOL branch remains `comp3 / solid3 / ht3 / c / state3 / std1`. `comp4`, `solid2`, `ht2`, `c2`, `state4`, `TFinal`, and multi-pore settings are ignored. The PINN plan should align conceptually with comp3 constants and physics, but it does not require line-by-line COMSOL platform matching.

The eventual COMSOL relation `k_d = g(d) * k0` is acknowledged but not part of the first heat PDE phase. See `tables/comsol_alignment_notes.csv`.

## 14. Source touch plan

This package modifies no behavior source code. Future implementation is expected to touch an isolated new heat PDE module, `config.py`, `train_mixed_tm.py`, `compute_energy_mixed_tm.py`, `history_field_mixed_tm.py`, `postprocess_results.py`, focused tests, and documentation only after the relevant gates pass.

See `tables/source_touch_plan.csv`.

## 15. Risks and blockers

The highest risk is unit conversion from SI thermal constants into the current mm/kN mechanics code. Other material risks are heat residual scaling, PINN instability from adding a T field, conflating heat PDE with damage-dependent conductivity, boundary-condition ambiguity, transient time-scale ambiguity, checkpoint compatibility, low-level alpha-background misinterpretation, and COMSOL convention mismatch.

See `tables/risk_register.csv`.

## 16. Decision gate

Before implementation, the reviewer should decide whether to approve Phase 1 heat PDE implementation, whether steady-state should precede transient, which solved-temperature representation to use, the exact unit-conversion convention, whether to keep damage-dependent conductivity deferred, and whether the prescribed-temperature fallback remains frozen.

See `tables/next_decision_gate.csv`.

## 17. Final classification

`heat PDE implementation plan complete`

This plan recommends adding heat PDE support in a staged way, starting with constant-conductivity heat transfer and patch tests, while preserving the prescribed-temperature branch and the no-thermal default route. Damage-dependent conductivity `k(d)=g(d)k0` should remain deferred until the constant-conductivity heat solve and solved-temperature-to-thermal-strain coupling are independently validated. The main implementation blocker is a precise unit-conversion convention for SI thermal constants in the current mm/kN mechanics code.

## 18. Exact next recommended task

Hold a reviewer decision-gate review of this package. If approved, implement only Phase 1 constant-conductivity heat PDE/unit-conversion infrastructure and patch tests in `examples/TM_comsol_thermal_micro`; do not implement damage-dependent conductivity, run D0040, run seed studies, run shear thermal studies, or modify `examples/TM_comsol_no_thermal_micro`.
