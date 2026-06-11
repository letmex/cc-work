# Existing Project Physical-Core Audit

## 1. Purpose

This audit checks whether the existing no-thermal PINN phase-field fracture route has a clean, physically interpretable core that can serve as the baseline for later thermal coupling. It does not attempt to make the PINN implementation identical to COMSOL line by line.

## 2. Existing no-thermal PINN route reviewed

The reviewed route is `mixedH_TM + tm_source + history` with AT2 phase field, default alpha initialization, top-u/free tension ansatz, the existing top-v-free shear ansatz, unit-box coordinate normalization, and checkpointed energy-conjugate reaction as the primary reaction route.

## 3. Relevant COMSOL branch

The only COMSOL reference branch used here is `comp3 / solid3 / ht3 / c / state3 / std1`. The COMSOL facts are treated as theoretical reference facts supplied by the task prompt, not as a requirement that every implementation detail must match.

## 4. Why comp4 is ignored

`comp4`, multi-pore settings, and `TFinal` are explicitly outside this single-notch audit. Mixing them into this baseline would create an invalid reference for the current no-thermal route.

## 5. Material parameters to keep unchanged

- `E0 = 81.5 GPa` / `81.5 kN/mm^2`
- `nu = 0.38`
- `Gf0 = 0.0024 MPa*mm` / `2.4e-6 kN/mm`
- `kappa = 1e-5`
- `l0 = 0.15 um` / `1.5e-4 mm`
- `eps_r = 1e-5`
- future thermal constants: `alpha_T=18.9 ppm/K`, `rho=1040 kg/m^3`, `k0=418 W/m/K`, `c=170 J/kg/K`, `Tref=273.15 K`, `T0=0 degC`

## 6. Same physical core assessment

The current no-thermal PINN route shares the required physical core for a baseline: material units, AT2 model class, `l0`, residual stiffness convention, mixed `psiI/psiII` drive, max-history irreversibility, degradation of crack-driving energy, and energy-conjugate reaction are internally consistent. The main caveat is the constitutive convention: COMSOL comp3 is documented as plane stress, while the current PINN uses 2D fields with `eps_zz` reconstruction and a Lame helper that should be documented and patch-tested before thermal strain is trusted.

## 7. Acceptable platform implementation differences

Acceptable differences include COMSOL external-stress degradation versus PINN energy degradation, FEM PDE solve versus neural energy/residual optimization, COMSOL boundary constraints versus PINN ansatz constraints, COMSOL heat transfer being active while the current baseline is intentionally no-thermal, and COMSOL postprocess reactions versus PINN checkpoint energy derivatives.

## 8. Unacceptable differences before thermal coupling

High-risk differences to avoid later are wrong material units, wrong `l0` units, using thermal strain as `alpha_T*T` instead of `alpha_T*(T-Tref)`, silently degrading compressive stress without justification, losing max-history irreversibility, reintroducing legacy top-sigma as the primary reaction metric, changing the shear ansatz, and mixing comp4/TFinal into the single-notch route.

## 9. Exact COMSOL matching

Exact line-by-line COMSOL matching is not required. Physical-model invariants must match, while solver mechanics, state storage, boundary enforcement, and postprocess extraction may differ when the physical meaning is preserved and documented.

## 10. Baseline classification

Final classification: `no-thermal physical core acceptable as thermal baseline with documented platform differences`.

The existing no-thermal PINN route is acceptable as the baseline for thermal reintroduction if the AT2 phase-field class, mixed history concept, degradation logic, material units, `l0`, and energy-conjugate reaction route remain internally consistent. Exact COMSOL implementation matching is not required. Differences caused by FEM versus PINN formulations are acceptable when the physical meaning is preserved and documented.

## 11. What must not change before thermal reintroduction

Do not change source physics, boundary conditions, shear ansatz, material parameters, `l0`, history logic, alpha initialization, training losses, reaction policy, or load schedules as part of this audit baseline. Do not run D0040 or a seed study as a substitute for thermal patch tests.

## 12. Safest next step

The safest next task is a prescribed-temperature thermal-strain branch with patch tests, not a full heat-equation/damage-conductivity coupling. Start with zero-damage thermoelastic tests for uniform free expansion, constrained thermal stress, zero-DeltaT equivalence to the current no-thermal route, and the `T-Tref` convention.

## Constraints Observed

- no source code was modified;
- no physics was changed;
- no boundary condition was changed;
- no shear ansatz was changed;
- no material parameter was changed;
- no `l0` was changed;
- no history logic was changed;
- no training loss was changed;
- no new training was run;
- no D0040 was run;
- no seed study was run;
- no heat PDE was implemented;
- no COMSOL exact-matching requirement was imposed.

## Evidence Status

- repo_memory: present
- project_memory: present
- current_config: present
- current_energy: present
- current_split: present
- current_history: present
- current_fields: present
- current_postprocess: present
- current_material: present
- current_pff: present
- current_readme: present
- postprocess_workflow: present
- project_structure: present
- s0090_handoff: present
- stress_audit_handoff: present
