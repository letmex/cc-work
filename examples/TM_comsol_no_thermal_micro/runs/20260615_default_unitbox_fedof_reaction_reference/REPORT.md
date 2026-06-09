# FE-DOF / energy-conjugate reaction reference audit

## Scope

This package builds a diagnostic finite-dimensional mechanics reference on the existing final_D0040 mesh using frozen alpha fields from seeds 7, 13, and 42. It does not extend loading, evolve alpha, change `l0`, change material constants, change TM split/history logic, or run a PINN split-domain replay.

The FE-DOF implementation is a linear CST plane-stress diagnostic with scalar frozen-alpha stiffness degradation. It is intended to audit reaction definitions and boundary sensitivity, not to replace the production mixed-driving mechanics formulation.

## Key summary

- Classification: **FE-DOF reference unresolved: energy-relaxed crack-band-void reaction collapses and does not reproduce persistent PINN reaction**.
- FE solve status: all requested solves solved = True.
- Continuous current-split with original top/bottom BC, top sigma-integral reactions [N]: [0.007959751516285682, 0.007461224731703281, 0.007362949225679203].
- Continuous current-split with original top/bottom BC, energy-conjugate reactions [N]: [0.00796929081143486, 0.007469936758453541, 0.007373852857067058].
- Continuous crack-band-void with original top/bottom BC, top sigma-integral reactions [N]: [-2.498043896949281e-14, 2.270166053847394e-15, -2.1801969067599804e-13].
- Continuous crack-band-void with original top/bottom BC, energy-conjugate reactions [N]: [-1.713039432527097e-14, 2.6129272356900657e-14, -2.2714035513571318e-13].
- Piecewise upper/lower crack-band-void with minimal rigid-body BC, energy-conjugate reactions [N]: [8.847089727481716e-14, -2.0079424234431542e-13, 1.43982048506075e-13].
- Max |piecewise/minimal energy-conjugate reaction| [N]: 2.00794e-13.
- Continuous void/original-BC nonzero top-reaction votes: 0/3 seeds.

## Answers

1. The FE-DOF frozen-alpha reference does not reproduce the previous nonzero post-crack top-boundary stress-integral reaction once the crack band is voided. Current-split cases retain a small nonzero reaction, while crack-band-void cases collapse to numerical zero in this reference solve.
2. The energy-conjugate generalized reaction also collapses in the continuous and piecewise crack-band-void references across the audited seeds.
3. In this FE-DOF reference, top-boundary stress integral and energy-conjugate reaction are consistent for current-split loading and both collapse after crack-band voiding. This does not confirm a top-vs-energy metric disagreement inside the FE-DOF reference itself.
4. Minimal rigid-body boundary treatment is still useful as a sensitivity check, but the original top/bottom BC already permits reaction collapse after crack-band voiding in the FE-DOF reference.
5. The previous PINN/split-domain residual reactions are therefore more consistent with saved-u/v branch, PINN ansatz, or replay relaxation effects than with a residual cracked-ligament load path in an energy-relaxed FE-DOF reference.
6. `reaction_N_tm_eff` should not be used alone to judge post-peak softening after through-crack formation.
7. Future stress-strain curves should report an energy-conjugate generalized reaction or constrained-DOF reaction, and should include bottom reaction / all-boundary residual checks for post-crack states.
8. No production physics change is justified directly from this reference audit.
9. Next minimal intervention: compute an energy-conjugate or constrained-DOF reaction on the actual saved PINN/replay branch and compare it with the legacy top-boundary sigma integral before changing mechanics or alpha evolution.

## Implementation limitations

- The FE-DOF reference uses a scalar frozen-alpha stiffness approximation rather than the exact nonlinear TM positive/negative spectral split tangent.
- The minimal rigid-body boundary treatment is diagnostic and not a proposed production boundary condition.
- Results are evidence about reaction definitions and boundary constraints, not physical validation of crack behavior.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260615_default_unitbox_fedof_reaction_reference\artifacts\run_fedof_reaction_reference.py`: passed.
