# Boundary/reaction/free-body audit

## Scope

This package audits the final_D0040 split-domain replay fields from seeds 7, 13, and 42. It does not extend loading, evolve alpha, change material parameters, change `l0`, change TM split, or retrain a production model.

## Key summary

- Classification: **reaction/boundary cause identified: boundary ansatz overconstrains separated subdomains and top reaction is a local boundary stress metric**.
- Crack-band-void replay top reactions [N]: [0.8748153152888245, 0.7423446550484991, 0.3961012804957055].
- Piecewise rigid upper/lower synthetic field, crack-band-void treatment, max |top reaction| [N]: 2.14039e-15.
- Zero-displacement-reference synthetic field mean top reaction [N]: 16.1607.
- Saved-u/v synthetic field mean top reaction under void treatment [N]: 0.795961.
- Mean void-replay upper/lower subdomain residual magnitude [N]: 2.30777.

## Answers

1. `reaction_N_tm_eff` is conjugate to the imposed vertical top displacement in the narrow sense that it integrates `sigma_yy_tm_eff` over the top boundary.
2. It is a top-boundary local stress integral; by itself it does not prove global load transfer across the cracked ligament.
3. After crack-band voiding, upper/lower subdomain free-body residuals are nonzero; see `tables/subdomain_free_body_audit.csv`.
4. Remaining force is concentrated in physical boundary and subdomain residual terms rather than in the voided crack band; see `tables/all_boundary_force_balance.csv` and `tables/reaction_vs_internal_cut_consistency.csv`.
5. Internal cut forces do not consistently explain the top-boundary reaction across all seeds and variants.
6. Synthetic fields show that boundary constraints can force stress in the zero-displacement-reference field, while a piecewise rigid upper/lower field nearly removes the top reaction under crack-band void treatment.
7. The prefit-to-saved-u/v step is a plausible contributor because saved-u/v synthetic fields retain nonzero top reaction under crack-band void treatment.
8. Current audit decision: **reaction/boundary cause identified: boundary ansatz overconstrains separated subdomains and top reaction is a local boundary stress metric**.
9. No production model change is justified directly from this diagnostic.
10. Next minimal intervention: ask ChatGPT to decide whether to audit the top/bottom displacement ansatz and reaction definition against a FE-DOF free-body calculation, before changing physics.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260614_default_unitbox_boundary_reaction_audit\artifacts\run_boundary_reaction_audit.py`: passed.
