# Exact alpha=0 FE mechanics audit

## Scope

This audit isolates the mechanics subproblem at `alpha=0` and `Delta = 1e-6`. It assembles a direct sparse P1/T3 plane-stress FE elasticity solve on the same mesh and boundary sets, then recomputes strain, mixed-mode postprocessing quantities, reactions, residuals, and energy decomposition for existing diagnostic fields.

No coupled phase-field full training was run. The audit does not modify `l0`, material parameters, `tm_source` split, phase-field notch behavior, alpha seeding, thermal terms, or history update logic.

## Exact FE baseline

The direct solve completed for both top-u-free and top-u-fixed variants. The top-u-free result is the priority baseline:

- standard internal energy: `2.4232909011199782e-11`
- current PINN mechanics energy: `3.67719431293434e-11`
- reaction top v: `0.04846581802239973 N`
- free-DOF residual L2: `1.7976327004734513e-18`
- max `He_current`: `4.646703018806875e-05`
- max `He_current` location: `(0.005021084355784554, 0.00499864349766533)`
- bulk/notch `He_current` ratio: `0.020359886917060196`
- bottom/notch `He_current` ratio: `0.04139856029550387`
- classification: `notch-amplified`

The top-u-fixed direct solve remains close to top-u-free:

- displacement relative RMSE vs top-u-free exact FE: `0.029431930862003085`
- strain relative RMSE: `0.06672102163110771`
- `He_current` correlation: `0.9994083343614097`
- classification: `notch-amplified`

This indicates that the exact alpha=0 mechanics baseline is not a broad/background drive field. It is notch-amplified at the explicit notch under the direct FE solve.

## Existing FE-DOF RPROP target

The existing FE-DOF RPROP free target is not close to the direct FE solution:

- displacement relative RMSE vs exact FE: `1832.902581161938`
- strain relative RMSE vs exact FE: `3001.860978848485`
- `He_current` correlation vs exact FE: `0.20084873839771597`
- standard internal energy: `0.0002884617688524471`
- standard energy ratio vs exact FE: `11903720.214487167`
- current PINN mechanics energy: `0.0003972291888203472`
- PINN energy ratio vs exact FE: `10802507.428642381`
- reaction top v: `-201.55319141665106 N`
- free-DOF residual L2: `0.03791821565707275`
- classification: `boundary-dominated`

This field should not currently be treated as the true linear elastic FE minimizer for the alpha=0 mechanics subproblem. The evidence points to a FE-DOF RPROP target validity issue, likely in boundary handling, scale, objective evaluation, or optimizer convergence for that diagnostic.

## Supervised PINN prefit fields

The supervised PINN prefit fields are close to the FE-DOF RPROP branch, not to the direct FE solution.

`pinn_prefit_disp_global`:

- displacement relative RMSE vs exact FE: `1833.2475346834515`
- strain relative RMSE vs exact FE: `3039.814884194919`
- standard energy ratio vs exact FE: `12046994.814370606`
- current PINN energy ratio vs exact FE: `10995418.349963877`
- reaction top v: `-204.29753644400517 N`

`pinn_prefit_disp_strain`:

- displacement relative RMSE vs exact FE: `1829.7429319245114`
- strain relative RMSE vs exact FE: `2993.9870732834393`
- standard energy ratio vs exact FE: `11854237.71228624`
- current PINN energy ratio vs exact FE: `10771701.104677316`
- reaction top v: `-201.55670114523252 N`

These results mean the prefit experiment did not prove that the current PINN ansatz cannot represent the exact FE mechanics branch. It mainly showed that the network can fit the high-energy FE-DOF RPROP target when supervised to do so.

## Collapsed PINN energy-continuation fields

The collapsed/energy-continuation fields are much closer to exact FE than the FE-DOF RPROP target in energy and residual terms.

`pinn_collapsed_pure_energy_log10`:

- displacement relative RMSE vs exact FE: `0.12206737304923156`
- strain relative RMSE vs exact FE: `0.5469429898505485`
- standard energy ratio vs exact FE: `1.2682575316331386`
- current PINN energy ratio vs exact FE: `1.1690362608339007`
- reaction top v: `0.046216834264280596 N`
- free-DOF residual L2: `2.920270854828983e-05`

`pinn_collapsed_energy_normalized`:

- displacement relative RMSE vs exact FE: `0.13916368450135955`
- strain relative RMSE vs exact FE: `0.5490109564656179`
- standard energy ratio vs exact FE: `1.2676652930363357`
- current PINN energy ratio vs exact FE: `1.1686940519914018`
- reaction top v: `0.050178962434244655 N`
- free-DOF residual L2: `2.0786548645814934e-05`

These fields are not exact FE solutions, and their `He_current` localization can differ from the exact FE postprocessing. However, their mechanics energies and reactions are orders of magnitude more consistent with the direct FE baseline than the FE-DOF RPROP target or supervised prefit branch.

## Energy formulation audit

The ranking under standard linear elastic internal energy and the current PINN alpha=0 mechanics energy is consistent for the compared fields:

- exact FE free: standard energy `1.0x`, current PINN mechanics energy `1.0x`
- exact FE fixed: standard energy `1.0229185259893885x`, current PINN mechanics energy `1.0225006795586402x`
- collapsed pure energy: standard energy `1.2682575316331386x`, current PINN mechanics energy `1.1690362608339007x`
- FE-DOF RPROP: standard energy `11903720.214487167x`, current PINN mechanics energy `10802507.428642381x`
- supervised prefit fields: about `1.18e7` to `1.20e7x` exact standard energy and about `1.08e7` to `1.10e7x` exact current PINN mechanics energy

The collapsed PINN fields do not have lower mechanics energy than the exact FE direct solve. They are higher than exact FE but far lower than the FE-DOF RPROP and supervised prefit fields.

For alpha=0, the current PINN mechanics energy tracks the standard FE energy ranking reasonably in this audit. The larger discrepancy is not that the collapsed PINN branch is falsely lower than exact FE. The larger discrepancy is that the FE-DOF RPROP target is extremely high energy relative to the direct FE solution.

## Boundary and reaction checks

The exact FE solves satisfy constrained displacements and have near-machine-precision free residuals. The top and bottom vertical reactions are balanced in sign and magnitude:

- exact FE free top v reaction: `0.04846581802239973 N`
- exact FE free bottom v reaction: `-0.048465818022399694 N`

The FE-DOF RPROP and supervised prefit fields have large negative top reactions and much larger free residuals:

- FE-DOF RPROP top v reaction: `-201.55319141665106 N`
- FE-DOF RPROP free residual L2: `0.03791821565707275`
- prefit global top v reaction: `-204.29753644400517 N`
- prefit global free residual L2: `0.053540398142271094`

This is incompatible with treating those fields as accurate alpha=0 linear elastic mechanics solutions under the audited FE assumptions.

## Answers to requested questions

1. The exact alpha=0 FE elasticity solve is `notch-amplified`, not broad or boundary-dominated.
2. The FE-DOF RPROP target is not close to exact FE. It has displacement/strain relative RMSE above `1e3`, energy ratios near `1e7`, and large residual/reaction inconsistencies.
3. The supervised PINN prefit is close to the FE-DOF RPROP target branch, not to exact FE.
4. The collapsed PINN field is not lower energy than exact FE under the same audit. It is about `1.17x` to `1.27x` exact energy, while the FE-DOF/prefit fields are about `1e7x` exact energy.
5. The current PINN mechanics energy is broadly consistent with standard linear elastic internal energy ranking for alpha=0 in this audit.
6. Boundary constraints and reaction signs are consistent for exact FE, but not for FE-DOF RPROP and supervised prefit fields.
7. The next step should target FE-DOF target validity, especially the RPROP nodal-DOF mechanics baseline, boundary constraints, displacement scale, and objective implementation. The next step should not preserve the FE-DOF RPROP target as a trusted supervision target.

## What this audit cannot conclude

- It does not validate the coupled phase-field full-run behavior.
- It does not prove the final physical crack path.
- It does not test new notch/lip loss, local weighting, enrichment, or displacement-jump supervision.
- It does not justify changing `l0`, material parameters, `tm_source` split, phase-field notch behavior, alpha seeding, thermal terms, or history update logic.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_exact_fe_elastic_solve.py` passed in `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q` passed in `D:\ProgramData\PINN\FEM-PINN-main`: `13 passed in 0.07s`.
- The same pytest command failed in the `cc-work` evidence repository because `examples\TM_comsol_no_thermal_micro\tests` is not present there.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q` failed in `D:\ProgramData\PINN\FEM-PINN-main` during test collection because `ref_files.Chinese_SENT_reproduction` is missing. The failing files were `tests/test_chinese_sent_boundary_ansatz.py`, `tests/test_chinese_sent_energy_terms.py`, `tests/test_chinese_sent_mesh_adapter.py`, `tests/test_chinese_sent_postprocess_contract.py`, `tests/test_chinese_sent_reaction_force.py`, and `tests/test_chinese_sent_training_config.py`.
