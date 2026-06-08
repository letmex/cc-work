# Exact FE target prefit and RPROP target audit

## Scope

This diagnostic uses the direct sparse FE alpha=0 top-u-free mechanics solution as the accepted target for `Delta = 1e-6`. It audits the previous FE-DOF RPROP target, reruns global-only PINN supervised prefit against the direct FE target, and performs short mechanics-only energy continuation from the exact-target strain-prefit checkpoint.

The run keeps the physical setup fixed: same mesh, same material constants, same `l0`, same `tm_source` split, alpha fixed to zero, no coupled phase-field training, no notch/lip loss, no local masks, no local weights, no enrichment, and no geometry-label guidance.

## Old FE-DOF RPROP target is rejected

The old target from `examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz/artifacts/fedof_free_log10_energy_e300_fields.npz` is rejected as a mechanics supervision target.

Guard check against the direct FE reference:

- displacement relative RMSE: `1832.902581161938`
- strain relative RMSE: `3001.860978848485`
- standard energy ratio: `11903720.214487167`
- PINN mechanics energy ratio: `10802507.428642381`
- top reaction: `-201.55319141665106 N`, while exact FE gives `+0.04846581802239973 N`
- free residual L2: `0.03791821565707275`, while exact FE gives `1.7976327004734513e-18`
- accepted target: `False`

The old field should only be used as a negative control unless a future task explicitly asks to audit that failed branch further.

## Why the old FE-DOF RPROP target failed

The audit did not find evidence that the old field was caused by an intentional boundary-condition change, objective sign reversal, or unit change. The old scripts apply bottom displacement constraints and top vertical displacement through a nodal transformation, and the direct FE audit uses the same mesh scale, `E = 81.5`, `nu = 0.38`, and `Delta = 1e-6`.

The primary failure mode is that the old FE-DOF RPROP baseline was an optimizer diagnostic, not a direct linear FE solve. Its default RPROP step size was `lr = 1e-3`, while the load scale was `Delta = 1e-6`, giving `lr/Delta = 1000`. Because RPROP uses sign-based steps, this is a poor scale for tiny nodal displacement DOFs. The exact FE field has much lower objective value than the old RPROP field:

- exact FE `log10(E)` under current PINN mechanics energy: `-10.4345`
- old RPROP `log10(E)`: `-3.40096`

Therefore the old RPROP field is not a lower-energy solution. It is a high-energy, high-residual optimizer artifact.

## Accepted direct sparse FE target

The accepted target is saved as:

`artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz`

Key target metrics:

- standard energy: `2.4232909011199782e-11`
- PINN mechanics energy: `3.67719431293434e-11`
- top reaction: `0.04846581802239973 N`
- bottom reaction: `-0.048465818022399694 N`
- free residual L2: `1.7976327004734513e-18`
- max `He_current`: `4.646703018806875e-05`
- max `He_current` location: `(0.005021084355784554, 0.00499864349766533)`
- bulk/notch `He_current` ratio: `0.020359886917060196`
- bottom/notch `He_current` ratio: `0.04139856029550387`
- classification: `notch-amplified`

The direct FE target passes all guard checks in `target_guard_check_summary.csv`.

## PINN prefit against direct FE target

Two global-only supervised prefit variants were run with the same 8x400 network and seed 2:

1. global displacement MSE
2. global displacement MSE plus global strain MSE with `strain_weight = 1e-5`

The displacement-only prefit fits the global displacement field well:

- displacement relative RMSE: `0.01977395825088024`
- `u_corr`: `0.9988618660643903`
- `v_corr`: `0.9994247454981122`
- strain relative RMSE: `1.1590726375579834`
- strain correlation: `0.634741505471522`
- `He_current` correlation: `0.07393151100870121`
- standard energy ratio vs exact: `2.045624326081702`
- PINN mechanics energy ratio vs exact: `2.0925167785646495`
- classification: `notch-amplified`, but max `He_current` is not at the exact notch location

The displacement-plus-strain prefit improves strain relative to displacement-only, but does not reconstruct the exact local drive:

- displacement relative RMSE: `0.3818899691104889`
- strain relative RMSE: `0.6868000030517578`
- strain correlation: `0.7825361279017198`
- `He_current` correlation: `0.2563174710837941`
- standard energy ratio vs exact: `1.9611950022562719`
- PINN mechanics energy ratio vs exact: `1.9477557393102667`
- classification: `other`

Conclusion: the current PINN ansatz can fit the exact FE displacement field globally, but global displacement fitting alone is not enough to reproduce the exact strain and `He_current` field. Adding the current global strain term improves strain metrics but still does not make the field exact-FE-like.

## Energy continuation from exact-target prefit

Starting from the displacement-plus-strain prefit checkpoint, three short mechanics-only continuation variants were run:

- raw energy
- log10 energy
- normalized energy

All three move toward a similar boundary-dominated branch:

| mode | standard energy ratio | PINN energy ratio | displacement rel RMSE | strain rel RMSE | He corr | reaction N | classification |
|---|---:|---:|---:|---:|---:|---:|---|
| raw | `1.8064347597606056` | `1.7101990217563494` | `0.3988787531852722` | `0.8754311800003052` | `0.022490745873069592` | `0.07989397396484685` | `boundary-dominated` |
| log10 | `1.770437445302307` | `1.7106252967909263` | `0.40112170577049255` | `0.861468493938446` | `-0.0058141753893173505` | `0.08073773576507202` | `boundary-dominated` |
| normalized | `1.7927911885646346` | `1.7105547227037499` | `0.4015549123287201` | `0.8718274831771851` | `0.04534709053378593` | `0.07856256388595736` | `boundary-dominated` |

Energy continuation does not preserve an exact-FE-like `He_current` field under these global-only conditions.

## Energy consistency

For alpha=0, standard FE energy and current PINN mechanics energy remain broadly consistent as ranking diagnostics:

- exact FE is the accepted lowest-energy reference.
- old FE-DOF RPROP is about `1e7` times exact under both energy measures.
- exact-target PINN prefit and continuation are about `1.7` to `2.1` times exact, much closer than the old target but not exact.

This supports using the direct sparse FE field as the mechanics supervision target and rejecting the old FE-DOF RPROP field.

## Answers to requested questions

1. The old FE-DOF RPROP target failed because it was a high-energy RPROP optimizer artifact. The strongest identified cause is optimizer scale: `lr = 1e-3` for `Delta = 1e-6`, plus sign-based RPROP updates on tiny nodal DOFs.
2. The direct sparse FE target is accepted by all guard checks.
3. PINN can prefit the direct sparse FE target in global displacement, but it does not accurately reconstruct strain and `He_current`.
4. Energy continuation does not preserve the exact-FE-like local drive; it moves to a boundary-dominated field.
5. Current PINN mechanics energy and standard FE energy are consistent enough for target ranking at alpha=0.
6. Going forward, mechanics pretraining should use the direct sparse FE target or a direct-solve-derived target, not the old FE-DOF RPROP target.
7. This package cannot validate a physical crack path, justify changing model physics, or prove coupled phase-field full-run behavior.

## Verification status

Verification commands are listed in `commands_run.txt`.

- `D:\anaconda3\envs\torch_env\python.exe -m py_compile validate_mechanics_target.py debug_exact_fe_target_prefit.py` passed.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q` passed in `D:\ProgramData\PINN\FEM-PINN-main`: `13 passed in 0.07s`.
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q` failed during collection because `ref_files.Chinese_SENT_reproduction` is missing. The failing files were `tests/test_chinese_sent_boundary_ansatz.py`, `tests/test_chinese_sent_energy_terms.py`, `tests/test_chinese_sent_mesh_adapter.py`, `tests/test_chinese_sent_postprocess_contract.py`, `tests/test_chinese_sent_reaction_force.py`, and `tests/test_chinese_sent_training_config.py`.
