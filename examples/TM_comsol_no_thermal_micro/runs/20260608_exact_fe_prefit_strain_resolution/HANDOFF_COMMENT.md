## Codex handoff: exact FE prefit strain-resolution diagnostic

Commit: 3821798166f1bbf7f09730a9d02d39693a28b46b
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_prefit_strain_resolution
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_prefit_strain_resolution/REPORT.md

### What changed
- Added `debug_exact_fe_prefit_strain_resolution.py` to audit why exact-FE displacement prefit does not necessarily reconstruct local strain and `He_current`.
- Used only the accepted direct sparse FE alpha=0 top-u-free target.
- Marked the old FE-DOF RPROP field as a rejected negative control.
- Tested global displacement, normalized displacement, normalized global strain-loss sweep, coordinate-scaled input, 10x500 capacity, and TrainableTanh variants without notch/lip geometry labels in the loss.
- Ran short mechanics-only energy continuation only from the successful coordinate-scaled prefit.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_prefit_strain_resolution\artifacts\debug_exact_fe_prefit_strain_resolution.py

D:\anaconda3\envs\torch_env\python.exe -X faulthandler examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_prefit_strain_resolution\artifacts\debug_exact_fe_prefit_strain_resolution.py --out-dir examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_prefit_strain_resolution --epochs 200 --capacity-epochs 100 --continuation-epochs 100

D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

### Key results
- Target guard accepts `accepted_direct_sparse_FE_topufree_alpha0_Delta1e-6` and rejects `rejected_old_FE_DOF_RPROP_free_log10_e300`.
- T3 target strain recomputation matches stored exact target strain to machine precision; baseline connectivity/order matches the target.
- Baseline 8x400 displacement-only prefit has good displacement RMSE (`0.01977`) but poor strain RMSE (`1.15907`) and poor `He_current` correlation (`0.07393`).
- Coordinate-scaled 8x400 TrainableReLU with global normalized displacement+strain loss reaches exact-target-like metrics: displacement RMSE `0.000621`, strain RMSE `0.009459`, strain corr `0.999952`, `He_current` corr `0.999977`, energy ratios about `1.0001`.
- 10x500 capacity and TrainableTanh without coordinate scaling do not solve the mismatch.
- Short energy continuation from the successful coordinate-scaled prefit keeps the max `He_current` at the notch and `He_current` corr above `0.993`, but strain RMSE drifts to about `0.274-0.278` and reaction ratio drops to about `0.76`, so it is not exact-target-like after continuation.
- Local project tests passed: `13 passed in 0.07s`.
- Full `pytest tests -q` still fails during collection because `ref_files.Chinese_SENT_reproduction` is missing in six Chinese_SENT test modules.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/prefit_variant_comparison.csv`
- `tables/coordinate_and_connectivity_checks.csv`
- `tables/element_quality_error_correlation.csv`
- `tables/optional_energy_continuation.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should coordinate input normalization be promoted as the next controlled mechanics ansatz implementation fix?
2. Before any coupled phase-field run, should Codex rerun exact-target prefit and longer mechanics-only energy continuation with coordinate normalization enabled in the normal training path?
3. Does the remaining post-continuation reaction/strain drift point to energy-loss scaling, optimizer trajectory, or another mechanics-only issue?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not add notch/lip loss, local masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance unless explicitly requested.
- Do not run coupled phase-field full training yet.
- Do not claim physical validation from this diagnostic.
