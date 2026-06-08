## Codex handoff: exact FE target prefit diagnostic

Commit: 80c358b8a6f480e2cb1b19c264709dd77ed84e36
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_target_prefit/REPORT.md

### What changed
- Rejected the old FE-DOF RPROP target as a mechanics supervision target.
- Added an accepted direct sparse FE alpha=0 top-u-free target artifact.
- Added reusable mechanics target guard checker `validate_mechanics_target.py`.
- Ran global-only PINN prefit against the direct FE target using 8x400 network, seed 2, alpha fixed to zero.
- Ran short mechanics-only raw/log10/normalized energy continuation from the exact-target strain-prefit checkpoint.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe -m py_compile validate_mechanics_target.py debug_exact_fe_target_prefit.py

D:\anaconda3\envs\torch_env\python.exe debug_exact_fe_target_prefit.py --repo-root 'd:\Desktop\新建文件夹\cc-work' --out-dir 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_target_prefit' --delta 1e-6 --seed 2 --hidden-layers 8 --neurons 400 --prefit-epochs 800 --continuation-epochs 200

D:\anaconda3\envs\torch_env\python.exe validate_mechanics_target.py --candidate 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_mechanics_only_notch_ansatz\artifacts\fedof_free_log10_energy_e300_fields.npz' --exact 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_target_prefit\artifacts\exact_fe_topufree_alpha0_Delta1e-6_fields.npz' --out 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_target_prefit\tables\validate_old_fedof_against_exact.csv' --candidate-id rejected_old_FE_DOF_RPROP_free_log10_e300 --top-u-mode free --delta 1e-6

D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

gh --version
gh auth status
```

### Key results
- Old FE-DOF RPROP target is rejected: displacement relative RMSE `1832.902581161938`, strain relative RMSE `3001.860978848485`, standard energy ratio `11903720.214487167`, PINN mechanics energy ratio `10802507.428642381`, reaction `-201.55319141665106 N`, free residual L2 `0.03791821565707275`.
- Direct sparse FE target is accepted: standard energy `2.4232909011199782e-11`, PINN mechanics energy `3.67719431293434e-11`, top reaction `0.04846581802239973 N`, free residual L2 `1.7976327004734513e-18`, classification `notch-amplified`.
- Likely old RPROP failure cause: optimizer scale. Old RPROP `lr=1e-3` while `Delta=1e-6`, so `lr/Delta=1000`; exact FE has much lower objective than the old RPROP field.
- PINN displacement-only prefit can fit global displacement: displacement relative RMSE `0.01977395825088024`, but strain relative RMSE is `1.1590726375579834` and `He_current` correlation is only `0.07393151100870121`.
- PINN displacement-plus-strain prefit improves strain relative RMSE to `0.6868000030517578`, but `He_current` correlation remains low at `0.2563174710837941`.
- Energy continuation from exact-target strain prefit does not preserve exact-FE-like drive. Raw/log10/normalized variants all become `boundary-dominated` with `He_current` correlation near zero and energy ratios about `1.7x` to `1.8x` exact.
- Verification: `py_compile` passed; `examples\TM_comsol_no_thermal_micro\tests` passed with `13 passed`; full `tests -q` failed during collection because `ref_files.Chinese_SENT_reproduction` is missing.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/target_guard_check_summary.csv`
- `tables/fedof_rprop_audit.csv`
- `tables/exact_fe_target_summary.csv`
- `tables/exact_target_prefit_metrics.csv`
- `tables/exact_target_energy_continuation.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should the old FE-DOF RPROP target now be permanently treated as a negative control and excluded from future mechanics supervision?
2. Should the next Codex task replace mechanics pretraining targets with direct sparse FE targets and add guard checks before any target is accepted?
3. Given that global displacement prefit succeeds but strain/He reconstruction fails, what global-only strategy should be tested next before considering local enrichment or local weighting?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, enrichment, or geometry-label guidance from this package alone.
- Do not run coupled phase-field full training from this package alone.
- Do not claim physical validation from diagnostic runs.
