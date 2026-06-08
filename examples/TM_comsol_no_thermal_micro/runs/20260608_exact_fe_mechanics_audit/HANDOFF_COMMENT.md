## Codex handoff: exact alpha=0 FE mechanics audit

Commit: PENDING_COMMIT_SHA
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_mechanics_audit
Main report: examples/TM_comsol_no_thermal_micro/runs/20260608_exact_fe_mechanics_audit/REPORT.md

### What changed
- Added an evidence package for a direct sparse FE alpha=0 mechanics audit at `Delta = 1e-6`.
- Compared exact FE top-u-free/top-u-fixed baselines against FE-DOF RPROP, supervised PINN prefit, and collapsed PINN energy-continuation fields.
- Added exact/candidate tables for mechanics field comparison, energy decomposition, residuals, and boundary reactions.
- Included the diagnostic script snapshot in `artifacts/debug_exact_fe_elastic_solve.py`.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe debug_exact_fe_elastic_solve.py --repo-root 'd:\Desktop\新建文件夹\cc-work' --out-dir 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_mechanics_audit' --delta 1e-6

Copy-Item -LiteralPath 'D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\debug_exact_fe_elastic_solve.py' -Destination 'd:\Desktop\新建文件夹\cc-work\examples\TM_comsol_no_thermal_micro\runs\20260608_exact_fe_mechanics_audit\artifacts\debug_exact_fe_elastic_solve.py' -Force

D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_exact_fe_elastic_solve.py

D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

gh --version
gh auth status
```

### Key results
- Exact FE top-u-free is `notch-amplified`: max `He_current = 4.646703018806875e-05` at `(0.005021084355784554, 0.00499864349766533)`.
- Exact FE top-u-free has standard internal energy `2.4232909011199782e-11`, current PINN mechanics energy `3.67719431293434e-11`, top reaction `0.04846581802239973 N`, and free residual L2 `1.7976327004734513e-18`.
- FE-DOF RPROP target is not close to exact FE: displacement relative RMSE `1832.902581161938`, strain relative RMSE `3001.860978848485`, standard energy ratio `11903720.214487167`, PINN mechanics energy ratio `10802507.428642381`, top reaction `-201.55319141665106 N`, free residual L2 `0.03791821565707275`.
- Supervised PINN prefit fields are close to the FE-DOF RPROP branch, not exact FE: energy ratios remain near `1e7` and reactions are about `-201` to `-204 N`.
- Collapsed PINN energy-continuation fields are much closer to exact FE in energy than FE-DOF/prefit fields: about `1.17x` to `1.55x` current PINN mechanics energy, not `1e7x`.
- The collapsed fields are not lower energy than exact FE. The direct FE solution remains the lowest-energy audited mechanics field.
- Current PINN mechanics energy broadly tracks standard linear elastic energy ranking for alpha=0; the dominant issue is FE-DOF RPROP target validity.
- Verification: `py_compile` passed; local PINN project `examples\TM_comsol_no_thermal_micro\tests` passed with `13 passed`; full `tests -q` failed during collection because `ref_files.Chinese_SENT_reproduction` is missing.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/exact_fe_summary.csv`
- `tables/mechanics_field_comparison.csv`
- `tables/energy_decomposition_comparison.csv`
- `tables/residual_comparison.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should the next Codex task focus on auditing `debug_fedof_energy_baseline.py` / the FE-DOF RPROP boundary and objective implementation, rather than adding notch-lip enrichment or preserving the old FE-DOF target?
2. Which guard checks should become mandatory before a field is accepted as a mechanics supervision target: direct FE energy ratio, free residual, reaction sign/magnitude, displacement scale, or all of them?
3. Should future supervised PINN prefit use the direct sparse FE solution as target and explicitly reject the old RPROP target?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, enrichment, or geometry-label guidance from this package alone.
- Do not run coupled phase-field full training from this package alone.
- Do not claim physical validation from medium/diagnostic runs.
