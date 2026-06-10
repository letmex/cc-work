## Codex handoff: example-local corrected stress-strain promotion

Commit: ed10d9f
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_example_local_stress_strain_promotion
Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_example_local_stress_strain_promotion/REPORT.md

### What changed
- Corrected the implementation scope: the active code change is only in `D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py`.
- Confirmed shared `source/postprocess_tm.py` no longer contains the previous corrected-curve promotion symbols.
- Promoted `nominal_stress_energy_exact_MPa` as the primary stress-strain source when a corrected CSV is supplied or discovered.
- Kept legacy top-boundary sigma curves only as diagnostics.
- Added a fallback status of `reaction_metric_unavailable` when corrected reaction data is missing.
- Avoided writing non-ASCII parent paths into generated metadata to prevent mojibake.
- Did not run or process D0040.

### Commands run
```powershell
git -C <cc-work> pull origin main
D:/anaconda3/envs/torch_env/python.exe -m py_compile D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py
D:/anaconda3/envs/torch_env/python.exe D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py --result-dir <TM_comsol_no_thermal_micro>/results/full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_42_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_full_D0020_seed42_history_default_unitbox --out-dir <TM_comsol_no_thermal_micro>/results/clean_figures/codex_corrected_curve_smoke_seed42 --run-label corrected_curve_smoke_seed42 --corrected-stress-strain-csv <cc-work>/examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_stress_strain_curve_fix/tables/corrected_stress_strain_by_step.csv --corrected-seed 42 --dpi 120
D:/anaconda3/envs/torch_env/python.exe -m pytest D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/tests -q
```

### Key results
- `py_compile` passed for the example-local plotting script.
- Existing example-local tests passed: 18 passed.
- Smoke output table reports `stress_strain_primary_metric = nominal_stress_energy_exact_MPa`.
- Smoke output table reports `stress_strain_metric_status = energy_conjugate_primary`.
- Smoke output table reports `legacy_curve_status = legacy_diagnostic_only`.
- Seed 42 smoke summary: primary final/peak = 0.00453097273492207; legacy final/peak = 0.945459761653825.
- This is a plotting/source-promotion fix only; no new physical-model claim is made.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/primary_source_check.csv`
- `tables/stress_strain_data_corrected_curve_smoke_seed42.csv`
- `figures/figure_summary.md`
- `artifacts/plot_clean_tm_results_after.py`

### Question for ChatGPT
1. Is the example-local promotion sufficient to treat corrected energy-conjugate stress-strain curves as the main plotting path?
2. Should the next Codex task add a regression test for corrected CSV priority in `plot_clean_tm_results.py`?
3. Should older result packages be regenerated through this example-local plotting flow, or should only future packages use it?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from medium/diagnostic runs.
- Do not run D0040 for this issue unless explicitly requested.

