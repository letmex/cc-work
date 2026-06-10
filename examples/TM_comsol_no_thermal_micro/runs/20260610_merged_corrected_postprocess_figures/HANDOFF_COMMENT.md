## Codex handoff: Merge corrected CSV and figure postprocessing

Commit: 0ea303f
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260610_merged_corrected_postprocess_figures
Main report: examples/TM_comsol_no_thermal_micro/runs/20260610_merged_corrected_postprocess_figures/REPORT.md

### What changed
- Changed active local source at `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`.
- `corrected_reaction_postprocess.py` now generates clean PNG figures after generating corrected CSVs.
- Normal postprocessing no longer requires a second manual `plot_clean_tm_results.py` command.
- Added `generate_clean_figures_for_corrected_curve(...)` and a guarded figure attachment path.
- New CLI options:
  - `--figure-dir`
  - `--run-label`
  - `--corrected-seed`
  - `--figure-dpi`
  - `--no-figures`
  - `--fail-on-figure-error`
- Figure generation is enabled by default. CSV success is preserved if figure generation fails, with `figure_status` reporting the failure unless `--fail-on-figure-error` is set.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m pytest tests\test_corrected_reaction_postprocess.py::test_corrected_postprocess_generates_clean_figures_from_corrected_curve -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile corrected_reaction_postprocess.py plot_clean_tm_results.py tests\test_corrected_reaction_postprocess.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
$run='full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_23_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_D0020_seed23_history_default_unitbox'
D:\anaconda3\envs\torch_env\python.exe corrected_reaction_postprocess.py --model-dir $run --result-dir "results\$run" --out-dir "results\$run\curves" --figure-dir "results\$run\figures" --run-label "seed23_D0020_corrected" --device cpu --figure-dpi 160
```

### Key results
- New test first failed before implementation because `generate_clean_figures_for_corrected_curve` did not exist.
- After implementation, the focused test passed.
- Full local example tests passed: `24 passed, 8 warnings`.
- `py_compile` passed.
- Seed 23 postprocess returned:
  - `status: energy_exact_primary`
  - `exact_reaction_computable: True`
  - `figure_status: generated`
- The same `corrected_reaction_postprocess.py` command generated corrected CSVs and PNG figures.
- Generated seed23 figures include:
  - `stress_strain_seed23_D0020_corrected.png`
  - `reaction_strain_seed23_D0020_corrected.png`
  - `final_fields_panel_seed23_D0020_corrected.png`

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/merged_postprocess_summary.csv`
- `tables/seed23_generated_figures.csv`
- `tables/seed23_corrected_reaction_availability.csv`
- `tables/seed23_corrected_stress_strain_by_step.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should the merged postprocess command be treated as the only normal user-facing postprocess path?
2. Should figure generation failure remain non-fatal by default, or should full-run reporting require `--fail-on-figure-error`?
3. Should `main.py` pass a shorter default `run_label` so training-completion figures have shorter filenames?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this workflow/output verification.
- Do not run D0040 for this issue.
