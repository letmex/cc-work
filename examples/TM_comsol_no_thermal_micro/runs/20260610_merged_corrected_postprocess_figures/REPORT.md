# Merged Corrected Postprocess Figures

## Purpose

The previous workflow required two separate commands:

1. `corrected_reaction_postprocess.py` for corrected CSVs.
2. `plot_clean_tm_results.py` for figures.

This was unnecessary friction. The postprocess entry point now runs both stages by default.

## Code Change

`corrected_reaction_postprocess.py` now includes:

- `generate_clean_figures_for_corrected_curve(...)`
- `_attach_figure_outputs(...)`
- new CLI arguments:
  - `--figure-dir`
  - `--run-label`
  - `--corrected-seed`
  - `--figure-dpi`
  - `--no-figures`
  - `--fail-on-figure-error`

Default behavior:

```text
corrected_reaction_postprocess.py -> corrected CSVs -> clean figures
```

Figures are enabled by default. Use `--no-figures` only when CSV-only postprocessing is desired.

## Verified Seed 23 Command

Executed in:

```powershell
D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro
```

Command:

```powershell
$run='full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_23_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_D0020_seed23_history_default_unitbox'
D:\anaconda3\envs\torch_env\python.exe corrected_reaction_postprocess.py --model-dir $run --result-dir "results\$run" --out-dir "results\$run\curves" --figure-dir "results\$run\figures" --run-label "seed23_D0020_corrected" --device cpu --figure-dpi 160
```

Observed result:

```text
status: energy_exact_primary
exact_reaction_computable: True
figure_status: generated
```

Generated figure outputs include:

- `stress_strain_seed23_D0020_corrected.png`
- `reaction_strain_seed23_D0020_corrected.png`
- `final_fields_panel_seed23_D0020_corrected.png`
- final alpha/u/v/disp/HI/HII/He/He_current/He_history/mechanics_drive field maps

## Tests

New test:

`tests/test_corrected_reaction_postprocess.py::test_corrected_postprocess_generates_clean_figures_from_corrected_curve`

It constructs a minimal field `.npz` and corrected CSV, then verifies that the merged postprocess helper creates stress-strain and reaction-strain PNGs.

Verification:

```powershell
D:\anaconda3\envs\torch_env\python.exe -m py_compile corrected_reaction_postprocess.py plot_clean_tm_results.py tests\test_corrected_reaction_postprocess.py
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

Result:

```text
24 passed, 8 warnings
```

## Interpretation

The user-facing postprocess command now produces both numerical and image results. The corrected stress-strain image is generated from `nominal_stress_energy_exact_MPa`; legacy top-boundary reaction remains diagnostic.

This is a workflow/output change only. It does not modify the physical model, training loss, material parameters, `l0`, TM split, alpha initialization, or history update logic.
