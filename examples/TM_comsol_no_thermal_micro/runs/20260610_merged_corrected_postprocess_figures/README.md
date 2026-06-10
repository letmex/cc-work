# Merged Corrected Postprocess Figures

This package documents the mainflow change that merges corrected CSV generation and clean figure generation into one command.

Changed active local source tree:

`D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`

Main result:

`corrected_reaction_postprocess.py` now generates:

- corrected reaction CSV
- corrected stress-strain CSV
- clean field figures
- corrected stress-strain PNG
- corrected reaction-strain PNG

The separate `plot_clean_tm_results.py` command is no longer required for normal postprocessing.

Read first:

1. `REPORT.md`
2. `tables/merged_postprocess_summary.csv`
3. `tables/seed23_generated_figures.csv`
4. `figures/figure_summary.md`
