# Example-local corrected stress-strain promotion

This package documents the corrected stress-strain curve source promotion after restricting the implementation scope to:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro`

The shared `source/postprocess_tm.py` path was checked and left without the corrected-curve promotion logic. The active code change is in the example-local plotting entry point:

`examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py`

No D0040 run was started or processed for this task.

## Files

- `REPORT.md`: main summary and decision.
- `tables/stress_strain_data_corrected_curve_smoke_seed42.csv`: smoke output table from the updated plotting flow.
- `tables/primary_source_check.csv`: compact verification summary for the promoted source.
- `tables/stress_strain_source_corrected_curve_smoke_seed42.txt`: source metadata written by the updated plotting flow.
- `figures/stress_strain_corrected_curve_smoke_seed42.png`: stress-strain smoke figure using the corrected primary metric.
- `figures/reaction_strain_corrected_curve_smoke_seed42.png`: reaction-strain smoke figure using the corrected primary metric.
- `artifacts/plot_clean_tm_results_after.py`: example-local script after the patch.

