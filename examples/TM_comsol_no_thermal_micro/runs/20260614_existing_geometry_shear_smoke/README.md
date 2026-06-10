# Existing-Geometry Shear Smoke

Package: `examples/TM_comsol_no_thermal_micro/runs/20260614_existing_geometry_shear_smoke`

This package contains the evidence for one seed-only shear smoke test on the existing `TM_comsol_no_thermal_micro` geometry.

Read first:

1. `REPORT.md`
2. `tables/shear_smoke_run_summary.csv`
3. `tables/shear_damage_drive_summary.csv`
4. `tables/shear_top_v_free_diagnostic.csv`
5. `figures/figure_summary.md`

Main conclusion: `shear smoke not convincing`. The implementation path ran and produced checkpointed energy-conjugate shear reaction, but the final mechanics-drive maximum is boundary/corner dominated and the smoke does not show a post-peak drop or alpha>=0.8 through-crack.

No D0040 run, seed study, material change, `l0` change, TM split change, alpha-init-intact route, staggered route, imposed alpha notch, local/lip/jump loss, or legacy top-sigma output is included.
