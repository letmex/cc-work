# Drive Broadening Summary: D0040_seed7_default_unitbox

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_7_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_softgate_D0040_seed7_history_default_unitbox`

Steps analyzed: 55

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **A: alpha broadens before bulk drive ratios**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 6 | 3e-05 | 0.054407317330406574 |
| first_alpha_mean_gt_0p1 | 10 | 4.7e-05 | 0.10219351497924775 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 |  |  |  |
| first_bulk_He_ratio_gt_0p5 |  |  |  |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.346950663602911e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.346950663602911e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.540193732642361e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 2.508827519704937e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.346950663602911e-05 |

## Final Step Snapshot

- step: 54
- Delta: 0.0002
- alpha_mean: 0.11327689389253934
- alpha_std: 0.19153456213632275
- alpha_max: 1.001736044883728
- bulk_He_current_p95_over_notch_tip_He_current_max: 3.6010541281609292e-06
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 3.6010541281609292e-06
- bottom_right_He_current_max_over_notch_tip_He_current_max: 4.124155915412001e-06
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 4.124155915412001e-06
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 6.305523043363498e-06
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 6.305523043363498e-06
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 4.176999024873181e-06
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 4.176999024873181e-06
- reaction_N_tm_eff: 0.9327040910720825
- top_u_mode: free
- top_u_abs_max: 1.1515173355292063e-05
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
