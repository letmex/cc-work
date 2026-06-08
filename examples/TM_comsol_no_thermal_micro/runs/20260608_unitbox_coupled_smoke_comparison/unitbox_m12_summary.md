# Drive Broadening Summary: unitbox_alpha_init_history_m12

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\medium_hl_4_Neurons_100_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_rprop_300_lbfgs_0_gradient_numerical_coord_cmp_unitbox_m12`

Steps analyzed: 12

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **A: alpha broadens before bulk drive ratios**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 6 | 3.025e-05 | 0.05477371088292406 |
| first_alpha_mean_gt_0p1 | 11 | 4.6e-05 | 0.10395352512594197 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 |  |  |  |
| first_bulk_He_ratio_gt_0p5 |  |  |  |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.328224895289168e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.328224895289168e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.657596879402263e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5.875e-06 | 3.575510309019591e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.328224895289168e-05 |

## Final Step Snapshot

- step: 11
- Delta: 4.6e-05
- alpha_mean: 0.10395352512594197
- alpha_std: 0.1553344023275423
- alpha_max: 1.0004647970199585
- bulk_He_current_p95_over_notch_tip_He_current_max: 0.0003442060911469014
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 0.0003442060911469014
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.00027926170498501746
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.00027926170498501746
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 0.0003623192324007572
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 0.0003623192324007572
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.00027926170498501746
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.00027926170498501746
- reaction_N_tm_eff: 0.7061354517936707
- top_u_mode: free
- top_u_abs_max: 1.0231022315565497e-05
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
