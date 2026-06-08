# Drive Broadening Summary: default_none

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\medium_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_rprop_300_lbfgs_0_gradient_numerical_coordnorm2x2_default_none`

Steps analyzed: 12

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **B: He_current broadens before alpha threshold**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 5 | 2.5e-05 | 0.056088695116375535 |
| first_alpha_mean_gt_0p1 | 7 | 3.5e-05 | 0.10438987217407045 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 | 0 | 1e-06 | 1.0023765143290477 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 1.0023765143290477 |
| first_bulk_drive_ratio_gt_0p25 | 0 | 1e-06 | 1.0023765143290477 |
| first_bulk_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.0023765143290477 |
| first_bottom_He_ratio_gt_0p5 | 0 | 1e-06 | 1.2591907384876726 |
| first_bottom_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.2591907384876726 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 6.369940592776402e-07 |
| first_notch_He_gt_1e-6 | 1 | 5e-06 | 1.8569557141745463e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 6.385078847870318e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 1.8604112119646744e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 6.369940592776402e-07 |

## Final Step Snapshot

- step: 11
- Delta: 4.9e-05
- alpha_mean: 0.1862220222089454
- alpha_std: 0.0006317162148737792
- alpha_max: 0.18720340728759766
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.0031199911467112
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.0031199911467112
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.9756832310374752
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.9756832310374752
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 1.0031199911467112
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 1.0031199911467112
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.9756832310374752
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.9756832310374752
- reaction_N_tm_eff: 2.1160924434661865
- top_u_mode: free
- top_u_abs_max: 3.636685050878441e-07
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
