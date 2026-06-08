# Drive Broadening Summary: intact_unitbox

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\medium_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_rprop_300_lbfgs_0_gradient_numerical_coordnorm2x2_intact_unitbox`

Steps analyzed: 12

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **A: alpha broadens before bulk drive ratios**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 6 | 3e-05 | 0.05425423547302808 |
| first_alpha_mean_gt_0p1 | 10 | 4.7e-05 | 0.10279266215690187 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 |  |  |  |
| first_bulk_He_ratio_gt_0p5 |  |  |  |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.202832133159973e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.202832133159973e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.564077461414851e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 2.5426841966691422e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.202832133159973e-05 |

## Final Step Snapshot

- step: 11
- Delta: 4.9e-05
- alpha_mean: 0.10605955268744101
- alpha_std: 0.17095768947833184
- alpha_max: 1.0002297163009644
- bulk_He_current_p95_over_notch_tip_He_current_max: 0.00018946118509804652
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 0.00018946118509804652
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.00013273933545836832
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.00013273933545836832
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 0.00019177185582191546
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 0.00019177185582191546
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.00013273933545836832
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.00013273933545836832
- reaction_N_tm_eff: 0.7263418436050415
- top_u_mode: free
- top_u_abs_max: 8.949039511207957e-06
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
