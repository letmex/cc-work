# Drive Broadening Summary: none_alpha_init_history_m12

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\medium_hl_4_Neurons_100_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_rprop_300_lbfgs_0_gradient_numerical_coord_cmp_none_m12`

Steps analyzed: 12

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **B: He_current broadens before alpha threshold**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 5 | 2.5375e-05 | 0.05754536478703618 |
| first_alpha_mean_gt_0p1 | 7 | 3.5125e-05 | 0.10485912604707377 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 | 0 | 1e-06 | 0.9992500896350739 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 0.9992500896350739 |
| first_bulk_drive_ratio_gt_0p25 | 0 | 1e-06 | 0.9992500896350739 |
| first_bulk_drive_ratio_gt_0p5 | 0 | 1e-06 | 0.9992500896350739 |
| first_bottom_He_ratio_gt_0p5 | 0 | 1e-06 | 1.2330237463457647 |
| first_bottom_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.2330237463457647 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 6.503361191789736e-07 |
| first_notch_He_gt_1e-6 | 1 | 5.875e-06 | 2.5913243007380515e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 6.498484253825154e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5.875e-06 | 2.596412832645001e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 6.503361191789736e-07 |

## Final Step Snapshot

- step: 11
- Delta: 4.6e-05
- alpha_mean: 0.16766955388317134
- alpha_std: 0.002103774252875565
- alpha_max: 0.17443208396434784
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.0054566391752635
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.0054566391752635
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.9634849437150499
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.9634849437150499
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 1.0054566391752635
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 1.0054566391752635
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.9634849437150499
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.9634849437150499
- reaction_N_tm_eff: 2.157921314239502
- top_u_mode: free
- top_u_abs_max: 3.727605530912115e-07
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
