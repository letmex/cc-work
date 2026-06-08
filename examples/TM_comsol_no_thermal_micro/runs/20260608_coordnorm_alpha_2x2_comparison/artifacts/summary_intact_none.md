# Drive Broadening Summary: intact_none

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\medium_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_rprop_300_lbfgs_0_gradient_numerical_coordnorm2x2_intact_none`

Steps analyzed: 12

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **C: alpha and drive broaden in the same step**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 5 | 2.5e-05 | 0.0558920006234986 |
| first_alpha_mean_gt_0p1 | 7 | 3.5e-05 | 0.10011364357268596 |
| first_alpha_mean_gt_0p2 | 11 | 4.9e-05 | 0.20455976271586374 |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 | 5 | 2.5e-05 | 0.39298349542886873 |
| first_bulk_He_ratio_gt_0p5 | 8 | 4e-05 | 0.5914707480971291 |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.442606950760819e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.442606950760819e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.515492169181006e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 2.5486102003924312e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.442606950760819e-05 |

## Final Step Snapshot

- step: 11
- Delta: 4.9e-05
- alpha_mean: 0.20455976271586374
- alpha_std: 0.06396470635641321
- alpha_max: 0.507638692855835
- bulk_He_current_p95_over_notch_tip_He_current_max: 0.045655732918922866
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 0.045655732918922866
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.03556848350714233
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.03556848350714233
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 0.05319847250746222
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 0.05319847250746222
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.03475457432995184
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.03475457432995184
- reaction_N_tm_eff: 1.728469967842102
- top_u_mode: free
- top_u_abs_max: 7.460855613317108e-06
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
