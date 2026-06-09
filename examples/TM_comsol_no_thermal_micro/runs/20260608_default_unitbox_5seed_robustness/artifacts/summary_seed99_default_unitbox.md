# Drive Broadening Summary: seed99_default_unitbox

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_99_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_full_D0020_seed99_history_default_unitbox`

Steps analyzed: 34

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **A: alpha broadens before bulk drive ratios**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 6 | 3e-05 | 0.05424363073727731 |
| first_alpha_mean_gt_0p1 | 10 | 4.7e-05 | 0.10249010482526527 |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 |  |  |  |
| first_bulk_He_ratio_gt_0p5 |  |  |  |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.3439889850560576e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.3439889850560576e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.534740516097372e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 2.532049820729298e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.3439889850560576e-05 |

## Final Step Snapshot

- step: 33
- Delta: 0.0001
- alpha_mean: 0.11440181488093627
- alpha_std: 0.1913102885042248
- alpha_max: 1.0018606185913086
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.5139933729647668e-05
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.5139933729647668e-05
- bottom_right_He_current_max_over_notch_tip_He_current_max: 1.3930665273125853e-05
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 1.3930665273125853e-05
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 3.0569841331796174e-05
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 3.0569841331796174e-05
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 1.99467438217579e-05
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 1.99467438217579e-05
- reaction_N_tm_eff: 0.9658390283584595
- top_u_mode: free
- top_u_abs_max: 4.233422714605695e-06
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
