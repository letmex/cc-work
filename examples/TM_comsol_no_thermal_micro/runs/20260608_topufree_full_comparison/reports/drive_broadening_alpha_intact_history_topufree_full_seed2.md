# Drive Broadening Summary: alpha_intact_history_topufree_full_seed2

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_gradient_numerical_history_alpha_init_intact_topufree_D0020_seed2`

Steps analyzed: 34

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **B: He_current broadens before alpha threshold**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 5 | 2.5e-05 | 0.05596251604757409 |
| first_alpha_mean_gt_0p1 | 7 | 3.5e-05 | 0.10429016859677688 |
| first_alpha_mean_gt_0p2 | 13 | 5.3e-05 | 0.2109841780265099 |
| first_alpha_mean_gt_0p4 | 28 | 8.6e-05 | 0.4135702809329571 |
| first_bulk_He_ratio_gt_0p25 | 0 | 1e-06 | 1.022799927576047 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 1.022799927576047 |
| first_bulk_drive_ratio_gt_0p25 | 0 | 1e-06 | 1.022799927576047 |
| first_bulk_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.022799927576047 |
| first_bottom_He_ratio_gt_0p5 | 0 | 1e-06 | 1.6301958071691651 |
| first_bottom_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.6301958071691651 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 6.344892540255387e-07 |
| first_notch_He_gt_1e-6 | 1 | 5e-06 | 1.845618498919066e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 6.489555630651012e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 1.8586757141747513e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 6.344892540255387e-07 |

## Final Step Snapshot

- step: 33
- Delta: 0.0001
- alpha_mean: 0.48817826120409724
- alpha_std: 4.0094150429576516e-05
- alpha_max: 0.4882843494415283
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.0008025829954588
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.0008025829954588
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.9995179868457842
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.9995179868457842
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 1.0008025829954588
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 1.0008025829954588
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.9995179868457842
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.9995179868457842
- reaction_N_tm_eff: -1.7486419677734375
- top_u_mode: free
- top_u_abs_max: 4.0447113747177355e-07
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
