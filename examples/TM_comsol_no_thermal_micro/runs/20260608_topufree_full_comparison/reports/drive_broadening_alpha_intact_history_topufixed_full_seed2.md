# Drive Broadening Summary: alpha_intact_history_topufixed_full_seed2

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_gradient_numerical_history_alpha_init_intact_D0020_seed2`

Steps analyzed: 34

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **B: He_current broadens before alpha threshold**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 5 | 2.5e-05 | 0.05593346455947702 |
| first_alpha_mean_gt_0p1 | 7 | 3.5e-05 | 0.10431312158431172 |
| first_alpha_mean_gt_0p2 | 13 | 5.3e-05 | 0.21107617140950877 |
| first_alpha_mean_gt_0p4 | 28 | 8.6e-05 | 0.4136078409579342 |
| first_bulk_He_ratio_gt_0p25 | 0 | 1e-06 | 1.0137009507038128 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 1.0137009507038128 |
| first_bulk_drive_ratio_gt_0p25 | 0 | 1e-06 | 1.0137009507038128 |
| first_bulk_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.0137009507038128 |
| first_bottom_He_ratio_gt_0p5 | 0 | 1e-06 | 1.2000986143914512 |
| first_bottom_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.2000986143914512 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 6.402885901479749e-07 |
| first_notch_He_gt_1e-6 | 1 | 5e-06 | 1.864367732196115e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 6.490611525578061e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 1.8643012663233093e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 6.402885901479749e-07 |

## Final Step Snapshot

- step: 33
- Delta: 0.0001
- alpha_mean: 0.48825829612572075
- alpha_std: 5.682864245045179e-05
- alpha_max: 0.48843449354171753
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.000672508082938
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.000672508082938
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.9961673784409758
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.9961673784409758
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 1.000672508082938
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 1.000672508082938
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.9961673784409758
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.9961673784409758
- reaction_N_tm_eff: -1.7251943349838257
- top_u_mode: fixed
- top_u_abs_max: 4.371138867531599e-12
- top_v_error_max: 2.526212488436659e-12
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
