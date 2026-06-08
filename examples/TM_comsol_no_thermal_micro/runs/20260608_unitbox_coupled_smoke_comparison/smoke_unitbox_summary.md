# Drive Broadening Summary: smoke_unitbox_coupled

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\smoke_hl_2_Neurons_20_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_topUfree_coordUnitBox_gradient_numerical_coord_unitbox_coupled_smoke`

Steps analyzed: 1

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **B: He_current broadens before alpha threshold**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 |  |  |  |
| first_alpha_mean_gt_0p1 |  |  |  |
| first_alpha_mean_gt_0p2 |  |  |  |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 | 0 | 1e-06 | 1.0440095112954246 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 1.0440095112954246 |
| first_bulk_drive_ratio_gt_0p25 | 0 | 1e-06 | 1.0440095112954246 |
| first_bulk_drive_ratio_gt_0p5 | 0 | 1e-06 | 1.0440095112954246 |
| first_bottom_He_ratio_gt_0p5 | 0 | 1e-06 | 0.6756040198292792 |
| first_bottom_drive_ratio_gt_0p5 | 0 | 1e-06 | 0.6756040198292792 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 8.585474233768764e-07 |
| first_notch_He_gt_1e-6 |  |  |  |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.963316759036387e-07 |
| first_bulk_He_p95_gt_1e-6 |  |  |  |
| first_step_where_ratio_valid | 0 | 1e-06 | 8.585474233768764e-07 |

## Final Step Snapshot

- step: 0
- Delta: 1e-06
- alpha_mean: 9.145217669284894e-05
- alpha_std: 1.1574305507826968e-05
- alpha_max: 0.00011753042781492695
- bulk_He_current_p95_over_notch_tip_He_current_max: 1.0440095112954246
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 1.0440095112954246
- bottom_right_He_current_max_over_notch_tip_He_current_max: 0.6756040198292792
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 0.6756040198292792
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 1.0440095112954246
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 1.0440095112954246
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 0.6756040198292792
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 0.6756040198292792
- reaction_N_tm_eff: 0.09202530235052109
- top_u_mode: free
- top_u_abs_max: 1.648657956820898e-07
- top_v_error_max: 0.0
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
