# Drive Broadening Summary: old_history_full_seed2_reference

Run directory: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\results\full_hl_8_Neurons_400_activation_TrainableReLU_coeff_3.0_Seed_2_PFFmodel_AT2_l0_0.00015_comsolMicroNoThermal_mixedH_TM_split_tm_source_mech_history_tmEpsR_1em05_gradient_numerical_full_seed2_history_D0020`

Steps analyzed: 34

## Event Ordering

Ratio validity threshold: `notch_tip_He_current_max > 1e-08`.

Ordering label: **A: alpha broadens before bulk drive ratios**

| event | step | Delta | value |
|---|---:|---:|---:|
| first_alpha_mean_gt_0p05 | 6 | 3e-05 | 0.0585224844554778 |
| first_alpha_mean_gt_0p1 | 8 | 4e-05 | 0.10749028498904266 |
| first_alpha_mean_gt_0p2 | 18 | 6.3e-05 | 0.20267066921659202 |
| first_alpha_mean_gt_0p4 |  |  |  |
| first_bulk_He_ratio_gt_0p25 |  |  |  |
| first_bulk_He_ratio_gt_0p5 |  |  |  |
| first_bulk_drive_ratio_gt_0p25 |  |  |  |
| first_bulk_drive_ratio_gt_0p5 |  |  |  |
| first_bottom_He_ratio_gt_0p5 |  |  |  |
| first_bottom_drive_ratio_gt_0p5 |  |  |  |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 4.8979756684275344e-05 |
| first_notch_He_gt_1e-6 | 0 | 1e-06 | 4.8979756684275344e-05 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 8.941119403971241e-07 |
| first_bulk_He_p95_gt_1e-6 | 1 | 5e-06 | 2.757050342552246e-05 |
| first_step_where_ratio_valid | 0 | 1e-06 | 4.8979756684275344e-05 |

## Final Step Snapshot

- step: 33
- Delta: 0.0001
- alpha_mean: 0.2144645048950973
- alpha_std: 0.24571941633918865
- alpha_max: 1.0018107891082764
- bulk_He_current_p95_over_notch_tip_He_current_max: 3.798645554151973e-05
- bulk_He_current_p95_over_notch_tip_He_current_max_valid: 3.798645554151973e-05
- bottom_right_He_current_max_over_notch_tip_He_current_max: 1.4269191047924905e-05
- bottom_right_He_current_max_over_notch_tip_He_current_max_valid: 1.4269191047924905e-05
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max: 9.725279424628874e-05
- bulk_mechanics_drive_p95_over_notch_tip_mechanics_drive_max_valid: 9.725279424628874e-05
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max: 3.6918911819923204e-05
- bottom_right_mechanics_drive_max_over_notch_tip_mechanics_drive_max_valid: 3.6918911819923204e-05
- reaction_N_tm_eff: 0.5903642773628235
- top_u_mode: fixed
- top_u_abs_max: 4.371138867531599e-12
- top_v_error_max: 2.526212488436659e-12
- bottom_u_abs_max: 0.0
- bottom_v_abs_max: 0.0

This diagnostic does not claim physical validation.
