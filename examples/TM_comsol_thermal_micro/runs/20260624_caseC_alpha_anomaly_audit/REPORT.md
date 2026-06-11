# Case C Alpha Anomaly Audit

## 1. Purpose

This package audits the suspicious broad low-level alpha field in Case C from the existing stronger prescribed-temperature tension diagnostic. It uses only completed outputs from `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic` and does not run training, change source behavior, or modify the no-thermal project.

## 2. Existing strong diagnostic being audited

- Case A: `20260623_strong_A_off_seed23`, thermal mode `off`, delta_T `0 K`.
- Case B: `20260623_strong_B_deltaT0_seed23`, thermal mode `uniform`, delta_T `0 K`.
- Case C: `20260623_strong_C_deltaT20_seed23`, thermal mode `uniform`, delta_T `+20 K`.
- Source package: `examples/TM_comsol_thermal_micro/runs/20260623_stronger_prescribed_temperature_tension_diagnostic`.
- Field source: `examples/TM_comsol_thermal_micro/outputs/results`.

## 3. Why Case C alpha looked suspicious

The previous final alpha view made Case C look spatially broad in the right half of the specimen, while A/B had a much larger notch-tip peak. The visual concern was plausible because a low or Case-C-like colorbar can saturate A/B and amplify low-amplitude Case C background.

## 4. Raw-data versus plotting-scale assessment

The broad low-level Case C values are present in the raw `alpha_elem` field, so the effect is not purely an image export artifact. However, its visual prominence is strongly scale dependent. Final alpha maxima are A `0.158222526312`, B `0.158222526312`, and C `0.0365089587867`. Case C reaches only `0.231` of the A/B peak.

## 5. Global-scale alpha interpretation

`figures/final_alpha_global_scale.png` uses a shared true global scale. On that scale, A/B peak alpha dominates and Case C is visibly lower. This supports the existing strong diagnostic result that positive prescribed temperature reduces the effective tensile damage drive at the final displacement.

## 6. Low-range alpha interpretation

`figures/final_alpha_low_range_scale.png` uses a shared `0..0.04` scale. This is useful for inspecting Case C low-level structure, but it clips A/B by design. The low-range figure should not be used to compare fracture severity or peak damage.

## 7. C-minus-A and C-minus-B alpha difference interpretation

The final `C_minus_A` distribution has min `-0.122059`, median `-0.00623816`, max `0.0107422`, positive fraction `0.1661`, and negative fraction `0.8339`. `C_minus_B` is the same within A/B equality: min `-0.122059`, median `-0.00623816`, max `0.0107422`. Negative differences include the reduced notch peak in Case C; positive differences mark low-level background locations where C exceeds the A/B baseline.

## 8. Threshold area/connectivity analysis

Connectivity uses element adjacency through shared raw mesh triangle edges. The notch seed window is centered at `(0.005, 0.005) mm` with half-width `0.0003 mm`.

| threshold | fraction above | largest component | notch-connected | x span | y span |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.587322 | 3178 | 3178 | 0.00594985 | 0.00976622 |
| 0.005 | 0.561079 | 3036 | 3036 | 0.00578349 | 0.00976622 |
| 0.01 | 0.433746 | 2347 | 2347 | 0.0056278 | 0.00976622 |
| 0.02 | 0.041212 | 223 | 223 | 0.00130174 | 0.00191941 |
| 0.03 | 0.013861 | 75 | 75 | 0.000636553 | 0.000750538 |

The final Case C thresholded area is therefore threshold-sensitive. Low thresholds capture broad low-amplitude output; higher thresholds quickly collapse toward the notch region or disappear.

## 9. Alpha versus HI/HII/He spatial correlation

For Case C final, alpha versus He is Pearson 0.574, Spearman 0.882. Full final correlations:

| field pair | Pearson | Spearman | interpretation |
|---|---:|---:|---|
| alpha vs HI | 0.576 | 0.882 | Strong spatial alignment. |
| alpha vs HII | 0.572 | 0.881 | Strong spatial alignment. |
| alpha vs He | 0.574 | 0.882 | Strong spatial alignment. |
| alpha vs mechanics_drive | 0.574 | 0.882 | Strong spatial alignment. |
| alpha vs elastic_energy_density | 0.584 | 0.878 | Strong spatial alignment. |
| alpha vs fracture_energy_density | 0.879 | 0.867 | Strong spatial alignment. |

The correlation evidence supports some relation to mechanical drive near high-alpha regions, but it does not by itself validate the broad low-amplitude background as physical fracture damage.

## 10. Case C alpha evolution through compensation crossing

Case C starts with reaction `-0.147555 N` at displacement `1e-06 mm`, is near/after crossing by step `3` with reaction `0.00151049 N`, and ends at `0.944113 N`. Final Case C alpha max is `0.036509`, mean `0.00706531`, and p95 `0.018331`. See `tables/caseC_alpha_evolution_summary.csv` and `figures/caseC_alpha_evolution_by_step.png`.

## 11. Sampling/texture artifact check

`figures/sampling_texture_diagnostic.png` compares Case C final alpha and threshold masks against element area and element index texture. This audit does not find enough evidence to treat the broad field as a pure interpolation artifact, because the tables use raw element values directly. It also does not validate the broad field as physical, because the low-amplitude region is scale-sensitive and single-run.

## 12. Artifact risk assessment

See `tables/artifact_risk_assessment.csv`. The highest immediate risk is colorbar clipping/low-range visual amplification. Medium residual risks remain for low-amplitude PINN background, path dependence through compensation, and insufficient single-run training evidence.

## 13. What is trustworthy from Case C

- The reaction/stress downward shift relative to A/B.
- The compressive-to-tensile reaction crossing through the compensation region.
- The reduced final notch-tip alpha peak relative to A/B.
- A/B equality under the same strong settings and zero prescribed temperature.

## 14. What is not yet trustworthy from Case C

- Treating the broad low-level Case C alpha cloud as physical fracture evidence.
- Comparing fracture severity from the low-range alpha colorbar.
- Using diffuse alpha area as a material conclusion without further validation.

## 15. Final classification

`caseC diffuse alpha likely plotting-scale artifact plus low-amplitude background`

Case C peak damage is lower than A/B and the reaction/stress shift remains physically interpretable. The visually broad Case C alpha region is partly amplified by low-range color scaling and should not be interpreted as stronger fracture damage. The trustworthy result is the reaction/stress shift and reduced notch-tip alpha peak; the broad low-level Case C alpha cloud should not be used as physical fracture evidence yet.

## 16. Recommended next task

Review this audit package first. If more validation is needed, run one moderate non-smoke prescribed-temperature tension diagnostic with a denser schedule around `3.0e-6` to `4.5e-6 mm`, still limited to A/B/C, still using checkpointed energy-conjugate reaction, and still without heat PDE, damage-dependent conductivity, D0040, seed study, shear extension, or S0110.

## Evidence files

- `tables/alpha_threshold_area_connectivity.csv`
- `tables/alpha_distribution_statistics.csv`
- `tables/alpha_difference_statistics.csv`
- `tables/alpha_drive_spatial_correlation.csv`
- `tables/caseC_alpha_evolution_summary.csv`
- `tables/plot_scale_audit.csv`
- `tables/artifact_risk_assessment.csv`
- `tables/no_new_training_guard.csv`
- `figures/figure_summary.md`
