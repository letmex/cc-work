# Moderate Prescribed-Temperature Tension Damage Probe

## 1. Purpose

Run one moderate, non-smoke A/B/C prescribed-temperature tension diagnostic to examine how prescribed uniform thermal strain affects notch-tip damage evolution. This is a diagnostic only, not physical validation.

## 2. Relationship to previous strong diagnostic and Case C alpha audit

This D0020 probe extends the prior D0015 strong diagnostic from `1.5e-5 mm` to `2.0e-5 mm` while preserving the compensation-region steps. It follows the Case C alpha audit conclusion: reaction/stress shift, compensation crossing, and notch-tip alpha peak reduction are meaningful within this diagnostic, while broad low-level diffuse alpha background is reported separately and is not used as fracture evidence.

## 3. Cases run

- Case A: `20260625_damage_probe_A_off_seed23`, thermal mode `off`, delta_T `0 K`.
- Case B: `20260625_damage_probe_B_deltaT0_seed23`, thermal mode `uniform`, delta_T `0 K`.
- Case C: `20260625_damage_probe_C_deltaT20_seed23`, thermal mode `uniform`, delta_T `+20 K`.

## 4. Schedule and why it is moderate rather than a through-crack extension

Schedule: `examples/TM_comsol_thermal_micro/load_schedules/load_schedule_D0020_tension_thermal_damage_probe.csv` with 11 displacements from `1.0e-6` to `2.0e-5 mm`. It preserves the `3.0e-6` and `3.8e-6 mm` compensation-region resolution and extends moderately beyond the previous endpoint. It is not a long through-crack schedule.

## 5. Training settings

All cases used `hidden_layers=8; neurons=400; seed=23; activation=TrainableReLU; init_coeff=3.0; full mesh; n_rprop=300; n_lbfgs=1; load_case=tension; checkpointed energy-conjugate reaction`. The run was full mode, seed 23, tension only, with checkpoints saved at every step.

## 6. A/B zero-thermal equivalence

Case B reproduces Case A under the D0020 schedule. A/B comparison status: `True`. Final A/B alpha max values are `0.344652026892` and `0.344652026892`; final nominal stresses are `158.589181956` and `158.589181956 MPa`.

## 7. C reaction/stress shift

Case C remains shifted downward relative to Case A. Final nominal stress is A `158.589181956 MPa` versus C `133.436103351 MPa`, giving C-A `-25.1530786045 MPa`.

## 8. C notch-tip alpha/damage evolution

Case C has reduced final notch-window alpha: A `0.344652026892` versus C `0.0781823396683`, giving C-A `-0.266469687223`. Final global alpha max is A `0.344652026892` versus C `0.0781823396683`.

## 9. High-threshold alpha/connectivity interpretation

Damage interpretation uses notch/high-threshold metrics, not low-threshold diffuse area. Case C final high-threshold summary:

| threshold | fraction above | notch-connected count | physical interpretation allowed |
|---:|---:|---:|---|
| 0.02 | 0.488449 | 2643 | true |
| 0.03 | 0.088523 | 464 | true |
| 0.05 | 0.029385 | 159 | true |
| 0.1 | 0.000000 | 0 | false |

## 10. Low-level diffuse alpha handling

Low-background thresholds `1e-4`, `1e-3`, `5e-3`, and `1e-2` are included only as diagnostic metrics. `figures/final_alpha_low_range_scale.png` is for background inspection only and explicitly clips high-alpha cases if applicable.

## 11. HI/HII/history interpretation

HI/HII remain finite. Final Case C HI peak is `0.00945452880114`, HII peak is `0.00597127713263`, and HII/HI ratio is `0.631578501502`. The final drive location classification is `notch_tip_region`.

## 12. Energy-conjugate reaction availability

All three cases generated 11 step checkpoints and the postprocess availability status is energy-conjugate for each run. The primary reaction metric is `reaction_N_energy`.

## 13. Heat PDE/damage conductivity guard

No heat PDE, damage-dependent conductivity, trainable/PDE temperature field, D0040, seed study, shear extension, or S0110 was implemented or run. See `tables/no_heat_pde_guard_summary.csv`.

## 14. Whether any legacy reaction metric was used as primary

No. Legacy top-sigma was not used as the primary reaction. The reported reaction and nominal stress use checkpointed energy-conjugate reaction.

## 15. Physical validation status

This is not physical validation. It is a moderate prescribed-temperature software/physics-route diagnostic.

## 16. Final classification

`moderate prescribed-temperature damage probe passed`

The moderate prescribed-temperature tension damage probe confirms that the prescribed `+20 K` thermal strain continues to shift the reaction/stress response downward and delays or reduces notch-tip/high-threshold alpha growth relative to the no-thermal baseline, while the zero-temperature branch remains equivalent to the no-thermal branch. Diffuse low-level alpha background is reported separately and is not used as fracture evidence.

## 17. Recommended next task

Review this package. If further validation is needed, run a focused review of high-threshold/notch metrics and reaction curves before deciding whether a denser compensation schedule is worth the runtime. Do not start heat PDE or damage-dependent conductivity work from this package alone.
