# Prescribed Thermal Strain Stage Summary

## 1. Purpose

This package consolidates the prescribed-temperature thermal-strain branch work completed so far in `examples/TM_comsol_thermal_micro`. It is a documentation and evidence synthesis package only. It does not run new training, rerun A/B/C diagnostics, introduce heat PDE physics, introduce damage-dependent conductivity, or modify the original no-thermal baseline project.

## 2. Scope boundaries

The scope is limited to the thermal subproject. The original `examples/TM_comsol_no_thermal_micro` project remains the frozen baseline. The summary reads existing reports, handoffs, current thermal source files, and existing package tables. The summary does not edit old packages, change source-model behavior, change the reaction route, or reintroduce legacy top-sigma as the primary reaction.

## 3. Timeline of completed thermal prescribed-strain work

The stage began with a thermal subproject scaffold copied from the verified no-thermal route. The prescribed thermal strain patch then added a default-off thermal mode and patch tests. Subsequent diagnostics exercised the branch with a smoke micro-notch case, a stronger D0015 tension case, a Case C alpha audit, and a moderate D0020 tension damage probe. See `tables/stage_milestone_summary.csv` for the reviewer-facing timeline.

## 4. What was implemented

The implemented branch is prescribed-temperature mechanics only. The current route computes `delta_T = T - Tref`, subtracts isotropic normal thermal strain from `exx` and `eyy`, leaves `exy` unchanged, and then passes the adjusted elastic strain through the existing TM split, history, energy, and checkpointed reaction route. CLI/config support exists for `thermal_mode`, prescribed absolute temperature, prescribed `delta_T`, `alpha_T`, and `Tref`.

## 5. What was validated by patch tests

Patch tests validated zero-`delta_T` equivalence, free uniform expansion, constrained uniform heating sign/scale under the current project convention, shear-component invariance, and guards showing no heat PDE, no trainable/PDE temperature field, and no damage-dependent conductivity. The final patch-test classification was `prescribed thermal strain branch implemented and patch tests passed`.

## 6. What was validated by micro-notch diagnostics

The micro-notch diagnostic confirmed that the branch can run the existing checkpointed mechanics route with A/B/C cases. Case B matched Case A within table precision. Case C shifted the final nominal stress from A `10.9504107968 MPa` to C `5.30404868186 MPa`, while alpha remained stable in that small smoke run. This is useful as a routing diagnostic, not physical validation.

## 7. What the stronger tension diagnostic showed

The stronger D0015 diagnostic used full training settings and a schedule around the thermal compensation region. Case B again matched Case A within table precision. Case C shifted final nominal stress from A `124.041421805 MPa` to C `94.4112776779 MPa`, and final alpha from A `0.158222526312` to C `0.0365089587867`. The observed Case C reaction zero crossing was `3.45037430392e-06 mm`, near the estimated compensation displacement `3.78e-06 mm`.

## 8. What the Case C alpha anomaly audit showed

The alpha audit found that Case C peak alpha was lower than A/B and the reaction/stress trend remained interpretable. It also found a broad low-level Case C alpha background in raw element values. The final C-minus-A alpha distribution had median `-0.00623816251755` and positive maximum `0.0107422318542`. Final Case C low-threshold area fractions were `>=0.001` `0.587322121604` and `>=0.01` `0.433746072815`. That background is a diagnostic warning and is not treated as physical fracture evidence.

## 9. What the moderate damage probe showed

The moderate D0020 probe extended the tension schedule to `2.0e-5 mm` while preserving compensation-region resolution. Case B matched Case A. Case C final nominal stress was `133.436103351 MPa` versus A `158.589181956 MPa`, a C-A shift of `-25.1530786045 MPa`. Case C final alpha was `0.0781823396683` versus A `0.344652026892`, and final Case C high-threshold area fractions were `>=0.02` `0.488449454814`, `>=0.03` `0.0885233783035`, `>=0.05` `0.0293845869525`, and `>=0.1` `0`.

## 10. Trusted conclusions

The trusted conclusions are that `thermal_mode=off` remains the default, `thermal_mode=uniform` with zero `delta_T` reproduces the no-thermal thermal-subproject route in completed diagnostics, prescribed `+20 K` shifts displacement-controlled tension reaction/stress downward, checkpointed `reaction_N_energy` remains available and primary, and no heat PDE or damage-dependent conductivity has been implemented. The moderate-probe damage reduction is trusted only within the diagnostic scope and is therefore rated medium rather than high.

## 11. Diagnostic-only conclusions

The broad low-level Case C alpha background, single-seed behavior, tension-only conclusions, and any implication about conduction or damage-dependent conductivity are diagnostic-only. These findings should guide review and future test design but should not be used as physical fracture evidence or as a transport-model conclusion.

## 12. Explicitly unimplemented physics

The following remain explicitly unimplemented or not run: full heat PDE, damage-dependent conductivity, trainable/PDE temperature, D0040, seed study, shear extension, S0110, no-thermal baseline modification, and legacy top-sigma as the primary reaction. See `tables/not_implemented_guard_summary.csv`.

## 13. Known limitations and risks

The branch is prescribed-temperature only, uses uniform temperature in the main diagnostics, has no transient thermal loading, has no independent seed validation, and has no physical validation against COMSOL or experiment. The low-level alpha background remains an artifact/interpretation risk. The thermal strain implementation follows the current project constitutive convention and is not a line-by-line COMSOL clone.

## 14. Whether this is physical validation

This is not physical validation. It is a software and physics-route validation stage for prescribed thermal strain in the thermal subproject. It supports preserving the branch as a baseline for review, but it does not establish quantitative agreement with COMSOL or experiment.

## 15. Recommendation on further prescribed-temperature diagnostics

Do not run more broad prescribed-temperature tension diagnostics by default. The evidence is now sufficient for a decision gate. A small seed repeat is optional only if the reviewer decides that robustness evidence is required before heat PDE planning.

## 16. Decision gate before heat PDE

The safest gate is a reviewer decision on whether to preserve this prescribed-temperature branch as the thermal baseline and begin a written heat PDE plan. Damage-dependent conductivity should remain deferred until a heat PDE branch is independently stable and tested.

## 17. Final classification

`prescribed thermal strain stage summary complete`

The prescribed-temperature thermal-strain branch has passed patch tests and multiple checkpointed tension diagnostics. The zero-temperature thermal branch reproduces the no-thermal branch, and +20 K prescribed uniform thermal strain consistently shifts the reaction/stress response downward while reducing notch-tip/high-threshold alpha growth in the moderate damage probe. The broad low-level Case C alpha background is a diagnostic warning and is not treated as physical fracture evidence. This stage remains prescribed-temperature only: no heat PDE, no trainable temperature field, and no damage-dependent conductivity have been implemented. The next step should be a decision-gate review before starting heat PDE work.

## 18. Exact next recommended task

Review `tables/next_decision_gate.csv`, this `REPORT.md`, and the existing strong/audit/moderate probe reports. If approved, write a heat PDE implementation and validation plan only; do not implement heat PDE or damage-dependent conductivity without that plan.
