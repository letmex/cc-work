# Frozen-alpha u/v re-optimization diagnostic

## Scope

This package freezes saved alpha, uses saved HI/HII as the old-history input, keeps the original trial-history max update logic, and preserves material constants, top-u-free boundary ansatz, unit-box coordinate normalization, TM split, and the saved displacement level. It re-optimizes only PINN `u,v` under diagnostic-only crack-band energy variants. No load extension and no alpha evolution were run. The current route has no thermal field, so no thermal input was introduced. Seeds: 7, 13, 42. Saved states: final_D0040, through_alpha0p8_onset.

## Final D0040 summary

| variant | mean reaction removal | seeds with >=30% reaction drop | mean crack-band traction removal | mean v-jump change |
|---|---:|---:|---:|---:|
| minus_degraded_in_crack_band | 0.79% | 0/3 | 100% | -2.0519e-07 |
| minus_removed_in_crack_band | -4.49% | 0/3 | 100% | -2.30564e-07 |
| full_degradation_all_energy | -16.3% | 0/3 | 100% | 1.62153e-07 |

## Answers

1. Re-optimization completed for all final D0040 seeds and variants with finite losses: True. Convergence statuses are listed in `tables/frozen_alpha_convergence.csv`; `budget_reached_finite` means the fixed iteration budget ended with finite loss but strict trace tolerance was not met.
2. Reaction changes are reported in `tables/frozen_alpha_reaction_comparison.csv`; the acceptance classification is given below.
3. Crack-band traction changes are reported in `tables/frozen_alpha_crack_band_traction.csv`.
4. Displacement jump proxy changes are reported in `tables/frozen_alpha_displacement_jump.csv` and summarized in the table above.
5. Energy changes are reported in `tables/frozen_alpha_energy_comparison.csv`; diagnostic variants are compared against the re-optimized baseline for the same seed/state.
6. Mechanism classification: **frozen-alpha reoptimization identifies dominant mechanism: Case B, continuous-field or boundary-condition bridging is dominant**.
7. A production model change is not justified directly from this package; all full/minus degradation variants remain diagnostic-only.
8. Next minimal intervention: ask ChatGPT to review the convergence and acceptance criteria. If Case B holds, the next diagnostic should test a discontinuous/enriched kinematic replay as non-production evidence; if Case C holds, improve replay convergence or initialization before changing physics.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260612_default_unitbox_frozen_alpha_uv_reopt\artifacts\run_frozen_alpha_uv_reopt.py`: passed.

No physical validation is claimed.
