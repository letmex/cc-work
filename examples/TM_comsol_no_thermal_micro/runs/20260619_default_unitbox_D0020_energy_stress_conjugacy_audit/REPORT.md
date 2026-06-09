# D0020 energy-stress conjugacy audit

## Scope

This package audits whether the stress fields used for reaction postprocessing are conjugate to the actual checkpoint mechanics energy. It uses existing D0020 seed 7/13/42 checkpoints only and does not retrain, extend loading, run D0040, or change physics.

## Checkpoints audited

- Seeds: 7, 13, 42.
- Availability rows: 3.
- Virtual-work identity rows: 408.
- Stress sample/selected rows: 129864.

## Classification

**stress postprocessing bug identified**.

## Main results

- Energy-autograd virtual-work median pre-through relative error: 8.36581e-08.
- Postprocessed sigma virtual-work median pre-through relative error: 1.08384.
- Median selected-checkpoint |sigma_yy_energy - sigma_yy_tm_eff|: 0.237099 kN/mm^2.
- Seeds where energy-autograd virtual work matches exact reaction: 3/3.
- Seeds where energy-autograd reaction collapses after through-crack: 3/3.

## Required questions

1. Does energy-autograd stress reproduce exact `dPi/dDelta` through virtual work?
   - Yes if the reported energy-autograd virtual-work relative error is near zero. See `tables/energy_autograd_virtual_work_identity.csv`.
2. Does postprocessed `sigma_eff` equal the energy-autograd stress for the same branch?
   - No when the stress-difference and postprocessed virtual-work error are nonzero. See `tables/energy_autograd_stress_vs_postprocessed_sigma.csv`.
3. Which formula path is responsible for the exact/legacy mismatch?
   - The mismatch is traced to using `sigma_total + (g_alpha - 1)*sigma_plus` as a postprocessed current stress while exact reaction differentiates `g_alpha*He_trial + psi_minus` with history/max branches and plane-stress auxiliary strain dependence.
4. Is the mismatch caused by history branch, stress split, shear convention, coordinate-gradient scaling, or another factor?
   - The audit points to stress formula/history-energy conjugacy. Shear and unit-box gradient conventions are internally consistent; see `tables/shear_and_gradient_scaling_audit.csv`.
5. Which reaction candidate is mathematically conjugate to the actual mechanics energy?
   - `R_energy_exact` and `R_virtual_energy_autograd_sigma` are conjugate to the checkpoint mechanics energy.
6. Which reaction candidate is suitable for future stress-strain curves?
   - This package identifies mathematically conjugate candidates but does not make a production policy change by itself.
7. Does the corrected energy-consistent reaction agree with legacy before through-crack onset?
   - See `tables/corrected_reaction_candidate_comparison.csv`; energy-consistent candidates retain the exact scaling rather than legacy scaling.
8. Does the corrected energy-consistent reaction collapse after alpha>=0.8 through-crack onset?
   - See `tables/corrected_reaction_candidate_summary.csv`.
9. Should `reaction_N_tm_eff` be demoted to legacy diagnostic?
   - The audit supports demoting it from energy-conjugate reaction status; ChatGPT should decide whether it remains a legacy diagnostic only.
10. Should D0040 remain deferred until this is resolved?
   - Yes.
11. Is any production mechanics change justified?
   - No mechanics change is made; a postprocessing/reaction metric change may be considered after review.

## Limits

- This is a reaction-definition audit, not physical validation.
- Energy-autograd boundary stress is a diagnostic stress path derived from selected energy branches.
- D0040 remains deferred.
