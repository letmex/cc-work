# D0020 reaction-mode audit

## Scope

This package audits the pre-through exact/legacy reaction scaling mismatch using existing checkpointed D0020 default-unitbox runs only. It does not retrain, extend loading, use D0040, or change physics.

## Checkpoints audited

- Seeds: 7, 13, 42.
- Checkpoint rows: 102.
- Reaction-mode rows: 408.

## Classification

**reaction-mode audit unresolved**.

## Main findings

- Global Delta pre-through exact/legacy median ratio: 2.17453.
- Pure top-vertical pre-through exact/legacy median ratio: 2.17276.
- No-horizontal Delta pre-through exact/legacy median ratio: 2.17478.
- Pure top-vertical collapse count above 50% post-peak drop: 3/3.
- Current-stress virtual-work median pre-through relative error for global Delta: 1.08384.
- Removing horizontal Delta contributions does not remove the approximately 2.17 pre-through ratio.
- The pure top-vertical diagnostic mode still gives an approximately 2.17 pre-through ratio, so legacy top reaction is not demoted by the acceptance rule.
- The remaining mismatch is most consistent with a current stress-postprocessing / actual-history-energy conjugacy mismatch, not a simple unit conversion or horizontal-mode issue.

## Required questions

1. Does `R_energy_exact` match the virtual work under the same Delta mode?
   - Not when virtual work is computed with the current postprocessed effective stress. Energy-consistent mode derivatives match by construction, but the current stress field is not conjugate to the exact history-energy branch. See `tables/virtual_work_identity_check.csv`.
2. What displacement/strain mode does Delta control under `top-u-mode free`?
   - Delta scales the top vertical affine mode and also learned horizontal/internal modes. See `tables/delta_loading_mode_decomposition.csv`.
3. Which term explains the 2.16-2.18 pre-through exact/legacy ratio?
   - The no-horizontal and pure-top-vertical ratios remain near 2.17, while current-stress virtual work and boundary work remain closer to legacy. The mismatch is therefore not explained by horizontal Delta mode alone; see the energy and virtual-work tables.
4. Is the mismatch a unit/scaling bug, loading-mode issue, history-branch issue, or unresolved?
   - Current classification: reaction-mode audit unresolved. No unit factor or horizontal-mode removal reconciles the mismatch; the stress-postprocessing / history-energy conjugacy remains unresolved.
5. Does pure top-vertical energy-conjugate reaction agree with legacy before through-crack onset?
   - No. The pure top-vertical pre-through median ratio remains about 2.17; see `tables/reaction_by_loading_mode_summary.csv`.
6. Does the pure top-vertical reaction collapse after through-crack onset?
   - Yes in this diagnostic, but because it does not agree with legacy before through-crack onset, this does not satisfy the legacy-demotion acceptance rule.
7. Should legacy `reaction_N_tm_eff` be demoted from primary softening gate?
   - Not by the acceptance rule in this package. Legacy remains non-softening after through-crack, but a corrected reaction definition still needs resolution.
8. Should global `dPi/dDelta` or corrected pure-top-vertical reaction be used going forward?
   - Neither is accepted as the final stress-strain reaction yet; the next minimal step should reconcile actual-energy derivatives with the stress/virtual-work path.
9. Is D0040 still deferred until D0020 reaction mode is settled?
   - Yes.
10. Is any production mechanics change justified?
   - No production mechanics change is justified by this package.

## Files

- `tables/delta_loading_mode_decomposition.csv`
- `tables/exact_reaction_energy_term_decomposition.csv`
- `tables/virtual_work_identity_check.csv`
- `tables/boundary_work_decomposition.csv`
- `tables/reaction_by_loading_mode.csv`
- `tables/reaction_unit_scaling_audit.csv`
- `tables/prethrough_linear_sanity_check.csv`
- `figures/figure_summary.md`

## Limits

- This is a reaction-definition diagnostic, not physical validation.
- Boundary work is estimated from elementwise effective stresses on boundary edges.
- Pure-top-vertical and no-horizontal modes are diagnostic perturbations on saved checkpoints, not retrained solutions.
