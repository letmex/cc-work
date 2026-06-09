# Checkpointed D0020 exact reaction diagnostic

## Scope

This package prioritizes D0020 checkpointed exact-reaction postprocessing. It uses the default-unitbox D0020 route: mixed-mechanics-mode history, default alpha initialization, top-u-mode free, coord-normalization unit_box, the same D0020 load schedule, material parameters, l0, TM split, and history logic.

No D0040 rerun is included in this package.

## Runs processed

- Seeds with exact rows: 7, 13, 42.
- Availability rows: 5.
- Exact reaction rows: 102.
- FD check rows: 714.

## History-state convention

For step j, exact autograd `dPi/dDelta` is computed from the checkpointed network at step j using the committed history from step j-1 as the fixed pre-step history. Step 0 uses zero history. This matches the production history objective more closely than differentiating through the post-step committed history at an equality point of the max-history operator.

## Seed 42 smoke answer

- First alpha>=0.8 through-crack step: 14.
- Exact post-peak drop [%]: 99.5469.
- Legacy top post-peak drop [%]: 5.45402.
- Exact final / legacy final absolute ratio: 0.00956098.
- FD p95 relative error: 0.313971.

## Classification

**exact reaction unresolved**.

## Acceptance criteria check

- Seeds with exact post-through collapse stronger than legacy: 3/3.
- Seeds with pre-through exact/legacy median ratio in [0.8, 1.25]: 0/3.
- Seeds with branch-stable FD p95 absolute error below 0.005 N: 3/3.
- The exact energy-conjugate reaction collapses after alpha>=0.8 through-crack for all processed primary seeds, while the legacy top-boundary sigma metric remains high.
- The acceptance condition requiring pre-through exact/legacy agreement is not met by the current checkpointed calculation, so this package keeps the conservative classification as unresolved.
- FD relative errors are inflated near small reactions and branch changes of the max-history operator; branch-stable absolute errors are small and are tabulated separately.

## Primary question

Does exact actual-PINN energy-conjugate reaction show post-through-crack softening or collapse in D0020, while legacy top-boundary sigma reaction remains high?

See `tables/exact_reaction_summary_by_seed.csv`, `tables/acceptance_criteria_check.csv`, `tables/pinn_energy_conjugate_reaction_by_checkpoint.csv`, and `figures/figure_summary.md`.

## Limits

- This is a reaction-metric diagnostic, not physical validation.
- Exact reaction is computed on saved checkpoint branches; it does not retrain or relax the branch after postprocessing.
- Optional robustness seeds 21 and 99 were not required and were not rerun in this package.
