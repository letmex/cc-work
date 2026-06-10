# D0020 Corrected Reaction Pipeline

## Scope

This package implements corrected energy-consistent reaction postprocessing for existing checkpointed D0020 TM runs. It uses seeds 7, 13, and 42 and does not run D0040, extend loading, retrain, or change physics.

## Classification

**corrected D0020 reaction pipeline validated; legacy reaction demoted**.

## Required Questions

1. Was corrected energy-consistent reaction postprocessing implemented?
   - Yes. `tables/corrected_reaction_by_step.csv` reports standardized corrected and legacy metric names.
2. Does `reaction_N_energy_exact` reproduce the previous exact D0020 reaction values for seeds 7/13/42?
   - Yes for 3/3 seeds; maximum absolute reproduction error is 2.22045e-16 N.
3. Does `reaction_N_energy_virtual_work` match exact autograd reaction within tolerance?
   - Yes for 3/3 seeds using relative-or-absolute tolerance; maximum absolute error is 2.32831e-07 N and maximum relative error is 3.23238e-06.
4. Does corrected D0020 softening gate pass for seeds 7/13/42?
   - It passes for 3/3 seeds using the corrected primary metric.
5. Does legacy top sigma reaction disagree with primary energy reaction after through-crack onset?
   - Yes in 3/3 seeds.
6. Is `reaction_N_tm_eff` formally demoted to legacy diagnostic-only status?
   - Yes. See `tables/legacy_reaction_metric_policy.csv` and `REACTION_METRIC_POLICY.md`.
7. How should old non-checkpointed D0020/D0040 no-softening conclusions be relabeled?
   - They should be relabeled as legacy-metric-only. Without checkpoints, primary softening classification is `reaction_metric_unavailable`, not `no_softening`.
8. Is D0040 still deferred?
   - Yes.
9. Is any production mechanics change justified?
   - No. This package changes postprocessing infrastructure and metric policy only.
10. What is the next minimal intervention?
   - Review whether to promote `reaction_N_energy_exact` into reusable checkpoint postprocessing code, then reprocess D0040 only after this D0020 policy is accepted.

## Key Tables

- `tables/corrected_reaction_by_step.csv`
- `tables/corrected_softening_gate_summary.csv`
- `tables/legacy_reaction_metric_policy.csv`
- `tables/exact_reaction_reproduction_check.csv`
- `tables/virtual_work_agreement_summary.csv`
- `tables/checkpoint_availability.csv`

## Limits

- This package does not claim physical validation.
- It does not modify mechanics training, material parameters, TM split, history logic, or load schedule.
- D0040 remains deferred.
