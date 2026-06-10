# Reaction Metric Policy

This note applies to checkpointed TM D0020 postprocessing in this package.

1. `reaction_N_tm_eff` is demoted because it integrates the legacy postprocessed current stress `sigma_tm_eff` on the top boundary. The prior conjugacy audit showed this stress path is not conjugate to the checkpoint history mechanics objective.
2. The energy-conjugate primary metric is `reaction_N_energy_exact`, computed as exact autograd `dPi/dDelta` from the checkpoint mechanics objective. `reaction_N_energy_virtual_work` is a validation-equivalent metric only when it matches the exact reaction within tolerance.
3. Checkpoint availability is required because the exact metric depends on reconstructing the model state, previous-step history fields, displacement mode, and checkpoint mechanics objective.
4. Old runs without checkpoints must be labeled `reaction_metric_unavailable` for primary softening classification. They must not be relabeled as `no_softening` from legacy `reaction_N_tm_eff` alone.
5. D0040 should remain deferred until the corrected D0020 reaction pipeline and metric policy are accepted. Later D0040 processing should use the same corrected metric names and checkpoint requirements.
6. Do not claim physical validation from this package. It is a postprocessing infrastructure and metric-policy update only.
