# Next questions

1. Should the mainflow always save step checkpoints for full runs so exact corrected reaction is computable by default?
2. Is the current fallback policy correct: write `reaction_metric_unavailable` when checkpoints are absent instead of using legacy top sigma as primary?
3. Should old D0020 packages be regenerated through this new example-local postprocessor, or only future runs?
