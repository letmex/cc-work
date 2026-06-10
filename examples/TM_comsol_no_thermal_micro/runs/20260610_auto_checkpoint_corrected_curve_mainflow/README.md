# Auto Checkpointed Corrected Curve Mainflow

This package documents the change that makes corrected stress-strain curves available immediately after `TM_comsol_no_thermal_micro` training.

Source tree changed locally:

`D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`

Main behavior change:

- `--save-step-checkpoints` now defaults to `true`.
- Training already had per-step checkpoint writing in `train_mixed_tm.py`; the missing link was the default config gate.
- Because `main.py` already calls `run_corrected_reaction_postprocess(...)` after training, a normal training run now has the checkpoint state needed to compute `dPi/dDelta` and write `results/<run>/curves/corrected_stress_strain_by_step.csv`.

Read first:

1. `REPORT.md`
2. `tables/default_checkpoint_smoke_summary.csv`
3. `tables/smoke_corrected_reaction_availability.csv`
4. `tables/smoke_corrected_stress_strain_by_step.csv`
5. `artifacts/source_patch.diff`
