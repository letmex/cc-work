# Report: auto corrected reaction mainflow

## Change summary

Implemented an example-local corrected reaction postprocess path.

New file:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/corrected_reaction_postprocess.py`

Updated file:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/main.py`

Updated plotting file:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py`

The training entry point now calls `run_corrected_reaction_postprocess(...)` after `train_mixed_tm(...)`. The postprocessor writes into:

`results/<run>/curves/corrected_stress_strain_by_step.csv`

When step checkpoints are available, it rebuilds the checkpointed model state, recomputes the actual PINN energy, and uses autograd to compute:

`reaction_N_energy_exact = dPi/dDelta`

When checkpoints are absent, it still writes a corrected-curve CSV with:

`stress_strain_primary_metric = reaction_metric_unavailable`

This prevents the plotting flow from silently treating legacy top-boundary sigma integration as the primary stress-strain response.

## Mainflow behavior

- Checkpoints available:
  - write `corrected_reaction_by_step.csv`
  - write `corrected_stress_strain_by_step.csv`
  - mark `exact_reaction_computable = True`
  - primary stress-strain metric is `nominal_stress_energy_exact_MPa`

- Checkpoints absent:
  - write `corrected_stress_strain_by_step.csv`
  - mark `stress_strain_primary_metric = reaction_metric_unavailable`
  - mark `exact_reaction_computable = False`
  - legacy top sigma remains diagnostic only

## Smoke verification

Smoke input:

Existing checkpointed D0020 seed 42 default-unitbox run, 34 step checkpoints.

Postprocess output:

- status: `energy_exact_primary`
- exact reaction computable: `True`
- corrected curve rows: 34
- primary metric: `nominal_stress_energy_exact_MPa`
- primary status: `energy_conjugate_primary`

Compact numerical check:

- primary peak: 182.820635382086 MPa
- primary final: 0.828355223347899 MPa
- primary final/peak: 0.00453097223744288
- legacy top peak: 91.6370766815224 MPa
- legacy top final: 86.6391686779654 MPa
- legacy final/peak: 0.945459761653825

Plotting chain check:

`plot_clean_tm_results.py` was run without `--corrected-stress-strain-csv` after the postprocessor wrote `result_dir/curves/corrected_stress_strain_by_step.csv`. It auto-discovered the corrected CSV from `result_dir/curves` and generated stress/reaction figures from the corrected primary metric.

## Tests

- New TDD tests initially failed for missing module and missing main hook.
- After implementation, `tests/test_corrected_reaction_postprocess.py` passed.
- A follow-up RED test showed that plotting did not auto-discover `result_dir/curves`; the plotting search path was then fixed.
- Full example-local test suite passed: 22 passed.
- `py_compile` passed for:
  - `corrected_reaction_postprocess.py`
  - `main.py`
  - `plot_clean_tm_results.py`

## Constraints preserved

- No D0040 run or processing.
- No physical-model change.
- No change to `l0`.
- No change to material parameters.
- No change to TM split.
- No imposed `alpha=1` notch.
- No shared `source/` modification.

## Interpretation

The previous gap was not that exact reaction could not be computed; it was not wired into the normal example-local flow. This change makes corrected CSV generation part of the example-local mainflow. If checkpoints exist, exact `dPi/dDelta` is computed. If checkpoints do not exist, the run records that the primary corrected reaction metric is unavailable instead of falling back to a misleading legacy primary curve.

This package is a pipeline verification, not new physical validation.
