# Project Cleanup Report

Classification: `project cleanup completed`

## Answers

1. Were default route options moved out of normal required command usage?
   Yes. `pff-model=AT2`, `mixed-mechanics-mode=history`, `top-u-mode=free`,
   and `coord-normalization=unit_box` are code defaults. They remain exposed as
   advanced overrides.

2. What is the new minimal normal D0020 command?

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-schedule-file load_schedule_D0020_extended.csv --run-suffix seed23_D0020
```

3. Were program files renamed away from `clean` / `corrected` naming?
   Yes. Normal entry points are `postprocess_results.py` and `plot_results.py`.
   The old filenames remain only as deprecated compatibility wrappers.

4. Were output files renamed to functional names?
   Yes. New postprocess outputs use `stress_strain_by_step.csv`,
   `reaction_by_step.csv`, `reaction_metric_availability.csv`, and figure names
   such as `stress_strain_seed23_D0020.png`.

5. Is `postprocess_results.py` now the normal postprocess entry point?
   Yes. It writes the stress-strain table and invokes figure generation by
   default.

6. Are metric names explicit and policy-compliant?
   Yes. Energy-exact, virtual-work, legacy top sigma, bottom sigma, and internal
   cut metrics remain explicitly named.

7. Is legacy top sigma diagnostic-only?
   Yes. `legacy_curve_status=legacy_diagnostic_only` is written in the
   stress-strain table; plotting labels it as a diagnostic curve.

8. Does training completion use the renamed postprocess path and short labels?
   Yes. `main.py` imports `run_results_postprocess` and passes
   `run_label=run_suffix or None`.

9. Do tests pass?
   Yes. Focused cleanup/postprocess tests passed and the full local test suite
   passed. See `tables/test_cleanup_summary.csv`.

10. Was any physics changed?
    No. The cleanup did not change material parameters, `l0`, TM split formulas,
    history logic, alpha initialization behavior, or training losses.

11. Is any additional seed run required?
    No. This was an interface cleanup. No new seed study was run.

## User-Facing Behavior

Training now saves step checkpoints by default and, on completion, calls the
same functional postprocess path that users can run manually:

```powershell
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir <model_dir> --result-dir <result_dir>
```

Defaults:

- `out-dir = <result_dir>/curves`
- `figure-dir = <result_dir>/figures`
- figures enabled by default
- figure failure non-fatal by default
- strict figure mode available through `--fail-on-figure-error`

`plot_results.py` remains available as an optional direct plotting helper, but
normal users do not need a second plotting command after training/postprocess.

## Limits

This package does not provide physical validation. It only verifies the cleaned
interface, naming, output policy, documentation, and tests.

