# Project Structure

The example root is source-only. Keep formal source files, tests, documentation,
mesh/load-schedule inputs, and managed top-level directories here.

Generated outputs are managed under `outputs/`:

- `outputs/checkpoints/<short_run_id>/` for model checkpoints and saved training states
- `outputs/results/<short_run_id>/` for field NPZ files, diagnostics, curves, and figures
- `outputs/results/<short_run_id>/curves/` for stress-strain and reaction tables
- `outputs/results/<short_run_id>/figures/` for generated figures
- `outputs/logs/<short_run_id>/` for TensorBoard and run logs
- `outputs/debug/<debug_task_id>/` for temporary diagnostics

Audit and handoff packages belong under `runs/<date_task_name>/`.

Debug scripts should not be left in the example root. Root one-off diagnostics should
either be converted into formal tests/tools or placed in a named external audit
package. Do not commit generated CSV, NPZ, checkpoint, log, or run folders unless
they are intentionally reduced into small audit tables.

Normal training uses short run IDs through `--run-suffix` and writes to managed
directories by default. Advanced users may override paths with `--output-root`,
`--model-dir`, `--result-dir`, `--log-dir`, `--figure-dir`, and `--curve-dir`.

`reaction_N_energy_exact` is the primary reaction metric when checkpoint exact
reaction is available. `reaction_N_legacy_top_sigma` is diagnostic only.

This structure policy does not claim physical validation and does not change
physics, material parameters, `l0`, TM split formulas, history logic, alpha
initialization behavior, or training losses.
