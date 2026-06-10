# Project Structure

The example root contains source files, tests, mesh input, and load schedules.
Generated artifacts should stay in managed folders.

```text
examples/TM_comsol_no_thermal_micro/
    main.py
    config.py
    train_mixed_tm.py
    compute_energy_mixed_tm.py
    mixed_mode_tm.py
    history_field_mixed_tm.py
    postprocess_results.py
    plot_results.py
    source/
    tests/
    outputs/
        checkpoints/
        results/
        figures/
        curves/
        logs/
        debug/
    runs/
```

## Managed Outputs

- `outputs/checkpoints/`: model settings, trained weights, and per-step checkpoints.
- `outputs/results/`: field NPZ/CSV data, diagnostics, curves, and figures for a run.
- `outputs/logs/`: TensorBoard logs.
- `outputs/debug/`: temporary development diagnostics.
- `runs/`: reduced evidence packages and handoff reports.

The root folder should not contain generated run directories, debug CSV/NPZ
files, cache directories, or one-off debug scripts.

## Normal Route

The single supported training route uses the `tm_source` energy split, history
mechanics objective, top-u-free ansatz, unit-box network input normalization, and
AT2 phase-field model. Postprocessing uses `postprocess_results.py` and writes
energy-conjugate reaction/stress-strain outputs with functional filenames.
