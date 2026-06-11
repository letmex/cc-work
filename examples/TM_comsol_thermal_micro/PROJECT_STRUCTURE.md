# Project Structure

This is a sibling scaffold copied from `examples/TM_comsol_no_thermal_micro`.
The initial source is intentionally no-thermal and is expected to match the
verified baseline before prescribed-temperature thermal-strain work begins.

```text
examples/TM_comsol_thermal_micro/
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
    load_schedules/
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
- `outputs/results/`: field data, diagnostics, curves, and figures for a run.
- `outputs/logs/`: TensorBoard logs.
- `outputs/debug/`: temporary development diagnostics.
- `runs/`: reduced evidence packages and handoff reports.

The root folder should not contain generated run directories, debug CSV/NPZ
files, cache directories, or one-off debug scripts.

## Current Route

The copied starting route uses the `tm_source` energy split, history mechanics
objective, top-u-free ansatz, unit-box network input normalization, and AT2
phase-field model. Postprocessing uses `postprocess_results.py` and writes
energy-conjugate reaction/stress-strain outputs with functional filenames.

## Thermal Work Boundary

Thermal strain, heat PDE, and damage-dependent conductivity are not implemented
in this scaffold. Future thermal changes should be made only in this copied
subproject while keeping `examples/TM_comsol_no_thermal_micro` as the frozen
baseline.
