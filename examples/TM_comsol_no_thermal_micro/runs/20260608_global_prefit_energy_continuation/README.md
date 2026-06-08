# Global Prefit Energy Continuation Diagnostic

This package tests whether the current PINN displacement ansatz can enter and preserve an FE-DOF-like mechanics branch without geometry-specific training guidance.

## Scope

- Project: `examples/TM_comsol_no_thermal_micro`
- Diagnostic mode: mechanics-only
- Target: FE-DOF alpha=0 mechanics baseline
- Delta: `1e-6`
- Top boundary mode: `top-u free`
- Alpha: fixed to zero during this diagnostic
- No coupled phase-field full run was performed.

## Training Modes

The diagnostic used four mechanics-only variants:

1. `random_init_energy`: random initialization followed by alpha-zero mechanics energy optimization.
2. `disp_global_prefit_then_energy`: global displacement prefit against FE-DOF target, then mechanics energy optimization.
3. `disp_strain_global_prefit_then_energy`: global displacement plus global element-strain prefit, then mechanics energy optimization.
4. `global_curriculum`: global displacement/strain prefit loss gradually ramped to mechanics energy loss.

The training loss did not use notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or any explicit geometry label guiding localization. Region metrics are postprocessing diagnostics only.

## Files

- `REPORT.md`: main interpretation.
- `tables/global_prefit_case_comparison.csv`: main numerical comparison.
- `tables/energy_continuation_metrics.csv`: energy-stage and branch classification metrics.
- `tables/global_strain_reconstruction_metrics.csv`: displacement/strain/He reconstruction metrics.
- `figures/figure_summary.md`: text summary of generated figures.
- `artifacts/debug_prefit_then_energy_mechanics.py`: script snapshot used for the diagnostic.
- `HANDOFF_COMMENT.md`: GitHub issue handoff text.

