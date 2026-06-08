# Exact FE Prefit Strain-Resolution Diagnostic

This package diagnoses why the PINN mechanics prefit can match the accepted direct sparse FE alpha=0 displacement target globally while failing to reconstruct local strain and `He_current`.

Package scope:

- Target: `runs/20260608_exact_fe_target_prefit/artifacts/exact_fe_topufree_alpha0_Delta1e-6_fields.npz`
- `alpha` fixed to zero
- `Delta = 1e-6`, top-u-free
- same material constants, `l0`, `tm_source` split, thermal terms, phase-field notch behavior, alpha seeding, and history update assumptions as previous diagnostics
- no coupled phase-field full run
- no notch/lip loss, local masks, local weights, displacement-jump target, enrichment, or geometry-label guidance

Read first:

1. `REPORT.md`
2. `tables/prefit_variant_comparison.csv`
3. `tables/coordinate_and_connectivity_checks.csv`
4. `tables/element_quality_error_correlation.csv`
5. `tables/optional_energy_continuation.csv`

Main result:

The current unscaled coordinate input is the dominant diagnosed issue. With the same 8x400 TrainableReLU network and a generic `[-1, 1]` input scaling wrapper, the prefit reaches exact-target-like strain and `He_current` reconstruction without any geometry-specific loss.
