# Drive-Broadening and Top-U-Free Diagnostic Package

This package records a diagnostic pass for `TM_comsol_no_thermal_micro`.

Scope:
- Analyze existing full D0020 seed 2 runs step by step.
- Check whether the drive field broadens before alpha grows, or alpha grows before drive broadens.
- Add and smoke-test boundary diagnostics for the existing `--top-u-mode free` ansatz route.
- Provide a compact handoff package for ChatGPT review.

This is not a physical validation package.

Model constants and physics left unchanged:
- No change to `l0`.
- No change to `Gc` or `GcII`.
- No change to `E`, `nu`, material parameters, or TM split.
- No phase-field notch was added.
- No `alpha=1` geometric notch seeding was imposed.
- No thermal expansion or thermal field setting was changed.

Top-u-free scope:
- The top-u-free smoke is a boundary-equivalence diagnostic only.
- It does not establish physical correctness.
- It is not a full D0020 production run.

Read first:
- `REPORT.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary.csv`
- `tables/topufree_smoke_summary.csv`
- `reports/drive_broadening_alpha_intact_history_full_seed2.md`
- `reports/drive_broadening_alpha_intact_current_split_full_seed2.md`
- `reports/drive_broadening_old_history_full_seed2.md`
- `reports/drive_broadening_old_current_split_full_seed2.md`
