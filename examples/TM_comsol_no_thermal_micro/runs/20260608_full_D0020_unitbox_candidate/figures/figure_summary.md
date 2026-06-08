# Figure Summary

## Cross-Case Figures

### `final_alpha_compare.png`

- What it plots: final alpha at full D0020 step 33 for the primary
  `alpha-init-intact + unit_box` run and optional `default-alpha + unit_box`
  run.
- Visual takeaway: both cases retain a notch-localized high-alpha zone; the
  default-alpha and alpha-init-intact branches look similar at final step.
- Evidence status: supports a full-schedule, single-seed diagnostic
  observation; not physical validation.

### `final_He_current_compare.png`

- What it plots: final `He_current` at step 33 for both full runs.
- Visual takeaway: both cases keep the maximum current drive at the notch
  neighborhood.
- Evidence status: supports the classification `12-step localized ->
  full-stage localized`.

### `final_mechanics_drive_compare.png`

- What it plots: final mechanics-drive field at step 33 for both full runs.
- Visual takeaway: mechanics-drive remains notch-centered in both branches.
- Evidence status: diagnostic observation only.

### `reaction_strain_compare.png`

- What it plots: `reaction_N_tm_eff` versus engineering strain for both full
  runs.
- Visual takeaway: reactions remain positive; the default-alpha branch has a
  slightly higher final reaction than the alpha-init-intact branch.
- Evidence status: branch diagnostic only; not a physical stiffness validation.

### `stress_strain_compare.png`

- What it plots: TM effective engineering stress versus engineering strain.
- Visual takeaway: the two unit-box branches follow similar stress-strain
  trends, with small separation at later steps.
- Evidence status: diagnostic observation only.

## Per-Case Figure Folders

### `figures/intact_unitbox_full/`

- Contains final alpha, displacement, `HI`, `HII`, `He`, `He_current`,
  `He_history`, mechanics-drive, field panel, reaction-strain, and
  stress-strain figures for the primary full run.
- Visual takeaway: primary full run remains notch-localized through final step.
- Evidence status: primary full-schedule candidate evidence for seed 2.

### `figures/default_unitbox_full/`

- Contains the same figure set for the optional default-alpha full run.
- Visual takeaway: default-alpha + unit_box also remains notch-localized
  through final step for seed 2.
- Evidence status: optional branch-selection comparison, not physical
  validation.

