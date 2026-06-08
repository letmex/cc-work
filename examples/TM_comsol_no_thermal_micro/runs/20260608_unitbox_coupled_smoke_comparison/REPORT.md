# Unit-Box Coupled Smoke and Controlled Comparison Report

## Purpose

The goal was to first verify that the coupled training path runs with
`coord-normalization unit_box`, and then compare two otherwise matched
controlled runs:

- `history + alpha-init-intact + coord-normalization none`
- `history + alpha-init-intact + coord-normalization unit_box`

This is a controlled diagnostic, not a physical validation run.

## Commands and Setup

The coupled smoke used:

```text
main.py 2 20 2 TrainableReLU 3 --smoke --alpha-init-intact --top-u-mode free --coord-normalization unit_box --run-suffix coord_unitbox_coupled_smoke
```

The controlled comparison used:

```text
main.py 4 100 2 TrainableReLU 3 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --alpha-init-intact --top-u-mode free --coord-normalization none --run-suffix coord_cmp_none_m12

main.py 4 100 2 TrainableReLU 3 --max-steps 12 --n-rprop 300 --n-lbfgs 0 --alpha-init-intact --top-u-mode free --coord-normalization unit_box --run-suffix coord_cmp_unitbox_m12
```

All runs used the existing COMSOL micro-notch mesh, `mixedH_TM`,
`tm_source`, `mixed-mechanics-mode history`, `AT2`, `l0=1.5e-4`, and the
existing history update logic.

## Smoke Result

The `unit_box` coupled smoke completed successfully and generated
`fields_mixed_tm_step_0000.npz` plus `diagnostics_mixed_tm_summary.csv`.

Key smoke checks:

- `coord_normalization = unit_box`
- `x_hat_min = -1`, `x_hat_max = 1`
- `y_hat_min = -1`, `y_hat_max = 1`
- `t3_gradients_use_physical_xy = True`
- `top_v_error_max = 0`
- `bottom_u_abs_max = 0`
- `bottom_v_abs_max = 0`

Conclusion: the `unit_box` coupled path is not broken at smoke level.

## Controlled Comparison

Main tables:

```text
tables/final_case_comparison.csv
tables/stepwise_summary.csv
tables/broadening_events.csv
```

Final-step comparison at step 11, `Delta = 4.6e-5`:

| case | coord | alpha_mean | alpha_std | alpha_max | bulk He / notch He | bottom He / notch He | reaction_N_tm_eff | classification |
|---|---|---:|---:|---:|---:|---:|---:|---|
| none_alpha_init_history_m12 | none | 0.167670 | 0.002104 | 0.174432 | 1.005457 | 0.963485 | 2.157921 | B-like still uniform |
| unitbox_alpha_init_history_m12 | unit_box | 0.103954 | 0.155334 | 1.000465 | 0.000344 | 0.000279 | 0.706135 | A-like localized |

The raw-coordinate run shows nearly uniform background damage: low alpha
standard deviation, alpha_max only slightly above alpha_mean, and bulk/notch
drive ratio near 1.

The unit-box run shows localized damage at the notch region: alpha_max reaches
about 1 while alpha_mean is lower, and bulk/bottom drive ratios relative to the
notch are about `3e-4`.

## Event Ordering

For the `none` case:

- `first_alpha_mean_gt_0p05`: step 5, `Delta = 2.5375e-5`
- `first_bulk_He_ratio_gt_0p25`: step 0
- final bulk/notch He ratio stays near 1

For the `unit_box` case:

- `first_alpha_mean_gt_0p05`: step 6, `Delta = 3.025e-5`
- `first_bulk_He_ratio_gt_0p25`: not reached
- final bulk/notch He ratio is `0.000344`

Interpretation: coordinate normalization changes the coupled alpha-init history
branch from a uniform/background branch to a notch-localized branch under this
12-step controlled diagnostic.

## Figures

Final field figures are included under:

```text
figures/none_m12/
figures/unitbox_m12/
```

The important files are:

- `figures/none_m12/final_alpha_none_m12.png`
- `figures/none_m12/final_He_current_none_m12.png`
- `figures/unitbox_m12/final_alpha_unitbox_m12.png`
- `figures/unitbox_m12/final_He_current_unitbox_m12.png`
- `figures/none_m12/final_fields_panel_none_m12.png`
- `figures/unitbox_m12/final_fields_panel_unitbox_m12.png`

The visual takeaway is summarized in `figures/figure_summary.md`.

## Verification

Project-local tests:

```text
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
18 passed
```

`py_compile` passed for the modified TM project files.

Repository-root tests were attempted:

```text
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

They failed during collection because `ref_files.Chinese_SENT_reproduction` is
not available. This is outside `TM_comsol_no_thermal_micro`.

## What Can Be Concluded

- `coord-normalization unit_box` works in the coupled alpha-init history path at
  smoke level.
- In the matched 12-step controlled comparison, `unit_box` changes the observed
  branch from uniform/background damage to notch-localized damage.
- The result is consistent with the previous mechanics-only evidence that raw
  physical-mm NN inputs were harming the displacement ansatz conditioning.

## What Cannot Be Concluded

- This does not prove physical validation.
- This does not prove robustness over seeds.
- This does not replace a full-run package.
- This does not justify changing `l0`, material parameters, TM split, alpha
  seeding, phase-field notch behavior, thermal terms, or history update logic.
- The reaction differs strongly between the two controlled branches, so reaction
  behavior still needs to be watched in any longer run.

## Suggested Next Step

If ChatGPT agrees, run a longer `unit_box` alpha-init history comparison with
the improved package structure, preferably with at least one additional seed or
a staged medium-to-full run, while preserving all physical/model settings.

