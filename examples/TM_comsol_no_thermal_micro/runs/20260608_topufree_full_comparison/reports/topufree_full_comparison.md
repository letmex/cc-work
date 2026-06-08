# Top-U-Free Full Comparison

## Scope

This package compares the old top-u fixed run against a controlled full top-u-free run for `history + alpha-init-intact + D0020 seed2`.

The top-u-free run changes only the top horizontal displacement ansatz route by adding `--top-u-mode free`. It keeps `history`, `alpha-init-intact`, `l0 = 1.5e-4 mm`, seed 2, D0020 extended schedule, 8x400 network, and full optimizer budget.

This is a boundary-equivalence diagnostic, not physical validation.

## Full Run Completion

The top-u-free full run completed with 34 saved `fields_mixed_tm_step_*.npz` files and a final diagnostics row at step 33, `Delta = 0.0001`.

## Absolute Drive Validity Check

The ratio validity threshold used in this package is:

`notch_tip_He_current_max > 1e-8`

For the top-u-free run:

| event | step | Delta | value |
|---|---:|---:|---:|
| first_step_where_ratio_valid | 0 | 1e-06 | 6.344892540255387e-07 |
| first_notch_He_gt_1e-8 | 0 | 1e-06 | 6.344892540255387e-07 |
| first_bulk_He_p95_gt_1e-8 | 0 | 1e-06 | 6.489555630651012e-07 |
| first_bulk_He_ratio_gt_0p5 | 0 | 1e-06 | 1.022799927576047 |

This means the step-0 broad-drive ratio is not only a zero-denominator artifact under the `1e-8` safety threshold. The notch-tip drive and bulk p95 drive both exceed `1e-8` at step 0, and the bulk/notch ratio is already greater than 0.5.

## Fixed vs Free Final Comparison

| metric | top-u fixed | top-u free |
|---|---:|---:|
| final alpha_mean | 0.48825829612572075 | 0.48817826120409724 |
| final alpha_std | 5.682864245045179e-05 | 4.0094150429576516e-05 |
| final alpha_max | 0.48843449354171753 | 0.4882843494415283 |
| final alpha_gt_0p5_area_fraction | 0.0 | 0.0 |
| final bulk/notch He ratio, valid | 1.000672508082938 | 1.0008025829954588 |
| final bottom/notch He ratio, valid | 0.9961673784409758 | 0.9995179868457842 |
| final reaction_N_tm_eff | -1.7251943349838257 | -1.7486419677734375 |
| final top_u_abs_max | 4.371138867531599e-12 | 4.0447113747177355e-07 |
| final top_v_error_max | 2.526212488436659e-12 | 0.0 |
| classification | B. medium-stage uniform -> full-stage still uniform | B. medium-stage uniform -> full-stage still uniform |

The top-u-free ansatz changes the sampled top horizontal displacement: `top_u_abs_max` increases from approximately `4.37e-12` to `4.04e-7`. The prescribed vertical displacement and bottom constraints remain sampled correctly: `top_v_error_max = 0.0`, `bottom_u_abs_max = 0.0`, and `bottom_v_abs_max = 0.0`.

## Reference: Old History Default-Alpha Run

The old history reference run remains qualitatively different:

| metric | old history reference |
|---|---:|
| final alpha_mean | 0.2144645048950973 |
| final alpha_std | 0.24571941633918865 |
| final alpha_max | 1.0018107891082764 |
| final bulk/notch He ratio, valid | 3.798645554151973e-05 |
| final bottom/notch He ratio, valid | 1.4269191047924905e-05 |
| final reaction_N_tm_eff | 0.5903642773628235 |
| classification | A. medium-stage uniform -> full-stage localized |

This reference is included only to show that notch-localized and background/uniform branches both exist in the current formulation. It is not evidence of physical validation.

## Required Answers

1. Did top-u-free full run complete?

Yes. The full D0020 seed2 top-u-free run completed with 34 analyzed steps.

2. Did top-u-free remove step-0 broad drive when using absolute-drive validity checks?

No. With the safety threshold `notch_tip_He_current_max > 1e-8`, the top-u-free run is ratio-valid at step 0. The top-u-free step-0 bulk/notch He ratio is `1.022799927576047`, and both notch-tip and bulk p95 He exceed `1e-8` at step 0.

3. Did top-u-free prevent final uniform alpha around 0.488?

No. The final top-u-free result has `alpha_mean = 0.48817826120409724`, `alpha_std = 4.0094150429576516e-05`, and `alpha_max = 0.4882843494415283`. This is still a near-uniform/background alpha field, not a localized crack.

4. Did top-u-free change reaction sign or magnitude?

It did not fix the sign. The fixed run has `reaction_N_tm_eff = -1.7251943349838257`, while the free run has `reaction_N_tm_eff = -1.7486419677734375`. The reaction remains negative and becomes slightly more negative in this run.

5. Is a further model change justified, or should the next step be another diagnostic?

This package does not justify a physical/model change by itself. It indicates that the top-boundary ansatz mismatch is not the main cause of the alpha-init uniform/background damage branch. The next step should be another minimal diagnostic targeted at why the alpha-init history run has broad/background drive from the first valid step, rather than changing `l0`, material parameters, TM split, or notch seeding.

## Bottom Line

Top-u-free is a platform-equivalence correction, but in this controlled full run it is insufficient to recover notch-localized damage. The alpha-init history run remains classified as:

`B. medium-stage uniform -> full-stage still uniform`
