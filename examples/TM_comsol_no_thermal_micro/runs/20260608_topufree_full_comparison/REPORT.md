# Top-U-Free Full Diagnostic Report

## 1. Did the Top-U-Free Full Run Complete?

Yes.

The controlled full D0020 seed2 run completed with:

- `history`
- `alpha-init-intact`
- `--top-u-mode free`
- `l0 = 1.5e-4 mm`
- seed 2
- D0020 extended load schedule
- 8 hidden layers, 400 neurons
- full optimizer budget

The analyzed result contains 34 saved field NPZ files and a final diagnostics row at step 33, `Delta = 0.0001`.

## 2. Did Top-U-Free Remove Step-0 Broad Drive?

No.

The stepwise analysis now uses a ratio validity threshold:

`notch_tip_He_current_max > 1e-8`

For the top-u-free run:

| event | step | Delta | value |
|---|---:|---:|---:|
| `first_step_where_ratio_valid` | 0 | 1e-06 | 6.344892540255387e-07 |
| `first_notch_He_gt_1e-8` | 0 | 1e-06 | 6.344892540255387e-07 |
| `first_bulk_He_p95_gt_1e-8` | 0 | 1e-06 | 6.489555630651012e-07 |
| `first_bulk_He_ratio_gt_0p5` | 0 | 1e-06 | 1.022799927576047 |

Therefore, under the chosen denominator safety threshold, the step-0 broad drive is not just a zero-denominator ratio artifact.

## 3. Did Top-U-Free Prevent Final Uniform Alpha Around 0.488?

No.

Final comparison:

| case | top_u_mode | final alpha_mean | final alpha_std | final alpha_max | final bulk/notch He ratio | classification |
|---|---|---:|---:|---:|---:|---|
| `alpha_intact_history_topufixed_full_seed2` | fixed | 0.48825829612572075 | 5.682864245045179e-05 | 0.48843449354171753 | 1.000672508082938 | B. medium-stage uniform -> full-stage still uniform |
| `alpha_intact_history_topufree_full_seed2` | free | 0.48817826120409724 | 4.0094150429576516e-05 | 0.4882843494415283 | 1.0008025829954588 | B. medium-stage uniform -> full-stage still uniform |
| `old_history_full_seed2_reference` | fixed | 0.2144645048950973 | 0.24571941633918865 | 1.0018107891082764 | 3.798645554151973e-05 | A. medium-stage uniform -> full-stage localized |

The top-u-free full result remains a near-uniform/background alpha field.

## 4. Did Top-U-Free Change Reaction Sign or Magnitude?

It did not fix the sign.

| case | reaction_N_tm_eff |
|---|---:|
| top-u fixed alpha-init history | -1.7251943349838257 |
| top-u free alpha-init history | -1.7486419677734375 |
| old history reference | 0.5903642773628235 |

The top-u-free run remains negative and is slightly more negative than the fixed run in this diagnostic.

## 5. Is Further Model Change Justified?

This package does not justify a physical/model change by itself.

It supports a narrower conclusion: the top-boundary ansatz mismatch is not the main cause of the alpha-init history run's step-0 broad/background drive and final near-uniform alpha around 0.488.

The next step should be another minimal diagnostic focused on why the alpha-init history run has broad/background drive from the first valid step. Do not change `l0`, material parameters, TM split, phase-field notch seeding, thermal field, or history update logic based only on this one-seed evidence.

## 6. Verification and GitHub Handoff

Verification passed:

- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q`
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py plot_clean_tm_results.py debug_recompute_he_current.py main.py`

Test result:

`13 passed in 0.05s`

GitHub CLI status:

- Bare `gh` was not available in the current PowerShell PATH.
- `C:\Program Files\GitHub CLI\gh.exe` exists and reports version `2.93.0`.
- `gh auth status` reports no logged-in GitHub hosts.
- `GH_TOKEN` and `GITHUB_TOKEN` were not present.
- Therefore this package uses markdown-only handoff and was not automatically posted to issue #1.

## Files to Read

- `reports/topufree_full_comparison.md`
- `tables/final_case_comparison.csv`
- `tables/stepwise_summary_topufixed_vs_topufree.csv`
- `tables/broadening_events_topufixed_vs_topufree.csv`
- `reports/drive_broadening_alpha_intact_history_topufree_full_seed2.md`
- `figures/figure_summary.md`

## Classification

The new top-u-free full run is classified as:

`B. medium-stage uniform -> full-stage still uniform`

This is a diagnostic classification, not physical validation.
