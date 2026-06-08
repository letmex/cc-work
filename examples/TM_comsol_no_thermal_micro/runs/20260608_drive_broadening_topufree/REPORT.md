# Drive-Broadening and Top-U-Free Diagnostic Report

## 1. Full Runs Found

Four existing full D0020 seed 2 runs were analyzed from the local PINN result folders:

| case | mode | alpha init | steps | final classification |
|---|---|---|---:|---|
| `alpha_intact_history_full_seed2` | history | intact | 34 | B. medium-stage uniform -> full-stage still uniform |
| `alpha_intact_current_split_full_seed2` | current_split | intact | 34 | B. medium-stage uniform -> full-stage still uniform |
| `old_history_full_seed2` | history | default | 34 | A. medium-stage uniform -> full-stage localized |
| `old_current_split_full_seed2` | current_split | default | 34 | C. medium-stage uniform -> full-stage boundary/corner damage |

No new expensive full training was started for this package.

## 2. Stepwise Drive-Broadening Findings

For `alpha_intact_history_full_seed2`, the drive field is already broad at step 0:

- `first_bulk_He_ratio_gt_0p5`: step 0, value 1.0137009507038128
- `first_bottom_He_ratio_gt_0p5`: step 0, value 1.2000986143914512
- `first_alpha_mean_gt_0p05`: step 5, value 0.05593346455947702

This ordering supports the diagnostic label:

`B: He_current broadens before alpha threshold`

For `alpha_intact_current_split_full_seed2`, the same ordering appears:

- `first_bulk_He_ratio_gt_0p5`: step 0, value 1.0137009507038128
- `first_bottom_He_ratio_gt_0p5`: step 0, value 1.2000986143914512
- `first_alpha_mean_gt_0p05`: step 5, value 0.05594045915242441

This also supports:

`B: He_current broadens before alpha threshold`

For `old_history_full_seed2`, alpha grows while bulk/bottom drive ratios remain below the thresholds through the analyzed final step:

- `first_alpha_mean_gt_0p05`: step 6, value 0.0585224844554778
- no `first_bulk_He_ratio_gt_0p25`
- no `first_bulk_He_ratio_gt_0p5`
- final `bulk_He_current_p95_over_notch_tip_He_current_max`: 3.798645554151973e-05

This supports:

`A: alpha broadens before bulk drive ratios`

For `old_current_split_full_seed2`, the drive field broadens early and then the final hotspot is boundary/corner dominated:

- `first_bulk_He_ratio_gt_0p5`: step 1, value 0.9990037154876901
- `first_bottom_He_ratio_gt_0p5`: step 1, value 1.0472766611377948
- `first_alpha_mean_gt_0p05`: step 6, value 0.05594097530856812
- final `bottom_right_He_current_max_over_notch_tip_He_current_max`: 95841.60507747109

This supports:

`C. medium-stage uniform -> full-stage boundary/corner damage`

## 3. Final Case Comparison

The final comparison table is in `tables/final_case_comparison.csv`.

Key final snapshots:

| case | final alpha_mean | final alpha_max | final drive pattern | reaction_N_tm_eff |
|---|---:|---:|---|---:|
| `alpha_intact_history_full_seed2` | 0.48825829612572075 | 0.48843449354171753 | broad/background | -1.7251943349838257 |
| `alpha_intact_current_split_full_seed2` | 0.4882849213420915 | 0.4885707497596741 | broad/background | -1.7272331714630127 |
| `old_history_full_seed2` | 0.2144645048950973 | 1.0018107891082764 | notch-localized drive | 0.5903642773628235 |
| `old_current_split_full_seed2` | 0.005038344523469671 | 1.0015006065368652 | boundary/corner drive | 0.01926925778388977 |

These classifications are diagnostic labels only. They do not prove physical validation.

## 4. Top-U-Free Smoke

The top-u-free smoke command completed successfully with:

- `--top-u-mode free`
- `--smoke`
- `--n-rprop 1`
- `--n-lbfgs 0`
- `--max-steps 1`
- `--delta-max 1e-6`

Boundary diagnostics were written to the smoke diagnostics CSV and summarized in `tables/topufree_smoke_summary.csv`.

Smoke summary:

| field | value |
|---|---:|
| `top_u_mode` | free |
| `top_u_abs_max` | 1.6637207167491397e-09 |
| `top_v_error_max` | 0.0 |
| `bottom_u_abs_max` | 0.0 |
| `bottom_v_abs_max` | 0.0 |
| `alpha_mean` | 5.062850050308043e-06 |
| `reaction_N_tm_eff` | 0.09519590437412262 |

The smoke confirms that the diagnostic columns exist and the imposed top-v/bottom constraints are sampled. It does not show whether top-u-free fixes the full-run broad-background damage.

## 5. Verification

Commands passed:

- top-u-free smoke run
- `D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q`
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py plot_clean_tm_results.py debug_recompute_he_current.py main.py analyze_drive_broadening_stepwise.py`

Test result:

`13 passed in 0.07s`

GitHub CLI status:

- Bare `gh` was not available in the current PowerShell PATH.
- `C:\Program Files\GitHub CLI\gh.exe` exists and reports version `2.93.0`.
- `gh auth status` reports no logged-in GitHub hosts.
- `GH_TOKEN` and `GITHUB_TOKEN` were not present.
- Therefore this package uses markdown-only handoff and was not automatically posted to issue #1.

## 6. What Cannot Be Concluded

This package cannot conclude:

- that the model is physically validated;
- that top-u-free fixes full D0020 behavior;
- that a single seed is sufficient;
- that any change to `l0`, material parameters, TM split, or notch seeding is justified.

## 7. Recommended Next Decision

ChatGPT should decide whether the top-u-free ansatz is worth one controlled full D0020 seed 2 run, based on:

- alpha-init full runs showing broad/background drive from step 0;
- old history full run remaining notch-localized by the final step;
- top-u-free smoke passing only as a boundary diagnostic.
