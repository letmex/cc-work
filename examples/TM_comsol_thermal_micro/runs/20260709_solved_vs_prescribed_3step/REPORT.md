# Solved vs Prescribed Multistep Main-Entrypoint Comparison

## Purpose

This package expands the project beyond one-step smoke by running real `main.py` mechanics schedules for three cases:

1. solved-temperature mechanics smoke (`--solved-temperature-mechanics-smoke`),
2. prescribed `linear_y` control matched to H1 (`--thermal-mode linear_y --thermal-grad-y 2000 --thermal-y0 0`),
3. no-thermal baseline.

It includes both a 3-step mini schedule and an 8-step probe. All runs used hidden layers `2`, neurons `20`, seed `7`, `n-rprop=2`, and `n-lbfgs=0`.

## Main Findings

For H1, solved-temperature and prescribed `linear_y` match in the main mechanics steps because both evaluate `delta_T = 2000*y` at element centroids. The measured agreement is tight:

- 3-step solved-prescribed max notch-tip mean DeltaT difference: `2.86102294922e-06 K`
- 8-step solved-prescribed max notch-tip mean DeltaT difference: `2.86102294922e-06 K`
- 8-step solved-prescribed max global `He_max` difference: `4.56348061562e-08`

Compared with no-thermal, the 8-step solved run shows a measurable thermal mechanics response at the final step:

- final `solved - no_thermal` `He_max`: `-3.1846575439e-06`
- final `solved - no_thermal` `mechanics_drive_max`: `-3.1846575439e-06`
- final `solved - no_thermal` `alpha_max`: `-1.37090682983e-06`
- max notch-tip mean DeltaT difference vs no-thermal: `9.99963283539 K`

## Important Caveat

A review subagent confirmed the main mechanics-step `delta_T` fields match, but noted that the full optimization trajectories are not mathematically identical: solved smoke applies the adapter after fine mesh prep, while prescribed thermal mode is active during pretraining too. Interpret solved-prescribed equality as a main-step thermal-field/response check, not as proof of identical end-to-end training objectives.

## Evidence Tables

- `tables/stepwise_case_summary.csv`: per-run-group, per-case, per-step metrics.
- `tables/stepwise_pairwise_differences.csv`: stepwise solved-prescribed and thermal-no-thermal differences.
- `tables/comparison_summary.csv`: max absolute and final-step differences for each metric.
- `tables/final_step_trend_summary.csv`: compact final-step trends for 3-step and 8-step groups.
- `tables/review_caveats.csv`: caveats carried forward from code-path review.

## Boundary

This is still not a full heat-fracture coupling result. H1 is a linear frozen lift, so equivalence with prescribed `linear_y` is expected. The next substantive physics step is a nontrivial solved heat field that cannot be represented by this prescribed linear control.
