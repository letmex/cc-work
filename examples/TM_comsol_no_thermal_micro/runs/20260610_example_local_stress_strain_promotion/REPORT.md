# Report: example-local stress-strain source promotion

## Scope correction

The implementation was moved out of the shared `source` path. The shared file `source/postprocess_tm.py` was checked for the previously attempted corrected-curve promotion symbols and no matches were found.

The active change is limited to the example-local file:

`D:/ProgramData/PINN/FEM-PINN-main/examples/TM_comsol_no_thermal_micro/plot_clean_tm_results.py`

## What changed

- Added `--corrected-stress-strain-csv` and `--corrected-seed` to the example-local plotting flow.
- When a corrected table is provided or discovered, `nominal_stress_energy_exact_MPa` is used as the primary stress-strain curve source.
- `nominal_stress_energy_virtual_work_MPa` is plotted only as a consistency check when available.
- Legacy top-boundary sigma metrics are retained only as dashed diagnostic curves.
- If corrected reaction data is unavailable, the output table marks the primary curve as `reaction_metric_unavailable` instead of implying `no_softening`.
- Metadata now avoids writing non-ASCII parent paths, preventing path mojibake in generated text outputs.

## Smoke verification

The updated flow was run on the D0020 seed 42 result directory with the corrected D0020 table from the earlier exact-reaction package.

The generated smoke CSV reports:

- `stress_strain_primary_metric = nominal_stress_energy_exact_MPa`
- `stress_strain_metric_status = energy_conjugate_primary`
- `legacy_curve_status = legacy_diagnostic_only`

Compact source check:

- rows: 34
- primary peak: 182.820635382086 MPa
- primary final: 0.82835531429736 MPa
- primary final/peak: 0.00453097273492207
- legacy top peak: 91.6370766815224 MPa
- legacy top final: 86.6391686779654 MPa
- legacy final/peak: 0.945459761653825

This smoke test verifies the plotting main flow now prefers the corrected energy-conjugate metric when it is available. It does not introduce new physical-model evidence.

## What this fixes

The previous curve source problem was that the plotted stress-strain response could still be driven by legacy top-boundary sigma integration. That metric remains high after D0020 through-crack onset. The corrected main plotting flow now promotes the energy-conjugate reaction as the primary stress-strain source, so the curve can show the post-through-crack softening already established by the D0020 exact-reaction diagnostics.

## What this does not do

- It does not modify the physical model.
- It does not change `l0`, material parameters, TM split, phase-field initialization, or history logic.
- It does not run or process D0040.
- It does not claim new physical validation.

