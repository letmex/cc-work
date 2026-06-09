# Negative-branch ablation and frozen-alpha replay diagnostic

## Scope

This package uses existing D0040 saved fields only. Alpha is frozen; no load extension and no production-route training were run. The mechanics replay is deterministic post-hoc evaluation on saved `u,v,alpha` fields, not a new `u,v` optimization.

## Key variant summaries

| variant | mean cut traction removed vs current | mean reaction removed vs baseline | mean v-jump proxy |
|---|---:|---:|---:|
| full_degradation_everywhere | 100% | -21.8% | 0.000178971 |
| minus_degraded_in_crack_band | 100% | 0% | 0.000178971 |
| minus_removed_in_crack_band | 99.9% | 0% | 0.000178971 |
| void_crack_band | 100% | N/A | N/A |

## Answers

1. Degrading/removing the negative branch inside the connected crack band removes most local crack-band traction: `minus_degraded_in_crack_band` removes 100% on average at the final state; `minus_removed_in_crack_band` removes 99.9%.
2. Full degradation of all elastic energy/stress gives a deterministic reaction-proxy removal of -21.8% at the final state.
3. Degrading only `psi_minus` inside the connected crack band gives a deterministic reaction-proxy removal of 0% at the final state.
4. Removing only `psi_minus` inside the connected crack band gives a deterministic reaction-proxy removal of 0% at the final state.
5. Cause classification: **dominant cause: continuous displacement-field or boundary-condition bridging**.
6. A production model change is not justified from this diagnostic alone. The variants are diagnostic-only and were evaluated on saved fields without re-optimizing `u,v`.
7. Next minimal intervention: run a focused frozen-alpha mechanics optimization/replay on the same states for baseline vs minus-degraded-in-crack-band, and compare whether re-optimized continuous `u,v` still bridges the crack. If bridging remains, test a diagnostic discontinuous/enriched kinematic representation, explicitly labeled non-production.

## Verification

- `D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q`: 18 passed.
- `D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260611_default_unitbox_negative_branch_ablation\artifacts\build_negative_branch_ablation_package.py`: passed.

## Energy summary

Mean final energy contributions by variant are written to `tables/variant_energy_comparison.csv`. The diagnostic variants reduce negative energy in the crack band as intended, but deterministic replay does not include relaxation of the displacement field. In this saved-field evaluation, negative reaction-removal values mean the top-boundary reaction proxy increases rather than collapses after the diagnostic stress definition is applied.

No physical validation is claimed.
