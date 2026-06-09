## Codex handoff: negative-branch ablation and frozen-alpha replay

Commit: PENDING
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_negative_branch_ablation
Main report: examples/TM_comsol_no_thermal_micro/runs/20260611_default_unitbox_negative_branch_ablation/REPORT.md

### What changed
- Used existing D0040 fields for seeds 7, 13, 42.
- Audited alpha>=0.8 through-onset step 14 and final D0040 step 54.
- Evaluated diagnostic-only stress/energy variants for negative-branch ablation.
- Frozen-alpha mechanics replay was deterministic post-hoc evaluation on saved `u,v,alpha`, not a new optimization.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe artifacts\build_negative_branch_ablation_package.py
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260611_default_unitbox_negative_branch_ablation\artifacts\build_negative_branch_ablation_package.py
```

### Key results
- Cause classification: **dominant cause: continuous displacement-field or boundary-condition bridging**.
- Mean final crack-band traction removed by `minus_degraded_in_crack_band`: 100%.
- Mean final reaction proxy removed by `minus_degraded_in_crack_band`: 0%.
- Mean final reaction proxy removed by `full_degradation_all_energy`: -21.8%.
- Negative reaction-removal values mean the saved-field reaction proxy increased rather than collapsed.
- `void_crack_band` was a post-hoc crack-band traction ablation only; no replay row was generated for it.
- Diagnostic replay did not re-optimize `u,v`; this limits production conclusions.
- Verification passed: `pytest examples\TM_comsol_no_thermal_micro\tests -q` reported 18 passed; package script `py_compile` passed.
- No physical validation is claimed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/posthoc_crack_band_stress_ablation.csv`
- `tables/frozen_alpha_mechanics_replay_summary.csv`
- `tables/variant_reaction_comparison.csv`
- `tables/variant_energy_comparison.csv`
- `tables/variant_displacement_jump_proxy.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this deterministic ablation support non-degraded negative branch as the dominant local crack-band load-transfer mechanism?
2. Is the next Codex task a true frozen-alpha mechanics re-optimization for baseline vs minus-degraded-in-crack-band?
3. What exact acceptance criterion should be used for the replay optimization?

### Constraints
- Do not extend loading.
- Do not evolve alpha.
- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.
- Diagnostic full/minus degradation variants are not production changes.
- Do not claim physical validation.
