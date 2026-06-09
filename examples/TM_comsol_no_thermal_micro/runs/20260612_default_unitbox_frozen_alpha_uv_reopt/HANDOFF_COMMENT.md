## Codex handoff: frozen-alpha u/v re-optimization

Commit: e6eb7ba
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260612_default_unitbox_frozen_alpha_uv_reopt
Main report: examples/TM_comsol_no_thermal_micro/runs/20260612_default_unitbox_frozen_alpha_uv_reopt/REPORT.md

### What changed
- Added and ran a true frozen-alpha PINN u/v re-optimization diagnostic.
- Used saved D0040 fields for seeds 7, 13, 42; states: final_D0040, through_alpha0p8_onset.
- Frozen inputs: alpha, saved HI/HII old-history fields, material constants, TM split, top-u-free/unit_box ansatz, and saved displacement level.
- The original trial-history `max(old,current)` mechanics logic was kept; alpha was not evolved.
- Re-optimized only `u,v` for baseline, minus-degraded crack band, minus-removed crack band, and full-degradation diagnostic variants.
- Initialization strategy used in the default run: global saved-uv prefit; no notch/lip/local/jump/geometry-guided training losses were used.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe artifacts\run_frozen_alpha_uv_reopt.py --states final_D0040 through_alpha0p8_onset
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260612_default_unitbox_frozen_alpha_uv_reopt\artifacts\run_frozen_alpha_uv_reopt.py
```

### Key results
- Mechanism classification: **frozen-alpha reoptimization identifies dominant mechanism: Case B, continuous-field or boundary-condition bridging is dominant**.
- See `tables/frozen_alpha_reaction_comparison.csv` for reaction removal and 10/30/50 percent collapse flags.
- See `tables/frozen_alpha_crack_band_traction.csv` for crack-band traction removal.
- See `tables/frozen_alpha_convergence.csv` for convergence statuses and finite-budget notes.
- Verification passed: `pytest examples\TM_comsol_no_thermal_micro\tests -q` reported 18 passed; package script `py_compile` passed.
- Diagnostic variants are not production model changes.
- No physical validation is claimed.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/frozen_alpha_uv_reopt_summary.csv`
- `tables/frozen_alpha_reaction_comparison.csv`
- `tables/frozen_alpha_crack_band_traction.csv`
- `tables/frozen_alpha_energy_comparison.csv`
- `tables/frozen_alpha_displacement_jump.csv`
- `tables/frozen_alpha_convergence.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Which acceptance case does this evidence support: Case A, Case B, or Case C?
2. Are finite-budget convergence statuses sufficient, or should Codex rerun selected variants with a larger budget?
3. What is the next minimal Codex diagnostic without changing physical parameters?

### Constraints
- Do not extend loading.
- Do not evolve alpha.
- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.
- Diagnostic full/minus degradation variants are not production changes.
- Do not use `--alpha-init-intact` as the main route.
- Do not claim physical validation.
