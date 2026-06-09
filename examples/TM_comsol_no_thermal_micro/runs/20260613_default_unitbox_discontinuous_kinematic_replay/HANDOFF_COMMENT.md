## Codex handoff: discontinuous/split-domain frozen-alpha kinematic replay

Commit: e13e490
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260613_default_unitbox_discontinuous_kinematic_replay
Main report: examples/TM_comsol_no_thermal_micro/runs/20260613_default_unitbox_discontinuous_kinematic_replay/REPORT.md

### What changed
- Added and ran a diagnostic-only split-domain/discontinuous frozen-alpha kinematic replay.
- Used saved D0040 fields for seeds 7, 13, 42; states: final_D0040.
- Frozen inputs: alpha, saved HI/HII old-history fields, material constants, TM split, top-u-free/unit_box ansatz, saved displacement level, and `l0`.
- Re-optimized only displacement fields. Alpha was not evolved.
- Constructed upper/lower domain labels from the connected alpha>=0.8 crack band.
- Evaluated continuous baseline, split-domain current split, split-domain minus-degraded crack band, and split-domain crack-band void diagnostics.
- No notch/lip/local/jump/geometry-guided training losses were used.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe artifacts\run_discontinuous_kinematic_replay.py --quick
D:\anaconda3\envs\torch_env\python.exe artifacts\run_discontinuous_kinematic_replay.py
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
D:\anaconda3\envs\torch_env\python.exe -m py_compile examples\TM_comsol_no_thermal_micro\runs\20260613_default_unitbox_discontinuous_kinematic_replay\artifacts\run_discontinuous_kinematic_replay.py
```

### Key results
- Diagnostic classification: **continuous-field bridging not confirmed**.
- See `tables/domain_split_geometry_audit.csv` for split construction validity.
- See `tables/discontinuous_reaction_comparison.csv` for reaction removal and 10/30/50 percent flags.
- See `tables/discontinuous_displacement_jump.csv` for crack opening/jump proxies.
- See `tables/discontinuous_crack_band_traction.csv` for crack-band traction suppression.
- See `tables/discontinuous_energy_comparison.csv` for energy changes.
- Verification note: the requested repo-relative pytest path is absent in this checkout; the actual project test path under `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests` reported 18 passed. `py_compile` passed.
- No physical validation is claimed.
- No production model change is justified directly.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/discontinuous_replay_summary.csv`
- `tables/discontinuous_reaction_comparison.csv`
- `tables/discontinuous_crack_band_traction.csv`
- `tables/discontinuous_displacement_jump.csv`
- `tables/discontinuous_energy_comparison.csv`
- `tables/domain_split_geometry_audit.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this evidence confirm continuous-field/boundary-condition bridging, fail to confirm it, or remain unresolved?
2. Is the split-domain geometry valid enough to interpret the reaction comparison?
3. What is the next minimal Codex diagnostic without changing physical parameters?

### Constraints
- Do not extend loading.
- Do not evolve alpha.
- Do not change `l0`, material parameters, thermal terms, TM split, or history update logic.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip loss, masks, local weights, displacement-jump targets, enrichment, or geometry-label guidance as a production route.
- The split-domain/discontinuous representation is diagnostic-only.
- Do not use `--alpha-init-intact` as the main route.
- Do not claim physical validation.
