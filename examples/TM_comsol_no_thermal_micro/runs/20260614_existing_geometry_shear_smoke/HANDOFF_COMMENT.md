## Codex handoff: Existing-geometry shear smoke

Commit: 324be2f801c3f8c44a5742cce2e0619235deb31a
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260614_existing_geometry_shear_smoke`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260614_existing_geometry_shear_smoke/REPORT.md`

### What changed
- Added a minimal `--load-case shear` route while keeping `tension` as default.
- Added shear ansatz: bottom `u=v=0`, top `u=Delta_s`, top `v` free through the network output.
- Added `load_schedules/load_schedule_S0020_shear.csv` with five smoke-level steps.
- Updated normal postprocessing/plotting for shear labels: `Delta_s`, `engineering_shear_strain`, `reaction_N_energy`, and `nominal_shear_stress_energy_MPa`.
- Kept the cleaned single route: `mixedH_TM + tm_source + history`; no D0040, no seed study, no legacy top sigma, no alpha-init-intact, no staggered route, and no local/lip/jump loss.

### Commands run
```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 20 --n-lbfgs 0 --load-case shear --load-schedule-file load_schedules/load_schedule_S0020_shear.csv --run-suffix seed23_S0020_shear
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0020_shear --result-dir outputs\results\seed23_S0020_shear --device cpu
D:\anaconda3\envs\torch_env\python.exe -m py_compile field_computation.py config.py main.py postprocess_results.py plot_results.py tests\test_shear_load_case.py
D:\anaconda3\envs\torch_env\python.exe -m pytest -p no:cacheprovider tests\test_shear_load_case.py -q
$env:PYTHONDONTWRITEBYTECODE='1'; D:\anaconda3\envs\torch_env\python.exe -B -m pytest -p no:cacheprovider tests -q
```

### Key results
- Seed used: 23 only.
- Load schedule: `load_schedules/load_schedule_S0020_shear.csv` with `Delta_s = 1e-6, 5e-6, 1e-5, 1.5e-5, 2e-5` mm.
- Checkpoint availability: 5 checkpoints, all found locally.
- Energy reaction status: `energy_conjugate`, `exact_reaction_computable=True`.
- Final shear strain: `2.000000e-03`.
- Final nominal shear stress: `29.297048` MPa.
- Final top `v`: mean `-1.621801e-06` mm, min `-1.022874e-05` mm, max `7.009092e-06` mm; finite and not identically zero.
- Final HII/HI peak ratio: `0.632`.
- Alpha max remains small: `1.420977e-02`; no alpha>=0.8 through-crack.
- Shear curve is monotonic over this short smoke; no peak/post-peak drop.
- Classification: `shear smoke not convincing`, because the mechanics-drive global maximum is near the lower boundary/corner rather than clearly notch-path dominated.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/shear_smoke_run_summary.csv`
- `tables/shear_damage_drive_summary.csv`
- `tables/shear_top_v_free_diagnostic.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Given this smoke evidence, should the next task diagnose the shear ansatz/boundary condition or the mixed-drive reconstruction first?
2. Is the moderate free top-v drift acceptable for the next diagnostic, or should a minimal top-v reference/average stabilization be tested separately?
3. What is the smallest next Codex task that can determine why shear mechanics drive is boundary/corner dominated?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this smoke run.
- Do not run D0040 or a seed study from this package.
