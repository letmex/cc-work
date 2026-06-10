## Codex handoff: Existing-geometry shear stronger training

Commit: 3f3c30870f606e8d6917d61f528295a0e2e696f9
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260615_existing_geometry_shear_stronger_training`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260615_existing_geometry_shear_stronger_training/REPORT.md`

### What changed
- Added `load_schedules/load_schedule_S0030_shear.csv` with 21 monotonic shear steps from `1e-6` to `5e-5` mm.
- Reran seed 23 shear with stronger training: `RPROP=300, LBFGS=1`.
- Kept the same shear ansatz, top-v-free boundary condition, material parameters, `l0`, TM split, history logic, and alpha initialization.
- Fixed plot label inference so fuller schedule codes like `S0030` keep shear labels/filenames.
- Did not run D0040, did not run a seed study, did not add legacy top sigma, and did not change physics.

### Commands run
```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0030_shear.csv --run-suffix seed23_S0030_shear
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0030_shear --result-dir outputs\results\seed23_S0030_shear --device cpu
D:\anaconda3\envs\torch_env\python.exe -m py_compile plot_results.py tests\test_shear_load_case.py
D:\anaconda3\envs\torch_env\python.exe -m pytest -p no:cacheprovider tests\test_shear_load_case.py -q
$env:PYTHONDONTWRITEBYTECODE='1'; D:\anaconda3\envs\torch_env\python.exe -B -m pytest -p no:cacheprovider tests -q
rg -n "phase_proximal|eta_eff|split_mode=|mechanics_mode=|legacy_top_sigma|reaction_N_legacy|corrected|clean|staggered|alpha_init_intact|apply_alpha_init_intact|voldev|current_split|MIXED_SPLIT_MODES|--mixed-mechanics-mode|--alpha-init-intact|--solve-scheme|--stagger-iters" config.py main.py mixed_mode_tm.py compute_energy_mixed_tm.py train_mixed_tm.py history_field_mixed_tm.py postprocess_results.py plot_results.py README.md POSTPROCESS_WORKFLOW.md PROJECT_STRUCTURE.md
rg -n "legacy_top_sigma|reaction_N_legacy|nominal_stress_energy_MPa" outputs\results\seed23_S0030_shear\curves outputs\results\seed23_S0030_shear\figures\stress_strain_source_seed23_shear.txt
```

### Key results
- Seed used: 23 only.
- Schedule used: `load_schedules/load_schedule_S0030_shear.csv`, 21 steps, final `Delta_s=5e-5` mm.
- Training settings: stronger `RPROP=300, LBFGS=1`; smoke was `RPROP=20, LBFGS=0`.
- Checkpoint availability: 21 checkpoints; checkpointed energy reaction computed at all steps.
- Final engineering shear strain: `5.000000e-03`.
- Final nominal shear stress: `27.285604` MPa; curve remains monotonic.
- Final alpha max: `0.358412` at the explicit notch-tip region; smoke final alpha max was `0.014210`.
- No alpha>=0.8 through-crack formed.
- Final HII/HI peak ratio: `0.632`; HII is active through the run.
- Final mechanics-drive maximum is notch-dominated at `(x,y)=(5.021085e-03, 4.998644e-03)` mm.
- Final top-v mean/min/max: `1.220332e-05`, `-2.455418e-05`, `5.065076e-05` mm; final `top_v_absmax/Delta_s=1.013`.
- Classification: `stronger shear run qualitatively improved`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/shear_training_run_summary.csv`
- `tables/shear_stronger_vs_smoke_comparison.csv`
- `tables/shear_damage_drive_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this stronger single-seed evidence justify continuing the same shear schedule/training path before changing boundary/ansatz/drive formulas?
2. Should the next minimal intervention be a controlled schedule extension/continuation, or a top-v drift diagnostic threshold?
3. What evidence is still missing before treating the shear response as more than a diagnostic branch?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this single-seed diagnostic run.
- Do not run D0040 or a seed study from this package.
