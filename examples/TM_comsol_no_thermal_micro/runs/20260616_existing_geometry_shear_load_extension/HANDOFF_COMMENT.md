## Codex handoff: Existing-geometry shear load extension

Commit: 2004149e2be130250c15017f60e94eadb00c21aa
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/REPORT.md`

### What changed
- Added controlled extended shear schedule `load_schedules/load_schedule_S0050_shear.csv` in the real run tree and included a snapshot in this package.
- Ran seed 23 only, using existing geometry, top-v-free shear boundary condition, and the same shear ansatz/physics as S0030.
- Used full rerun from step 0 (`continued_from_S0030=False`) because clean continuation was not implemented/was ambiguous.
- Ran normal postprocess with checkpointed energy-conjugate reaction.
- Did not run D0040, did not run a seed study, did not change `l0`, material parameters, TM split, history logic, alpha initialization, boundary conditions, shear ansatz, or losses.

### Commands run
```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0050_shear.csv --run-suffix seed23_S0050_shear
D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs\checkpoints\seed23_S0050_shear --result-dir outputs\results\seed23_S0050_shear --device cpu
```

### Key results
- Seed used: 23 only.
- Schedule used: `load_schedules/load_schedule_S0050_shear.csv`, 33 monotonic steps, final `Delta_s=8e-5` mm.
- Training settings: `RPROP=300, LBFGS=1`.
- Checkpoint availability: 33/33 step checkpoints; checkpointed energy reaction computed at all steps.
- Final engineering shear strain: `0.008`.
- Peak nominal shear stress: `29.9647` MPa at step `24`; final stress `28.4379` MPa, so post-peak drop is observed.
- Final alpha max: `1.00034` at the explicit notch-tip region; S0030 final alpha max was `0.358412`.
- `alpha>=0.5` notch-connected damage forms; `alpha>=0.8` notch-connected damage forms, but no alpha>=0.8 through-crack to the right boundary is detected.
- Final HII/HI peak ratio: `0.597398`; HII remains active and notch-localized.
- Mechanics-drive maximum remains notch-dominated.
- Final `top_v_absmax/Delta_s=1.08336`; maximum over run `1.08336`, below warning threshold 1.5.
- Classification: `shear extension successful with crack growth`.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/shear_extension_run_summary.csv`
- `tables/shear_extension_vs_S0030_comparison.csv`
- `tables/shear_damage_drive_summary.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should S0050 now replace S0030 as the main existing-geometry shear diagnostic result?
2. Since post-peak softening appears but no full alpha>=0.8 right-boundary through-crack forms, should the next minimal action be a slightly longer same-path shear extension or a connectivity/mesh-resolution audit?
3. Should top-v-free drift remain monitored only, given the ratio stayed below the 1.5 warning threshold?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this single-seed diagnostic run.
- Do not run D0040 or a seed study from this package.
