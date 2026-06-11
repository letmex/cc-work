## Codex handoff: Existing-geometry shear longer connectivity extension

Commit: 88034fb
Package folder: `examples/TM_comsol_no_thermal_micro/runs/20260618_existing_geometry_shear_longer_connectivity_extension`
Local main ahead of origin: `True`; after this handoff hash update, local `main` is ahead of `origin/main` by 4 commits. Commits pushed: no.

### Run identity
- Continuation: `continued_from_S0070=False`.
- Reason: clean continuation from the committed S0070 history state was not already implemented unambiguously; S0090 is a full rerun from step 0.
- Training command: `D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0090_shear.csv --run-suffix seed23_S0090_shear`
- Postprocess command: `D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs/checkpoints/seed23_S0090_shear --result-dir outputs/results/seed23_S0090_shear --device cpu`
- Load schedule: `load_schedules/load_schedule_S0090_shear.csv`.
- Seed: 23 only.
- Training settings: `RPROP=300, LBFGS=1`.

### Status
- Checkpoint availability: 53/53 checkpoints.
- Energy reaction status: energy-conjugate reaction at all steps.
- S0070 comparison: final S0070 stress 23.071 MPa, final S0090 stress 18.136 MPa.
- S0090 peak stress: 29.9647 MPa at step 24; final stress 18.136 MPa; post-peak drop 39.5%.
- Alpha/HII/mechanics drive: final alpha 1.0015; final HII/HI ratio 0.631579; drive `notch-dominated`.

### Connectivity by threshold and growth vs S0070
- alpha >= 0.3: final x-span 0.00183102 mm, growth vs S0070 0.000393403 mm
- alpha >= 0.5: final x-span 0.00159079 mm, growth vs S0070 0.000423795 mm
- alpha >= 0.8: final x-span 0.00133007 mm, growth vs S0070 0.000434216 mm
- alpha >= 0.95: final x-span 0.00106198 mm, growth vs S0070 0.000598153 mm

### Through-crack status
- alpha >= 0.3: no right-boundary reach
- alpha >= 0.5: no right-boundary reach
- alpha >= 0.8: no right-boundary reach
- alpha >= 0.95: no right-boundary reach

### Top-v diagnostic
- Final top_v_absmax/Delta_s: 1.18739.

### Generated tables and figures
- Tables are under `tables/`; key tables are `shear_longer_extension_run_summary.csv`, `shear_extension_vs_S0070_comparison.csv`, and `shear_connectivity_by_threshold.csv`.
- Figures are under `figures/`; read `figures/figure_summary.md`.

### Classification
`shear longer extension shows continued propagation`

### Constraints observed
- Physics changed: no.
- Seed study run: no.
- D0040 run: no.
- Commits pushed: no.

### Next recommended minimal intervention
If no right-boundary through-crack occurs, stop further single-seed shear escalation and interpret geometry/connectivity first.
