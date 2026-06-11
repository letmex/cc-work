## Codex handoff: Existing-geometry shear connectivity extension

Commit: TO_BE_FILLED_AFTER_COMMIT
Package folder: `examples/TM_comsol_no_thermal_micro/runs/20260617_existing_geometry_shear_connectivity_extension`
Memory file read confirmation: read `CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`, project-memory handoff files, and the S0050 shear extension report/tables before this package was built.

### Run identity
- Continuation: `continued_from_S0050=False`.
- Reason: clean continuation from the committed S0050 history state was not already implemented unambiguously; S0070 is a full rerun from step 0.
- Training command: `D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --n-rprop 300 --n-lbfgs 1 --load-case shear --load-schedule-file load_schedules/load_schedule_S0070_shear.csv --run-suffix seed23_S0070_shear`
- Postprocess command: `D:\anaconda3\envs\torch_env\python.exe postprocess_results.py --model-dir outputs/checkpoints/seed23_S0070_shear --result-dir outputs/results/seed23_S0070_shear --device cpu`
- Load schedule: `load_schedules/load_schedule_S0070_shear.csv`.
- Seed: 23 only.
- Training settings: `RPROP=300, LBFGS=1`.

### Status
- Checkpoint availability: 43/43 checkpoints.
- Energy reaction status: energy-conjugate reaction at all steps.
- S0050 comparison: final S0050 stress 28.4379 MPa, final S0070 stress 23.071 MPa.
- S0070 peak stress: 29.9647 MPa at step 24; final stress 23.071 MPa; post-peak drop 23%.
- Alpha/HII/mechanics drive: alpha remains notch-localized; final HII/HI ratio 0.627437; drive remains `notch-dominated`.

### Connectivity by threshold
- alpha >= 0.3: first notch-connected step 20, final x-span 0.00143761 mm
- alpha >= 0.5: first notch-connected step 22, final x-span 0.00116699 mm
- alpha >= 0.8: first notch-connected step 25, final x-span 0.000895859 mm
- alpha >= 0.95: first notch-connected step 28, final x-span 0.000463828 mm

### Through-crack status
- alpha >= 0.3: no right-boundary reach
- alpha >= 0.5: no right-boundary reach
- alpha >= 0.8: no right-boundary reach
- alpha >= 0.95: no right-boundary reach

### Top-v diagnostic
- Final top_v_absmax/Delta_s: 1.13531; below warning 1.5 and unstable 2.0 thresholds.

### Generated tables and figures
- Tables are under `tables/`; key tables are `shear_connectivity_extension_run_summary.csv`, `shear_extension_vs_S0050_comparison.csv`, and `shear_connectivity_by_threshold.csv`.
- Figures are under `figures/`; read `figures/figure_summary.md`.

### Classification
`shear extension shows propagating crack`

### Constraints observed
- Physics changed: no.
- Seed study run: no.
- D0040 run: no.

### Next recommended minimal intervention
Interpret the geometry/connectivity diagnostics before changing physics. If another run is needed, make only a small same-path extension after reviewing the threshold tables and final crack-path overlay.
