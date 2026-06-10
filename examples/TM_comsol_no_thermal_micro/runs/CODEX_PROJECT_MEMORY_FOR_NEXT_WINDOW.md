# Codex Project Memory for Next Window

## 1. Repository and working path

- repo: `letmex/cc-work`
- main project path in the repo: `examples/TM_comsol_no_thermal_micro`
- normal results root in the project: `examples/TM_comsol_no_thermal_micro/outputs/`
- run/handoff packages root: `examples/TM_comsol_no_thermal_micro/runs/`
- local execution tree used in this Windows environment: `D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro`
- evidence/handoff repo checkout used here: `D:\Desktop\新建文件夹\cc-work`

Do not confuse the evidence repo with any stale desktop mirror. Training and postprocessing should run from the real project tree when explicitly requested; evidence packages should be committed under `cc-work/examples/TM_comsol_no_thermal_micro/runs/`.

## 2. Current project status

The project has been cleaned to one normal verified route. The current normal route is:

- `mixedH_TM + tm_source + history`
- AT2 phase-field
- default alpha initialization
- existing material parameters
- existing `l0`
- unit-box coordinate normalization as default
- top-u-free tension route as default
- checkpointed mechanics-energy reaction
- normal postprocessing through `postprocess_results.py`
- normal plotting through `plot_results.py`
- managed outputs under `outputs/`
- no generated/debug artifacts in the example root

Current cleanup state:

- legacy top-boundary sigma reaction is removed from normal output
- old split/mechanics/solver branches were removed
- alpha-init-intact route was removed
- staggered/debug alternate route was removed
- `corrected` / `clean` program names were removed from the normal workflow

## 3. Important cleanup package

Package:

`examples/TM_comsol_no_thermal_micro/runs/20260613_single_verified_pipeline_cleanup`

Commit:

`00e52bcdb0750e924b8359d8b3db6c300a17907d`

Key facts:

- normal route simplified to `mixedH_TM + tm_source + history`
- normal outputs use `reaction_N_energy` and `nominal_stress_energy_MPa`
- `apply_alpha_init_intact` removed
- alternate split modes removed/collapsed
- legacy top sigma removed from normal CSV/figures
- root generated/debug artifacts removed
- tests passed: `38 passed, 8 warnings`
- no training run, no seed study, no D0040
- no physics formula changed

Files to read:

- `HANDOFF_COMMENT.md`
- `REPORT.md`
- `tables/forbidden_term_check.csv`
- `tables/reaction_metric_cleanup_audit.csv`
- `tables/cli_simplification_audit.csv`

## 4. Current verified reaction policy

The only normal reaction metric is the energy-conjugate reaction from checkpoint mechanics energy:

`reaction_N_energy = dPi / dDelta`

For shear:

`reaction_N_energy = dPi / dDelta_s`

Normal stress labels:

- tension: `nominal_stress_energy_MPa`
- shear: `nominal_shear_stress_energy_MPa`

Do not use these in normal outputs:

- `legacy_top_sigma_integral_N`
- `reaction_N_legacy_top_sigma`
- `reaction_N_tm_eff`
- old top-boundary sigma integral
- legacy top sigma diagnostic columns

If checkpointed energy reaction is unavailable, report reaction unavailable. Do not classify old non-checkpointed runs as no-softening based on legacy top reaction.

## 5. Existing shear implementation

The current shear load case was added and should be kept.

Shear ansatz:

```text
eta = (y - y_min) / H
bubble = eta * (1 - eta)
free_top_shape = eta + bubble

u = Delta_s * (eta + bubble * raw_u)
v = Delta_s * free_top_shape * raw_v
```

Boundary meaning:

- bottom: `u = 0`, `v = 0`
- top: `u = Delta_s`, `v = free`
- left/right: free

Do not change this boundary/ansatz unless explicitly instructed later.

Do not impose full top `v=0`.

Do not add top-v stabilization unless a later diagnostic clearly requires it.

Top-v monitor policy:

- monitor `top_v_absmax / Delta_s`
- warning threshold: `> 1.5`
- unstable threshold: `> 2.0` or visible field blow-up

## 6. Shear run history

### 6.1 Smoke shear run

Package:

`examples/TM_comsol_no_thermal_micro/runs/20260614_existing_geometry_shear_smoke`

Commit:

`324be2f801c3f8c44a5742cce2e0619235deb31a`

Key facts:

- seed 23 only
- schedule `load_schedules/load_schedule_S0020_shear.csv`
- 5 steps
- final `Delta_s=2e-5` mm
- smoke training: `RPROP=20, LBFGS=0`
- checkpointed energy reaction computed
- top `v` free and finite
- HII active
- final `alpha_max=0.014210`
- no through-crack
- curve monotonic
- mechanics-drive was boundary/corner dominated
- classification: `shear smoke not convincing`

Interpretation:

The smoke was too weak to judge shear capability.

### 6.2 Stronger shear run

Package:

`examples/TM_comsol_no_thermal_micro/runs/20260615_existing_geometry_shear_stronger_training`

Commit:

`3f3c30870f606e8d6917d61f528295a0e2e696f9`

Key facts:

- seed 23 only
- schedule `load_schedules/load_schedule_S0030_shear.csv`
- 21 steps
- final `Delta_s=5e-5` mm
- training: `RPROP=300, LBFGS=1`
- checkpointed energy reaction computed at all 21 steps
- final engineering shear strain: `0.005`
- final nominal shear stress: `27.285604 MPa`
- curve still monotonic
- final `alpha_max=0.358412`
- final alpha max at explicit notch-tip region
- mechanics-drive became notch-dominated
- final HII/HI peak ratio about `0.632`
- final `top_v_absmax/Delta_s=1.013`
- no alpha>=0.8 through-crack
- classification: `stronger shear run qualitatively improved`

Interpretation:

The smoke issue was training/schedule-limited. Do not change shear ansatz, boundary, or mixed-drive based on smoke.

### 6.3 S0050 shear load extension

Package:

`examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension`

Commit:

`2004149e2be130250c15017f60e94eadb00c21aa`

Key facts:

- seed 23 only
- full rerun from step 0, not continuation
- `continued_from_S0030=False`
- schedule `load_schedules/load_schedule_S0050_shear.csv`
- 33 steps
- final `Delta_s=8e-5` mm
- training: `RPROP=300, LBFGS=1`
- checkpointed energy reaction computed at all 33 steps
- final engineering shear strain: `0.008`
- peak nominal shear stress: `29.9647 MPa` at step 24 / shear strain `0.006`
- final nominal shear stress: `28.4379 MPa`
- post-peak drop observed
- final `alpha_max=1.00034`
- final alpha max near explicit notch-tip region
- `alpha>=0.5` notch-connected damage first appears at step 22
- `alpha>=0.8` notch-connected damage first appears at step 25
- no alpha>=0.8 through-crack to the right boundary detected
- final alpha>=0.8 connected x-span about `0.000384 mm`
- final HII/HI peak ratio about `0.597398`
- HII remains active and notch-localized
- mechanics-drive remains notch-dominated
- final and maximum `top_v_absmax/Delta_s=1.08336`
- below 1.5 top-v warning threshold
- classification: `shear extension successful with crack growth`

Interpretation:

S0050 is the current main existing-geometry shear diagnostic result. It shows notch-localized crack growth and post-peak softening, but no full through-crack to the right boundary. This is still diagnostic, not physical validation.

## 7. Current next recommended task

The next recommended task is not cleanup, not seed study, not D0040, and not boundary modification.

Next task:

Run one slightly longer same-path shear extension and add detailed crack-connectivity diagnostics.

Suggested package:

`examples/TM_comsol_no_thermal_micro/runs/20260617_existing_geometry_shear_connectivity_extension`

Suggested schedule:

`load_schedules/load_schedule_S0070_shear.csv`

Recommended target:

- include prior S0050 range up to `Delta_s=8e-5` mm
- extend gradually to around `1.0e-4` or `1.1e-4` mm
- about 41 to 51 total steps
- conservative increments after the prior peak/softening region

Use:

- seed 23 only
- existing geometry only
- same shear ansatz
- same top-v-free boundary
- same physical route
- training about `RPROP=300, LBFGS=1`
- checkpointed energy reaction
- normal postprocessing

If clean continuation from S0050 is not safely implemented, rerun from step 0 and document `continued_from_S0050=False`.

## 8. Required next diagnostics

For thresholds:

- alpha >= 0.3
- alpha >= 0.5
- alpha >= 0.8
- alpha >= 0.95

Compute by step:

- largest connected component count
- notch-connected component count
- notch-connected x-span
- notch-connected y-span
- notch-connected area fraction
- reaches right boundary yes/no
- reaches top/bottom boundary yes/no
- component min/max x
- component mean y
- approximate crack angle or principal direction
- first step where notch-connected component appears
- first step where right-boundary through-crack appears, if any

Required next tables:

- `tables/shear_connectivity_extension_run_summary.csv`
- `tables/shear_reaction_by_step.csv`
- `tables/shear_stress_strain_by_step.csv`
- `tables/shear_damage_drive_summary.csv`
- `tables/shear_connectivity_by_threshold.csv`
- `tables/shear_crack_path_geometry_by_step.csv`
- `tables/shear_top_v_free_diagnostic.csv`
- `tables/shear_checkpoint_availability.csv`
- `tables/shear_training_loss_summary.csv`
- `tables/shear_extension_vs_S0050_comparison.csv`
- `tables/shear_output_file_manifest.csv`

Required next figures:

- `figures/shear_stress_strain_seed23.png`
- `figures/shear_reaction_strain_seed23.png`
- `figures/final_fields_panel_seed23_shear.png`
- `figures/final_alpha_seed23_shear.png`
- `figures/final_u_seed23_shear.png`
- `figures/final_v_seed23_shear.png`
- `figures/final_HI_seed23_shear.png`
- `figures/final_HII_seed23_shear.png`
- `figures/final_mechanics_drive_seed23_shear.png`
- `figures/shear_alpha_max_by_step.png`
- `figures/shear_HII_HI_ratio_by_step.png`
- `figures/shear_top_v_absmax_over_Delta_by_step.png`
- `figures/shear_notch_drive_by_step.png`
- `figures/shear_connectivity_xspan_by_threshold.png`
- `figures/shear_connected_component_count_by_threshold.png`
- `figures/shear_crack_path_overlay_final.png`
- `figures/shear_through_crack_status_by_threshold.png`

## 9. Persistent constraints for future Codex tasks

Always obey:

- Do not claim physical validation from single-seed diagnostic runs.
- Do not run D0040 unless explicitly requested.
- Do not run seed studies unless explicitly requested.
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change material parameters unless a clear bug is found.
- Do not change TM split formulas unless a clear bug is found.
- Do not change history logic unless a clear bug is found.
- Do not change alpha initialization behavior unless explicitly requested.
- Do not change training losses.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not reintroduce legacy top sigma into normal outputs.
- Do not reintroduce old split modes.
- Do not reintroduce alpha-init-intact.
- Do not reintroduce staggered/debug alternate routes.
- Do not write generated/debug artifacts to the project root.
- Use managed output directories under `outputs/` and handoff packages under `runs/`.

## 10. Files the new Codex should read first

Tell the new Codex to read these first:

1. `examples/TM_comsol_no_thermal_micro/runs/CODEX_PROJECT_MEMORY_FOR_NEXT_WINDOW.md`
2. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/HANDOFF_COMMENT.md`
3. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/REPORT.md`
4. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/tables/shear_extension_run_summary.csv`
5. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/tables/shear_extension_vs_S0030_comparison.csv`
6. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/tables/shear_damage_drive_summary.csv`
7. `examples/TM_comsol_no_thermal_micro/runs/20260616_existing_geometry_shear_load_extension/figures/figure_summary.md`
8. `examples/TM_comsol_no_thermal_micro/runs/20260613_single_verified_pipeline_cleanup/HANDOFF_COMMENT.md`
9. `examples/TM_comsol_no_thermal_micro/runs/20260613_single_verified_pipeline_cleanup/REPORT.md`

## 11. Verification for this memory task

This memory task should only run lightweight checks:

- verify the file exists
- verify it contains the package paths above
- verify it contains current S0050 key numbers
- verify it contains the next recommended task
- do not run training
- do not run postprocessing

For the memory task itself, create a short handoff package under:

`examples/TM_comsol_no_thermal_micro/runs/20260617_project_memory_handoff/`

That package should state:

- commit hash
- memory file path
- whether any code changed
- whether any training was run
- whether any postprocessing was run
- next task for the new Codex window
