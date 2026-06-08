## Codex handoff: global anchor energy continuation

Commit: `45f1e890cb84e454bcff41af11dfbddbffee956b`
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_global_anchor_energy_continuation/REPORT.md`

### What changed
- Added a mechanics-only alpha=0 global anchor continuation diagnostic.
- Added `artifacts/debug_global_anchor_energy_continuation.py` as the script snapshot.
- Saved `disp_global` and `disp_strain_global` prefit checkpoints.
- Tested pure energy baseline, displacement anchors, strain anchors, combined displacement+strain anchors, trust-region continuation, and raw/normalized energy variants.
- No notch-lip loss, notch-tip/lip mask, local notch weight, displacement-jump target, or geometry-label guidance was used in the training loss.

### Commands run
```powershell
git pull origin main
```

```powershell
D:\anaconda3\envs\torch_env\python.exe debug_global_anchor_energy_continuation.py --skip-figures --out-dir <repo_root>\examples\TM_comsol_no_thermal_micro\runs\20260608_global_anchor_energy_continuation --target-free <repo_root>\examples\TM_comsol_no_thermal_micro\runs\20260608_mechanics_only_notch_ansatz\artifacts\fedof_free_log10_energy_e300_fields.npz --prefit-epochs 1000 --continuation-epochs 300 --trust-chunks 10 --trust-epochs-per-chunk 30 --delta 1e-6 --seed 2 --hidden-layers 8 --neurons 400
```

```powershell
D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_global_anchor_energy_continuation.py debug_prefit_then_energy_mechanics.py debug_pinn_prefit_fedof_mechanics.py debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py
```

Result: passed.

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q
```

Result: failed during collection because `ref_files.Chinese_SENT_reproduction` is missing in the current environment.

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests -q
```

Result: passed, `13 passed`.

```powershell
D:\anaconda3\envs\torch_env\python.exe -m pytest examples\TM_comsol_no_thermal_micro\tests tests\test_tm_comsol_no_thermal_micro_project.py -q
```

Result: `19 passed, 1 failed`; the failure is an existing config contract expecting `mixed_mechanics_mode=current_split` while current config is `history`.

GitHub CLI status: `gh` is available and reports version `2.93.0`, but `gh auth status` reports no logged-in GitHub host. No `GH_TOKEN` or `GITHUB_TOKEN` was present, so this package uses markdown-only handoff.

### Key results
- Both prefit checkpoints pass all diagnostic thresholds:
  - `prefit_disp_global`: displacement rel. RMSE `0.005637`, strain corr `0.984313`, He corr `0.893306`.
  - `prefit_disp_strain_global`: displacement rel. RMSE `0.006095`, strain corr `0.984722`, He corr `0.941575`.
- No continuation case passes all diagnostic thresholds.
- Best displacement-anchor case by He correlation: `lambda_u=0.1`, He corr `0.869763`, but displacement rel. RMSE is still `0.882046` and classification is `boundary-dominated`.
- Pure log10 energy continuation collapses: displacement rel. RMSE `0.999704`, strain corr `-0.016976`, He corr `-0.014971`.
- Trust-region continuation did not preserve the branch through 10 chunks.
- Raw/normalized/log10 energy variants all stay far from the target-like branch without an anchor.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/anchor_sweep_metrics.csv`
- `tables/success_threshold_summary.csv`
- `tables/continuation_checkpoint_metrics.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Should the next diagnostic directly compare mechanics energy decomposition for FE-DOF target, prefit PINN, and collapsed PINN fields under identical quadrature and boundary assumptions?
2. Is the current evidence enough to treat mechanics energy formulation / optimizer path as the main unresolved issue before coupled full training?
3. What global, non-geometry-specific continuation rule should be tested next, if any?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not add notch-lip loss, notch-tip/lip masks, local notch weights, displacement-jump targets, or geometry-label guidance to the training loss unless explicitly requested.
- Do not change TM split/material parameters unless a clear bug is found.
- Do not claim physical validation from this mechanics-only diagnostic.
