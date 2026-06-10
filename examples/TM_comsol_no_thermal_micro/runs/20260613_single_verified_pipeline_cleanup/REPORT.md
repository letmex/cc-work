# Report: Single Verified Pipeline Cleanup

Status: single verified pipeline cleanup completed

## Answers to Required Questions

1. Were alternative split modes removed or collapsed to the verified split?

Yes. `mixed_mode_tm.py` now exposes only the verified `tm_source` split path through `mixed_mode_energy_split(...)`. The normal CLI no longer exposes alternate split/mechanics modes.

2. Was `apply_alpha_init_intact` removed from normal code?

Yes. `alpha_initialization.py` was deleted, and `main.py` no longer imports or calls `apply_alpha_init_intact`.

3. Were obsolete staggered/debug paths removed or renamed?

Yes. The selectable solver-mode CLI and training branch were removed. Obsolete root reports were deleted. Root debug/cache artifacts were removed.

4. Was legacy top sigma removed from normal outputs?

Yes. `postprocess_results.py` and `plot_results.py` no longer write or plot the old top-boundary sigma metric in normal CSVs/figures.

5. Is energy-conjugate reaction now the only normal reaction metric?

Yes. Normal tables use `reaction_N_energy`, `nominal_stress_energy_MPa`, `reaction_metric_status`, and `is_energy_conjugate`.

6. Were postprocess scripts renamed by function?

The normal postprocess entry point is `postprocess_results.py`, and the plotting helper is `plot_results.py`. Normal outputs are `reaction_by_step.csv`, `stress_strain_by_step.csv`, `reaction_strain.png`, and `stress_strain.png` variants.

7. Were root generated artifacts and debug files deleted?

Yes. Root cache/debug/run artifacts were removed. `outputs/` keeps only managed directory skeletons.

8. Does the normal command no longer require default route flags?

Yes. The normal command is:

```powershell
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 23 TrainableReLU 3.0 --full --load-schedule-file load_schedule_D0020_extended.csv --run-suffix seed23_D0020
```

9. Do docs describe only the verified workflow?

Yes. `README.md`, `POSTPROCESS_WORKFLOW.md`, and `PROJECT_STRUCTURE.md` were rewritten around the single verified route.

10. Do tests pass?

Yes. `pytest -p no:cacheprovider tests -q` reported `38 passed, 8 warnings`.

11. Was any physics changed?

No new physics formula was added. The verified `tm_source` split and history mechanics drive were kept; unused alternate branches and output metrics were removed.

12. Were any new seed studies run?

No. No seed study, no D0040 run, and no training run was performed.

## Verification Summary

- Full tests: `38 passed, 8 warnings`.
- `py_compile` on changed scripts/tests: passed.
- Forbidden-term check on normal source/docs: passed.
- Root cleanliness check: passed.

## Main Code Changes

- `config.py`: removed user-selectable old route flags and sets verified defaults in code.
- `main.py`: removed alpha-init-intact branch.
- `history_field_mixed_tm.py`: removed proximal old columns and top-boundary reaction summary columns.
- `postprocess_results.py`: normal output now uses energy-conjugate checkpoint reaction only.
- `plot_results.py`: normal figures plot only energy reaction/stress and optional virtual-work check.
- Docs and tests were updated for the single verified pipeline.

## Limitations

This cleanup does not prove physical validation, softening correctness, or seed robustness. It only removes conflicting implementation paths and standardizes the normal route/output interface.
