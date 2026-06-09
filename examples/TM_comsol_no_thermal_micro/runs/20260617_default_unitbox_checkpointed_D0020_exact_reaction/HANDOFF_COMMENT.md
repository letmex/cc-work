## Codex handoff: checkpointed D0020 exact reaction diagnostic

Commit: 334e0f23fffa991675e364d59d2ee00ddc7379bf
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260617_default_unitbox_checkpointed_D0020_exact_reaction
Main report: examples/TM_comsol_no_thermal_micro/runs/20260617_default_unitbox_checkpointed_D0020_exact_reaction/REPORT.md

### What changed
- Prioritized D0020, not D0040, for checkpointed exact-reaction testing.
- Ran/processed checkpointed D0020 default-unitbox seed(s) using the same route as the previous robustness package.
- Computed exact actual-PINN `dPi/dDelta` from checkpointed branches with finite-difference checks.
- Compared exact reaction with legacy top-boundary sigma reaction, bottom reaction, internal cuts, and through-crack status.

### Commands run
```powershell
git pull origin main
Read AGENT_HANDOFF_WORKFLOW.md and CODEX_NO_GH_HANDOFF.md.
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 42 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix checkpointed_D0020_seed42_history_default_unitbox --save-step-checkpoints true --checkpoint-every-step true
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 7 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix checkpointed_D0020_seed7_history_default_unitbox --save-step-checkpoints true --checkpoint-every-step true
D:\anaconda3\envs\torch_env\python.exe main.py 8 400 13 TrainableReLU 3.0 --full --pff-model AT2 --mixed-mechanics-mode history --top-u-mode free --coord-normalization unit_box --load-schedule-file load_schedule_D0020_extended.csv --run-suffix checkpointed_D0020_seed13_history_default_unitbox --save-step-checkpoints true --checkpoint-every-step true
D:\anaconda3\envs\torch_env\python.exe artifacts\run_checkpointed_d0020_exact_reaction.py
```

### Key results
- Classification: **exact reaction unresolved**.
- D0040 was not used as the first required exact-reaction rerun.
- See `REPORT.md` and `tables/exact_reaction_summary_by_seed.csv` for seed-level acceptance criteria.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/exact_reaction_summary_by_seed.csv`
- `tables/acceptance_criteria_check.csv`
- `tables/pinn_energy_conjugate_reaction_by_checkpoint.csv`
- `tables/pinn_energy_reaction_finite_difference_check.csv`
- `tables/checkpoint_availability.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does the D0020 checkpointed exact reaction justify demoting `reaction_N_tm_eff`, despite the pre-through exact/legacy mismatch?
2. What is the next minimal diagnostic to resolve the pre-through reaction scaling mismatch?
3. Should D0040 remain deferred until the D0020 reaction definition is settled?

### Constraints
- Do not extend loading.
- Do not change `l0`, material parameters, thermal terms, TM split, history update logic, alpha initialization, or training losses.
- Do not use D0040 as the first required exact-reaction rerun.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
