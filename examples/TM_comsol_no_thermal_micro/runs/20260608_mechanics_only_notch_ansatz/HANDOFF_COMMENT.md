## Codex handoff: mechanics-only notch ansatz diagnostic

Commit: 92f187ca00bc91eeec9f6cb6f856a21dd0589f74
Data folder: `examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz`
Main report: `examples/TM_comsol_no_thermal_micro/runs/20260608_mechanics_only_notch_ansatz/REPORT.md`

### What changed
- Added a mechanics-only alpha-zero diagnostic package for the narrow explicit notch ansatz question.
- Added and ran `debug_mechanics_only_notch_ansatz.py` as a diagnostic-only script snapshot in `artifacts/`.
- Compared current PINN displacement ansatz against independent nodal FE-DOF mechanics on the same mesh, `Delta = 1e-6`, `l0 = 1.5e-4 mm`, material constants, and `tm_source` split.
- Included top-u fixed/free variants and `log10_energy` vs `raw_energy` loss scaling variants.
- Generated representative `He_current` figures plus CSV summaries.

### Commands run
```powershell
git pull origin main

D:\anaconda3\envs\torch_env\python.exe debug_mechanics_only_notch_ansatz.py --out-dir <package> --delta 1e-6 --seed 2 --hidden-layers 8 --neurons 400 --pinn-epochs 0 100 300 --fedof-epochs 300 --top-u-modes fixed free --loss-forms log10_energy raw_energy

D:\anaconda3\envs\torch_env\python.exe -m pytest tests -q

D:\anaconda3\envs\torch_env\python.exe -m py_compile debug_mechanics_only_notch_ansatz.py debug_step0_root_cause.py debug_fedof_energy_baseline.py debug_elastic_only_pinn.py debug_recompute_he_current.py analyze_drive_broadening_stepwise.py config.py field_computation.py compute_energy_mixed_tm.py mixed_mode_tm.py history_field_mixed_tm.py train_mixed_tm.py main.py

gh --version
gh auth status
& 'C:\Program Files\GitHub CLI\gh.exe' --version
& 'C:\Program Files\GitHub CLI\gh.exe' auth status
```

### Key results
- Representative PINN 300-epoch `log10_energy` cases remain broad/background or boundary/background: fixed `bulk/notch He = 1.01504`, free `bulk/notch He = 0.787217`.
- FE-DOF on the same mesh produces notch-amplified drive: fixed `bulk/notch He = 0.260135`, free `bulk/notch He = 0.234402`.
- FE-DOF / PINN notch `He_current` ratio is about `4.630918e+07` for top-u fixed and `4.038712e+07` for top-u free.
- FE-DOF / PINN notch-lip v-jump ratio is about `3.898051e+04` for top-u fixed and `3.241655e+04` for top-u free.
- Raw-energy loss scaling does not remove the PINN broad/background behavior.
- FE-DOF still has boundary/corner maxima, so this is classified as notch-amplified with boundary max, not a physical validation claim.
- Tests passed: `13 passed in 0.06s`.
- GitHub CLI exists at `C:\Program Files\GitHub CLI\gh.exe`, but it is unauthenticated and no token environment variable was present; issue #1 was not auto-commented.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/key_metrics_summary.csv`
- `tables/mechanics_only_comparison.csv`
- `tables/notch_lip_comparison.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Does this evidence justify a next diagnostic focused on local notch-lip displacement enrichment or independent local DOFs within the PINN mechanics representation?
2. Should Codex attempt a mechanics prefit to the FE-DOF displacement field before any further coupled full run?
3. Is there any alternative minimum diagnostic that better separates ansatz expressivity from optimizer path without changing physical parameters?

### Constraints
- Do not change `l0` unless explicitly requested.
- Do not impose `alpha=1` on the geometric notch unless explicitly testing an alternative model.
- Do not change TM split/material parameters unless a clear bug is found.
- Keep phase-field notch behavior, alpha seeding, and history update logic unchanged.
- Do not claim physical validation from this diagnostic package.


