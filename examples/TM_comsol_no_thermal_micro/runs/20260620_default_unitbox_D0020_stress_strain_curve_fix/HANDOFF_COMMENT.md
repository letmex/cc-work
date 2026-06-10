## Codex handoff: D0020 stress-strain curve fix

Commit: ae206f0
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_stress_strain_curve_fix
Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_default_unitbox_D0020_stress_strain_curve_fix/REPORT.md

### What changed
- Fixed the D0020 stress-strain curve output source: primary nominal stress now uses `reaction_N_energy_exact / reference_area`.
- Kept legacy top sigma only as a diagnostic overlay.
- Generated corrected stress-strain CSV, softening summary, source policy table, and three figures.
- Did not run or process D0040.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m py_compile artifacts\run_d0020_stress_strain_curve_fix.py
D:\anaconda3\envs\torch_env\python.exe artifacts\run_d0020_stress_strain_curve_fix.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
```

### Key results
- Classification: **D0020 stress-strain curve softening fixed**.
- Primary stress-strain metric: `nominal_stress_energy_exact_MPa`.
- Primary reaction source: `reaction_N_energy_exact`.
- Corrected primary stress-strain curve softens in 3/3 seeds.
- Legacy top-sigma diagnostic softens in 0/3 seeds by the same gate.
- Corrected and legacy conclusions disagree in 3/3 seeds.
- D0040 remains untouched.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/corrected_stress_strain_by_step.csv`
- `tables/stress_strain_softening_summary.csv`
- `tables/stress_strain_curve_source_policy.csv`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is this enough to treat the D0020 stress-strain non-softening issue as a curve-source bug rather than a physics/model rerun task?
2. Should Codex promote this curve-source convention into reusable plotting code next?
3. Should D0040 stay deferred until the user explicitly asks for it?

### Constraints
- Do not run D0040.
- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
