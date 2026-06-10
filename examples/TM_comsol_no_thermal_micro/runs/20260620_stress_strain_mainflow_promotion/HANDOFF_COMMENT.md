## Codex handoff: stress-strain mainflow promotion

Commit: 210a09a
Data folder: examples/TM_comsol_no_thermal_micro/runs/20260620_stress_strain_mainflow_promotion
Main report: examples/TM_comsol_no_thermal_micro/runs/20260620_stress_strain_mainflow_promotion/REPORT.md

### What changed
- Modified external `D:\ProgramData\PINN\FEM-PINN-main\source\postprocess_tm.py` to promote corrected energy-conjugate stress-strain curves as the main plotting source.
- Mainflow now prefers `curves/corrected_stress_strain_by_step.csv` / `nominal_stress_energy_exact_MPa` when available.
- Legacy `macro_stress` from top-boundary sigma is separated to `macro_stress_strain_legacy.png` and is no longer the primary stress-strain plot when corrected data is missing.
- Stored before/after source files and a diff in this package.
- Did not run or process D0040.

### Commands run
```powershell
git pull origin main
D:\anaconda3\envs\torch_env\python.exe -m py_compile D:\ProgramData\PINN\FEM-PINN-main\source\postprocess_tm.py
D:\anaconda3\envs\torch_env\python.exe -m py_compile artifacts\validate_postprocess_tm_curve_promotion.py
D:\anaconda3\envs\torch_env\python.exe artifacts\validate_postprocess_tm_curve_promotion.py
D:\anaconda3\envs\torch_env\python.exe -m pytest D:\ProgramData\PINN\FEM-PINN-main\examples\TM_comsol_no_thermal_micro\tests -q
```

### Key results
- Classification: **mainflow corrected stress-strain source promoted**.
- Validation checks passed: 6/6.
- Primary mainflow metric selected: `nominal_stress_energy_exact_MPa`.
- Legacy top-sigma curve remains diagnostic-only.
- D0040 remains untouched.

### Files to read first
- `README.md`
- `REPORT.md`
- `tables/mainflow_curve_source_selection.csv`
- `artifacts/postprocess_tm_mainflow_promotion.diff`
- `figures/figure_summary.md`

### Question for ChatGPT
1. Is the `postprocess_tm.py` source promotion sufficient to call the corrected stress-strain curve the main plotting flow?
2. Should any training CSV column names be changed next, or is plotting/source selection enough?
3. Should D0040 remain deferred until explicitly requested?

### Constraints
- Do not run D0040.
- Do not change `l0`, material parameters, thermal terms, TM split, history logic, alpha initialization, or training losses.
- Do not impose `alpha=1` on the geometric notch.
- Do not add notch/lip/local/jump/geometry-guided losses.
- Do not claim physical validation.
