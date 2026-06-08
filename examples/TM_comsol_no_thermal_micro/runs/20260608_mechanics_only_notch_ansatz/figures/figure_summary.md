# Figure Summary

The package includes four representative PNGs generated from `He_current` element values. All critical observations are also represented in CSV tables; do not rely on PNGs alone.

| filename | what it plots | visual takeaway | supports physical conclusion? |
|---|---|---|---|
| `pinn_fixed_log10_energy_e300_He_current.png` | log10(`He_current`) for PINN, top-u fixed, 300 epochs | Low-amplitude broad/background drive; no strong notch-tip amplification. | Diagnostic observation only. |
| `fedof_fixed_log10_energy_e300_He_current.png` | log10(`He_current`) for FE-DOF, top-u fixed, 300 epochs | FE-DOF forms a much higher drive field near the notch region while boundary/corner effects remain. | Diagnostic observation only. |
| `pinn_free_log10_energy_e300_He_current.png` | log10(`He_current`) for PINN, top-u free, 300 epochs | PINN remains low-amplitude and broad/background; freeing top horizontal displacement does not recover FE-DOF-like notch amplification. | Diagnostic observation only. |
| `fedof_free_log10_energy_e300_He_current.png` | log10(`He_current`) for FE-DOF, top-u free, 300 epochs | FE-DOF again forms notch-region amplification with remaining boundary/corner effects. | Diagnostic observation only. |
