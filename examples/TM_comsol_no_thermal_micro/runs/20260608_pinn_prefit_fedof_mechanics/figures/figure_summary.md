# Figure Summary

The package includes three top-u-free target-vs-prefit `He_current` comparison figures. Critical conclusions are encoded in CSV tables; PNGs are supporting diagnostics only.

| filename | what it plots | visual takeaway | supports physical conclusion? |
|---|---|---|---|
| `prefit_free_disp_only_e1000_target_vs_pred_He.png` | FE-DOF target and PINN prefit `log10(He_current)` for displacement-only prefit | The current PINN ansatz can reconstruct a notch-amplified `He_current` field from supervised `u/v` fitting. | Diagnostic observation only. |
| `prefit_free_disp_lip_e1000_target_vs_pred_He.png` | FE-DOF target and PINN prefit `log10(He_current)` for lip-weighted prefit | Lip weighting enforces notch-lip jump but over-concentrates the predicted drive and degrades global field agreement. | Diagnostic observation only. |
| `prefit_free_disp_strain_e1000_target_vs_pred_He.png` | FE-DOF target and PINN prefit `log10(He_current)` for displacement+strain prefit | Adding strain loss improves strain/He correlation and preserves notch-amplified reconstruction. | Diagnostic observation only. |
