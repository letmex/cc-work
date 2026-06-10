# Figure Summary

All figures are curve-output diagnostics only and do not constitute physical validation.

| filename | what it plots | visual takeaway | conclusion support |
|---|---|---|---|
| `D0020_corrected_nominal_stress_strain.png` | Primary nominal stress-strain curves using `reaction_N_energy_exact / reference_area`. | The corrected D0020 stress-strain curves soften after peak for seeds 7, 13, and 42. | Supports fixing the non-softening curve output for checkpointed D0020. |
| `D0020_corrected_vs_legacy_stress_strain.png` | Corrected primary stress, energy virtual-work check, and legacy top-sigma diagnostic. | The legacy curve stays high while the corrected primary curve drops. | Supports demoting legacy top sigma from the primary stress-strain curve. |
| `D0020_stress_strain_softening_gate.png` | Post-peak stress-drop percentage for corrected and legacy curves. | Corrected curve passes the softening gate in 3/3 seeds; legacy does not. | Supports the curve-source fix. |
