# Figure Summary

All figures are diagnostic only and do not constitute physical validation.

| filename | what it plots | visual takeaway | conclusion support |
|---|---|---|---|
| `corrected_reaction_candidates.png` | Seed 42 reaction curves for candidate corrected reactions. | Compares exact, energy-autograd, current trial, legacy, and postprocessed virtual paths. | Supports metric selection discussion. |
| `energy_autograd_stress_vs_postprocessed_sigma.png` | Elementwise sigma_yy from energy autograd versus postprocessed sigma_yy_tm_eff. | Shows whether postprocessed stress is the derivative of the exact energy density. | Supports stress formula audit. |
| `energy_autograd_virtual_work_identity.png` | Exact dPi/dmode, energy-autograd virtual work, and postprocessed sigma virtual work. | Checks internal consistency of exact reaction and identifies postprocessed-stress mismatch. | Supports acceptance decision. |
| `history_branch_reaction_comparison.png` | Global mode reactions under current, pre-step history, post-step history, and frozen branch energies. | Shows how history branch choice changes the energy-conjugate reaction. | Supports branch audit. |
| `local_patch_stress_comparison.png` | Patch stress comparison for canonical and representative strain states. | Checks formula mismatch on simple local states. | Supports local stress audit. |
| `postthrough_collapse_corrected_candidates.png` | Post-through collapse percentage by candidate and seed. | Shows which candidates collapse after alpha>=0.8 through-crack. | Diagnostic comparison. |
| `prethrough_ratio_corrected_candidates.png` | Pre-through candidate ratios to legacy by seed. | Shows which candidates agree with or differ from legacy before through-crack. | Diagnostic comparison. |
| `shear_gradient_scaling_sanity.png` | Text summary of shear and coordinate-gradient conventions. | Documents tensor shear and physical-gradient conventions. | Sanity check only. |
