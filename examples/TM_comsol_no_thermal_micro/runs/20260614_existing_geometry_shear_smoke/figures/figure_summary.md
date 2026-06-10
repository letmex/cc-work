# Figure Summary

## shear_stress_strain_seed23.png
- What it plots: energy-conjugate nominal shear stress versus engineering shear strain for the seed 23 S0020 smoke.
- Visual takeaway: the curve increases monotonically from about 1.83 MPa to 29.30 MPa over five steps.
- Conclusion support: diagnostic observation only; it confirms shear labels and energy-reaction plotting, but it is not physical validation.

## shear_reaction_strain_seed23.png
- What it plots: checkpointed energy-conjugate reaction_N_energy versus engineering shear strain.
- Visual takeaway: reaction_N_energy increases monotonically from about 0.018 N to 0.293 N.
- Conclusion support: diagnostic observation only; no post-peak drop is observed in this short smoke.

## final_fields_panel_seed23_shear.png
- What it plots: final alpha, u, v, displacement magnitude, HI, HII, He, history, current drive, and mechanics drive fields.
- Visual takeaway: u shows the imposed horizontal shear gradient and v is finite/free, but the strongest He/mechanics-drive regions appear near lower boundary/corner areas rather than exclusively at the notch tip.
- Conclusion support: supports the diagnostic classification that implementation works but the shear smoke is not yet convincing.

## final_alpha_seed23_shear.png
- What it plots: final damage alpha.
- Visual takeaway: alpha remains small, with final max about 0.014; no alpha>=0.8 through-crack forms.
- Conclusion support: diagnostic observation only.

## final_u_seed23_shear.png
- What it plots: final horizontal displacement field.
- Visual takeaway: u ranges from 0 at the bottom to Delta_s at the top, consistent with imposed shear.
- Conclusion support: supports that the shear ansatz imposes the intended horizontal displacement.

## final_v_seed23_shear.png
- What it plots: final vertical displacement field.
- Visual takeaway: v is not identically zero on the top boundary and remains smooth/finite; final top-v range is about [-1.02e-5, 7.01e-6] mm.
- Conclusion support: supports top-v free-boundary diagnostic only.

## final_displacement_seed23_shear.png
- What it plots: final displacement magnitude.
- Visual takeaway: displacement magnitude follows the imposed shear scale without obvious blow-up.
- Conclusion support: diagnostic observation only.

## final_HI_seed23_shear.png
- What it plots: final HI driving field.
- Visual takeaway: HI is nonzero under shear and has strong lower-boundary/corner contributions.
- Conclusion support: diagnostic observation only.

## final_HII_seed23_shear.png
- What it plots: final HII driving field.
- Visual takeaway: HII is active; final HII/HI peak ratio is about 0.632.
- Conclusion support: supports that the shear load activates the HII branch, but not validation.

## final_mechanics_drive_seed23_shear.png
- What it plots: final mechanics-drive field.
- Visual takeaway: the global maximum is near the lower-left boundary/corner, while the notch-tip drive is lower.
- Conclusion support: supports the classification `shear smoke not convincing`.
