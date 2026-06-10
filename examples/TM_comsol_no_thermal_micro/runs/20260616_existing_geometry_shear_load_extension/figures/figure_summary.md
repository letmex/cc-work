# Figure Summary

## shear_stress_strain_seed23.png
- What it plots: checkpointed energy-conjugate nominal shear stress versus engineering shear strain for the S0050 seed 23 extension.
- Visual takeaway: stress peaks at step 24, engineering shear strain 0.006, about 29.9647 MPa, then drops to 28.4379 MPa at the final step.
- Conclusion support: supports a diagnostic post-peak response, not physical validation.

## shear_reaction_strain_seed23.png
- What it plots: energy-conjugate reaction_N_energy versus engineering shear strain.
- Visual takeaway: reaction peaks and then declines after the same peak step as the stress curve.
- Conclusion support: supports that checkpointed reaction is available and shows post-peak behavior.

## final_fields_panel_seed23_shear.png
- What it plots: final alpha, displacement, HI, HII, He/history/current drive, and mechanics-drive fields.
- Visual takeaway: alpha and drive remain concentrated in the explicit notch-tip region at the final step.
- Conclusion support: supports notch-localized crack growth as a diagnostic observation.

## final_alpha_seed23_shear.png
- What it plots: final damage alpha.
- Visual takeaway: alpha reaches about 1.00034 near the notch tip.
- Conclusion support: supports crack growth beyond S0030; no full alpha>=0.8 through-crack to the right boundary is detected.

## final_u_seed23_shear.png
- What it plots: final horizontal displacement field.
- Visual takeaway: the intended bottom-to-top shear displacement remains active.
- Conclusion support: verifies the shear loading path.

## final_v_seed23_shear.png
- What it plots: final vertical displacement field.
- Visual takeaway: top v remains finite; final top |v|max/Delta_s is 1.08336.
- Conclusion support: diagnostic top-v monitor only.

## final_HI_seed23_shear.png
- What it plots: final HI field.
- Visual takeaway: HI is localized near the notch tip.
- Conclusion support: supports notch-localized mixed driving.

## final_HII_seed23_shear.png
- What it plots: final HII field.
- Visual takeaway: HII remains active and notch-localized; final HII/HI peak ratio is 0.597398.
- Conclusion support: supports that shear continues to activate the HII branch.

## final_mechanics_drive_seed23_shear.png
- What it plots: final mechanics-drive field.
- Visual takeaway: the global mechanics-drive maximum remains at the explicit notch-tip region.
- Conclusion support: supports notch-dominated drive.

## shear_alpha_max_by_step.png
- What it plots: alpha_max versus engineering shear strain.
- Visual takeaway: alpha grows beyond the S0030 final value and reaches approximately 1.00034.
- Conclusion support: diagnostic evidence of continued crack growth.

## shear_HII_HI_ratio_by_step.png
- What it plots: HII/HI peak ratio versus engineering shear strain.
- Visual takeaway: HII remains active, with the ratio near 0.6 over most of the extension.
- Conclusion support: diagnostic evidence for active HII contribution.

## shear_top_v_absmax_over_Delta_by_step.png
- What it plots: top-boundary |v|max normalized by Delta_s.
- Visual takeaway: the ratio remains below the 1.5 warning threshold; maximum observed value is 1.08336.
- Conclusion support: supports finite non-runaway top-v behavior in this run.

## shear_notch_drive_by_step.png
- What it plots: notch-tip and bottom-right mechanics-drive maxima versus engineering shear strain.
- Visual takeaway: the notch-tip drive strongly dominates the bottom-right monitor region.
- Conclusion support: supports notch-dominated mechanics drive.

## shear_through_crack_status_by_step.png
- What it plots: notch-connected alpha>=0.5 and alpha>=0.8 component counts, plus alpha>=0.8 through-to-right flag.
- Visual takeaway: connected damage grows near the notch, but the alpha>=0.8 component does not reach the right boundary.
- Conclusion support: supports crack growth without full through-crack classification.
