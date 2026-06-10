# Figure Summary

## shear_stress_strain_seed23.png
- What it plots: energy-conjugate nominal shear stress versus engineering shear strain for the stronger S0030 seed 23 run.
- Visual takeaway: the curve remains monotonic through engineering shear strain 0.005, reaching about 27.29 MPa.
- Conclusion support: diagnostic observation only; no post-peak softening is shown in this schedule.

## shear_reaction_strain_seed23.png
- What it plots: checkpointed energy-conjugate reaction_N_energy versus engineering shear strain.
- Visual takeaway: reaction increases monotonically to about 0.273 N.
- Conclusion support: confirms reaction computability for all 21 checkpoints; not physical validation.

## final_fields_panel_seed23_shear.png
- What it plots: final alpha, u, v, displacement magnitude, HI, HII, He, history, current drive, and mechanics-drive fields.
- Visual takeaway: alpha and driving fields are concentrated at the explicit notch tip, unlike the prior smoke where the global drive was boundary/corner dominated.
- Conclusion support: supports `stronger shear run qualitatively improved` as a diagnostic classification.

## final_alpha_seed23_shear.png
- What it plots: final damage alpha.
- Visual takeaway: alpha grows to about 0.358 at the notch tip, materially above the smoke final alpha of about 0.014.
- Conclusion support: diagnostic observation only; no alpha>=0.8 through-crack forms.

## final_u_seed23_shear.png
- What it plots: final horizontal displacement field.
- Visual takeaway: u shows the intended bottom-to-top shear displacement.
- Conclusion support: verifies the shear kinematic path remains active.

## final_v_seed23_shear.png
- What it plots: final vertical displacement field.
- Visual takeaway: top v is nonzero and finite; final top |v|max/Delta_s is about 1.013.
- Conclusion support: diagnostic only; the top-v-free boundary did not blow up but should be monitored.

## final_HI_seed23_shear.png
- What it plots: final HI field.
- Visual takeaway: HI is sharply concentrated near the notch tip.
- Conclusion support: supports improved notch localization.

## final_HII_seed23_shear.png
- What it plots: final HII field.
- Visual takeaway: HII remains active and similarly notch-localized; final HII/HI peak ratio is about 0.632.
- Conclusion support: supports that shear activates the HII branch in the notch region.

## final_mechanics_drive_seed23_shear.png
- What it plots: final mechanics drive field.
- Visual takeaway: the global mechanics-drive maximum lies at the explicit notch tip region.
- Conclusion support: supports that the stronger run is qualitatively improved relative to the smoke.

## shear_alpha_max_by_step.png
- What it plots: alpha_max versus load step.
- Visual takeaway: alpha_max grows materially after the early steps and reaches about 0.358 by the final step.
- Conclusion support: diagnostic stepwise evidence of damage growth.

## shear_HII_HI_ratio_by_step.png
- What it plots: HII/HI peak ratio versus step.
- Visual takeaway: the ratio stays near 0.63 across the run.
- Conclusion support: diagnostic evidence that HII remains active throughout the load schedule.

## shear_top_v_absmax_over_Delta_by_step.png
- What it plots: top-boundary |v|max normalized by Delta_s.
- Visual takeaway: the ratio stays finite, trending near 1.0 at the later steps.
- Conclusion support: supports non-runaway top-v behavior, with a monitor flag for high ratio.
