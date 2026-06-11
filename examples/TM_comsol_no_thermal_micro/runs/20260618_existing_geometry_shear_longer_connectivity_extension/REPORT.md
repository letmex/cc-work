# Existing-Geometry Shear Longer Connectivity Extension Report

## Scope

This package analyzes S0090, one more same-path shear extension using seed 23 and the existing geometry. No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, shear ansatz, boundary conditions, split modes, or training losses were changed.

## Required Questions

1. Did the S0090 longer shear extension complete?  
Yes. S0090 has 53 stress-strain rows and 53 field files.

2. Was this a continuation from S0070 or a full rerun?  
It was a full rerun from step 0: `continued_from_S0070=False`. Clean continuation from the committed S0070 history state was not already implemented unambiguously, so no continuation framework was added.

3. What schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0090_shear.csv`, ending at `Delta_s=0.00012` mm. Training: `RPROP=300, LBFGS=1`.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. `reaction_N_energy = dPi/dDelta_s` is available at all 53 steps.

5. Does post-peak softening continue beyond S0070?  
The final nominal shear stress is 18.136 MPa, compared with S0070 final 23.071 MPa.

6. What are peak stress, final stress, post-peak drop amount, and post-peak drop percent?  
Peak nominal shear stress is 29.9647 MPa at step 24. Final stress is 18.136 MPa. Post-peak drop is 11.8288 MPa, or 39.5%.

7. Does alpha remain notch-localized?  
Final alpha maximum is near (0.005083784, 0.0049602254).

8. Does alpha stay saturated near 1, and does the connected damaged region grow?  
Final `alpha_max=1.0015`. X-span growth by threshold is listed below.

9. How do the notch-connected x-spans evolve for thresholds 0.3, 0.5, 0.8, and 0.95?  
- alpha >= 0.3: final x-span 0.00183102 mm, growth vs S0070 0.000393403 mm
- alpha >= 0.5: final x-span 0.00159079 mm, growth vs S0070 0.000423795 mm
- alpha >= 0.8: final x-span 0.00133007 mm, growth vs S0070 0.000434216 mm
- alpha >= 0.95: final x-span 0.00106198 mm, growth vs S0070 0.000598153 mm

10. How much did each x-span grow relative to S0070?  
- alpha >= 0.3: final x-span 0.00183102 mm, growth vs S0070 0.000393403 mm
- alpha >= 0.5: final x-span 0.00159079 mm, growth vs S0070 0.000423795 mm
- alpha >= 0.8: final x-span 0.00133007 mm, growth vs S0070 0.000434216 mm
- alpha >= 0.95: final x-span 0.00106198 mm, growth vs S0070 0.000598153 mm

11. Does any threshold reach the right boundary?  
- alpha >= 0.3: no right-boundary reach
- alpha >= 0.5: no right-boundary reach
- alpha >= 0.8: no right-boundary reach
- alpha >= 0.95: no right-boundary reach

12. Does the crack path propagate away from the notch, or remain a local notch-tip damage zone?  
Classification: `shear longer extension shows continued propagation`.

13. Is HII still active and notch-localized?  
Final HII/HI peak ratio is 0.631579.

14. Does mechanics drive remain notch-dominated?  
Final mechanics-drive maximum location is (0.005083784, 0.0049602254), classified as `notch-dominated`.

15. Does top `v` remain below warning/unstable thresholds?  
Final `top_v_absmax/Delta_s=1.18739`.

16. Compared with S0070, is the longer extension more informative?  
It is more informative if the x-span growth and stress-softening changes in `tables/shear_extension_vs_S0070_comparison.csv` are material.

17. If no right-boundary through-crack occurs, should the next step be another extension, a geometry interpretation, or stopping the single-seed shear diagnostic?  
If no right-boundary through-crack occurs, stop further single-seed shear escalation and interpret geometry/connectivity first.

18. Was any physics changed?  
No.

19. Was any seed study or D0040 run performed?  
No. Only seed 23 S0090 is reported here; no D0040 run and no seed study were performed.

20. Were local commits pushed? If not, state that local main remains ahead of origin.  
No commits were pushed. Git status at package generation: `## main...origin/main [ahead 2]`.

## Classification

`shear longer extension shows continued propagation`

Through-crack at any threshold: `False`.
