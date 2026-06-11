# Existing-Geometry Shear Connectivity Extension Report

## Scope

This package analyzes one longer same-path shear extension, S0070, using the existing geometry and seed 23. No physics, material parameters, `l0`, TM split formulas, history logic, alpha initialization behavior, shear ansatz, boundary conditions, split modes, or training losses were changed.

Notch-connected is defined as a thresholded connected component whose element-centroid set intersects a box of half-width `3.0e-4 mm` around the explicit notch tip at `(0.005, 0.005) mm`. Right-boundary reach is defined as any element centroid in the notch-connected component satisfying `x >= 0.01 - 0.00025 mm`.

## Required Questions

1. Did the S0070 longer shear extension complete?  
Yes. S0070 has 43 stress-strain rows and 43 field files through final step 42.

2. Was this a continuation from S0050 or a full rerun?  
It was a full rerun from step 0: `continued_from_S0050=False`. Clean continuation from the committed S0050 history state was not already implemented unambiguously, so no continuation framework was added.

3. What schedule and training settings were used?  
Schedule: `load_schedules/load_schedule_S0070_shear.csv`, ending at `Delta_s=0.0001` mm. Training: `RPROP=300, LBFGS=1`.

4. Did checkpointed energy-conjugate shear reaction compute at all available steps?  
Yes. `reaction_N_energy = dPi/dDelta_s` is available at all 43 steps.

5. Does the shear stress-strain curve show continued post-peak softening?  
Yes. The peak stress occurs before the final step and the final stress remains below the peak.

6. What are peak stress, final stress, post-peak drop amount, and post-peak drop percent?  
Peak nominal shear stress is 29.9647 MPa at step 24. Final stress is 23.071 MPa. Post-peak drop is 6.89377 MPa, or 23%.

7. Does alpha remain notch-localized?  
Yes. The final alpha maximum is near (0.005083784, 0.0049602254), in the explicit notch-tip region.

8. Does alpha grow beyond S0050 final `alpha_max=1.00034`, or is it saturated?  
Final `alpha_max=1.00091`. This is essentially saturated relative to S0050 final `alpha_max=1.00034`.

9. How do the notch-connected x-spans evolve for alpha thresholds 0.3, 0.5, 0.8, and 0.95?  
- alpha >= 0.3: first notch-connected step 20, final x-span 0.00143761 mm
- alpha >= 0.5: first notch-connected step 22, final x-span 0.00116699 mm
- alpha >= 0.8: first notch-connected step 25, final x-span 0.000895859 mm
- alpha >= 0.95: first notch-connected step 28, final x-span 0.000463828 mm

10. Does any threshold reach the right boundary?  
- alpha >= 0.3: no right-boundary reach
- alpha >= 0.5: no right-boundary reach
- alpha >= 0.8: no right-boundary reach
- alpha >= 0.95: no right-boundary reach

11. Does the crack path propagate away from the notch, or stay as a local notch-tip damage zone?  
The high-threshold x-span grows beyond S0050, so the diagnostic shows propagation away from the notch tip, but it does not reach the right boundary in this package.

12. Is HII still active and notch-localized?  
Yes. The final HII/HI peak ratio is 0.627437, and the field figures keep HII concentrated near the notch region.

13. Does mechanics drive remain notch-dominated?  
Yes. Final mechanics-drive maximum location is (0.005083784, 0.0049602254), classified as `notch-dominated`.

14. Does top `v` remain below warning/unstable thresholds?  
Yes. Final `top_v_absmax/Delta_s=1.13531`, below warning threshold 1.5 and unstable threshold 2.0.

15. Compared with S0050, is the longer extension more informative?  
Yes. It extends the stress-strain softening branch and grows the connected high-alpha x-span beyond S0050 while keeping reaction and top-v diagnostics finite.

16. If no right-boundary through-crack occurs, is the next step another small extension, a geometry/connectivity interpretation, or stopping the shear diagnostic?  
The next step should be geometry/connectivity interpretation before changing physics or escalating load again.

17. Was any physics changed?  
No.

18. Was any seed study or D0040 run performed?  
No. Only seed 23 S0070 is reported here; no D0040 run and no seed study were performed.

## Classification

`shear extension shows propagating crack`

Through-crack at any threshold: `False`.
