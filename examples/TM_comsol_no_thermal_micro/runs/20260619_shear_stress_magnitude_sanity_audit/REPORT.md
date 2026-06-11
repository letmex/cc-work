# Shear Stress Magnitude Sanity Audit

## Scope

This audit reads existing S0050/S0070/S0090 shear handoff packages and source snippets. It does not run training, postprocessing, D0040, a seed study, or any further shear extension. No physics, boundary condition, shear ansatz, material parameter, `l0`, history logic, alpha initialization, or training loss was changed.

## Required Questions

1. Is `23.071 MPa` the maximum stress?  
No. `23.071 MPa is S0070 final stress, not the maximum stress.`

2. What is the actual peak nominal shear stress in S0070 and S0090?  
The actual peak stress reported in S0070/S0090 is approximately `29.9647 MPa`, at step 24 and engineering shear strain 0.006.

3. How is `nominal_shear_stress_energy_MPa` computed?  
It is computed as `reaction_N_energy / reference_area_mm2`, using `reaction_N_energy = dPi/dDelta_s` converted from kN to N.

4. Does recomputing stress from `reaction_N_energy / reference_area_mm2` reproduce the reported values?  
Yes. Maximum absolute error is 1.421e-14 MPa and maximum relative error is 1.439e-14.

5. What reference area is used?  
`reference_area_mm2=0.01`, corresponding to gross specimen width 0.01 mm times unit thickness 1 mm.

6. Is this gross-area nominal stress or local notch-tip stress?  
The reported stress is nominal gross-area shear stress, not local notch-tip stress and not net-ligament shear stress.

7. What is the early elastic shear slope from the numerical curves?  
- S0050: measured initial gross-area slope 5903.36 MPa, expected material G 29529 MPa, ratio 0.199917
- S0070: measured initial gross-area slope 5903.36 MPa, expected material G 29529 MPa, ratio 0.199917
- S0090: measured initial gross-area slope 5903.36 MPa, expected material G 29529 MPa, ratio 0.199917

8. What is the expected material shear modulus `G = E/[2(1+nu)]`?  
With `E=81.5 kN/mm^2` and `nu=0.38`, `G=29529 MPa`.

9. Is the early slope consistent with `G`?  
It is not equal to the bulk material `G`; it is much lower because the plotted value is a structure-level gross-area generalized reaction stress from a notched, top-v-free shear diagnostic, not a homogeneous pure-shear material coupon stress.

10. Is there any evidence of unit conversion error?  
No stress postprocess unit error was found. The kN-to-N conversion occurs before stress formation, and `1 N/mm^2 = 1 MPa` is applied correctly.

11. Is there any evidence of reference-area error?  
No internal reference-area error was found. The postprocess consistently uses gross area 0.01 mm^2. The result should not be interpreted as net-ligament or local stress.

12. Is there an explicit target shear strength in the project?  
No explicit target shear strength was found.

13. Can the approximately `29.9647 MPa` peak be called physically reasonable?  
Not from these packages alone. No explicit target shear strength was found, so the 29.9647 MPa nominal shear peak can only be judged as internally consistent or inconsistent, not physically calibrated.

14. What should the reviewer conclude about stress magnitude?  
The reviewer should conclude that `23.071 MPa` is S0070 final post-peak stress, not maximum stress; the peak is about `29.9647 MPa`. Reaction-to-area recomputation passes exactly, so the stress magnitude is internally consistent as an energy-conjugate gross-area nominal shear stress, while the physical strength interpretation remains uncalibrated.

## Peak And Final Values

- S0070 peak: 29.9647 MPa; S0070 final: 23.071 MPa.
- S0090 peak: 29.9647 MPa; S0090 final: 18.136 MPa.

## Classification

`stress magnitude internally consistent`
