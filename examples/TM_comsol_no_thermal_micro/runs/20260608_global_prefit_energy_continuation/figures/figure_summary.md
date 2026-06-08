# Figure Summary

Each figure compares the FE-DOF target `He_current` field against the PINN-reconstructed `He_current` field for one saved stage. The visual observations below are supported by the CSV metrics; they are diagnostic observations, not physical validation.

## `free_random_init_energy_energy_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the energy-optimized PINN field from random initialization.
- Visual takeaway: The random energy-only field is not target-like; table metrics classify it as `broad/background`.
- Conclusion type: Diagnostic observation only.

## `free_disp_global_prefit_then_energy_prefit_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the PINN field after global displacement prefit.
- Visual takeaway: The prefit stage is close to the target-like branch, with `He_current_corr=0.926113`.
- Conclusion type: Supports the diagnostic statement that global displacement supervision can fit the FE-DOF-like branch.

## `free_disp_global_prefit_then_energy_energy_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the PINN field after global displacement prefit followed by energy-only continuation.
- Visual takeaway: The post-energy field no longer matches the FE-DOF target by correlation, although the postprocessing ratios classify it as weakly `notch-amplified`.
- Conclusion type: Diagnostic observation only.

## `free_disp_strain_global_prefit_then_energy_prefit_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the PINN field after global displacement plus global strain prefit.
- Visual takeaway: This is the closest prefit stage, with `He_current_corr=0.984394` and `strain_corr=0.997029`.
- Conclusion type: Supports the diagnostic statement that the current ansatz can represent the FE-DOF-like branch under global supervision.

## `free_disp_strain_global_prefit_then_energy_energy_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the PINN field after displacement+strain prefit followed by energy-only continuation.
- Visual takeaway: Target agreement collapses after energy-only continuation; table metrics classify the field as `broad/background`.
- Conclusion type: Diagnostic observation only.

## `free_global_curriculum_curriculum_end_target_vs_after_energy_He.png`

- What it plots: FE-DOF target `He_current` versus the final field from a global curriculum ramping to pure energy.
- Visual takeaway: The simple curriculum does not preserve the target-like branch; table metrics classify it as `boundary-dominated`.
- Conclusion type: Diagnostic observation only.

