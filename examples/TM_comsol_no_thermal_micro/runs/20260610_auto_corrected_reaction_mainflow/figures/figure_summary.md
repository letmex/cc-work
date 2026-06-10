# Figure summary

## stress_strain_auto_corrected_chain_smoke_seed42.png

- What it plots: engineering strain versus engineering stress for D0020 seed 42 using the newly generated corrected CSV.
- Visual takeaway: the primary plotted curve is sourced from `nominal_stress_energy_exact_MPa`; legacy top sigma is diagnostic.
- Supports: pipeline verification that checkpoint exact reaction can feed the plotting path.
- Does not support: new physical validation.

## reaction_strain_auto_corrected_chain_smoke_seed42.png

- What it plots: engineering strain versus reaction for D0020 seed 42 using the newly generated corrected CSV.
- Visual takeaway: the primary reaction is `reaction_N_energy_exact`; legacy top sigma remains diagnostic.
- Supports: pipeline verification of corrected reaction source selection.
- Does not support: new physical validation.

