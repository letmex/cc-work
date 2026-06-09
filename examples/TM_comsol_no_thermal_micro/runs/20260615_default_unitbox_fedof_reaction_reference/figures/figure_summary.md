# Figure Summary

Figures are diagnostic only. They summarize the FE-DOF frozen-alpha reaction reference audit and do not support physical validation.

| filename | what it plots | visual takeaway | conclusion support |
|---|---|---|---|
| `reaction_metric_comparison_by_seed_variant.png` | Top sigma integral, energy-conjugate reaction, and internal cut force by seed/variant/BC | Compares whether reaction metrics collapse under voiding or piecewise separation. | Diagnostic reaction-metric evidence. |
| `energy_conjugate_reaction_vs_top_sigma_integral.png` | Energy-conjugate reaction against top-boundary stress integral | Points away from or toward agreement between a generalized load and a local top stress metric. | Diagnostic only. |
| `boundary_condition_sensitivity_top_reaction.png` | Original top/bottom BC versus minimal rigid-body BC | Shows whether the top reaction is sensitive to boundary treatment. | Boundary-condition evidence. |
| `fedof_displacement_field_seed7_*.png` | Example FE-DOF displacement magnitude fields for continuous and piecewise variants | Shows the diagnostic displacement mode used for reaction comparison. | Diagnostic illustration. |
| `fedof_stress_map_seed7_*.png` | Example sigma_yy maps for original and minimal BC treatments | Shows whether stress remains near boundary regions or ligament paths. | Diagnostic observation. |
| `upper_lower_free_body_residual_fedof.png` | Whole/upper/lower residual magnitudes for piecewise void minimal BC | Checks force-balance consistency of separated subdomains. | Diagnostic force-balance evidence. |
| `internal_cut_force_vs_top_reaction.png` | Internal cut/top reaction ratio for continuous crack-band void FE-DOF solves | Tests whether internal cut forces explain top sigma reaction. | Diagnostic reaction consistency evidence. |
