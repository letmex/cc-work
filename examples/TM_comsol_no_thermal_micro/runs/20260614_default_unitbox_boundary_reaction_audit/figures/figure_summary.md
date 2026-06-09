# Figure Summary

Figures are diagnostic only and support the boundary/reaction audit.

| filename | what it plots | visual takeaway | conclusion support |
|---|---|---|---|
| `all_boundary_force_vectors_seed*.png` | Physical-boundary force vectors for each seed and replay variant | Shows whether top reaction is balanced by bottom/side forces. | Diagnostic force-balance evidence. |
| `top_reaction_vs_internal_cut_force.png` | Internal horizontal cut force divided by top reaction | Checks whether internal cuts explain the top-boundary reaction. | Diagnostic reaction consistency evidence. |
| `upper_lower_subdomain_free_body_residual.png` | Upper/lower free-body residual magnitudes for void replay | Shows whether split subdomains are in force balance. | Diagnostic only. |
| `boundary_condition_map.png` | Displacement ansatz boundary conditions | Summarizes top/bottom/side constraints. | Boundary-condition evidence. |
| `rigid_body_sanity_reaction_comparison.png` | Synthetic-field top reaction under crack-band void treatment | Tests whether boundary ansatz alone can force nonzero reaction. | Diagnostic sanity evidence. |
| `split_domain_labels_boundaries_seed*.png` | Split labels, crack band, and physical boundaries | Audits geometry used for subdomain calculations. | Geometry support. |
| `stress_map_void_high_reaction_seed*.png` | sigma_yy map for crack-band void replay | Shows where stress remains when crack-band traction is zero. | Diagnostic observation. |
