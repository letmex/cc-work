# Figure Summary

Figures compare reaction metrics computed from existing saved fields only. They do not support physical validation.

| filename | what it plots | visual takeaway | conclusion support |
|---|---|---|---|
| `reaction_metric_curves_D0040.png` | D0040 reaction/load curves for legacy top sigma, saved-field energy proxy, virtual-work proxy, bottom reaction, and internal cut force | Shows whether metric choice changes apparent post-peak behavior. | Diagnostic postprocessing evidence only. |
| `reaction_metric_curves_D0020.png` | Same reaction metrics for D0020 5-seed robustness runs | Shows whether D0020 no-softening is metric-dependent in saved-field proxies. | Diagnostic only. |
| `legacy_vs_energy_reaction_D0040_by_seed.png` | Legacy top reaction against saved-field energy proxy by seed with alpha>=0.8 through-onset marker | Highlights divergence after through-crack onset, if present. | Proxy diagnostic only. |
| `legacy_vs_energy_reaction_D0020_by_seed.png` | Same comparison for D0020 seeds | Audits robustness package under alternative reaction proxy. | Proxy diagnostic only. |
| `reaction_metric_drop_summary.png` | Post-peak drop percentage by case and metric | Summarizes softening/no-softening sensitivity to reaction metric. | Diagnostic summary. |
| `boundary_cut_consistency_summary.png` | Final top, bottom, internal cut, and boundary residual force metrics | Checks whether boundary/cut metrics agree at final saved step. | Boundary/cut consistency evidence. |
