# Next Questions

1. Should future full-run packages treat `results/<run>/figures/stress_strain_<label>.png` as the canonical image output from the corrected postprocess command?
2. Should the postprocess command fail hard if figures cannot be generated, or keep the current behavior where CSV success is preserved and figure failure is reported in `figure_status`?
3. Should `main.py` pass a default `run_label` into the postprocess call so training-completion figures have shorter filenames?
