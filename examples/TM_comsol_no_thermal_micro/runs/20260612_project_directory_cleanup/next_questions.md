# Next Questions

1. Should any old local generated results be regenerated under `outputs/` only when needed, or should the next task proceed without recreating old artifacts?
2. Is the current functional naming (`postprocess_results.py`, `plot_results.py`) acceptable as the stable user-facing workflow?
3. Should a small CI-style check be added outside this local folder later, or is the local pytest root-cleanliness test sufficient for now?
