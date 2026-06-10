"""Deprecated compatibility wrapper for plot_results.

Use plot_results.py directly only when a separate plotting step is needed.
Normal training and postprocessing invoke plotting through postprocess_results.
"""

from __future__ import annotations

import warnings

from plot_results import *  # noqa: F401,F403
from plot_results import main as _main


warnings.warn(
    "plot_clean_tm_results.py is deprecated; use plot_results.py.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    _main()
