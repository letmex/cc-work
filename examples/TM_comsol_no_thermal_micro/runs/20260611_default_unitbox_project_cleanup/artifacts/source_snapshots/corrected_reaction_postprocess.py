"""Deprecated compatibility wrapper for postprocess_results.

Use postprocess_results.py for normal TM postprocessing. This wrapper is kept
temporarily so older command lines fail softly while user-facing docs migrate.
"""

from __future__ import annotations

import warnings

from postprocess_results import *  # noqa: F401,F403
from postprocess_results import main as _main


warnings.warn(
    "corrected_reaction_postprocess.py is deprecated; use postprocess_results.py.",
    DeprecationWarning,
    stacklevel=2,
)


if __name__ == "__main__":
    _main()
