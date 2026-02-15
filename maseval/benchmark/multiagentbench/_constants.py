"""Shared constants for MultiAgentBench module.

This module exists to avoid circular imports between multiagentbench.py,
environment.py, and adapters/marble_adapter.py.
"""

import sys
from pathlib import Path

# Shared error message for MARBLE import failures
MARBLE_IMPORT_ERROR = "MARBLE is not available. Clone MARBLE to maseval/benchmark/multiagentbench/marble/\nOriginal error: {error}"

# Root of the vendored MARBLE clone (contains marble/ Python package)
_MARBLE_ROOT = str(Path(__file__).parent / "marble")


def ensure_marble_on_path() -> None:
    """Add vendored MARBLE clone root to sys.path.

    MARBLE's internal code uses absolute imports like ``from marble.environments...``.
    Since it's vendored (not installed), we add its clone root to sys.path so Python
    can resolve these imports.
    """
    if _MARBLE_ROOT not in sys.path:
        sys.path.insert(0, _MARBLE_ROOT)
