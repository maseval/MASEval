"""Shared constants for MultiAgentBench module.

This module exists to avoid circular imports between multiagentbench.py,
environment.py, and adapters/marble_adapter.py.
"""

# Shared error message for MARBLE import failures
MARBLE_IMPORT_ERROR = "MARBLE is not available. Clone MARBLE to maseval/benchmark/multiagentbench/marble/\nOriginal error: {error}"
