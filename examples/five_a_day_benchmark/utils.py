"""Utility functions for the 5-A-Day Benchmark.

This module contains helper functions for the benchmark example.
"""


def sanitize_name(name: str) -> str:
    """Sanitize name to be a valid Python identifier.

    Converts human-readable agent names to valid identifiers that work across
    all agent frameworks.

    Args:
        name: Agent name to sanitize (may contain spaces, hyphens, etc.)

    Returns:
        Valid Python identifier
    """
    sanitized = name.replace(" ", "_").replace("-", "_")
    if not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = "_" + sanitized
    return sanitized
