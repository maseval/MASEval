"""Prompt template loader for the ConVerse benchmark.

Loads .txt prompt files from the subdirectories of this package.
"""

from functools import lru_cache
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(category: str, name: str) -> str:
    """Load a prompt template from a .txt file.

    Args:
        category: Subdirectory name (``"judge"``, ``"assistant"``,
            or ``"external"``).
        name: Template filename without the ``.txt`` extension.

    Returns:
        Template content as a string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = _TEMPLATES_DIR / category / f"{name}.txt"
    return path.read_text()
