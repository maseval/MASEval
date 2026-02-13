"""Utility functions for Tau 2 benchmark.

This module provides helper functions for:
- Database state hashing (critical for deterministic evaluation)
- Pydantic model utilities
- File loading utilities

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Hashing Functions
# =============================================================================


def get_dict_hash(data: Dict[str, Any]) -> str:
    """Compute a deterministic hash for a dictionary.

    Uses JSON serialization with sorted keys to ensure consistent hashing
    regardless of dictionary key order.

    Args:
        data: Dictionary to hash

    Returns:
        SHA-256 hash hex string

    Example:
        >>> get_dict_hash({"a": 1, "b": 2})
        '...'
    """
    # Sort keys and use separators without spaces for compact, consistent output
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_pydantic_hash(model: BaseModel) -> str:
    """Compute a deterministic hash for a Pydantic model.

    Uses Pydantic's built-in JSON serialization which handles datetime
    and other complex types properly.

    Args:
        model: Pydantic model instance

    Returns:
        SHA-256 hash hex string

    Example:
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     x: int
        >>> get_pydantic_hash(MyModel(x=1))
        '...'
    """
    # Use Pydantic's model_dump_json for proper datetime serialization
    # Sort keys by dumping to dict first, sorting, then re-serializing
    data = model.model_dump(mode="json")  # Converts datetimes to ISO strings
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# Pydantic Utilities
# =============================================================================


def update_pydantic_model_with_dict(model: T, update_data: Dict[str, Any]) -> T:
    """Update a Pydantic model with values from a dictionary.

    Recursively updates nested models. Creates a new model instance.

    Args:
        model: Pydantic model to update
        update_data: Dictionary with update values

    Returns:
        New model instance with updates applied

    Example:
        >>> from pydantic import BaseModel
        >>> class Config(BaseModel):
        ...     x: int
        ...     y: str
        >>> config = Config(x=1, y="old")
        >>> updated = update_pydantic_model_with_dict(config, {"y": "new"})
        >>> updated.y
        'new'
    """
    if not update_data:
        return model

    # Get current data
    current_data = model.model_dump()

    # Deep merge update_data into current_data
    merged = _deep_merge(current_data, update_data)

    # Create new model with merged data
    return model.__class__.model_validate(merged)


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        updates: Dictionary with updates

    Returns:
        New merged dictionary
    """
    result = base.copy()

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# File Loading Utilities
# =============================================================================


def load_file(path: Union[str, Path]) -> Any:
    """Load a structured file (JSON, TOML, YAML).

    Detects format by file extension.

    Args:
        path: Path to file

    Returns:
        Parsed file content

    Raises:
        ValueError: If file extension is not supported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif suffix == ".toml":
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:
            import tomli as tomllib  # type: ignore
        with path.open("rb") as f:
            return tomllib.load(f)
    elif suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML files: pip install pyyaml")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .json, .toml, .yml, .yaml")


def dump_file(path: Union[str, Path], data: Any, **kwargs: Any) -> None:
    """Dump data to a structured file (JSON, TOML).

    Detects format by file extension.

    Args:
        path: Path to file
        data: Data to dump
        **kwargs: Additional arguments passed to dump function

    Raises:
        ValueError: If file extension is not supported
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, **kwargs)
    elif suffix == ".toml":
        try:
            import tomli_w  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("tomli_w is required to write TOML files: pip install tomli_w")
        with path.open("wb") as f:
            tomli_w.dump(data, f, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for writing: {suffix}. Supported: .json, .toml")


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_tool_calls(
    expected_name: str,
    expected_args: Dict[str, Any],
    actual_name: str,
    actual_args: Dict[str, Any],
    compare_args: list[str] | None = None,
) -> bool:
    """Compare a tool call against expected values.

    Args:
        expected_name: Expected tool name
        expected_args: Expected arguments
        actual_name: Actual tool name
        actual_args: Actual arguments
        compare_args: List of argument names to compare. If None, compares all.

    Returns:
        True if tool calls match, False otherwise
    """
    if expected_name != actual_name:
        return False

    if compare_args is None:
        # Compare all arguments present in actual_args
        compare_args = list(actual_args.keys())

    if not compare_args:
        return True

    expected_subset = {k: v for k, v in expected_args.items() if k in compare_args}
    actual_subset = {k: v for k, v in actual_args.items() if k in compare_args}

    return expected_subset == actual_subset
