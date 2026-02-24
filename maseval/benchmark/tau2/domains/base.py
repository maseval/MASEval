"""Tau 2 Benchmark - Base Domain Classes.

Base classes for domain databases and tools.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from:
- src/tau2/environment/db.py
- src/tau2/environment/toolkit.py
"""

import inspect
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union

from typing_extensions import Self

from pydantic import BaseModel, ConfigDict

from maseval.benchmark.tau2.utils import get_pydantic_hash, load_file, update_pydantic_model_with_dict


# =============================================================================
# Database Base Class
# =============================================================================


class DB(BaseModel):
    """Base class for domain databases.

    Provides standard load/dump/hash operations for Pydantic-based databases.

    Adapted from: tau2-bench src/tau2/environment/db.py
    """

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """Load the database from a structured file (JSON, TOML, YAML).

        Args:
            path: Path to database file

        Returns:
            Database instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
        """
        data = load_file(path)
        return cls.model_validate(data)

    def get_hash(self) -> str:
        """Get a deterministic hash of the database state.

        Used for comparing database states before/after tool execution.

        Returns:
            SHA-256 hash hex string
        """
        return get_pydantic_hash(self)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the database.

        Override in subclass to provide domain-specific statistics.

        Returns:
            Dictionary of statistics
        """
        return {}

    def copy_deep(self) -> Self:
        """Create a deep copy of the database.

        Returns:
            New database instance with copied data
        """
        return self.__class__.model_validate(self.model_dump())


T = TypeVar("T", bound=DB)


# =============================================================================
# Tool Types
# =============================================================================


class ToolType(str, Enum):
    """Type of a tool based on its behavior.

    Adapted from: tau2-bench src/tau2/environment/toolkit.py
    """

    READ = "read"  # Read-only operations
    WRITE = "write"  # State-modifying operations
    THINK = "think"  # Reasoning/thinking operations
    GENERIC = "generic"  # Other operations (e.g., calculator, transfer)


TOOL_ATTR = "__tau2_tool__"
TOOL_TYPE_ATTR = "__tau2_tool_type__"


def is_tool(tool_type: ToolType = ToolType.READ) -> Callable[[Callable], Callable]:
    """Decorator to mark a function as a tool.

    Args:
        tool_type: Type of tool (READ, WRITE, THINK, GENERIC)

    Returns:
        Decorator function

    Example:
        >>> class MyTools(ToolKitBase):
        ...     @is_tool(ToolType.READ)
        ...     def get_item(self, item_id: str) -> Item:
        ...         return self.db.items[item_id]
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, TOOL_ATTR, True)
        setattr(func, TOOL_TYPE_ATTR, tool_type)
        return func

    return decorator


# =============================================================================
# JSON Schema Helpers (for get_tool_metadata)
# =============================================================================


def _resolve_node(node: Any, defs: Dict[str, Any]) -> Any:
    """Resolve a JSON schema node, inlining ``$ref`` references and simplifying ``anyOf``."""
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        if ref_name in defs:
            return _resolve_node(dict(defs[ref_name]), defs)
        return node

    # Simplify anyOf (typically Optional[X] -> X with nullable flag)
    if "anyOf" in node:
        variants = node["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(non_null) == 1:
            resolved = _resolve_node(non_null[0], defs)
            resolved["nullable"] = True
            if "description" in node and "description" not in resolved:
                resolved["description"] = node["description"]
            return resolved
        if non_null:
            return _resolve_node(non_null[0], defs)

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "title", "default"):
            continue
        if isinstance(value, dict):
            out[key] = _resolve_node(value, defs)
        elif isinstance(value, list):
            out[key] = [_resolve_node(v, defs) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def _resolve_schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resolved per-parameter schemas from a JSON Schema ``properties`` block."""
    defs = schema.get("$defs", {})
    properties = schema.get("properties", {})

    resolved: Dict[str, Any] = {}
    for name, prop in properties.items():
        resolved[name] = _resolve_node(prop, defs)
    return resolved


# =============================================================================
# ToolKit Base Class
# =============================================================================


class ToolKitMeta(type):
    """Metaclass that discovers methods decorated with @is_tool.

    Adapted from: tau2-bench src/tau2/environment/toolkit.py:ToolKitType
    """

    def __init__(cls, name: str, bases: tuple, attrs: Dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)

        # Collect all methods marked as tools
        func_tools: Dict[str, Callable] = {}
        for method_name, method in attrs.items():
            if isinstance(method, property):
                method = method.fget
            if hasattr(method, TOOL_ATTR):
                func_tools[method_name] = method

        # Create property to access tools
        @property
        def _func_tools(self) -> Dict[str, Callable]:  # type: ignore[misc]
            all_func_tools = func_tools.copy()
            # Include parent class tools - iterate through MRO to find parent _func_tools
            for base in cls.__mro__[1:]:
                if hasattr(base, "_func_tools") and base is not ToolKitBase:
                    try:
                        # Access parent's _func_tools directly via the descriptor
                        parent_prop = getattr(base, "_func_tools", None)
                        if parent_prop is not None and hasattr(parent_prop, "fget"):
                            parent_tools = parent_prop.fget(self)  # type: ignore[arg-type]
                            all_func_tools.update(parent_tools)
                    except (AttributeError, TypeError):
                        pass
                    break
            return all_func_tools

        cls._func_tools = _func_tools  # type: ignore[attr-defined]


class ToolKitBase(Generic[T], metaclass=ToolKitMeta):
    """Base class for domain tool kits.

    Provides infrastructure for tool discovery, invocation, and database access.

    Adapted from: tau2-bench src/tau2/environment/toolkit.py:ToolKitBase
    """

    db: Optional[T]
    _func_tools: Dict[str, Callable]  # Set by metaclass

    def __init__(self, db: Optional[T] = None) -> None:
        """Initialize the toolkit.

        Args:
            db: Domain database instance
        """
        self.db = db

    @property
    def tools(self) -> Dict[str, Callable]:
        """Get all available tools in this toolkit.

        Returns:
            Dictionary mapping tool names to bound methods
        """
        return {name: getattr(self, name) for name in self._func_tools.keys()}

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by name.

        Args:
            tool_name: Name of the tool
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name](**kwargs)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool exists, False otherwise
        """
        return tool_name in self.tools

    def tool_type(self, tool_name: str) -> ToolType:
        """Get the type of a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolType enum value
        """
        return getattr(self.tools[tool_name], TOOL_TYPE_ATTR)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the toolkit.

        Returns:
            Dictionary with tool counts by type
        """
        num_tools = len(self.tools)
        num_read = sum(self.tool_type(name) == ToolType.READ for name in self.tools)
        num_write = sum(self.tool_type(name) == ToolType.WRITE for name in self.tools)
        num_think = sum(self.tool_type(name) == ToolType.THINK for name in self.tools)
        num_generic = sum(self.tool_type(name) == ToolType.GENERIC for name in self.tools)
        return {
            "num_tools": num_tools,
            "num_read_tools": num_read,
            "num_write_tools": num_write,
            "num_think_tools": num_think,
            "num_generic_tools": num_generic,
        }

    def update_db(self, update_data: Optional[Dict[str, Any]] = None) -> None:
        """Update the database with new values.

        Args:
            update_data: Dictionary of updates to apply

        Raises:
            ValueError: If database not initialized
        """
        if update_data is None:
            return
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        self.db = update_pydantic_model_with_dict(self.db, update_data)  # type: ignore[assignment]

    def get_db_hash(self) -> str:
        """Get hash of current database state.

        Returns:
            SHA-256 hash hex string

        Raises:
            ValueError: If database not initialized
        """
        if self.db is None:
            raise ValueError("Database has not been initialized.")
        return self.db.get_hash()

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools.

        Returns:
            Dictionary mapping tool names to their docstrings
        """
        descriptions = {}
        for name, tool in self.tools.items():
            doc = tool.__doc__ or ""
            descriptions[name] = doc.strip()
        return descriptions

    def get_tool_metadata(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata for a tool including description and input schema.

        Uses ``pydantic.create_model`` + ``model_json_schema`` for correct
        handling of complex types (``List[str]``, ``Optional[int]``, enums,
        nested Pydantic models).  Matches the schema generation approach from
        the original tau2-bench ``Tool.openai_schema``
        (``src/tau2/environment/tool.py``).

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with tool metadata:
                - description: Tool docstring
                - inputs: Dictionary of input parameters with JSON Schema types
                - tool_type: ToolType enum value

        Raises:
            ValueError: If tool not found
        """
        from typing import Any as TypingAny

        from docstring_parser import parse as parse_docstring
        from pydantic import Field as PydanticField
        from pydantic import create_model

        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")

        tool = self.tools[tool_name]
        doc = parse_docstring(tool.__doc__ or "")

        description = doc.short_description or f"Execute {tool_name}"
        if doc.long_description:
            description += "\n\n" + doc.long_description

        # Build pydantic model from signature + docstring params
        # Matches original tau2-bench as_tool() in src/tau2/environment/tool.py
        sig = inspect.signature(tool)
        doc_params = {p.arg_name: p for p in doc.params}
        model_fields: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            anno = param.annotation
            default = param.default
            if default is param.empty:
                default = ...
            if param_name in doc_params:
                default = PydanticField(default, description=doc_params[param_name].description)
                if (anno is param.empty) and (doc_params[param_name].type_name is not None):
                    anno = doc_params[param_name].type_name
            if anno is param.empty:
                anno = TypingAny
            model_fields[param_name] = (anno, default)

        params_model = create_model("parameters", **model_fields)
        schema = params_model.model_json_schema()

        # Resolve $ref/$defs and extract per-parameter schemas
        inputs = _resolve_schema_properties(schema)

        return {
            "description": description,
            "inputs": inputs,
            "tool_type": self.tool_type(tool_name),
        }
