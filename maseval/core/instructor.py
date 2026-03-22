"""Instructor library integration for structured LLM outputs.

Provides helpers to create instructor-patched clients from provider SDK clients
and to generate flattened JSON schemas from Pydantic models.

Instructor adds ``response_model`` support with automatic validation and retries
to any supported LLM provider.

Example:
    ```python
    from maseval.core.instructor import create_instructor_client

    # Wrap an OpenAI client
    import openai
    client = openai.OpenAI()
    instructor_client = create_instructor_client(client, provider="openai")
    ```
"""

from __future__ import annotations

from typing import Any, Optional, Dict


def create_instructor_client(
    client: Any,
    provider: str,
    mode: Optional[str] = None,
) -> Any:
    """Create an instructor-patched client from a provider SDK client.

    All patched clients expose a unified API:
    ``client.chat.completions.create(response_model=..., messages=...)``.

    Args:
        client: The provider SDK client instance (e.g., ``openai.OpenAI()``,
            ``anthropic.Anthropic()``). For LiteLLM, pass ``litellm.completion``.
        provider: Provider name. One of: ``"openai"``, ``"litellm"``.
            For other providers, use ``instructor.from_provider()`` directly.
        mode: Optional instructor mode override. If None, uses the default
            for the provider.

    Returns:
        An instructor-patched client supporting ``response_model``.

    Raises:
        ValueError: If provider is not recognized.
    """
    import instructor

    kwargs: Dict[str, Any] = {}
    if mode is not None:
        kwargs["mode"] = getattr(instructor.Mode, mode.upper(), mode)

    if provider == "openai":
        return instructor.from_openai(client, **kwargs)
    elif provider == "litellm":
        return instructor.from_litellm(client, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Use instructor.from_provider() directly for other providers.")


def flatten_model_schema(model: type) -> Dict[str, Any]:
    """Generate a flattened JSON schema from a Pydantic model.

    Uses instructor's ``openai_schema`` to produce a clean schema, then
    applies additional flattening to remove ``anyOf`` (for ``Optional``
    fields) and other constructs that some providers reject.

    Args:
        model: A Pydantic BaseModel subclass.

    Returns:
        A flat JSON schema dict suitable for LLM tool parameters.
    """
    import instructor

    schema_obj = instructor.openai_schema(model)  # ty: ignore[invalid-argument-type]
    schema = schema_obj.openai_schema["parameters"]

    # instructor's openai_schema still produces anyOf for Optional fields.
    # Flatten those for provider compatibility (especially Google GenAI).
    return _resolve_schema(schema)


def _resolve_schema(node: Any) -> Any:
    """Recursively resolve anyOf and strip unsupported keys from a schema."""
    if not isinstance(node, dict):
        return node

    _STRIP_KEYS = {"$defs", "additionalProperties", "title", "default"}

    # Simplify anyOf (Optional[X] -> X with nullable)
    if "anyOf" in node:
        variants = node["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(non_null) == 1:
            resolved = _resolve_schema(non_null[0])
            if isinstance(resolved, dict):
                resolved["nullable"] = True
                if "description" in node and "description" not in resolved:
                    resolved["description"] = node["description"]
            return resolved
        if non_null:
            return _resolve_schema(non_null[0])

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIP_KEYS or key == "anyOf":
            continue
        if isinstance(value, dict):
            out[key] = _resolve_schema(value)
        elif isinstance(value, list):
            out[key] = [_resolve_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out
