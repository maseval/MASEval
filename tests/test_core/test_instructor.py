"""Tests for maseval.core.instructor module.

Tests the schema flattening logic (``flatten_model_schema``) and the
instructor client factory (``create_instructor_client``).

Schema tests are pure unit tests — they exercise Pydantic model → JSON
schema conversion without any mocking or network access.
"""

import pytest
from typing import Optional, List
from pydantic import BaseModel, Field

from maseval.core.instructor import flatten_model_schema, create_instructor_client

# ── Test models ──────────────────────────────────────────────────────────────


class SimpleModel(BaseModel):
    """Model with all required fields."""

    name: str = Field(description="The name")
    age: int = Field(description="The age")
    score: float = Field(description="The score")


class OptionalFieldsModel(BaseModel):
    """Model with Optional fields that produce anyOf in Pydantic v2 schemas."""

    required_field: str = Field(description="Always required")
    optional_field: Optional[str] = Field(default=None, description="May be absent")
    optional_int: Optional[int] = Field(default=None, description="Optional number")


class Address(BaseModel):
    street: str
    city: str


class NestedModel(BaseModel):
    """Model with a nested sub-model that produces $ref/$defs."""

    name: str = Field(description="Person name")
    address: Address = Field(description="Home address")


class ListFieldModel(BaseModel):
    """Model with list fields."""

    tags: List[str] = Field(description="A list of tags")
    scores: List[float] = Field(description="A list of scores")


# ── flatten_model_schema tests ───────────────────────────────────────────────

STRIPPED_KEYS = {"$defs", "additionalProperties", "title", "default"}


def _assert_no_stripped_keys(schema: dict) -> None:
    """Recursively verify no stripped keys exist anywhere in the schema."""
    for key in STRIPPED_KEYS:
        assert key not in schema, f"Found stripped key {key!r} in schema"
    for value in schema.values():
        if isinstance(value, dict):
            _assert_no_stripped_keys(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _assert_no_stripped_keys(item)


@pytest.mark.core
class TestFlattenModelSchema:
    """Tests for flatten_model_schema()."""

    def test_simple_model_produces_flat_schema(self):
        """Simple model with all required fields produces correct types and descriptions."""
        schema = flatten_model_schema(SimpleModel)

        assert schema["type"] == "object"
        props = schema["properties"]

        assert props["name"]["type"] == "string"
        assert props["name"]["description"] == "The name"
        assert props["age"]["type"] == "integer"
        assert props["score"]["type"] == "number"

        assert "name" in schema["required"]
        assert "age" in schema["required"]
        assert "score" in schema["required"]

    def test_optional_fields_resolved_to_nullable(self):
        """Optional[X] fields have anyOf removed and nullable added."""
        schema = flatten_model_schema(OptionalFieldsModel)
        props = schema["properties"]

        # Required field is straightforward
        assert props["required_field"]["type"] == "string"
        assert "nullable" not in props["required_field"]

        # Optional fields should have base type + nullable
        assert props["optional_field"]["type"] == "string"
        assert props["optional_field"]["nullable"] is True

        assert props["optional_int"]["type"] == "integer"
        assert props["optional_int"]["nullable"] is True

        # No anyOf should remain anywhere
        assert "anyOf" not in str(schema)

    def test_optional_field_preserves_description(self):
        """Description on Optional fields survives the anyOf resolution."""
        schema = flatten_model_schema(OptionalFieldsModel)
        props = schema["properties"]

        assert props["optional_field"]["description"] == "May be absent"
        assert props["optional_int"]["description"] == "Optional number"

    def test_nested_model_defs_stripped(self):
        """Nested model $defs are stripped from the schema.

        Note: $ref references are preserved but $defs definitions are removed.
        In practice this doesn't arise — Tau2 uses create_model() with simple
        types (str, int, float) which don't produce $ref.
        """
        schema = flatten_model_schema(NestedModel)

        # $defs should be stripped
        assert "$defs" not in schema

        # Description is preserved on the nested field
        assert schema["properties"]["address"]["description"] == "Home address"

    def test_stripped_keys_removed(self):
        """$defs, additionalProperties, title, default are stripped recursively."""
        schema = flatten_model_schema(NestedModel)
        _assert_no_stripped_keys(schema)

    def test_list_fields_preserved(self):
        """List[X] fields produce correct array schemas."""
        schema = flatten_model_schema(ListFieldModel)
        props = schema["properties"]

        assert props["tags"]["type"] == "array"
        assert props["scores"]["type"] == "array"


# ── create_instructor_client tests ───────────────────────────────────────────


@pytest.mark.core
class TestCreateInstructorClient:
    """Tests for create_instructor_client()."""

    def test_unknown_provider_raises_value_error(self):
        """Unsupported provider raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_instructor_client(object(), provider="not-a-real-provider")

    def test_openai_provider_returns_patched_client(self):
        """OpenAI client is wrapped and exposes chat.completions.create."""
        from openai import OpenAI

        client = OpenAI(api_key="test-key-not-real")
        patched = create_instructor_client(client, provider="openai")

        assert hasattr(patched, "chat")
        assert hasattr(patched.chat, "completions")
        assert callable(patched.chat.completions.create)

    def test_litellm_provider_returns_patched_client(self):
        """LiteLLM completion function is wrapped and exposes chat.completions.create."""
        litellm = pytest.importorskip("litellm")

        patched = create_instructor_client(litellm.completion, provider="litellm")

        assert hasattr(patched, "chat")
        assert hasattr(patched.chat, "completions")
        assert callable(patched.chat.completions.create)
