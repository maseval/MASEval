"""Unit tests for Tau2 utils module."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

from maseval.benchmark.tau2.utils import (
    get_dict_hash,
    get_pydantic_hash,
    update_pydantic_model_with_dict,
    load_file,
    dump_file,
    compare_tool_calls,
)


# =============================================================================
# Hash Function Tests
# =============================================================================


@pytest.mark.benchmark
class TestGetDictHash:
    """Tests for get_dict_hash function."""

    def test_consistent_hash(self):
        """Same dict produces same hash."""
        data = {"a": 1, "b": 2, "c": "test"}

        hash1 = get_dict_hash(data)
        hash2 = get_dict_hash(data)

        assert hash1 == hash2

    def test_key_order_independent(self):
        """Hash is independent of key order."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}

        assert get_dict_hash(data1) == get_dict_hash(data2)

    def test_different_data_different_hash(self):
        """Different data produces different hash."""
        data1 = {"a": 1}
        data2 = {"a": 2}

        assert get_dict_hash(data1) != get_dict_hash(data2)

    def test_hash_is_sha256(self):
        """Hash is proper SHA-256 format."""
        hash_value = get_dict_hash({"test": "data"})

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_nested_dict(self):
        """Handles nested dictionaries."""
        data = {"outer": {"inner": {"deep": 42}}}

        hash_value = get_dict_hash(data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_empty_dict(self):
        """Handles empty dictionary."""
        hash_value = get_dict_hash({})

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


@pytest.mark.benchmark
class TestGetPydanticHash:
    """Tests for get_pydantic_hash function."""

    def test_consistent_hash(self):
        """Same model produces same hash."""

        class TestModel(BaseModel):
            x: int
            y: str

        model = TestModel(x=1, y="test")

        hash1 = get_pydantic_hash(model)
        hash2 = get_pydantic_hash(model)

        assert hash1 == hash2

    def test_different_values_different_hash(self):
        """Different values produce different hash."""

        class TestModel(BaseModel):
            x: int

        model1 = TestModel(x=1)
        model2 = TestModel(x=2)

        assert get_pydantic_hash(model1) != get_pydantic_hash(model2)

    def test_hash_is_sha256(self):
        """Hash is proper SHA-256 format."""

        class TestModel(BaseModel):
            x: int

        hash_value = get_pydantic_hash(TestModel(x=1))

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


# =============================================================================
# Pydantic Model Update Tests
# =============================================================================


@pytest.mark.benchmark
class TestUpdatePydanticModel:
    """Tests for update_pydantic_model_with_dict function."""

    def test_update_simple_field(self):
        """Updates a simple field."""

        class Config(BaseModel):
            x: int
            y: str

        config = Config(x=1, y="old")
        updated = update_pydantic_model_with_dict(config, {"y": "new"})

        assert updated.x == 1  # Unchanged
        assert updated.y == "new"  # Updated

    def test_update_multiple_fields(self):
        """Updates multiple fields at once."""

        class Config(BaseModel):
            a: int
            b: int
            c: int

        config = Config(a=1, b=2, c=3)
        updated = update_pydantic_model_with_dict(config, {"a": 10, "c": 30})

        assert updated.a == 10
        assert updated.b == 2
        assert updated.c == 30

    def test_empty_update(self):
        """Empty update returns equivalent model."""

        class Config(BaseModel):
            x: int

        config = Config(x=1)
        updated = update_pydantic_model_with_dict(config, {})

        assert updated.x == config.x

    def test_nested_update(self):
        """Updates nested model fields."""

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner
            name: str

        outer = Outer(inner=Inner(value=1), name="test")
        updated = update_pydantic_model_with_dict(outer, {"inner": {"value": 42}})

        assert updated.inner.value == 42
        assert updated.name == "test"


# =============================================================================
# File Loading Tests
# =============================================================================


@pytest.mark.benchmark
class TestLoadFile:
    """Tests for load_file function."""

    def test_load_json(self):
        """Loads JSON file correctly."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text('{"key": "value"}')

            result = load_file(path)

            assert result == {"key": "value"}

    def test_load_nonexistent_file(self):
        """Raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent/path/file.json")

    def test_load_unsupported_format(self):
        """Raises error for unsupported format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"
            path.write_text("content")

            with pytest.raises(ValueError, match="Unsupported"):
                load_file(path)


@pytest.mark.benchmark
class TestDumpFile:
    """Tests for dump_file function."""

    def test_dump_json(self):
        """Dumps JSON file correctly."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            dump_file(path, data)

            assert path.exists()
            loaded = load_file(path)
            assert loaded == data

    def test_dump_unsupported_format(self):
        """Raises error for unsupported format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.xyz"

            with pytest.raises(ValueError, match="Unsupported"):
                dump_file(path, {"data": "value"})


# =============================================================================
# Tool Call Comparison Tests
# =============================================================================


@pytest.mark.benchmark
class TestCompareToolCalls:
    """Tests for compare_tool_calls function."""

    def test_exact_match(self):
        """Identical tool calls match."""
        result = compare_tool_calls(
            expected_name="get_user",
            expected_args={"user_id": "123"},
            actual_name="get_user",
            actual_args={"user_id": "123"},
        )

        assert result is True

    def test_different_name(self):
        """Different tool names don't match."""
        result = compare_tool_calls(
            expected_name="get_user",
            expected_args={"user_id": "123"},
            actual_name="get_order",
            actual_args={"user_id": "123"},
        )

        assert result is False

    def test_different_args(self):
        """Different arguments don't match."""
        result = compare_tool_calls(
            expected_name="get_user",
            expected_args={"user_id": "123"},
            actual_name="get_user",
            actual_args={"user_id": "456"},
        )

        assert result is False

    def test_subset_args_with_compare_args(self):
        """Subset of args matches when using compare_args."""
        result = compare_tool_calls(
            expected_name="update_user",
            expected_args={"user_id": "123", "name": "John", "email": "john@test.com"},
            actual_name="update_user",
            actual_args={"user_id": "123", "name": "John"},
            compare_args=["user_id", "name"],
        )

        assert result is True

    def test_empty_compare_args(self):
        """Empty compare_args matches any arguments."""
        result = compare_tool_calls(
            expected_name="get_user",
            expected_args={"user_id": "123"},
            actual_name="get_user",
            actual_args={"different": "args"},
            compare_args=[],
        )

        assert result is True

    def test_extra_actual_args(self):
        """Extra args in actual are compared by default."""
        result = compare_tool_calls(
            expected_name="update_user",
            expected_args={"user_id": "123"},
            actual_name="update_user",
            actual_args={"user_id": "123", "extra": "arg"},
        )

        # By default compares all actual args against expected
        # Extra arg in actual but not in expected should be fine
        # as long as expected is subset of actual
        assert result is False  # extra is in actual but not expected

    def test_missing_expected_arg(self):
        """Missing expected arg in actual fails."""
        result = compare_tool_calls(
            expected_name="update_user",
            expected_args={"user_id": "123", "required": "value"},
            actual_name="update_user",
            actual_args={"user_id": "123"},
            compare_args=["user_id", "required"],
        )

        # If compare_args specified, only those are compared
        assert result is False  # required is missing from actual

    def test_nested_args(self):
        """Handles nested argument values."""
        result = compare_tool_calls(
            expected_name="create_order",
            expected_args={"items": [{"id": "1"}, {"id": "2"}]},
            actual_name="create_order",
            actual_args={"items": [{"id": "1"}, {"id": "2"}]},
        )

        assert result is True

    def test_nested_args_different(self):
        """Detects differences in nested arguments."""
        result = compare_tool_calls(
            expected_name="create_order",
            expected_args={"items": [{"id": "1"}]},
            actual_name="create_order",
            actual_args={"items": [{"id": "2"}]},
        )

        assert result is False
