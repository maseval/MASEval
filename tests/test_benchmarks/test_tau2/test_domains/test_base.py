"""Unit tests for Tau2 domains/base.py — DB and ToolKitBase."""

import pytest

from maseval.benchmark.tau2.domains.base import DB, ToolKitBase, ToolType, is_tool


# =============================================================================
# DB Base Class Tests
# =============================================================================


@pytest.mark.benchmark
class TestDBBase:
    """Tests for DB base class methods."""

    def test_get_statistics_returns_empty(self):
        """Base DB.get_statistics() returns empty dict."""

        class MinimalDB(DB):
            x: int = 1

        db = MinimalDB()
        assert db.get_statistics() == {}

    def test_copy_deep_returns_equal(self):
        """copy_deep() returns a model with same hash."""

        class SimpleDB(DB):
            value: int
            name: str

        db = SimpleDB(value=42, name="test")
        copy = db.copy_deep()
        assert copy.get_hash() == db.get_hash()
        assert copy.value == db.value
        assert copy.name == db.name

    def test_copy_deep_is_independent(self):
        """copy_deep() returns an independent copy."""

        class MutableDB(DB):
            items: list

        db = MutableDB(items=[1, 2, 3])
        copy = db.copy_deep()
        copy.items.append(4)
        assert len(db.items) == 3
        assert len(copy.items) == 4

    def test_get_hash_consistent(self):
        """get_hash() is consistent for same state."""

        class SimpleDB(DB):
            x: int

        db = SimpleDB(x=1)
        assert db.get_hash() == db.get_hash()

    def test_get_hash_differs_for_different_state(self):
        """get_hash() differs for different data."""

        class SimpleDB(DB):
            x: int

        db1 = SimpleDB(x=1)
        db2 = SimpleDB(x=2)
        assert db1.get_hash() != db2.get_hash()


@pytest.mark.benchmark
@pytest.mark.live
class TestDBWithRealData:
    """Tests for DB methods with real domain databases."""

    def test_retail_copy_deep(self, retail_db):
        """Retail DB copy_deep produces same hash."""
        copy = retail_db.copy_deep()
        assert copy.get_hash() == retail_db.get_hash()

    def test_retail_copy_deep_independent(self, retail_db):
        """Retail DB copy_deep is independent of original."""
        copy = retail_db.copy_deep()
        first_user_id = next(iter(copy.users))
        copy.users[first_user_id].email = "MODIFIED@test.com"
        assert retail_db.users[first_user_id].email != "MODIFIED@test.com"

    def test_airline_copy_deep(self, airline_db):
        """Airline DB copy_deep produces same hash."""
        copy = airline_db.copy_deep()
        assert copy.get_hash() == airline_db.get_hash()

    def test_telecom_copy_deep(self, telecom_db):
        """Telecom DB copy_deep produces same hash."""
        copy = telecom_db.copy_deep()
        assert copy.get_hash() == telecom_db.get_hash()


# =============================================================================
# ToolKitBase Tests
# =============================================================================


class SampleDB(DB):
    value: int = 0


class SampleToolKit(ToolKitBase[SampleDB]):
    """Sample toolkit for testing."""

    @is_tool(ToolType.READ)
    def get_value(self) -> int:
        """Get the current value."""
        return self.db.value  # type: ignore[union-attr]

    @is_tool(ToolType.WRITE)
    def set_value(self, new_value: int) -> str:
        """Set the value.

        Args:
            new_value: The new value to set
        """
        self.db.value = new_value  # type: ignore[union-attr]
        return "done"

    @is_tool(ToolType.GENERIC)
    def calculate(self, a: float, b: float) -> float:
        """Calculate the sum of two numbers."""
        return a + b

    @is_tool(ToolType.READ)
    def all_types(self, name: str, count: int, price: float, active: bool, items: list, data: dict) -> str:
        """Tool with all supported param types."""
        return "ok"


class ChildToolKit(SampleToolKit):
    """Child toolkit inheriting from SampleToolKit."""

    @is_tool(ToolType.READ)
    def child_only(self) -> str:
        """A tool only in the child."""
        return "child"


@pytest.mark.benchmark
class TestToolKitBase:
    """Tests for ToolKitBase base class methods."""

    def test_get_statistics(self):
        """get_statistics returns correct counts by type."""
        tk = SampleToolKit(SampleDB())
        stats = tk.get_statistics()
        assert stats["num_tools"] == 4
        assert stats["num_read_tools"] == 2
        assert stats["num_write_tools"] == 1
        assert stats["num_generic_tools"] == 1
        assert stats["num_think_tools"] == 0

    def test_get_tool_descriptions(self):
        """get_tool_descriptions returns docstrings for all tools."""
        tk = SampleToolKit(SampleDB())
        descriptions = tk.get_tool_descriptions()
        assert len(descriptions) == 4
        assert "Get the current value" in descriptions["get_value"]
        assert "Set the value" in descriptions["set_value"]
        assert "Calculate the sum" in descriptions["calculate"]
        assert "Tool with all supported" in descriptions["all_types"]

    def test_get_tool_metadata(self):
        """get_tool_metadata returns description, inputs, tool_type."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("get_value")
        assert "Get the current value" in meta["description"]
        assert meta["tool_type"] == ToolType.READ
        assert isinstance(meta["inputs"], dict)

    def test_get_tool_metadata_with_params(self):
        """get_tool_metadata includes parameter types."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("calculate")
        assert "a" in meta["inputs"]
        assert "b" in meta["inputs"]
        assert meta["inputs"]["a"]["type"] == "number"
        assert meta["inputs"]["b"]["type"] == "number"

    def test_get_tool_metadata_int_param(self):
        """get_tool_metadata correctly types int params."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("set_value")
        assert meta["inputs"]["new_value"]["type"] == "integer"

    def test_get_tool_metadata_not_found(self):
        """get_tool_metadata raises ValueError for unknown tool."""
        tk = SampleToolKit(SampleDB())
        with pytest.raises(ValueError, match="not found"):
            tk.get_tool_metadata("nonexistent")

    def test_tools_property(self):
        """tools property returns dict of bound methods."""
        tk = SampleToolKit(SampleDB(value=42))
        tools = tk.tools
        assert "get_value" in tools
        assert callable(tools["get_value"])
        assert tools["get_value"]() == 42

    def test_use_tool(self):
        """use_tool invokes tool by name."""
        tk = SampleToolKit(SampleDB(value=10))
        result = tk.use_tool("get_value")
        assert result == 10

    def test_use_tool_not_found(self):
        """use_tool raises ValueError for unknown tool."""
        tk = SampleToolKit(SampleDB())
        with pytest.raises(ValueError, match="not found"):
            tk.use_tool("missing")

    def test_has_tool(self):
        """has_tool returns True for existing, False for missing."""
        tk = SampleToolKit(SampleDB())
        assert tk.has_tool("get_value") is True
        assert tk.has_tool("nonexistent") is False

    def test_tool_type(self):
        """tool_type returns correct ToolType."""
        tk = SampleToolKit(SampleDB())
        assert tk.tool_type("get_value") == ToolType.READ
        assert tk.tool_type("set_value") == ToolType.WRITE
        assert tk.tool_type("calculate") == ToolType.GENERIC

    def test_update_db(self):
        """update_db applies update to database."""
        tk = SampleToolKit(SampleDB(value=1))
        tk.update_db({"value": 99})
        assert tk.db.value == 99  # type: ignore[union-attr]

    def test_update_db_none(self):
        """update_db with None is a no-op."""
        tk = SampleToolKit(SampleDB(value=1))
        tk.update_db(None)
        assert tk.db.value == 1  # type: ignore[union-attr]

    def test_update_db_no_db_raises(self):
        """update_db without db raises ValueError."""
        tk = SampleToolKit(None)
        with pytest.raises(ValueError, match="not been initialized"):
            tk.update_db({"value": 1})

    def test_get_db_hash(self):
        """get_db_hash returns hash string."""
        tk = SampleToolKit(SampleDB(value=42))
        h = tk.get_db_hash()
        assert isinstance(h, str)
        assert len(h) == 64

    def test_get_db_hash_no_db_raises(self):
        """get_db_hash without db raises ValueError."""
        tk = SampleToolKit(None)
        with pytest.raises(ValueError, match="not been initialized"):
            tk.get_db_hash()

    def test_all_types_metadata_bool(self):
        """get_tool_metadata maps bool annotation to 'boolean'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["active"]["type"] == "boolean"

    def test_all_types_metadata_list(self):
        """get_tool_metadata maps list annotation to 'array'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["items"]["type"] == "array"

    def test_all_types_metadata_dict(self):
        """get_tool_metadata maps dict annotation to 'object'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["data"]["type"] == "object"

    def test_all_types_metadata_str(self):
        """get_tool_metadata maps str annotation to 'string'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["name"]["type"] == "string"

    def test_all_types_metadata_int(self):
        """get_tool_metadata maps int annotation to 'integer'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["count"]["type"] == "integer"

    def test_all_types_metadata_float(self):
        """get_tool_metadata maps float annotation to 'number'."""
        tk = SampleToolKit(SampleDB())
        meta = tk.get_tool_metadata("all_types")
        assert meta["inputs"]["price"]["type"] == "number"


@pytest.mark.benchmark
class TestChildToolKit:
    """Tests for ChildToolKit — verifies parent tool inheritance via MRO."""

    def test_child_has_own_tools(self):
        """Child toolkit has its own tools."""
        tk = ChildToolKit(SampleDB())
        assert tk.has_tool("child_only")
        assert tk.use_tool("child_only") == "child"

    def test_child_inherits_parent_tools(self):
        """Child toolkit inherits parent tools via MRO."""
        tk = ChildToolKit(SampleDB(value=42))
        assert tk.has_tool("get_value")
        assert tk.use_tool("get_value") == 42

    def test_child_statistics_include_all(self):
        """Child toolkit statistics include both child and parent tools."""
        tk = ChildToolKit(SampleDB())
        stats = tk.get_statistics()
        # Should include parent's 4 tools + child's 1 tool = 5
        assert stats["num_tools"] == 5
        # Parent has 2 READ (get_value, all_types) + child has 1 READ (child_only) = 3
        assert stats["num_read_tools"] == 3

    def test_child_tool_descriptions_include_all(self):
        """Child toolkit descriptions include both child and parent tools."""
        tk = ChildToolKit(SampleDB())
        descs = tk.get_tool_descriptions()
        assert "child_only" in descs
        assert "get_value" in descs
        assert "set_value" in descs
        assert len(descs) == 5


@pytest.mark.benchmark
@pytest.mark.live
class TestToolKitWithRealData:
    """Tests for ToolKitBase methods with real domain toolkits."""

    def test_retail_get_statistics(self, retail_toolkit):
        """Retail toolkit has expected stats."""
        stats = retail_toolkit.get_statistics()
        assert stats["num_tools"] == 15
        assert stats["num_read_tools"] == 6
        assert stats["num_write_tools"] == 7

    def test_retail_get_tool_descriptions(self, retail_toolkit):
        """Retail toolkit descriptions are non-empty."""
        descs = retail_toolkit.get_tool_descriptions()
        assert len(descs) == 15
        for name, desc in descs.items():
            assert isinstance(desc, str)
            assert len(desc) > 0, f"Tool {name} has empty description"

    def test_retail_get_tool_metadata(self, retail_toolkit):
        """Retail toolkit metadata includes inputs and type."""
        meta = retail_toolkit.get_tool_metadata("get_user_details")
        assert "description" in meta
        assert "inputs" in meta
        assert "tool_type" in meta
        assert "user_id" in meta["inputs"]

    def test_airline_get_statistics(self, airline_toolkit):
        """Airline toolkit has expected stats."""
        stats = airline_toolkit.get_statistics()
        assert stats["num_tools"] == 14

    def test_telecom_get_statistics(self, telecom_toolkit):
        """Telecom toolkit has expected stats."""
        stats = telecom_toolkit.get_statistics()
        assert stats["num_tools"] == 13
