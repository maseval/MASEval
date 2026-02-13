"""Tests for Gaia2GenericTool and wrap_are_tools.

Tests the framework-agnostic tool wrapper that provides MASEval tracing
for ARE tools following the MACSGenericTool pattern.
"""

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Test Gaia2GenericTool
# =============================================================================


@pytest.mark.benchmark
class TestGaia2GenericToolInit:
    """Tests for Gaia2GenericTool initialization."""

    def test_extracts_name_from_are_tool(self, mock_are_tool):
        """Test wrapper extracts _public_name from ARE tool (tool_utils.py:546)."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        assert wrapper.name == mock_are_tool._public_name

    def test_extracts_description_from_are_tool(self, mock_are_tool):
        """Test wrapper builds full description with app prefix and return type suffix."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        # ARE tool_utils.py:550,581-582: "Acts on app {app_name}: {desc} Returns: {type}"
        assert wrapper.description.startswith(f"Acts on app {mock_are_tool.app_name}: ")
        assert mock_are_tool._public_description in wrapper.description

    def test_extracts_inputs_flat_dict(self, mock_are_tool):
        """Test wrapper extracts inputs as flat dict matching ARE's AppToolAdapter format."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        # ARE tool_utils.py:572-578 — flat dict keyed by arg name
        assert "param1" in wrapper.inputs
        assert wrapper.inputs["param1"]["type"] == "string"
        assert "param2" in wrapper.inputs
        assert wrapper.inputs["param2"]["type"] == "integer"
        # param2 has a default
        assert wrapper.inputs["param2"]["default"] == 0

    def test_initializes_empty_history(self, mock_are_tool):
        """Test wrapper initializes with empty history."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        assert len(wrapper.history.to_list()) == 0


@pytest.mark.benchmark
class TestGaia2GenericToolCall:
    """Tests for Gaia2GenericTool.__call__()."""

    def test_delegates_to_are_tool(self, mock_are_tool):
        """Test wrapper delegates call to ARE tool."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        result = wrapper(arg1="test_value")

        assert result == "Tool executed successfully"
        assert len(mock_are_tool.calls) == 1
        assert mock_are_tool.calls[0] == {"arg1": "test_value"}

    def test_records_invocation_in_history(self, mock_are_tool):
        """Test wrapper records invocation in history."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 100.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        wrapper(arg1="value")

        history = wrapper.history.to_list()
        assert len(history) == 1
        assert history[0]["inputs"] == {"arg1": "value"}
        assert history[0]["outputs"] == "Tool executed successfully"
        assert history[0]["status"] == "success"

    def test_records_simulation_time(self, mock_are_tool):
        """Test wrapper records simulation time before and after."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.side_effect = [10.0, 15.0]  # before, after

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        wrapper(arg1="value")

        history = wrapper.history.to_list()
        meta = history[0]["meta"]
        assert meta["simulation_time_before"] == 10.0
        assert meta["simulation_time_after"] == 15.0
        assert meta["simulation_time_elapsed"] == 5.0

    def test_handles_tool_error(self, mock_are_tool):
        """Test wrapper handles tool execution error."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_are_tool._return_value = lambda **kw: (_ for _ in ()).throw(ValueError("Tool error"))

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        with pytest.raises(ValueError):
            wrapper(arg1="value")

        # Error should still be recorded
        history = wrapper.history.to_list()
        assert len(history) == 1
        assert history[0]["status"] == "error"
        assert "Tool error" in str(history[0]["outputs"])

    def test_handles_missing_simulation_time(self, mock_are_tool):
        """Test wrapper handles missing simulation time gracefully."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.side_effect = Exception("Time not available")

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        # Should not raise
        result = wrapper(arg1="value")

        assert result == "Tool executed successfully"
        history = wrapper.history.to_list()
        assert history[0]["meta"]["simulation_time_before"] is None


@pytest.mark.benchmark
class TestGaia2GenericToolTracing:
    """Tests for Gaia2GenericTool tracing methods."""

    def test_gather_traces_includes_invocations(self, mock_are_tool):
        """Test gather_traces includes invocation history."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        wrapper(arg1="first")
        wrapper(arg1="second")

        traces = wrapper.gather_traces()

        assert traces["name"] == mock_are_tool.name
        assert traces["total_invocations"] == 2
        assert len(traces["invocations"]) == 2

    def test_gather_config_includes_schema(self, mock_are_tool):
        """Test gather_config includes tool configuration."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        config = wrapper.gather_config()

        assert config["name"] == mock_are_tool._public_name
        assert "Acts on app TestTool:" in config["description"]
        assert "input_schema" in config


@pytest.mark.benchmark
class TestGaia2GenericToolSchemaExtraction:
    """Tests for schema extraction from ARE's args attribute."""

    def test_extracts_from_are_args_attribute(self, mock_are_tool):
        """Test extraction from ARE's 'args' attribute produces flat inputs dict."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        # inputs is flat dict (ARE tool_utils.py:572-578)
        assert "param1" in wrapper.inputs
        assert wrapper.inputs["param1"]["type"] == "string"
        assert wrapper.inputs["param1"]["description"] == "First parameter"

        # input_schema keeps JSON schema format for tracing
        assert "properties" in wrapper.input_schema
        assert "param1" in wrapper.input_schema["properties"]

    def test_extracts_description_with_app_prefix(self, mock_are_tool):
        """Test description includes app name prefix and return type suffix."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_are_tool, mock_env)

        # ARE tool_utils.py:550 — prefix with app name
        assert wrapper.description.startswith("Acts on app TestTool: ")
        # ARE tool_utils.py:547-548 — includes public description
        assert "A test tool that does something" in wrapper.description

    def test_handles_empty_args(self):
        """Test handling when args list is empty."""
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        mock_tool = MagicMock(spec=["name", "_public_name", "_public_description", "app_name", "return_type", "args", "__call__"])
        mock_tool.name = "test_tool"
        mock_tool._public_name = "test_tool"
        mock_tool._public_description = "Test"
        mock_tool.app_name = "TestApp"
        mock_tool.return_type = None
        mock_tool.args = []

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = Gaia2GenericTool(mock_tool, mock_env)

        assert wrapper.inputs == {}


# =============================================================================
# Test wrap_are_tools
# =============================================================================


@pytest.mark.benchmark
class TestWrapAreTools:
    """Tests for wrap_are_tools helper function."""

    def test_wraps_multiple_tools(self, mock_are_tools):
        """Test wrapping multiple ARE tools."""
        from maseval.benchmark.gaia2.tool_wrapper import wrap_are_tools, Gaia2GenericTool

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapped = wrap_are_tools(mock_are_tools, mock_env)

        assert len(wrapped) == len(mock_are_tools)
        for tool in mock_are_tools:
            assert tool.name in wrapped
            assert isinstance(wrapped[tool.name], Gaia2GenericTool)

    def test_returns_dict_keyed_by_name(self, mock_are_tools):
        """Test wrapped tools are keyed by name."""
        from maseval.benchmark.gaia2.tool_wrapper import wrap_are_tools

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapped = wrap_are_tools(mock_are_tools, mock_env)

        assert "Calendar__get_events" in wrapped
        assert "Email__send" in wrapped
        assert "SystemApp__get_current_time" in wrapped

    def test_handles_empty_list(self):
        """Test handling empty tools list."""
        from maseval.benchmark.gaia2.tool_wrapper import wrap_are_tools

        mock_env = MagicMock()

        wrapped = wrap_are_tools([], mock_env)

        assert wrapped == {}

    def test_wrapped_tools_are_callable(self, mock_are_tools):
        """Test wrapped tools can be called."""
        from maseval.benchmark.gaia2.tool_wrapper import wrap_are_tools

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapped = wrap_are_tools(mock_are_tools, mock_env)

        # Call a wrapped tool
        result = wrapped["Calendar__get_events"]()

        assert result == []  # From mock_are_tools fixture
