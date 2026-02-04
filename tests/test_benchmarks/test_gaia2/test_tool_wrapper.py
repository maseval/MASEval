"""Tests for AREToolWrapper and wrap_are_tools.

Tests the tool wrapper that provides MASEval tracing for ARE tools.
"""

import pytest
from unittest.mock import MagicMock


# =============================================================================
# Test AREToolWrapper
# =============================================================================


@pytest.mark.benchmark
class TestAREToolWrapperInit:
    """Tests for AREToolWrapper initialization."""

    def test_extracts_name_from_are_tool(self, mock_are_tool):
        """Test wrapper extracts name from ARE tool."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        assert wrapper.name == mock_are_tool.name

    def test_extracts_description_from_are_tool(self, mock_are_tool):
        """Test wrapper extracts description from ARE tool."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        assert wrapper.description == mock_are_tool.description

    def test_extracts_inputs_schema(self, mock_are_tool):
        """Test wrapper extracts inputs schema."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        assert "properties" in wrapper.inputs

    def test_initializes_empty_history(self, mock_are_tool):
        """Test wrapper initializes with empty history."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        assert len(wrapper.history.to_list()) == 0


@pytest.mark.benchmark
class TestAREToolWrapperCall:
    """Tests for AREToolWrapper.__call__()."""

    def test_delegates_to_are_tool(self, mock_are_tool):
        """Test wrapper delegates call to ARE tool."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        result = wrapper(arg1="test_value")

        assert result == "Tool executed successfully"
        assert len(mock_are_tool.calls) == 1
        assert mock_are_tool.calls[0] == {"arg1": "test_value"}

    def test_records_invocation_in_history(self, mock_are_tool):
        """Test wrapper records invocation in history."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 100.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        wrapper(arg1="value")

        history = wrapper.history.to_list()
        assert len(history) == 1
        assert history[0]["inputs"] == {"arg1": "value"}
        assert history[0]["outputs"] == "Tool executed successfully"
        assert history[0]["status"] == "success"

    def test_records_simulation_time(self, mock_are_tool):
        """Test wrapper records simulation time before and after."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.side_effect = [10.0, 15.0]  # before, after

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        wrapper(arg1="value")

        history = wrapper.history.to_list()
        meta = history[0]["meta"]
        assert meta["simulation_time_before"] == 10.0
        assert meta["simulation_time_after"] == 15.0
        assert meta["simulation_time_elapsed"] == 5.0

    def test_handles_tool_error(self, mock_are_tool):
        """Test wrapper handles tool execution error."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_are_tool._return_value = lambda **kw: (_ for _ in ()).throw(ValueError("Tool error"))

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        with pytest.raises(ValueError):
            wrapper(arg1="value")

        # Error should still be recorded
        history = wrapper.history.to_list()
        assert len(history) == 1
        assert history[0]["status"] == "error"
        assert "Tool error" in str(history[0]["outputs"])

    def test_handles_missing_simulation_time(self, mock_are_tool):
        """Test wrapper handles missing simulation time gracefully."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.side_effect = Exception("Time not available")

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        # Should not raise
        result = wrapper(arg1="value")

        assert result == "Tool executed successfully"
        history = wrapper.history.to_list()
        assert history[0]["meta"]["simulation_time_before"] is None


@pytest.mark.benchmark
class TestAREToolWrapperTracing:
    """Tests for AREToolWrapper tracing methods."""

    def test_gather_traces_includes_invocations(self, mock_are_tool):
        """Test gather_traces includes invocation history."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        wrapper(arg1="first")
        wrapper(arg1="second")

        traces = wrapper.gather_traces()

        assert traces["name"] == mock_are_tool.name
        assert traces["total_invocations"] == 2
        assert len(traces["invocations"]) == 2

    def test_gather_config_includes_schema(self, mock_are_tool):
        """Test gather_config includes tool configuration."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_are_tool, mock_env)

        config = wrapper.gather_config()

        assert config["name"] == mock_are_tool.name
        assert config["description"] == mock_are_tool.description
        assert "inputs_schema" in config


@pytest.mark.benchmark
class TestAREToolWrapperSchemaExtraction:
    """Tests for schema extraction from different ARE tool formats."""

    def test_extracts_from_inputs_attribute(self):
        """Test extraction from 'inputs' attribute."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.inputs = {"key": "value"}

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_tool, mock_env)

        assert wrapper.inputs == {"key": "value"}

    def test_extracts_from_parameters_attribute(self):
        """Test extraction from 'parameters' attribute when 'inputs' is absent."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        # Use spec to ensure tool only has specified attributes (no 'inputs')
        mock_tool = MagicMock(spec=["name", "description", "parameters", "__call__"])
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"
        mock_tool.parameters = {"param": "schema"}

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_tool, mock_env)

        assert wrapper.inputs == {"param": "schema"}

    def test_handles_missing_schema(self):
        """Test handling when no schema is available."""
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        mock_tool = MagicMock(spec=[])  # No attributes
        mock_tool.name = "test_tool"
        mock_tool.description = "Test"

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapper = AREToolWrapper(mock_tool, mock_env)

        assert wrapper.inputs == {}


# =============================================================================
# Test wrap_are_tools
# =============================================================================


@pytest.mark.benchmark
class TestWrapAreTools:
    """Tests for wrap_are_tools helper function."""

    def test_wraps_multiple_tools(self, mock_are_tools):
        """Test wrapping multiple ARE tools."""
        from maseval.benchmark.gaia2.tool_wrapper import wrap_are_tools, AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.return_value = 0.0

        wrapped = wrap_are_tools(mock_are_tools, mock_env)

        assert len(wrapped) == len(mock_are_tools)
        for tool in mock_are_tools:
            assert tool.name in wrapped
            assert isinstance(wrapped[tool.name], AREToolWrapper)

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
