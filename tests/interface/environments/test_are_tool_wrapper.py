"""Tests for AREToolWrapper."""

from unittest.mock import MagicMock, patch
import pytest

from maseval.interface.environments.are_tool_wrapper import AREToolWrapper


@pytest.fixture(autouse=True)
def mock_app_tool_adapter():
    """Mock AppToolAdapter so AREToolWrapper can initialize without real ARE validation."""

    def make_adapter(are_tool):
        adapter = MagicMock()
        adapter.name = are_tool.name
        adapter.description = are_tool.description
        adapter.inputs = are_tool.inputs
        adapter.output_type = are_tool.output_type
        adapter.actual_return_type = None
        return adapter

    with patch("maseval.interface.environments.are_tool_wrapper.AppToolAdapter", side_effect=make_adapter):
        yield


class TestAREToolWrapper:
    """Tests for AREToolWrapper."""

    def _make_mock_are_tool(
        self,
        name="Calendar__create_event",
        description="Create a calendar event",
        inputs=None,
        output_type="string",
        return_value="Event created",
    ):
        """Create a mock ARE tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputs = inputs or {"title": {"type": "string", "description": "Event title"}}
        tool.output_type = output_type
        tool.return_value = return_value
        tool.__call__ = MagicMock(return_value=return_value)
        return tool

    def test_metadata_from_are_tool(self):
        """Wrapper exposes ARE tool metadata."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        assert wrapper.name == "Calendar__create_event"
        assert wrapper.description == "Create a calendar event"
        assert wrapper.inputs == {"title": {"type": "string", "description": "Event title"}}
        assert wrapper.output_type == "string"

    def test_call_delegates_to_are_tool(self):
        """Calling wrapper delegates to underlying ARE tool."""
        are_tool = self._make_mock_are_tool(return_value="Event created")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        result = wrapper(title="Standup")

        are_tool.assert_called_once_with(title="Standup")
        assert result == "Event created"

    def test_call_records_success_in_history(self):
        """Successful calls are recorded in invocation history."""
        are_tool = self._make_mock_are_tool(return_value="OK")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        wrapper(title="Test")

        assert len(wrapper.history) == 1
        record = wrapper.history.to_list()[0]
        assert record["inputs"] == {"title": "Test"}
        assert record["outputs"] == "OK"
        assert record["status"] == "success"

    def test_call_records_error_in_history(self):
        """Failed calls are recorded in invocation history and re-raised."""
        are_tool = self._make_mock_are_tool()
        are_tool.side_effect = ValueError("Invalid title")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        with pytest.raises(ValueError, match="Invalid title"):
            wrapper(title="")

        assert len(wrapper.history) == 1
        record = wrapper.history.to_list()[0]
        assert record["status"] == "error"
        assert "Invalid title" in record["outputs"]

    def test_gather_traces(self):
        """gather_traces returns structured trace data."""
        are_tool = self._make_mock_are_tool(return_value="Done")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        wrapper(title="Test1")
        wrapper(title="Test2")

        traces = wrapper.gather_traces()
        assert traces["type"] == "AREToolWrapper"
        assert traces["name"] == "Calendar__create_event"
        assert traces["total_invocations"] == 2
        assert len(traces["invocations"]) == 2

    def test_gather_config(self):
        """gather_config returns tool configuration."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        config = wrapper.gather_config()

        assert config["name"] == "Calendar__create_event"
        assert config["description"] == "Create a calendar event"
        assert "input_schema" in config

    def test_input_schema_from_to_open_ai(self):
        """input_schema is extracted from AppTool.to_open_ai() format."""
        are_tool = self._make_mock_are_tool()
        are_tool.to_open_ai.return_value = {
            "type": "function",
            "function": {
                "name": "Calendar__create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                    },
                    "required": ["title"],
                },
            },
        }
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        assert wrapper.input_schema == {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
            },
            "required": ["title"],
        }

    def test_repr(self):
        """String representation shows tool signature."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        r = repr(wrapper)

        assert "AREToolWrapper" in r
        assert "Calendar__create_event" in r
