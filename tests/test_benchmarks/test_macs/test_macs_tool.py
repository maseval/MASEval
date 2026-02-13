"""Unit tests for MACSGenericTool."""

import pytest
from unittest.mock import patch

from maseval.benchmark.macs import MACSGenericTool

from conftest import DummyModelAdapter


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.benchmark
class TestMACSGenericToolInit:
    """Tests for MACSGenericTool initialization."""

    def test_init_and_defaults(self, simple_tool_spec, minimal_tool_spec, macs_model):
        """Tool initializes correctly and handles defaults."""
        # Standard initialization
        tool = MACSGenericTool(simple_tool_spec, macs_model)
        assert tool.name == "search_flights"
        assert tool.description == "Search for available flights"
        assert tool.output_type == "string"
        assert "origin" in tool.inputs
        assert "destination" in tool.inputs
        assert tool.simulator is not None
        assert tool.simulator.tool_name == "search_flights"
        assert len(tool.history.to_list()) == 0

        # Minimal spec with defaults
        minimal_tool = MACSGenericTool(minimal_tool_spec, macs_model)
        assert minimal_tool.name == "simple_action"
        assert minimal_tool.description == ""
        assert minimal_tool.inputs == {}


# =============================================================================
# Unit Tests: Schema Conversion
# =============================================================================


@pytest.mark.benchmark
class TestSchemaToInputs:
    """Tests for _schema_to_inputs static method."""

    def test_schema_to_inputs_basic(self):
        """JSON schema correctly converted to inputs format."""
        schema = {
            "properties": {
                "origin": {"type": "string", "description": "Origin city"},
                "count": {"type": "integer", "description": "Number of items"},
            }
        }
        result = MACSGenericTool._schema_to_inputs(schema)

        assert "origin" in result
        assert result["origin"]["type"] == "string"
        assert result["origin"]["description"] == "Origin city"
        assert result["count"]["type"] == "integer"

    def test_schema_to_inputs_empty(self):
        """Empty schema returns empty inputs."""
        assert MACSGenericTool._schema_to_inputs({}) == {}
        assert MACSGenericTool._schema_to_inputs({"properties": {}}) == {}

    def test_schema_to_inputs_data_type_field(self):
        """Handles data_type field (MACS format) over type."""
        schema = {
            "properties": {
                "date": {"data_type": "date", "type": "string", "description": "Date"},
            }
        }
        result = MACSGenericTool._schema_to_inputs(schema)

        # data_type takes precedence
        assert result["date"]["type"] == "date"

    def test_schema_to_inputs_missing_description(self):
        """Handles missing description gracefully."""
        schema = {
            "properties": {
                "field": {"type": "string"},
            }
        }
        result = MACSGenericTool._schema_to_inputs(schema)

        assert result["field"]["description"] == ""

    def test_schema_to_inputs_missing_type(self):
        """Defaults to string type when missing."""
        schema = {
            "properties": {
                "field": {"description": "Some field"},
            }
        }
        result = MACSGenericTool._schema_to_inputs(schema)

        assert result["field"]["type"] == "string"


# =============================================================================
# Unit Tests: Tool Invocation
# =============================================================================


@pytest.mark.benchmark
class TestMACSGenericToolInvocation:
    """Tests for tool invocation behavior."""

    def test_call_invokes_model(self, simple_tool_spec):
        """Calling tool invokes the model via simulator."""
        # Create model that returns valid JSON (ToolLLMSimulator expects {"text": ..., "details": ...})
        model = DummyModelAdapter(responses=['{"text": "Found flights", "details": {}}'])
        tool = MACSGenericTool(simple_tool_spec, model)

        _ = tool(origin="LAX", destination="JFK")

        # Model should have been called
        assert model._call_count >= 1

    def test_call_returns_response(self, simple_tool_spec):
        """Tool call returns the simulated response."""
        # Create model that returns valid JSON
        model = DummyModelAdapter(responses=['{"text": "Flight found: AA123", "details": {}}'])
        tool = MACSGenericTool(simple_tool_spec, model)

        result = tool(origin="LAX", destination="JFK")

        # Result should contain the response text
        assert "Flight found: AA123" in result

    def test_call_records_history(self, simple_tool_spec):
        """Tool invocation recorded in history."""
        model = DummyModelAdapter(responses=['{"text": "success", "details": {"booking_id": "123"}}'])
        tool = MACSGenericTool(simple_tool_spec, model)

        tool(origin="LAX", destination="JFK")

        history = tool.history.to_list()
        assert len(history) == 1
        assert history[0]["inputs"] == {"origin": "LAX", "destination": "JFK"}
        assert history[0]["outputs"] == "success"
        assert history[0]["status"] == "success"
        assert history[0]["meta"] == {"booking_id": "123"}

    def test_multiple_invocations(self, simple_tool_spec):
        """Multiple calls tracked in history."""
        model = DummyModelAdapter(responses=['{"text": "success", "details": {}}'])
        tool = MACSGenericTool(simple_tool_spec, model)

        tool(origin="LAX", destination="JFK")
        tool(origin="SFO", destination="ORD")
        tool(origin="BOS", destination="MIA")

        history = tool.history.to_list()
        assert len(history) == 3
        assert history[0]["inputs"]["origin"] == "LAX"
        assert history[1]["inputs"]["origin"] == "SFO"
        assert history[2]["inputs"]["origin"] == "BOS"


# =============================================================================
# Unit Tests: Tracing and Config
# =============================================================================


@pytest.mark.benchmark
class TestMACSGenericToolTracing:
    """Tests for trace and config gathering."""

    def test_gather_traces(self, simple_tool_spec, macs_model):
        """Traces include name and invocations."""
        tool = MACSGenericTool(simple_tool_spec, macs_model)

        with patch.object(tool.simulator, "__call__", return_value=("Response", {})):
            tool(origin="LAX", destination="JFK")

        traces = tool.gather_traces()

        assert traces["name"] == "search_flights"
        assert "invocations" in traces
        assert len(traces["invocations"]) == 1
        assert "gathered_at" in traces  # From TraceableMixin

    def test_gather_config(self, simple_tool_spec, macs_model):
        """Config includes name, description, schema."""
        tool = MACSGenericTool(simple_tool_spec, macs_model)

        config = tool.gather_config()

        assert config["name"] == "search_flights"
        assert config["description"] == "Search for available flights"
        assert "input_schema" in config
        assert "gathered_at" in config  # From ConfigurableMixin

    def test_gather_traces_empty_history(self, simple_tool_spec, macs_model):
        """Traces work with empty invocation history."""
        tool = MACSGenericTool(simple_tool_spec, macs_model)

        traces = tool.gather_traces()

        assert traces["name"] == "search_flights"
        assert traces["invocations"] == []


# =============================================================================
# Unit Tests: String Representation
# =============================================================================


@pytest.mark.benchmark
class TestMACSGenericToolRepr:
    """Tests for string representation."""

    def test_repr(self, simple_tool_spec, macs_model):
        """String representation is informative."""
        tool = MACSGenericTool(simple_tool_spec, macs_model)

        repr_str = repr(tool)

        assert "MACSGenericTool" in repr_str
        assert "search_flights" in repr_str
        assert "origin" in repr_str
        assert "destination" in repr_str
        assert "string" in repr_str

    def test_repr_no_inputs(self, minimal_tool_spec, macs_model):
        """Repr handles tool with no inputs."""
        tool = MACSGenericTool(minimal_tool_spec, macs_model)

        repr_str = repr(tool)

        assert "MACSGenericTool" in repr_str
        assert "simple_action" in repr_str
        assert "()" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSGenericToolIntegration:
    """Integration tests for MACSGenericTool."""

    def test_tool_with_complex_spec(self, complex_tool_spec, macs_model):
        """Tool works with complex specification."""
        tool = MACSGenericTool(complex_tool_spec, macs_model)

        assert tool.name == "book_hotel"
        assert "city" in tool.inputs
        assert "check_in" in tool.inputs
        assert tool.inputs["check_in"]["type"] == "date"

    def test_end_to_end_flow(self, simple_tool_spec):
        """Complete flow from creation to trace gathering."""
        # Create model with specific response
        model = DummyModelAdapter(responses=['{"status": "found", "flights": ["AA123", "UA456"]}'])
        tool = MACSGenericTool(simple_tool_spec, model)

        # Invoke tool (simulator will use the model)
        # Note: actual response depends on ToolLLMSimulator's parsing
        with patch.object(tool.simulator, "__call__", return_value=("Found 2 flights", {"parsed": True})):
            _ = tool(origin="LAX", destination="JFK")

        # Check traces
        traces = tool.gather_traces()
        assert traces["name"] == "search_flights"
        assert len(traces["invocations"]) == 1

        # Check config
        config = tool.gather_config()
        assert config["name"] == "search_flights"
