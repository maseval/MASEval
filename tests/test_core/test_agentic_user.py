"""Tests for AgenticLLMUser and AgenticLLMUserLLMSimulator."""

import pytest
import json
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock
from maseval.core.user import AgenticLLMUser
from maseval.core.model import ModelAdapter, ChatResponse


class MockModelAdapter(ModelAdapter):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
        self.call_count = 0

    @property
    def model_id(self):
        return "mock-model"

    def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Return mock response as ChatResponse."""
        response_text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # Create ChatResponse with the text content
        chat_response = ChatResponse()
        chat_response.content = response_text
        chat_response.tool_calls = None
        return chat_response


class DummyAgenticLLMUser(AgenticLLMUser):
    def get_tool(self):
        return None


@pytest.fixture
def mock_tool():
    tool = MagicMock(return_value="Tool executed successfully")
    tool.__doc__ = "A mock tool."
    return tool


@pytest.mark.core
def test_agentic_user_tool_usage(mock_tool):
    """Test that AgenticLLMUser can execute a tool loop."""

    # Mock model responses:
    # 1. Thought + Tool Call
    # 2. Final response (after tool execution)

    response_1 = json.dumps({"text": "I need to check something.", "tool_calls": [{"name": "mock_tool", "arguments": {"arg": "val"}}]})

    response_2 = json.dumps({"text": "The tool worked.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticLLMUser(name="AgenticTester", model=model, user_profile={}, scenario="Testing tools", tools={"mock_tool": mock_tool})

    # Simulate response
    final_response = user.respond("Please check.")

    # Verification
    assert final_response == "The tool worked."
    assert mock_tool.called
    assert mock_tool.call_args[1] == {"arg": "val"}

    # Check history
    # The shared history should only contain the final response
    assert len(user.messages) == 2
    assert user.messages[-1]["content"] == "The tool worked."


@pytest.mark.core
def test_agentic_user_max_steps(mock_tool):
    """Test that AgenticLLMUser respects max_internal_steps."""

    # Always return tool call
    response = json.dumps({"text": "Looping...", "tool_calls": [{"name": "mock_tool", "arguments": {}}]})

    model = MockModelAdapter([response])

    user = DummyAgenticLLMUser(
        name="Looper", model=model, user_profile={}, scenario="Infinite loop", tools={"mock_tool": mock_tool}, max_internal_steps=3
    )

    final_response = user.respond("Go.")

    assert "I need to stop" in final_response
    assert mock_tool.call_count == 3


@pytest.mark.core
def test_agentic_user_no_tools():
    """Test that AgenticLLMUser works without tools."""
    response = json.dumps({"text": "Hello! How can I help?", "tool_calls": []})
    model = MockModelAdapter([response])

    user = DummyAgenticLLMUser(
        name="NoToolUser",
        model=model,
        user_profile={},
        scenario="Simple response",
        tools={},
    )

    final_response = user.respond("Hi there!")

    assert final_response == "Hello! How can I help?"
    assert len(user.messages) == 2


@pytest.mark.core
def test_agentic_user_tool_error(mock_tool):
    """Test that AgenticLLMUser handles tool execution errors gracefully."""
    # Make tool raise an error
    mock_tool.side_effect = ValueError("Tool failed!")

    response_1 = json.dumps({"text": "Let me use the tool.", "tool_calls": [{"name": "mock_tool", "arguments": {}}]})
    response_2 = json.dumps({"text": "I see there was an error.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticLLMUser(
        name="ErrorHandler",
        model=model,
        user_profile={},
        scenario="Handle errors",
        tools={"mock_tool": mock_tool},
    )

    final_response = user.respond("Please check.")

    # Should continue despite error
    assert final_response == "I see there was an error."
    assert mock_tool.called


@pytest.mark.core
def test_agentic_user_stop_token():
    """Test that AgenticLLMUser can emit stop tokens."""
    # Response with stop token (no tool calls and contains satisfaction indicator)
    response = json.dumps({"text": "I am satisfied with the response. <STOP>", "tool_calls": []})
    model = MockModelAdapter([response])

    user = DummyAgenticLLMUser(
        name="StopUser",
        model=model,
        user_profile={},
        scenario="Test stop",
        tools={},
    )

    final_response = user.respond("Here is your answer.")

    assert "satisfied" in final_response


@pytest.mark.core
def test_agentic_user_parse_error():
    """Test that AgenticLLMUser handles parse errors with retry."""
    # First response is invalid JSON, second is valid
    invalid_response = "This is not valid JSON"
    valid_response = json.dumps({"text": "Now valid.", "tool_calls": []})

    model = MockModelAdapter([invalid_response, valid_response])

    user = DummyAgenticLLMUser(
        name="ParseRetry",
        model=model,
        user_profile={},
        scenario="Retry on error",
        tools={},
    )

    # The behavior depends on implementation - may return error or retry
    final_response = user.respond("Test.")

    # Should handle the invalid JSON somehow
    assert final_response is not None


@pytest.mark.core
def test_agentic_user_multiple_tool_calls(mock_tool):
    """Test that AgenticLLMUser can execute multiple tool calls."""
    response_1 = json.dumps(
        {
            "text": "I need to call multiple tools.",
            "tool_calls": [
                {"name": "mock_tool", "arguments": {"arg": "val1"}},
                {"name": "mock_tool", "arguments": {"arg": "val2"}},
            ],
        }
    )
    response_2 = json.dumps({"text": "Both tools executed.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticLLMUser(
        name="MultiTool",
        model=model,
        user_profile={},
        scenario="Multiple calls",
        tools={"mock_tool": mock_tool},
    )

    final_response = user.respond("Execute all.")

    assert final_response == "Both tools executed."
    assert mock_tool.call_count == 2


@pytest.mark.core
def test_agentic_user_unknown_tool():
    """Test that AgenticLLMUser handles unknown tool names."""
    response_1 = json.dumps({"text": "Let me use an unknown tool.", "tool_calls": [{"name": "nonexistent_tool", "arguments": {}}]})
    response_2 = json.dumps({"text": "That tool didn't work.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticLLMUser(
        name="UnknownTool",
        model=model,
        user_profile={},
        scenario="Unknown tool",
        tools={},
    )

    final_response = user.respond("Test.")

    # Should continue despite unknown tool
    assert final_response == "That tool didn't work."


@pytest.mark.core
def test_agentic_user_gather_traces(mock_tool):
    """Test that AgenticLLMUser can gather traces."""
    response_1 = json.dumps({"text": "Using tool.", "tool_calls": [{"name": "mock_tool", "arguments": {"key": "value"}}]})
    response_2 = json.dumps({"text": "Done.", "tool_calls": []})

    model = MockModelAdapter([response_1, response_2])

    user = DummyAgenticLLMUser(
        name="TraceTester",
        model=model,
        user_profile={},
        scenario="Gather traces",
        tools={"mock_tool": mock_tool},
    )

    user.respond("Start.")
    traces = user.gather_traces()

    # Check that traces is a dict with expected structure
    assert isinstance(traces, dict)
    # Should have messages or logs
    assert "messages" in traces or "logs" in traces


@pytest.mark.core
class TestGenerateToolDefinitions:
    """Tests for tool definition generation."""

    def test_generate_tool_definitions_empty(self):
        """Empty tools dict generates empty definitions."""
        model = MockModelAdapter([json.dumps({"text": "Hi", "tool_calls": []})])
        user = DummyAgenticLLMUser(
            name="Empty",
            model=model,
            user_profile={},
            scenario="Test",
            tools={},
        )

        definitions = user._generate_tool_definitions()
        assert definitions == [] or definitions == {}

    def test_generate_tool_definitions_with_tools(self, mock_tool):
        """Tools generate proper definitions."""
        model = MockModelAdapter([json.dumps({"text": "Hi", "tool_calls": []})])
        user = DummyAgenticLLMUser(
            name="WithTools",
            model=model,
            user_profile={},
            scenario="Test",
            tools={"mock_tool": mock_tool},
        )

        definitions = user._generate_tool_definitions()

        # Should have at least one definition
        assert len(definitions) >= 1
