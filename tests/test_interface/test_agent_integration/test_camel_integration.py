"""Integration tests for CAMEL-AI.

These tests require camel-ai to be installed.
Run with: pytest -m camel
"""

import pytest
from unittest.mock import Mock

# Import shared CAMEL mocks from conftest
from conftest import MockCamelMemory, MockCamelResponse, create_mock_camel_agent

# Skip entire module if camel-ai not installed
pytest.importorskip("camel")

# Mark all tests in this file as requiring camel
pytestmark = [pytest.mark.interface, pytest.mark.camel]


# =============================================================================
# Basic Creation and Configuration Tests
# =============================================================================


def test_camel_adapter_creation():
    """Test that CamelAgentAdapter can be created with a mock agent."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    # Create adapter with mock agent
    agent_adapter = CamelAgentAdapter(agent_instance=object(), name="test_agent")

    assert agent_adapter.name == "test_agent"
    assert agent_adapter.agent is not None


def test_camel_user_creation():
    """Test that CamelLLMUser can be created."""
    from maseval.interface.agents.camel import CamelLLMUser

    # Create user with required parameters
    mock_model = Mock()
    user = CamelLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="test prompt",
    )

    assert user is not None
    assert user.name == "test_user"


def test_camel_adapter_gather_traces_basic():
    """Test that CamelAgentAdapter.gather_traces() returns expected base fields."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    agent_adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    traces = agent_adapter.gather_traces()

    assert "type" in traces
    assert "gathered_at" in traces
    assert "name" in traces
    assert traces["name"] == "test_agent"
    assert "agent_type" in traces
    assert "messages" in traces
    assert "logs" in traces


def test_camel_adapter_gather_traces_with_response():
    """Test that gather_traces captures response metadata when available."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Simulate a response being stored (new API uses _responses list)
    mock_response = MockCamelResponse(
        content="Response", terminated=True, info={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    )
    adapter._responses.append(mock_response)

    traces = adapter.gather_traces()

    # New API uses last_terminated and aggregated stats
    assert "last_terminated" in traces
    assert traces["last_terminated"] is True
    assert "total_steps" in traces
    assert traces["total_steps"] == 1
    assert "total_tokens" in traces
    assert traces["total_tokens"] == 15


def test_camel_adapter_gather_config_basic():
    """Test that CamelAgentAdapter.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    config = adapter.gather_config()

    assert "type" in config
    assert "gathered_at" in config
    assert "name" in config
    assert config["name"] == "test_agent"
    assert "agent_type" in config
    assert "adapter_type" in config
    assert config["adapter_type"] == "CamelAgentAdapter"


def test_camel_adapter_gather_config_with_system_message():
    """Test that gather_config captures system message."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent(system_message="You are a helpful assistant.")
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    config = adapter.gather_config()

    assert "camel_config" in config
    assert "system_message" in config["camel_config"]
    assert config["camel_config"]["system_message"] == "You are a helpful assistant."


def test_camel_adapter_gather_config_with_tools():
    """Test that gather_config captures tool information."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    # Create mock tools with both name attribute and get_function_name method
    mock_tool1 = Mock()
    mock_tool1.name = "search"
    mock_tool1.description = "Search the web"
    mock_tool1.get_function_name = Mock(return_value="search")
    mock_tool1.get_function_description = Mock(return_value="Search the web")

    mock_tool2 = Mock()
    mock_tool2.name = "calculator"
    mock_tool2.description = "Perform calculations"
    mock_tool2.get_function_name = Mock(return_value="calculator")
    mock_tool2.get_function_description = Mock(return_value="Perform calculations")

    mock_agent = create_mock_camel_agent(tools=[mock_tool1, mock_tool2])
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    config = adapter.gather_config()

    assert "camel_config" in config
    assert "tools" in config["camel_config"]
    assert len(config["camel_config"]["tools"]) == 2
    assert config["camel_config"]["tools"][0]["name"] == "search"
    assert config["camel_config"]["tools"][1]["name"] == "calculator"


def test_camel_adapter_get_messages_empty():
    """Test that get_messages returns empty MessageHistory when no messages."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from maseval.core.history import MessageHistory

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    messages = adapter.get_messages()

    assert isinstance(messages, MessageHistory)
    assert len(messages) == 0


def test_camel_adapter_get_messages_with_memory():
    """Test that get_messages correctly converts memory messages."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    from maseval.core.history import MessageHistory

    memory_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    mock_agent = create_mock_camel_agent(memory_messages=memory_messages)
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    messages = adapter.get_messages()

    assert isinstance(messages, MessageHistory)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there!"


def test_camel_adapter_logs_initially_empty():
    """Test that adapter logs are initially empty."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    assert isinstance(adapter.logs, list)
    assert len(adapter.logs) == 0


def test_camel_user_get_tool():
    """Test that CamelLLMUser.get_tool() returns a CAMEL FunctionTool."""
    from maseval.interface.agents.camel import CamelLLMUser
    from camel.toolkits import FunctionTool

    # Create mock model
    mock_model = Mock()

    # Create user
    user = CamelLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "customer"},
        scenario="Testing scenario",
        initial_query="Hello",
    )

    # Get the tool
    tool = user.get_tool()

    # Verify it's a FunctionTool
    assert isinstance(tool, FunctionTool)


def test_camel_adapter_extract_final_answer():
    """Test that _extract_final_answer correctly extracts content from response."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    mock_response = MockCamelResponse(content="This is the answer")
    answer = adapter._extract_final_answer(mock_response)

    assert answer == "This is the answer"


def test_camel_adapter_extract_final_answer_from_msg():
    """Test extraction when response has msg instead of msgs."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    mock_response = MockCamelResponse(content="Single message answer", use_msg=True)
    answer = adapter._extract_final_answer(mock_response)

    assert answer == "Single message answer"


def test_camel_adapter_convert_memory_messages_with_tool_calls():
    """Test that tool-related fields are preserved during conversion."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Messages with tool calls
    memory_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "Search results", "tool_call_id": "call_1", "name": "search"},
    ]

    # Convert
    converted = adapter._convert_memory_messages(memory_messages)

    # Verify tool-related fields are preserved
    assert len(converted) == 2
    assert "tool_calls" in converted[0]
    assert converted[1]["role"] == "tool"
    assert converted[1]["tool_call_id"] == "call_1"
    assert converted[1]["name"] == "search"


# =============================================================================
# CamelAgentUser Tests
# =============================================================================


def test_camel_agent_user_creation():
    """Test CamelAgentUser creation with a mock agent."""
    from maseval.interface.agents.camel import CamelAgentUser
    from maseval.core.user import User

    mock_agent = create_mock_camel_agent()

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello, I need help",
        name="test_user",
        max_turns=5,
    )

    assert user.name == "test_user"
    assert user._max_turns == 5
    assert user._turn_count == 0
    assert isinstance(user, User)


def test_camel_agent_user_get_initial_query():
    """Test CamelAgentUser.get_initial_query() returns the initial query."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Test initial query",
    )

    assert user.get_initial_query() == "Test initial query"


def test_camel_agent_user_is_done():
    """Test CamelAgentUser.is_done() respects max_turns."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=2,
    )

    assert user.is_done() is False

    user._turn_count = 1
    assert user.is_done() is False

    user._turn_count = 2
    assert user.is_done() is True


def test_camel_agent_user_respond():
    """Test CamelAgentUser.respond() delegates to the CAMEL agent."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent(responses=["Agent response"])

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=5,
    )

    response = user.respond("What is your question?")

    assert response == "Agent response"
    assert user._turn_count == 1
    assert mock_agent.step.called


def test_camel_agent_user_respond_when_done():
    """Test CamelAgentUser.respond() returns empty string when done."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=1,
    )

    # Set as done
    user._turn_count = 1

    response = user.respond("Question?")

    assert response == ""
    assert not mock_agent.step.called


def test_camel_agent_user_get_tool():
    """Test CamelAgentUser.get_tool() returns a FunctionTool."""
    from maseval.interface.agents.camel import CamelAgentUser
    from camel.toolkits import FunctionTool

    mock_agent = create_mock_camel_agent()
    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
    )

    tool = user.get_tool()

    assert isinstance(tool, FunctionTool)


def test_camel_agent_user_gather_traces():
    """Test CamelAgentUser.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent()
    mock_agent.__class__.__name__ = "MockChatAgent"

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        name="test_user",
        max_turns=5,
    )

    traces = user.gather_traces()

    assert traces["type"] == "CamelAgentUser"
    assert traces["name"] == "test_user"
    assert traces["max_turns"] == 5
    assert traces["turns_used"] == 0
    assert "gathered_at" in traces
    assert "logs" in traces


def test_camel_agent_user_gather_config():
    """Test CamelAgentUser.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_agent = create_mock_camel_agent()
    mock_agent.__class__.__name__ = "MockChatAgent"

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Test query",
        name="test_user",
        max_turns=10,
    )

    config = user.gather_config()

    assert config["type"] == "CamelAgentUser"
    assert config["name"] == "test_user"
    assert config["max_turns"] == 10
    assert "initial_query" in config


# =============================================================================
# Execution Loop Tests
# =============================================================================


def test_execution_loop_basic():
    """Test camel_role_playing_execution_loop with mock RolePlaying."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop

    mock_assistant_response = MockCamelResponse(content="Final answer", terminated=True)
    mock_user_response = MockCamelResponse(terminated=False)

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    result = camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=5,
    )

    assert result == "Final answer"
    assert mock_role_playing.step.call_count == 1  # Should stop after first step (terminated)


def test_execution_loop_with_tracer():
    """Test camel_role_playing_execution_loop records steps in tracer."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop, CamelRolePlayingTracer

    mock_assistant_response = MockCamelResponse(content="Answer", terminated=True)
    mock_user_response = MockCamelResponse(terminated=False)

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=5,
        tracer=tracer,
    )

    assert tracer._step_count == 1
    assert tracer._termination_reason == "assistant_terminated"


def test_execution_loop_max_steps():
    """Test camel_role_playing_execution_loop respects max_steps."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop

    mock_assistant_response = MockCamelResponse(content="Non-terminal answer", terminated=False)
    mock_user_response = MockCamelResponse(terminated=False)

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=3,
    )

    assert mock_role_playing.step.call_count == 3


# =============================================================================
# RolePlaying Tracer Tests
# =============================================================================


def test_role_playing_tracer_creation():
    """Test CamelRolePlayingTracer creation."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    assert tracer.name == "test_tracer"
    assert tracer._step_count == 0
    assert tracer._termination_reason is None


def test_role_playing_tracer_record_step():
    """Test CamelRolePlayingTracer.record_step() tracks progress."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing)

    # Create mock responses
    mock_assistant = MockCamelResponse(terminated=False)
    mock_user = MockCamelResponse(terminated=False)

    tracer.record_step(mock_assistant, mock_user)

    assert tracer._step_count == 1
    assert tracer._termination_reason is None

    # Record another step with termination
    mock_assistant_terminated = MockCamelResponse(terminated=True)
    tracer.record_step(mock_assistant_terminated, mock_user)

    assert tracer._step_count == 2
    assert tracer._termination_reason == "assistant_terminated"


def test_role_playing_tracer_gather_traces():
    """Test CamelRolePlayingTracer.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    mock_response = MockCamelResponse(terminated=False)
    tracer.record_step(mock_response, mock_response)
    tracer.record_step(mock_response, mock_response)

    traces = tracer.gather_traces()

    assert traces["name"] == "test_tracer"
    assert traces["trace_type"] == "role_playing_orchestration"
    assert traces["step_count"] == 2
    assert "step_logs" in traces
    assert len(traces["step_logs"]) == 2


def test_role_playing_tracer_gather_config():
    """Test CamelRolePlayingTracer.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    mock_role_playing = Mock()
    mock_role_playing.task_prompt = "Test task prompt"
    mock_role_playing.assistant_role_name = "Assistant"
    mock_role_playing.user_role_name = "User"

    tracer = CamelRolePlayingTracer(mock_role_playing, name="test_tracer")

    config = tracer.gather_config()

    assert config["name"] == "test_tracer"
    assert config["task_prompt"] == "Test task prompt"
    assert config["assistant_role"] == "Assistant"
    assert config["user_role"] == "User"


# =============================================================================
# Workforce Tracer Tests
# =============================================================================


def test_workforce_tracer_creation():
    """Test CamelWorkforceTracer creation."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    mock_workforce = Mock()
    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    assert tracer.name == "test_workforce"


def test_workforce_tracer_gather_traces():
    """Test CamelWorkforceTracer.gather_traces() returns expected fields."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Create mock workforce with internal state
    mock_workforce = Mock()
    mock_workforce._assignees = {"task_1": "worker_a", "task_2": "worker_b"}
    mock_workforce._pending_tasks = []
    mock_workforce._completed_tasks = []
    mock_workforce._task_dependencies = {"task_1": [], "task_2": ["task_1"]}

    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    traces = tracer.gather_traces()

    assert traces["name"] == "test_workforce"
    assert traces["trace_type"] == "workforce_orchestration"
    assert "task_decomposition" in traces
    assert "worker_assignments" in traces
    assert "completed_tasks" in traces
    assert traces["pending_tasks_count"] == 0


def test_workforce_tracer_gather_config():
    """Test CamelWorkforceTracer.gather_config() returns expected fields."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Create mock workforce
    mock_worker1 = Mock()
    mock_worker1.name = "worker_a"
    mock_worker2 = Mock()
    mock_worker2.name = "worker_b"

    mock_workforce = Mock()
    mock_workforce.mode = "collaborative"
    mock_workforce._children = [mock_worker1, mock_worker2]

    tracer = CamelWorkforceTracer(mock_workforce, name="test_workforce")

    config = tracer.gather_config()

    assert config["name"] == "test_workforce"
    assert config["mode"] == "collaborative"
    assert config["workers"] == ["worker_a", "worker_b"]


def test_workforce_tracer_extract_completed_tasks():
    """Test CamelWorkforceTracer extracts completed task info."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Create mock tasks
    mock_task = Mock()
    mock_task.id = "task_123"
    mock_task.content = "Do something"
    mock_task.result = "Done"

    mock_workforce = Mock()
    mock_workforce._completed_tasks = [mock_task]
    mock_workforce._pending_tasks = []
    mock_workforce._assignees = {}
    mock_workforce._task_dependencies = {}

    tracer = CamelWorkforceTracer(mock_workforce)

    traces = tracer.gather_traces()

    assert len(traces["completed_tasks"]) == 1
    assert traces["completed_tasks"][0]["id"] == "task_123"
    assert traces["completed_tasks"][0]["content"] == "Do something"
    assert traces["completed_tasks"][0]["result"] == "Done"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


def test_camel_adapter_run_agent_error_handling():
    """Test CamelAgentAdapter._run_agent() error handling path."""
    from maseval.interface.agents.camel import CamelAgentAdapter
    import pytest

    mock_agent = create_mock_camel_agent(raise_on_step=RuntimeError("Agent execution failed"))
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    with pytest.raises(RuntimeError, match="Agent execution failed"):
        adapter.run("Test query")

    assert len(adapter.logs) == 1
    assert adapter.logs[0]["status"] == "error"
    assert "Agent execution failed" in adapter.logs[0]["error"]
    assert adapter.logs[0]["error_type"] == "RuntimeError"


def test_camel_adapter_run_agent_with_token_usage():
    """Test CamelAgentAdapter._run_agent() captures token usage from response info."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    # Create mock agent with custom response handling for token usage
    usage_info = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    }
    mock_response = MockCamelResponse(content="Test response", terminated=False, info=usage_info)

    # Create agent without side_effect, then set return_value directly
    mock_agent = Mock()
    mock_agent.memory = None
    mock_agent.step = Mock(return_value=mock_response)

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")
    adapter.run("Test query")

    assert len(adapter.logs) == 1
    assert adapter.logs[0]["input_tokens"] == 10
    assert adapter.logs[0]["output_tokens"] == 20
    assert adapter.logs[0]["total_tokens"] == 30


def test_camel_adapter_convert_base_message():
    """Test CamelAgentAdapter._convert_base_message() with BaseMessage objects."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Test with role_type having value attribute
    mock_msg_1 = Mock()
    mock_msg_1.role_type = Mock()
    mock_msg_1.role_type.value = "USER"
    mock_msg_1.content = "User message"

    result_1 = adapter._convert_base_message(mock_msg_1)
    assert result_1["role"] == "user"
    assert result_1["content"] == "User message"

    # Test with role_type having name attribute (no value)
    mock_msg_2 = Mock()
    mock_msg_2.role_type = Mock(spec=["name"])  # Only has name, not value
    mock_msg_2.role_type.name = "ASSISTANT"
    mock_msg_2.content = "Assistant message"

    result_2 = adapter._convert_base_message(mock_msg_2)
    assert result_2["role"] == "assistant"
    assert result_2["content"] == "Assistant message"

    # Test with CAMEL-specific roles (critic, embodiment)
    mock_msg_3 = Mock()
    mock_msg_3.role_type = Mock()
    mock_msg_3.role_type.value = "CRITIC"
    mock_msg_3.content = "Critic message"

    result_3 = adapter._convert_base_message(mock_msg_3)
    assert result_3["role"] == "assistant"  # Mapped to assistant


def test_camel_adapter_convert_memory_messages_with_base_message():
    """Test _convert_memory_messages handles BaseMessage objects."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Create a BaseMessage-like object (not a dict)
    mock_base_msg = Mock()
    mock_base_msg.role_type = Mock()
    mock_base_msg.role_type.value = "USER"
    mock_base_msg.content = "Hello"

    # Mix of dict and BaseMessage
    memory_messages = [
        {"role": "system", "content": "System prompt"},
        mock_base_msg,  # Not a dict, should trigger BaseMessage conversion
    ]

    converted = adapter._convert_memory_messages(memory_messages)

    assert len(converted) == 2
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"
    assert converted[1]["content"] == "Hello"


def test_camel_adapter_gather_traces_with_non_serializable_info():
    """Test gather_traces handles response with minimal/unusual info gracefully."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent()
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Create a response with non-standard info that might not be dict-like
    mock_response = Mock()
    mock_response.terminated = True
    mock_response.msgs = []
    # Mock info that is not a dict (edge case)
    mock_response.info = "not a dict"

    adapter._responses.append(mock_response)

    traces = adapter.gather_traces()

    # Should still work and capture the terminated status
    assert "last_terminated" in traces
    assert traces["last_terminated"] is True
    assert "total_steps" in traces
    assert traces["total_steps"] == 1


def test_camel_adapter_get_messages_memory_access_failure():
    """Test get_messages handles memory access failure gracefully."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_memory = MockCamelMemory()
    mock_memory.get_context = Mock(side_effect=Exception("Memory access failed"))  # type: ignore[method-assign]

    mock_agent = create_mock_camel_agent()
    mock_agent.memory = mock_memory

    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    messages = adapter.get_messages()
    assert len(messages) == 0


def test_camel_adapter_gather_config_with_model():
    """Test gather_config captures model information."""
    from maseval.interface.agents.camel import CamelAgentAdapter

    mock_agent = create_mock_camel_agent(model_type="gpt-4")
    adapter = CamelAgentAdapter(agent_instance=mock_agent, name="test_agent")

    config = adapter.gather_config()

    assert "camel_config" in config
    assert config["camel_config"]["model_type"] == "gpt-4"


def test_camel_user_get_tool_invocation():
    """Test CamelLLMUser.get_tool() returns a working tool."""
    from maseval.interface.agents.camel import CamelLLMUser
    from camel.toolkits import FunctionTool

    mock_model = Mock()

    user = CamelLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "customer"},
        scenario="Testing scenario",
        initial_query="Hello",
    )

    tool = user.get_tool()
    assert isinstance(tool, FunctionTool)

    # Verify the tool function is callable
    assert callable(tool.func)


def test_camel_agent_user_respond_error_handling():
    """Test CamelAgentUser.respond() error handling."""
    from maseval.interface.agents.camel import CamelAgentUser
    import pytest

    mock_agent = create_mock_camel_agent(raise_on_step=RuntimeError("Agent failed"))

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
        max_turns=5,
    )

    with pytest.raises(RuntimeError, match="Agent failed"):
        user.respond("Question?")

    assert len(user.logs) == 1
    assert user.logs[0]["status"] == "error"
    assert "Agent failed" in user.logs[0]["error"]


def test_camel_agent_user_respond_with_msg_not_msgs():
    """Test CamelAgentUser.respond() handles response.msg (singular) attribute."""
    from maseval.interface.agents.camel import CamelAgentUser

    mock_response = MockCamelResponse(content="Response from msg", use_msg=True)

    # Create agent without side_effect to allow direct return_value
    mock_agent = Mock()
    mock_agent.step = Mock(return_value=mock_response)

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
    )

    response = user.respond("Question?")

    assert response == "Response from msg"


def test_camel_agent_user_respond_fallback_to_str():
    """Test CamelAgentUser.respond() falls back to str() for unknown response format."""
    from maseval.interface.agents.camel import CamelAgentUser

    # Response with neither msgs nor msg
    mock_response = Mock()
    mock_response.msgs = []
    mock_response.msg = None
    mock_response.__str__ = Mock(return_value="Fallback string response")

    # Create agent without side_effect to allow direct return_value
    mock_agent = Mock()
    mock_agent.step = Mock(return_value=mock_response)

    user = CamelAgentUser(
        user_agent=mock_agent,
        initial_query="Hello",
    )

    response = user.respond("Question?")

    assert "Fallback string response" in response or "Mock" in response


def test_execution_loop_with_msg_attribute():
    """Test camel_role_playing_execution_loop handles response.msg (singular)."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop

    mock_assistant_response = MockCamelResponse(content="Answer from msg", use_msg=True, terminated=True)
    mock_user_response = MockCamelResponse(terminated=False)

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_assistant_response, mock_user_response)

    result = camel_role_playing_execution_loop(
        mock_role_playing,
        Mock(),
        max_steps=5,
    )

    assert result == "Answer from msg"


def test_execution_loop_calls_init_chat():
    """Test camel_role_playing_execution_loop calls init_chat if available."""
    from maseval.interface.agents.camel import camel_role_playing_execution_loop

    mock_response = MockCamelResponse(content="Response", terminated=True)

    mock_role_playing = Mock()
    mock_role_playing.step.return_value = (mock_response, mock_response)

    camel_role_playing_execution_loop(mock_role_playing, Mock())

    mock_role_playing.init_chat.assert_called_once()


def test_role_playing_tracer_user_termination():
    """Test CamelRolePlayingTracer records user termination."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    mock_role_playing = Mock()
    tracer = CamelRolePlayingTracer(mock_role_playing)

    # First step: neither terminates
    mock_assistant = MockCamelResponse(terminated=False)
    mock_user = MockCamelResponse(terminated=False)
    tracer.record_step(mock_assistant, mock_user)

    assert tracer._termination_reason is None

    # Second step: user terminates
    mock_user_terminated = MockCamelResponse(terminated=True)
    tracer.record_step(mock_assistant, mock_user_terminated)

    assert tracer._termination_reason == "user_terminated"


def test_role_playing_tracer_config_missing_attributes():
    """Test CamelRolePlayingTracer.gather_config() handles missing attributes."""
    from maseval.interface.agents.camel import CamelRolePlayingTracer

    # RolePlaying without optional attributes
    mock_role_playing = Mock(spec=[])  # No attributes

    tracer = CamelRolePlayingTracer(mock_role_playing, name="test")

    config = tracer.gather_config()

    assert config["name"] == "test"
    # Should not raise, just not include missing fields
    assert "task_prompt" not in config
    assert "assistant_role" not in config


def test_workforce_tracer_empty_state():
    """Test CamelWorkforceTracer handles empty/missing internal state."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Workforce with no internal state
    mock_workforce = Mock(spec=[])  # No attributes

    tracer = CamelWorkforceTracer(mock_workforce)

    traces = tracer.gather_traces()

    assert traces["task_decomposition"] == {}
    assert traces["worker_assignments"] == {}
    assert traces["completed_tasks"] == []
    assert traces["pending_tasks_count"] == 0


def test_workforce_tracer_config_missing_mode():
    """Test CamelWorkforceTracer.gather_config() handles missing mode."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    mock_workforce = Mock(spec=[])  # No mode attribute

    tracer = CamelWorkforceTracer(mock_workforce)

    config = tracer.gather_config()

    assert "mode" not in config
    assert "workers" not in config


def test_workforce_tracer_task_without_result():
    """Test CamelWorkforceTracer handles tasks without result attribute."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Task with only id and content, no result
    mock_task = Mock(spec=["id", "content"])
    mock_task.id = "task_1"
    mock_task.content = "Do something"

    mock_workforce = Mock()
    mock_workforce._completed_tasks = [mock_task]
    mock_workforce._pending_tasks = []
    mock_workforce._assignees = {}
    mock_workforce._task_dependencies = {}

    tracer = CamelWorkforceTracer(mock_workforce)

    traces = tracer.gather_traces()

    assert len(traces["completed_tasks"]) == 1
    assert traces["completed_tasks"][0]["id"] == "task_1"
    assert "result" not in traces["completed_tasks"][0]


def test_workforce_tracer_truncates_long_content():
    """Test CamelWorkforceTracer truncates long task content."""
    from maseval.interface.agents.camel import CamelWorkforceTracer

    # Task with very long content
    mock_task = Mock()
    mock_task.id = "task_1"
    mock_task.content = "x" * 500  # 500 chars, should be truncated to 200
    mock_task.result = "y" * 500

    mock_workforce = Mock()
    mock_workforce._completed_tasks = [mock_task]
    mock_workforce._pending_tasks = []
    mock_workforce._assignees = {}
    mock_workforce._task_dependencies = {}

    tracer = CamelWorkforceTracer(mock_workforce)

    traces = tracer.gather_traces()

    assert len(traces["completed_tasks"][0]["content"]) == 200
    assert len(traces["completed_tasks"][0]["result"]) == 200
