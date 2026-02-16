"""Integration tests for smolagents.

These tests require smolagents to be installed.
Run with: pytest -m smolagents
"""

import pytest

# Skip entire module if smolagents not installed
pytest.importorskip("smolagents")

# Mark all tests in this file as requiring smolagents
pytestmark = [pytest.mark.interface, pytest.mark.smolagents]


def test_smolagents_adapter_import():
    """Test that SmolAgentAdapter can be imported when smolagents is installed."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter, SmolAgentLLMUser

    assert SmolAgentAdapter is not None
    assert SmolAgentLLMUser is not None


def test_smolagents_in_agents_all():
    """Test that smolagents appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "SmolAgentAdapter" in maseval.interface.agents.__all__
    assert "SmolAgentLLMUser" in maseval.interface.agents.__all__


def test_check_smolagents_installed_function():
    """Test that _check_smolagents_installed doesn't raise when smolagents is installed."""
    from maseval.interface.agents.smolagents import _check_smolagents_installed

    # Should not raise
    _check_smolagents_installed()


def test_smolagents_adapter_creation():
    """Test that SmolAgentAdapter can be created."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    # Create adapter with mock agent
    agent_adapter = SmolAgentAdapter(agent_instance=object(), name="test_agent")

    assert agent_adapter.name == "test_agent"
    assert agent_adapter.agent is not None


def test_smolagents_user_creation():
    """Test that SmolAgentLLMUser can be created."""
    from maseval.interface.agents.smolagents import SmolAgentLLMUser
    from unittest.mock import Mock

    # Create user with required parameters
    mock_model = Mock()
    user = SmolAgentLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="test prompt",
    )

    assert user is not None
    assert user.name == "test_user"


def test_smolagents_adapter_gather_traces_with_monitoring():
    """Test that SmolAgentAdapter.gather_traces() captures token and timing data."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add some ActionSteps with monitoring data
    start_time = time.time()

    # Step 1: ActionStep with token usage and timing
    step1 = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.5),
        observations_images=[],
    )
    step1.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    step1.observations = "Observation from step 1"
    step1.action_output = "Output from step 1"
    mock_agent.memory.steps.append(step1)

    # Step 2: Another ActionStep
    step2 = ActionStep(
        step_number=2,
        timing=Timing(start_time=start_time + 0.5, end_time=start_time + 1.2),
        observations_images=[],
    )
    step2.token_usage = TokenUsage(input_tokens=200, output_tokens=100)
    step2.observations = "Observation from step 2"
    step2.action_output = "Output from step 2"
    mock_agent.memory.steps.append(step2)

    # Mock write_memory_to_messages to return empty list (we're testing gather_traces, not get_messages)
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    agent_adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = agent_adapter.gather_traces()

    # Verify aggregated statistics
    assert "total_steps" in traces
    assert traces["total_steps"] == 2

    assert "total_input_tokens" in traces
    assert traces["total_input_tokens"] == 300  # 100 + 200

    assert "total_output_tokens" in traces
    assert traces["total_output_tokens"] == 150  # 50 + 100

    assert "total_tokens" in traces
    assert traces["total_tokens"] == 450  # 300 + 150

    assert "total_duration_seconds" in traces
    assert traces["total_duration_seconds"] == pytest.approx(1.2, abs=0.01)  # 0.5 + 0.7

    # Verify step details
    assert "steps_detail" in traces
    assert len(traces["steps_detail"]) == 2

    # Check step 1 details
    step1_detail = traces["steps_detail"][0]
    assert step1_detail["step_number"] == 1
    assert step1_detail["input_tokens"] == 100
    assert step1_detail["output_tokens"] == 50
    assert step1_detail["total_tokens"] == 150
    assert step1_detail["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert step1_detail["observations"] == "Observation from step 1"
    assert step1_detail["action_output"] == "Output from step 1"

    # Check step 2 details
    step2_detail = traces["steps_detail"][1]
    assert step2_detail["step_number"] == 2
    assert step2_detail["input_tokens"] == 200
    assert step2_detail["output_tokens"] == 100
    assert step2_detail["total_tokens"] == 300
    assert step2_detail["duration_seconds"] == pytest.approx(0.7, abs=0.01)
    assert step2_detail["observations"] == "Observation from step 2"
    assert step2_detail["action_output"] == "Output from step 2"


def test_smolagents_adapter_gather_traces_without_monitoring():
    """Test that gather_traces works when agent has no monitoring data."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import AgentMemory
    from unittest.mock import Mock

    # Create a mock agent with empty memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    agent_adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = agent_adapter.gather_traces()

    # Verify aggregated statistics show zero usage
    assert "total_steps" in traces
    assert traces["total_steps"] == 0

    assert "total_input_tokens" in traces
    assert traces["total_input_tokens"] == 0

    assert "total_output_tokens" in traces
    assert traces["total_output_tokens"] == 0

    assert "total_tokens" in traces
    assert traces["total_tokens"] == 0

    assert "total_duration_seconds" in traces
    assert traces["total_duration_seconds"] == 0.0

    assert "steps_detail" in traces
    assert len(traces["steps_detail"]) == 0


def test_smolagents_adapter_gather_traces_with_planning_step():
    """Test that gather_traces captures PlanningStep data correctly."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import PlanningStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add a PlanningStep
    start_time = time.time()
    planning_step = PlanningStep(
        timing=Timing(start_time=start_time, end_time=start_time + 1.0),
        model_input_messages=[],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Planning output"),
        plan="This is my plan",
    )
    planning_step.token_usage = TokenUsage(input_tokens=500, output_tokens=200)
    mock_agent.memory.steps.append(planning_step)

    # Mock write_memory_to_messages
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    agent_adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Call gather_traces
    traces = agent_adapter.gather_traces()

    # Verify aggregated statistics
    assert traces["total_steps"] == 1
    assert traces["total_input_tokens"] == 500
    assert traces["total_output_tokens"] == 200
    assert traces["total_tokens"] == 700
    assert traces["total_duration_seconds"] == pytest.approx(1.0, abs=0.01)

    # Verify step details
    assert len(traces["steps_detail"]) == 1
    step_detail = traces["steps_detail"][0]
    # PlanningStep may not have step_number, so it could be None
    assert step_detail["input_tokens"] == 500
    assert step_detail["output_tokens"] == 200
    assert step_detail["total_tokens"] == 700
    assert step_detail["duration_seconds"] == pytest.approx(1.0, abs=0.01)
    assert step_detail["plan"] == "This is my plan"
    # PlanningStep should not have action_output or observations
    assert "action_output" not in step_detail
    assert "observations" not in step_detail


def test_smolagents_adapter_extract_current_logs():
    """Test that SmolAgentAdapter._extract_current_logs() returns converted memory steps.

    This test validates that the log extraction correctly extracts all relevant
    information from smolagents' internal memory system, including:
    - Step types (ActionStep, PlanningStep)
    - Timing information (start_time, end_time, duration)
    - Token usage (input_tokens, output_tokens, total_tokens)
    - Model input/output messages
    - Tool calls and observations
    - Error information
    """
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, PlanningStep, AgentMemory, ToolCall
    from smolagents.monitoring import TokenUsage, Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add an ActionStep with comprehensive data
    start_time = time.time()
    step1 = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.5),
        observations_images=[],
    )
    step1.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    step1.observations = "Tool returned: success"
    step1.action_output = "Final output from action"
    step1.tool_calls = [ToolCall(name="test_tool", arguments={"arg": "value"}, id="call_123")]
    step1.model_input_messages = [
        ChatMessage(role=MessageRole.USER, content="Execute this task"),
        ChatMessage(role=MessageRole.SYSTEM, content="System context"),
    ]
    mock_agent.memory.steps.append(step1)

    # Add a PlanningStep
    step2 = PlanningStep(
        timing=Timing(start_time=start_time + 0.5, end_time=start_time + 1.0),
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="What should I do?")],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Here's the plan"),
        plan="Step 1: Do this\nStep 2: Do that",
    )
    step2.token_usage = TokenUsage(input_tokens=200, output_tokens=150)
    mock_agent.memory.steps.append(step2)

    # Mock write_memory_to_messages
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Use _extract_current_logs() to test the conversion logic
    # (logs property returns _accumulated_logs, populated only via _run_agent())
    logs = adapter._extract_current_logs()

    # Verify logs structure
    assert isinstance(logs, list)
    assert len(logs) == 2

    # Verify ActionStep log entry
    action_log = logs[0]
    assert action_log["step_type"] == "ActionStep"
    assert action_log["step_number"] == 1
    assert action_log["input_tokens"] == 100
    assert action_log["output_tokens"] == 50
    assert action_log["total_tokens"] == 150
    assert action_log["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert action_log["observations"] == "Tool returned: success"
    assert action_log["action_output"] == "Final output from action"
    assert "tool_calls" in action_log
    assert len(action_log["tool_calls"]) == 1
    assert action_log["tool_calls"][0]["name"] == "test_tool"

    # Verify model_input_messages are converted
    assert "model_input_messages" in action_log
    assert isinstance(action_log["model_input_messages"], list)
    assert len(action_log["model_input_messages"]) == 2
    assert action_log["model_input_messages"][0]["role"] == "user"
    assert action_log["model_input_messages"][0]["content"] == "Execute this task"
    assert action_log["model_input_messages"][1]["role"] == "system"

    # Verify PlanningStep log entry
    planning_log = logs[1]
    assert planning_log["step_type"] == "PlanningStep"
    assert planning_log["input_tokens"] == 200
    assert planning_log["output_tokens"] == 150
    assert planning_log["total_tokens"] == 350
    assert planning_log["duration_seconds"] == pytest.approx(0.5, abs=0.01)
    assert planning_log["plan"] == "Step 1: Do this\nStep 2: Do that"

    # Verify model_input_messages for planning step
    assert "model_input_messages" in planning_log
    assert len(planning_log["model_input_messages"]) == 1
    assert planning_log["model_input_messages"][0]["content"] == "What should I do?"

    # PlanningStep should not have action-specific fields
    assert "action_output" not in planning_log
    assert "observations" not in planning_log
    assert "tool_calls" not in planning_log


def test_smolagents_adapter_logs_with_errors():
    """Test that _extract_current_logs() captures error information from failed steps."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents import AgentError
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import Timing
    from unittest.mock import Mock
    import time

    # Create a mock agent with memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    # Add an ActionStep with an error
    start_time = time.time()
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.2),
        observations_images=[],
    )
    # Create a proper AgentError object with mock logger
    mock_logger = Mock()
    step.error = AgentError("Tool execution failed: Connection timeout", logger=mock_logger)
    mock_agent.memory.steps.append(step)

    # Mock write_memory_to_messages
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Use _extract_current_logs() to test the conversion logic
    logs = adapter._extract_current_logs()

    # Verify error is captured
    assert len(logs) == 1
    assert "error" in logs[0]
    assert logs[0]["error"] == "Tool execution failed: Connection timeout"


def test_smolagents_adapter_extract_current_logs_empty_when_no_steps():
    """Test that _extract_current_logs() returns empty list when no execution has occurred."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import AgentMemory
    from unittest.mock import Mock

    # Create a mock agent with empty memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Use _extract_current_logs() to test the conversion logic
    logs = adapter._extract_current_logs()

    # Should be empty
    assert isinstance(logs, list)
    assert len(logs) == 0


# =============================================================================
# gather_config() Tests
# =============================================================================


def test_smolagents_gather_config_with_to_dict():
    """Test gather_config() uses agent.to_dict() when available."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])
    to_dict_data = {"max_steps": 10, "model": {"class": "FakeModel"}, "tools": []}
    mock_agent.to_dict = Mock(return_value=to_dict_data)

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="config_agent")
    config = adapter.gather_config()

    # Base keys
    assert config["name"] == "config_agent"
    assert config["adapter_type"] == "SmolAgentAdapter"
    assert "type" in config
    assert "gathered_at" in config
    assert "agent_type" in config

    # smolagents_config from to_dict()
    assert config["smolagents_config"] == to_dict_data


def test_smolagents_gather_config_fallback_without_to_dict():
    """Test gather_config() falls back to manual attribute collection when to_dict is absent."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock(
        spec=[
            "memory",
            "write_memory_to_messages",
            "step_callbacks",
            "max_steps",
            "planning_interval",
            "name",
            "description",
            "additional_authorized_imports",
            "executor_type",
        ]
    )
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])
    mock_agent.max_steps = 5
    mock_agent.planning_interval = 3
    mock_agent.name = "my_agent"
    mock_agent.description = "A test agent"
    mock_agent.additional_authorized_imports = ["os"]
    mock_agent.executor_type = "local"

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="fallback_agent")
    config = adapter.gather_config()

    assert "smolagents_config" in config
    smolagents_config = config["smolagents_config"]
    assert smolagents_config["max_steps"] == 5
    assert smolagents_config["planning_interval"] == 3
    assert smolagents_config["name"] == "my_agent"
    assert smolagents_config["description"] == "A test agent"
    assert smolagents_config["additional_authorized_imports"] == ["os"]
    assert smolagents_config["executor_type"] == "local"


def test_smolagents_gather_config_to_dict_raises_falls_back():
    """Test gather_config() falls back to attributes when to_dict() raises."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])
    mock_agent.to_dict = Mock(side_effect=RuntimeError("serialization failed"))
    mock_agent.max_steps = 7

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="error_agent")
    config = adapter.gather_config()

    assert "smolagents_config" in config
    assert config["smolagents_config"]["max_steps"] == 7


# =============================================================================
# _run_agent() and logs Tests
# =============================================================================


def test_smolagents_run_populates_accumulated_logs():
    """Test that run() populates accumulated logs from agent memory."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import Timing
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")

    start_time = time.time()
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.1),
        observations_images=[],
    )
    mock_agent.memory.steps.append(step)
    mock_agent.run = Mock(return_value="final answer")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="run_agent")
    result = adapter.run("test query")

    assert result == "final answer"
    assert len(adapter.logs) == 1
    assert adapter.logs[0]["step_type"] == "ActionStep"
    assert adapter.logs[0]["step_number"] == 1


# =============================================================================
# _extract_current_logs() TaskStep Branch
# =============================================================================


def test_smolagents_extract_current_logs_task_step():
    """Test _extract_current_logs() handles TaskStep with and without images."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import TaskStep, AgentMemory
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # TaskStep without images
    task_step = TaskStep(task="Solve the puzzle")
    mock_agent.memory.steps.append(task_step)

    # TaskStep with images
    task_step_with_images = TaskStep(task="Analyze the image")
    task_step_with_images.task_images = [Mock(), Mock()]
    mock_agent.memory.steps.append(task_step_with_images)

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="task_agent")
    logs = adapter._extract_current_logs()

    assert len(logs) == 2

    assert logs[0]["step_type"] == "TaskStep"
    assert logs[0]["task"] == "Solve the puzzle"
    assert "task_images_count" not in logs[0]

    assert logs[1]["step_type"] == "TaskStep"
    assert logs[1]["task"] == "Analyze the image"
    assert logs[1]["task_images_count"] == 2


# =============================================================================
# SmolAgentLLMUser.get_tool() Test
# =============================================================================


def test_smolagents_llm_user_get_tool():
    """Test SmolAgentLLMUser.get_tool() returns a SmolAgentUserSimulationInputTool."""
    from maseval.interface.agents.smolagents import SmolAgentLLMUser
    from maseval.interface.agents.smolagents_optional import SmolAgentUserSimulationInputTool
    from unittest.mock import Mock

    mock_model = Mock()
    user = SmolAgentLLMUser(
        name="tool_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="hello",
    )

    tool = user.get_tool()

    assert isinstance(tool, SmolAgentUserSimulationInputTool)
    assert hasattr(tool, "forward")


# =============================================================================
# Message Conversion with Tool Calls
# =============================================================================


def test_smolagents_message_conversion_tool_call_attributes():
    """Test _convert_smolagents_messages preserves tool_calls from ChatMessage objects."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="msg_agent")

    # Create ChatMessage with tool_calls attribute
    msg = ChatMessage(role=MessageRole.ASSISTANT, content="Using tool")
    msg.tool_calls = [{"id": "call_1", "function": {"name": "search", "arguments": '{"q": "test"}'}}]

    history = adapter._convert_smolagents_messages([msg])

    assert len(history) == 1
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "Using tool"
    assert "tool_calls" in history[0]
    assert history[0]["tool_calls"][0]["id"] == "call_1"
    assert history[0]["tool_calls"][0]["function"]["name"] == "search"


# =============================================================================
# Phase 2 Hook: _on_step() Tests
# =============================================================================


def test_smolagents_on_step_action_step():
    """Test _on_step() callback handles ActionStep with tool calls."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, ToolCall
    from smolagents.monitoring import Timing
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="hook_agent")

    # Create an ActionStep
    t = time.time()
    action_step = ActionStep(step_number=3, timing=Timing(start_time=t, end_time=t + 0.1), observations_images=[])
    action_step.error = None
    action_step.tool_calls = [ToolCall(name="search", arguments={"q": "test"}, id="tc_1")]

    # Create a mock agent with a name attribute
    mock_calling_agent = Mock()
    mock_calling_agent.name = "sub_agent"

    # Call _on_step directly
    adapter._on_step(action_step, agent=mock_calling_agent)

    assert len(adapter._trace_buffer) == 1
    entry = adapter._trace_buffer[0]
    assert entry["source"] == "smolagents_step_callback"
    assert entry["step_type"] == "ActionStep"
    assert entry["agent_name"] == "sub_agent"
    assert entry["step_number"] == 3
    assert entry["has_error"] is False
    assert entry["tool_calls"] == ["search"]


def test_smolagents_on_step_planning_step():
    """Test _on_step() callback handles PlanningStep."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import PlanningStep
    from smolagents.monitoring import Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="hook_agent")

    t = time.time()
    planning_step = PlanningStep(
        timing=Timing(start_time=t, end_time=t + 0.1),
        model_input_messages=[],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan"),
        plan="Step 1\nStep 2\nStep 3",
    )

    adapter._on_step(planning_step, agent=None)

    assert len(adapter._trace_buffer) == 1
    entry = adapter._trace_buffer[0]
    assert entry["source"] == "smolagents_step_callback"
    assert entry["step_type"] == "PlanningStep"
    assert entry["agent_name"] is None
    assert entry["plan_length"] == len("Step 1\nStep 2\nStep 3")  # 20


def test_smolagents_message_conversion_dict_format_with_tool_fields():
    """Test _convert_smolagents_messages handles dict-format messages with tool fields."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="dict_msg_agent")

    # Dict-format messages with tool_calls, tool_call_id, name, metadata
    dict_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": "{}"}}],
            "metadata": {"source": "test"},
        },
        {
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_1",
            "name": "calc",
        },
        {
            "role": "user",
            "content": "Thanks",
        },
    ]

    history = adapter._convert_smolagents_messages(dict_messages)

    assert len(history) == 3

    # Assistant with tool_calls and metadata
    assert history[0]["role"] == "assistant"
    assert history[0]["tool_calls"][0]["id"] == "call_1"
    assert history[0]["metadata"]["source"] == "test"

    # Tool message with tool_call_id and name
    assert history[1]["role"] == "tool"
    assert history[1]["content"] == "42"
    assert history[1]["tool_call_id"] == "call_1"
    assert history[1]["name"] == "calc"

    # Regular user message
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Thanks"


def test_smolagents_message_conversion_non_string_role():
    """Test _convert_smolagents_messages handles roles that are neither enums nor strings."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = Mock()
    mock_agent.memory.steps = []
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="role_agent")

    # Create a ChatMessage-like object with a role that's not an enum and not a string
    msg = Mock()
    msg.role = 42  # integer role (edge case)
    msg.content = "test"
    msg.tool_calls = None
    msg.tool_call_id = None

    history = adapter._convert_smolagents_messages([msg])

    assert len(history) == 1
    assert history[0]["role"] == "42"  # converted via str().lower()
    assert history[0]["content"] == "test"
