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


def test_smolagents_adapter_logs_property():
    """Test that SmolAgentAdapter.logs property returns converted memory steps.

    This test validates that the logs property correctly extracts all relevant
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

    # Access logs property
    logs = adapter.logs

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
    """Test that adapter.logs captures error information from failed steps."""
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

    # Access logs property
    logs = adapter.logs

    # Verify error is captured
    assert len(logs) == 1
    assert "error" in logs[0]
    assert logs[0]["error"] == "Tool execution failed: Connection timeout"


def test_smolagents_adapter_logs_empty_when_no_steps():
    """Test that adapter.logs returns empty list when no execution has occurred."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import AgentMemory
    from unittest.mock import Mock

    # Create a mock agent with empty memory
    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    # Create adapter
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    # Access logs property
    logs = adapter.logs

    # Should be empty
    assert isinstance(logs, list)
    assert len(logs) == 0
