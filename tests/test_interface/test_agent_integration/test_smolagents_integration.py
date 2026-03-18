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

    # Verify aggregated statistics (token totals moved to gather_usage)
    assert "total_steps" in traces
    assert traces["total_steps"] == 2

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

    # Verify aggregated statistics (token totals moved to gather_usage)
    assert "total_steps" in traces
    assert traces["total_steps"] == 0

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

    # Verify aggregated statistics (token totals moved to gather_usage)
    assert traces["total_steps"] == 1
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


def test_smolagents_adapter_step_status_error_on_crashed_step():
    """Test that steps with no output fields are marked as 'error'.

    When smolagents raises AgentGenerationError, the step is added to memory
    via the finally block but step.error is never set. The step has
    model_input_messages but no model_output_message, tool_calls, observations,
    action_output, or is_final_answer=True. MASEval should detect this and
    report status='error' rather than 'success'.
    """
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test system prompt")

    start_time = time.time()

    # Step 1: Normal successful step (has observations and action_output)
    step1 = ActionStep(
        step_number=1,
        timing=Timing(start_time=start_time, end_time=start_time + 0.5),
        observations_images=[],
    )
    step1.observations = "Tool returned data"
    step1.action_output = "Result"
    mock_agent.memory.steps.append(step1)

    # Step 2: Incomplete step — simulates AgentGenerationError scenario
    # Has model_input_messages but no output fields, error is None
    step2 = ActionStep(
        step_number=2,
        timing=Timing(start_time=start_time + 0.5, end_time=start_time + 0.6),
        observations_images=[],
    )
    step2.model_input_messages = [
        ChatMessage(role=MessageRole.USER, content="Do something"),
    ]
    # Crucially: no model_output_message, no tool_calls, no observations,
    # no action_output, is_final_answer=False (default), error=None (default)
    mock_agent.memory.steps.append(step2)

    # Step 3: Error step (has step.error set)
    step3 = ActionStep(
        step_number=3,
        timing=Timing(start_time=start_time + 0.6, end_time=start_time + 0.7),
        observations_images=[],
    )
    mock_logger = Mock()
    from smolagents import AgentError

    step3.error = AgentError("Something failed", logger=mock_logger)
    mock_agent.memory.steps.append(step3)

    mock_agent.write_memory_to_messages = Mock(return_value=[])
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    logs = adapter.logs

    assert len(logs) == 3
    assert logs[0]["status"] == "success"
    assert logs[1]["status"] == "error"
    assert logs[2]["status"] == "error"


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


# =============================================================================
# gather_usage() Tests
# =============================================================================


def test_smolagents_adapter_gather_usage_with_steps():
    """Test that gather_usage() aggregates token usage across all memory steps."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage
    from smolagents.memory import ActionStep, PlanningStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from smolagents.models import ChatMessage, MessageRole
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")

    start = time.time()

    # ActionStep with usage
    step1 = ActionStep(
        step_number=1,
        timing=Timing(start_time=start, end_time=start + 0.5),
        observations_images=[],
    )
    step1.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    mock_agent.memory.steps.append(step1)

    # PlanningStep with usage
    step2 = PlanningStep(
        timing=Timing(start_time=start + 0.5, end_time=start + 1.0),
        model_input_messages=[],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="Plan"),
        plan="My plan",
    )
    step2.token_usage = TokenUsage(input_tokens=200, output_tokens=80)
    mock_agent.memory.steps.append(step2)

    mock_agent.write_memory_to_messages = Mock(return_value=[])
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    usage = adapter.gather_usage()

    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 300  # 100 + 200
    assert usage.output_tokens == 130  # 50 + 80
    assert usage.total_tokens == 430


def test_smolagents_adapter_gather_usage_no_steps():
    """Test that gather_usage() returns empty Usage when no steps exist."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import Usage
    from smolagents.memory import AgentMemory
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    usage = adapter.gather_usage()

    assert isinstance(usage, Usage)
    assert usage.cost == 0.0
    assert usage.input_tokens == 0 if hasattr(usage, "input_tokens") else True


def test_smolagents_adapter_gather_usage_steps_without_token_usage():
    """Test that gather_usage() returns empty Usage when steps have no token_usage."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import Usage
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import Timing
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")

    start = time.time()
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=start, end_time=start + 0.5),
        observations_images=[],
    )
    # No token_usage set — defaults to None
    mock_agent.memory.steps.append(step)
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent")

    usage = adapter.gather_usage()

    # Should return plain Usage (not TokenUsage) since no usage data
    assert isinstance(usage, Usage)
    assert usage.cost == 0.0


# =============================================================================
# End-to-End Usage Collection Tests
# (real ToolCallingAgent execution, not pre-populated mock data)
# =============================================================================


class _FakeModelForUsageTest:
    """Deterministic fake model that returns canned responses with token usage.

    Subclasses smolagents Model via duck-typing (same generate() signature).
    Uses a list of responses; cycles the last one if calls exceed the list.
    """

    def __init__(self, responses=None):
        from smolagents.models import (
            ChatMessage,
            ChatMessageToolCall,
            ChatMessageToolCallFunction,
            MessageRole,
        )
        from smolagents.monitoring import TokenUsage

        self.model_id = "fake-model-for-test"
        self._call_count = 0
        self._responses = responses or [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Here is the answer.",
                tool_calls=[
                    ChatMessageToolCall(
                        function=ChatMessageToolCallFunction(
                            name="final_answer",
                            arguments={"answer": "42"},
                        ),
                        id="call_001",
                        type="function",
                    )
                ],
                token_usage=TokenUsage(input_tokens=150, output_tokens=30),
            )
        ]

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def test_e2e_smolagents_gather_usage_single_step():
    """Run a real ToolCallingAgent → adapter.run() → gather_usage() returns real token counts."""
    from smolagents import ToolCallingAgent
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage

    agent = ToolCallingAgent(tools=[], model=_FakeModelForUsageTest(), max_steps=3, verbosity_level=0)
    adapter = SmolAgentAdapter(agent_instance=agent, name="test_agent")

    result = adapter.run("What is the meaning of life?")
    assert result == "42"

    usage = adapter.gather_usage()
    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 150
    assert usage.output_tokens == 30
    assert usage.total_tokens == 180


def test_e2e_smolagents_gather_usage_multi_step():
    """Run a real agent through tool call + final answer, verify usage aggregation."""
    from smolagents import ToolCallingAgent
    from smolagents.models import (
        ChatMessage,
        ChatMessageToolCall,
        ChatMessageToolCallFunction,
        MessageRole,
    )
    from smolagents.monitoring import TokenUsage
    from smolagents.tools import Tool
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage

    class AddTool(Tool):
        name = "add_numbers"
        description = "Adds two numbers"
        inputs = {"a": {"type": "number", "description": "First"}, "b": {"type": "number", "description": "Second"}}
        output_type = "number"

        def forward(self, a, b):
            return a + b

    responses = [
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Let me add.",
            tool_calls=[
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="add_numbers", arguments={"a": 20, "b": 22}),
                    id="call_001",
                    type="function",
                )
            ],
            token_usage=TokenUsage(input_tokens=200, output_tokens=40),
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The sum is 42.",
            tool_calls=[
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "42"}),
                    id="call_002",
                    type="function",
                )
            ],
            token_usage=TokenUsage(input_tokens=350, output_tokens=20),
        ),
    ]

    agent = ToolCallingAgent(tools=[AddTool()], model=_FakeModelForUsageTest(responses), max_steps=5, verbosity_level=0)
    adapter = SmolAgentAdapter(agent_instance=agent, name="test_agent")

    result = adapter.run("What is 20 + 22?")
    assert result == "42"

    usage = adapter.gather_usage()
    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 550  # 200 + 350
    assert usage.output_tokens == 60  # 40 + 20
    assert usage.total_tokens == 610


def test_e2e_smolagents_gather_usage_empty_before_run():
    """Verify gather_usage() returns empty Usage before run, real TokenUsage after."""
    from smolagents import ToolCallingAgent
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import Usage, TokenUsage as MasevalTokenUsage

    agent = ToolCallingAgent(tools=[], model=_FakeModelForUsageTest(), max_steps=3, verbosity_level=0)
    adapter = SmolAgentAdapter(agent_instance=agent, name="test_agent")

    # Before run: no usage
    usage_before = adapter.gather_usage()
    assert isinstance(usage_before, Usage)
    assert not isinstance(usage_before, MasevalTokenUsage)

    # After run: real usage from the model
    adapter.run("test query")
    usage_after = adapter.gather_usage()
    assert isinstance(usage_after, MasevalTokenUsage)
    assert usage_after.input_tokens > 0
    assert usage_after.output_tokens > 0


# --- Cost calculation tests ---


def test_smolagents_adapter_cost_with_explicit_calculator():
    """Test that passing a cost_calculator computes cost from token usage."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage, StaticPricingCalculator
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")
    mock_agent.model.model_id = "gpt-4o-mini"

    start = time.time()
    step = ActionStep(step_number=1, timing=Timing(start_time=start, end_time=start + 0.5), observations_images=[])
    step.token_usage = TokenUsage(input_tokens=1000, output_tokens=500)
    mock_agent.memory.steps.append(step)

    calculator = StaticPricingCalculator({"gpt-4o-mini": {"input": 0.00001, "output": 0.00002}})

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test_agent", cost_calculator=calculator)
    usage = adapter.gather_usage()

    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    # Cost = 1000 * 0.00001 + 500 * 0.00002 = 0.01 + 0.01 = 0.02
    assert usage.cost == pytest.approx(0.02)


def test_smolagents_adapter_cost_with_explicit_model_id():
    """Test that explicit model_id overrides auto-detected one."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import StaticPricingCalculator
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from unittest.mock import Mock
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")
    mock_agent.model.model_id = "wrong-model"  # Auto-detected, but overridden

    start = time.time()
    step = ActionStep(step_number=1, timing=Timing(start_time=start, end_time=start + 0.5), observations_images=[])
    step.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    mock_agent.memory.steps.append(step)

    calculator = StaticPricingCalculator({"my-model": {"input": 0.001, "output": 0.002}})

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test", cost_calculator=calculator, model_id="my-model")
    usage = adapter.gather_usage()

    # Should use "my-model" pricing, not "wrong-model"
    assert usage.cost == pytest.approx(0.001 * 100 + 0.002 * 50)


def test_smolagents_adapter_resolve_model_id():
    """Test that _resolve_model_id() reads from agent.model.model_id."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.model.model_id = "gpt-4o"
    mock_agent.write_memory_to_messages = Mock(return_value=[])

    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test")
    assert adapter._resolve_model_id() == "gpt-4o"


def test_smolagents_adapter_resolve_model_id_missing():
    """Test that _resolve_model_id() returns None when model has no model_id."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock(spec=[])  # No attributes at all
    adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test")
    assert adapter._resolve_model_id() is None


def test_smolagents_adapter_no_cost_without_calculator():
    """Test that cost stays 0.0 when no calculator is available and auto-create fails."""
    from maseval.interface.agents.smolagents import SmolAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage
    from smolagents.memory import ActionStep, AgentMemory
    from smolagents.monitoring import TokenUsage, Timing
    from unittest.mock import Mock, patch
    import time

    mock_agent = Mock()
    mock_agent.memory = AgentMemory(system_prompt="Test")
    mock_agent.model.model_id = "some-model"

    start = time.time()
    step = ActionStep(step_number=1, timing=Timing(start_time=start, end_time=start + 0.5), observations_images=[])
    step.token_usage = TokenUsage(input_tokens=100, output_tokens=50)
    mock_agent.memory.steps.append(step)

    # Patch LiteLLMCostCalculator to simulate litellm not installed
    with patch("maseval.interface.agents.smolagents.SmolAgentAdapter._resolve_cost_calculator", return_value=None):
        adapter = SmolAgentAdapter(agent_instance=mock_agent, name="test")
        usage = adapter.gather_usage()

    assert isinstance(usage, MasevalTokenUsage)
    assert usage.cost == 0.0
    assert usage.input_tokens == 100
