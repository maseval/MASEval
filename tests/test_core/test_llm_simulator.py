"""Test LLM Simulator functionality.

These tests verify that LLMSimulator retry logic and tracing work correctly.
"""

import pytest
from maseval.core.simulator import (
    ToolLLMSimulator,
    SimulatorCallStatus,
    ToolSimulatorError,
)


@pytest.mark.core
class TestLLMSimulator:
    """Tests for LLMSimulator retry and tracing."""

    def test_llm_simulator_retry_logic(self, dummy_model):
        """Test that simulator retries on parsing errors."""
        # Model returns invalid JSON first, then valid
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(
            responses=[
                "invalid json",
                '{"text": "Tool executed successfully", "details": {"result": "success"}}',
            ]
        )

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        result = simulator(actual_inputs={"param": "test"})

        # Should eventually succeed - ToolLLMSimulator returns (text, details) tuple
        assert result is not None
        assert isinstance(result, tuple)
        text, details = result
        assert isinstance(details, dict)
        assert details.get("result") == "success"

        # Should have 2 attempts captured in logs (1 fail, 1 success)
        assert len(simulator.logs) == 2

    def test_llm_simulator_parsing_error_retry(self, dummy_model):
        """Test that parsing errors trigger retries and raise SimulatorError on exhaustion."""
        from conftest import DummyModelAdapter

        # All responses are invalid JSON
        model = DummyModelAdapter(responses=["bad", "bad", "bad"])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        # Should raise ToolSimulatorError after max_try attempts
        with pytest.raises(ToolSimulatorError) as exc_info:
            simulator(actual_inputs={"param": "test"})

        # Verify exception details
        err = exc_info.value
        assert err.attempts == 3
        assert err.last_error is not None
        assert len(err.logs) == 3  # All 3 attempts in exception logs
        assert len(simulator.logs) == 3  # All 3 attempts logged in simulator

    def test_llm_simulator_max_attempts_respected(self, dummy_model):
        """Test that max_try limit is respected."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["invalid"] * 10)

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=2,  # Only allow 2 attempts
        )

        # Should raise after 2 attempts
        with pytest.raises(ToolSimulatorError) as exc_info:
            simulator(actual_inputs={"param": "test"})

        # Should stop after 2 attempts, not continue to 10
        err = exc_info.value
        assert len(simulator.logs) == 2
        assert err.attempts == 2

    def test_llm_simulator_history_structure(self, dummy_model):
        """Test that history entries have correct structure."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        # Check log entry structure
        entry = simulator.logs[0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "input" in entry
        assert "prompt" in entry
        assert "raw_output" in entry
        assert "parsed_output" in entry
        assert "status" in entry

    def test_llm_simulator_status_tracking(self, dummy_model):
        """Test that status is correctly tracked."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        entry = simulator.logs[0]
        assert entry["status"] == SimulatorCallStatus.Successful.value

    def test_llm_simulator_gather_traces(self, dummy_model):
        """Test that gather_traces includes complete history."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

        traces = simulator.gather_traces()

        assert "simulator_type" in traces
        assert "total_calls" in traces
        assert "successful_calls" in traces
        assert "failed_calls" in traces
        assert "logs" in traces
        assert traces["successful_calls"] == 1


@pytest.mark.core
class TestUserLLMSimulatorValidation:
    """Tests for UserLLMSimulator early stopping validation."""

    def test_stop_token_without_condition_raises(self, dummy_model):
        """ValueError raised when stop_token set but early_stopping_condition is None."""
        from maseval.core.simulator import UserLLMSimulator

        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                stop_token="</stop>",
            )

    def test_condition_without_stop_token_raises(self, dummy_model):
        """ValueError raised when early_stopping_condition set but stop_token is None."""
        from maseval.core.simulator import UserLLMSimulator

        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                early_stopping_condition="goals are met",
            )

    def test_both_none_is_valid(self, dummy_model):
        """No error when both stop_token and early_stopping_condition are None."""
        from maseval.core.simulator import UserLLMSimulator

        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
        )
        assert simulator.stop_token is None
        assert simulator.early_stopping_condition is None

    def test_both_set_is_valid(self, dummy_model):
        """No error when both stop_token and early_stopping_condition are set."""
        from maseval.core.simulator import UserLLMSimulator

        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
            stop_token="</stop>",
            early_stopping_condition="all goals accomplished",
        )
        assert simulator.stop_token == "</stop>"
        assert simulator.early_stopping_condition == "all goals accomplished"


# =============================================================================
# UserLLMSimulator Response Tests
# =============================================================================


@pytest.mark.core
class TestUserLLMSimulatorResponse:
    """Tests for UserLLMSimulator response generation."""

    def test_user_simulator_generates_response(self, dummy_model):
        """UserLLMSimulator generates a response from conversation history."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import UserLLMSimulator

        # UserLLMSimulator expects JSON output with "text" field
        model = DummyModelAdapter(responses=['{"text": "I need help with my order."}'])

        simulator = UserLLMSimulator(
            model=model,
            user_profile={"name": "John", "issue": "order problem"},
            scenario="Customer calling about an order issue",
        )

        result = simulator(conversation_history=[{"role": "agent", "content": "How can I help?"}])

        assert result is not None
        assert isinstance(result, str)
        assert result == "I need help with my order."

    def test_user_simulator_fills_template(self, dummy_model):
        """UserLLMSimulator fills prompt template with profile and scenario."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import UserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "Test response"}'])

        simulator = UserLLMSimulator(
            model=model,
            user_profile={"name": "Jane", "account_id": "12345"},
            scenario="Account inquiry scenario",
        )

        # Call to trigger prompt filling
        simulator(conversation_history=[{"role": "agent", "content": "Hello"}])

        # Check that the prompt was filled (via logs)
        assert len(simulator.logs) > 0
        prompt = simulator.logs[0].get("prompt", "")
        assert "Jane" in prompt or "12345" in prompt or "Account inquiry" in prompt

    def test_user_simulator_with_early_stopping(self, dummy_model):
        """UserLLMSimulator includes early stopping instructions when configured."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import UserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "Thanks, goodbye! </end>"}'])

        simulator = UserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test scenario",
            stop_token="</end>",
            early_stopping_condition="issue is resolved",
        )

        result = simulator(conversation_history=[{"role": "agent", "content": "Your issue is fixed."}])

        assert result is not None
        # The prompt should include early stopping instructions
        prompt = simulator.logs[0].get("prompt", "")
        assert "</end>" in prompt or "issue is resolved" in prompt


# =============================================================================
# AgenticUserLLMSimulator Tests
# =============================================================================


@pytest.mark.core
class TestAgenticUserLLMSimulatorValidation:
    """Tests for AgenticUserLLMSimulator initialization and validation."""

    def test_agentic_user_simulator_initialization(self, dummy_model):
        """AgenticUserLLMSimulator initializes with required parameters."""
        from maseval.core.simulator import AgenticUserLLMSimulator

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test", "phone": "555-1234"},
            scenario="testing phone features",
        )

        assert simulator.user_profile == {"name": "test", "phone": "555-1234"}
        assert simulator.scenario == "testing phone features"
        assert simulator.tools == []

    def test_agentic_user_simulator_with_tools(self, dummy_model):
        """AgenticUserLLMSimulator initializes with tools."""
        from maseval.core.simulator import AgenticUserLLMSimulator

        tools = [
            {"name": "check_balance", "description": "Check account balance", "inputs": {}},
            {"name": "make_payment", "description": "Make a payment", "inputs": {"amount": {"type": "number"}}},
        ]

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="payment scenario",
            tools=tools,
        )

        assert len(simulator.tools) == 2
        assert simulator.tools[0]["name"] == "check_balance"

    def test_agentic_user_stop_token_validation(self, dummy_model):
        """AgenticUserLLMSimulator validates stop_token and early_stopping_condition."""
        from maseval.core.simulator import AgenticUserLLMSimulator

        # stop_token without condition should raise
        with pytest.raises(ValueError, match="must both be set or both be None"):
            AgenticUserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test",
                stop_token="</stop>",
            )

        # condition without stop_token should raise
        with pytest.raises(ValueError, match="must both be set or both be None"):
            AgenticUserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test",
                early_stopping_condition="done",
            )

    def test_agentic_user_both_early_stopping_params_valid(self, dummy_model):
        """AgenticUserLLMSimulator accepts both early stopping params together."""
        from maseval.core.simulator import AgenticUserLLMSimulator

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test",
            stop_token="</done>",
            early_stopping_condition="task completed",
        )

        assert simulator.stop_token == "</done>"
        assert simulator.early_stopping_condition == "task completed"


@pytest.mark.core
class TestAgenticUserLLMSimulatorResponse:
    """Tests for AgenticUserLLMSimulator response generation."""

    def test_agentic_user_generates_text_response(self, dummy_model):
        """AgenticUserLLMSimulator generates text response from JSON output."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "I need to check my balance.", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "John"},
            scenario="Account inquiry",
        )
        simulator.model = model  # Override with our test model

        result = simulator(conversation_history=[{"role": "agent", "content": "How can I help?"}])

        assert isinstance(result, tuple)
        text, tool_calls = result
        assert text == "I need to check my balance."
        assert tool_calls == []

    def test_agentic_user_generates_tool_calls(self, dummy_model):
        """AgenticUserLLMSimulator generates tool calls from JSON output."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "Let me check.", "tool_calls": [{"name": "check_signal", "arguments": {}}]}'])

        tools = [{"name": "check_signal", "description": "Check phone signal"}]

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Jane"},
            scenario="Phone issue",
            tools=tools,
        )
        simulator.model = model

        result = simulator(conversation_history=[{"role": "agent", "content": "What's the problem?"}])

        text, tool_calls = result
        assert text == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "check_signal"

    def test_agentic_user_parses_markdown_json(self, dummy_model):
        """AgenticUserLLMSimulator parses JSON wrapped in markdown code blocks."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        # Response wrapped in markdown code block
        model = DummyModelAdapter(responses=['```json\n{"text": "Parsed correctly", "tool_calls": []}\n```'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
        )
        simulator.model = model

        result = simulator(conversation_history=[])

        text, tool_calls = result
        assert text == "Parsed correctly"

    def test_agentic_user_invalid_json_raises(self, dummy_model):
        """AgenticUserLLMSimulator raises on invalid JSON after retries."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator, UserSimulatorError

        model = DummyModelAdapter(responses=["not valid json", "still not valid", "nope"])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
            max_try=3,
        )
        simulator.model = model

        with pytest.raises(UserSimulatorError) as exc_info:
            simulator(conversation_history=[])

        assert exc_info.value.attempts == 3


@pytest.mark.core
class TestAgenticUserLLMSimulatorPrompt:
    """Tests for AgenticUserLLMSimulator prompt template filling."""

    def test_prompt_includes_user_profile(self, dummy_model):
        """Prompt template includes user profile information."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Alice", "customer_id": "C12345"},
            scenario="Customer support call",
        )
        simulator.model = model

        simulator(conversation_history=[{"role": "agent", "content": "Hello"}])

        prompt = simulator.logs[0].get("prompt", "")
        assert "Alice" in prompt or "C12345" in prompt

    def test_prompt_includes_scenario(self, dummy_model):
        """Prompt template includes scenario description."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Billing dispute about overcharges",
        )
        simulator.model = model

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "Billing dispute" in prompt or "overcharges" in prompt

    def test_prompt_includes_tool_instructions(self, dummy_model):
        """Prompt template includes tool instructions when tools are provided."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        tools = [
            {"name": "toggle_wifi", "description": "Toggle WiFi on/off", "inputs": {"enabled": {"type": "boolean"}}},
        ]

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
            tools=tools,
        )
        simulator.model = model

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "toggle_wifi" in prompt or "Toggle WiFi" in prompt

    def test_prompt_includes_early_stopping_instructions(self, dummy_model):
        """Prompt template includes early stopping instructions when configured."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
            stop_token="</complete>",
            early_stopping_condition="problem is solved",
        )
        simulator.model = model

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "</complete>" in prompt or "problem is solved" in prompt

    def test_prompt_includes_conversation_history(self, dummy_model):
        """Prompt template includes formatted conversation history."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
        )
        simulator.model = model

        history = [
            {"role": "agent", "content": "Welcome to support."},
            {"role": "user", "content": "I have a problem."},
            {"role": "agent", "content": "What's the issue?"},
        ]

        simulator(conversation_history=history)

        prompt = simulator.logs[0].get("prompt", "")
        assert "Welcome to support" in prompt or "I have a problem" in prompt


@pytest.mark.core
class TestAgenticUserLLMSimulatorTracing:
    """Tests for AgenticUserLLMSimulator tracing and logging."""

    def test_logs_successful_calls(self, dummy_model):
        """AgenticUserLLMSimulator logs successful calls."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator, SimulatorCallStatus

        model = DummyModelAdapter(responses=['{"text": "Success", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
        )
        simulator.model = model

        simulator(conversation_history=[])

        assert len(simulator.logs) == 1
        assert simulator.logs[0]["status"] == SimulatorCallStatus.Successful.value

    def test_logs_failed_calls(self, dummy_model):
        """AgenticUserLLMSimulator logs failed parsing attempts."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator, SimulatorCallStatus, UserSimulatorError

        model = DummyModelAdapter(responses=["bad json", "still bad"])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
            max_try=2,
        )
        simulator.model = model

        with pytest.raises(UserSimulatorError):
            simulator(conversation_history=[])

        assert len(simulator.logs) == 2
        for log in simulator.logs:
            assert log["status"] == SimulatorCallStatus.ModelParsingError.value

    def test_gather_traces_returns_complete_info(self, dummy_model):
        """gather_traces returns complete tracing information."""
        from conftest import DummyModelAdapter
        from maseval.core.simulator import AgenticUserLLMSimulator

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "Test"},
            scenario="Test",
        )
        simulator.model = model

        simulator(conversation_history=[])

        traces = simulator.gather_traces()

        assert "simulator_type" in traces
        assert traces["simulator_type"] == "AgenticUserLLMSimulator"
        assert "total_calls" in traces
        assert traces["total_calls"] == 1
        assert "successful_calls" in traces
        assert traces["successful_calls"] == 1
