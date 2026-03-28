"""Test LLM Simulator functionality.

These tests verify that LLMSimulator structured output and tracing work correctly.
Simulators use instructor for structured output via response_model, so retries
happen inside the model adapter's _structured_chat method, not in the simulator loop.
"""

import pytest
from maseval.core.simulator import (
    ToolLLMSimulator,
    UserLLMSimulator,
    AgenticUserLLMSimulator,
    SimulatorCallStatus,
    ToolSimulatorError,
    UserSimulatorError,
    ToolSimulatorResponse,
    UserSimulatorResponse,
    AgenticUserSimulatorResponse,
)


@pytest.mark.core
class TestSimulatorResponseModels:
    """Test that simulator response Pydantic models work correctly."""

    def test_tool_simulator_response_model(self):
        resp = ToolSimulatorResponse(text="success", details={"key": "value"})
        assert resp.text == "success"
        assert resp.details == {"key": "value"}

    def test_user_simulator_response_model(self):
        resp = UserSimulatorResponse(text="I need help")
        assert resp.text == "I need help"

    def test_agentic_user_simulator_response_model(self):
        resp = AgenticUserSimulatorResponse(
            text="Let me check",
            tool_calls=[{"name": "check_status", "arguments": {}}],
        )
        assert resp.text == "Let me check"
        assert len(resp.tool_calls) == 1


@pytest.mark.core
class TestLLMSimulator:
    """Tests for LLMSimulator structured output and tracing."""

    def test_llm_simulator_success(self, dummy_model):
        """Test that simulator succeeds with valid JSON."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "Tool executed successfully", "details": {"result": "success"}}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        result = simulator(actual_inputs={"param": "test"})

        assert result is not None
        assert isinstance(result, tuple)
        text, details = result
        assert text == "Tool executed successfully"
        assert details.get("result") == "success"
        assert len(simulator.logs) == 1
        assert simulator.logs[0]["status"] == SimulatorCallStatus.Successful.value

    def test_llm_simulator_retry_logic(self, dummy_model):
        """Test that instructor retries on invalid JSON and eventually succeeds."""
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

        # Should succeed after instructor retries internally
        assert result is not None
        assert isinstance(result, tuple)
        text, details = result
        assert details.get("result") == "success"

        # Simulator sees 1 successful call (retries are internal to _structured_chat)
        assert len(simulator.logs) == 1

    def test_llm_simulator_parsing_error_raises(self, dummy_model):
        """Test that all-invalid JSON raises SimulatorError after retries."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["bad", "bad", "bad"])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        with pytest.raises(ToolSimulatorError):
            simulator(actual_inputs={"param": "test"})

        # Simulator logs 1 entry (the failed call)
        assert len(simulator.logs) == 1
        assert simulator.logs[0]["status"] == SimulatorCallStatus.ModelCallError.value

    def test_llm_simulator_history_structure(self, dummy_model):
        """Test that history entries have correct structure."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "ok", "details": {}}'])

        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string"}},
            max_try=3,
        )

        _ = simulator(actual_inputs={"param": "test"})

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

        model = DummyModelAdapter(responses=['{"text": "ok", "details": {}}'])

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

        model = DummyModelAdapter(responses=['{"text": "ok", "details": {}}'])

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
        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                stop_token="</stop>",
            )

    def test_condition_without_stop_token_raises(self, dummy_model):
        """ValueError raised when early_stopping_condition set but stop_token is None."""
        with pytest.raises(ValueError, match="must both be set or both be None"):
            UserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test scenario",
                early_stopping_condition="goals are met",
            )

    def test_both_none_is_valid(self, dummy_model):
        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
        )
        assert simulator.stop_token is None
        assert simulator.early_stopping_condition is None

    def test_both_set_is_valid(self, dummy_model):
        simulator = UserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test"},
            scenario="test scenario",
            stop_token="</stop>",
            early_stopping_condition="all goals accomplished",
        )
        assert simulator.stop_token == "</stop>"
        assert simulator.early_stopping_condition == "all goals accomplished"


@pytest.mark.core
class TestUserLLMSimulatorResponse:
    """Tests for UserLLMSimulator response generation."""

    def test_user_simulator_generates_response(self, dummy_model):
        from conftest import DummyModelAdapter

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
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "Test response"}'])

        simulator = UserLLMSimulator(
            model=model,
            user_profile={"name": "Jane", "account_id": "12345"},
            scenario="Account inquiry scenario",
        )

        simulator(conversation_history=[{"role": "agent", "content": "Hello"}])

        assert len(simulator.logs) > 0
        prompt = simulator.logs[0].get("prompt", "")
        assert "Jane" in prompt or "12345" in prompt or "Account inquiry" in prompt

    def test_user_simulator_with_early_stopping(self, dummy_model):
        from conftest import DummyModelAdapter

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
        prompt = simulator.logs[0].get("prompt", "")
        assert "</end>" in prompt or "issue is resolved" in prompt


# =============================================================================
# AgenticUserLLMSimulator Tests
# =============================================================================


@pytest.mark.core
class TestAgenticUserLLMSimulatorValidation:
    """Tests for AgenticUserLLMSimulator initialization and validation."""

    def test_agentic_user_simulator_initialization(self, dummy_model):
        simulator = AgenticUserLLMSimulator(
            model=dummy_model,
            user_profile={"name": "test", "phone": "555-1234"},
            scenario="testing phone features",
        )

        assert simulator.user_profile == {"name": "test", "phone": "555-1234"}
        assert simulator.scenario == "testing phone features"
        assert simulator.tools == []

    def test_agentic_user_simulator_with_tools(self, dummy_model):
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
        with pytest.raises(ValueError, match="must both be set or both be None"):
            AgenticUserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test",
                stop_token="</stop>",
            )

        with pytest.raises(ValueError, match="must both be set or both be None"):
            AgenticUserLLMSimulator(
                model=dummy_model,
                user_profile={"name": "test"},
                scenario="test",
                early_stopping_condition="done",
            )

    def test_agentic_user_both_early_stopping_params_valid(self, dummy_model):
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
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "I need to check my balance.", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "John"},
            scenario="Account inquiry",
        )

        result = simulator(conversation_history=[{"role": "agent", "content": "How can I help?"}])

        assert isinstance(result, tuple)
        text, tool_calls = result
        assert text == "I need to check my balance."
        assert tool_calls == []

    def test_agentic_user_generates_tool_calls(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "Let me check.", "tool_calls": [{"name": "check_signal", "arguments": {}}]}'])

        tools = [{"name": "check_signal", "description": "Check phone signal"}]

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Jane"},
            scenario="Phone issue",
            tools=tools,
        )

        result = simulator(conversation_history=[{"role": "agent", "content": "What's the problem?"}])

        text, tool_calls = result
        assert text == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "check_signal"

    def test_agentic_user_invalid_json_raises(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["not valid json", "still not valid", "nope"])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
            max_try=3,
        )

        with pytest.raises(UserSimulatorError):
            simulator(conversation_history=[])


@pytest.mark.core
class TestAgenticUserLLMSimulatorPrompt:
    """Tests for AgenticUserLLMSimulator prompt template filling."""

    def test_prompt_includes_user_profile(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Alice", "customer_id": "C12345"},
            scenario="Customer support call",
        )

        simulator(conversation_history=[{"role": "agent", "content": "Hello"}])

        prompt = simulator.logs[0].get("prompt", "")
        assert "Alice" in prompt or "C12345" in prompt

    def test_prompt_includes_scenario(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Billing dispute about overcharges",
        )

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "Billing dispute" in prompt or "overcharges" in prompt

    def test_prompt_includes_tool_instructions(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        tools = [
            {"name": "toggle_wifi", "description": "Toggle WiFi on/off", "inputs": {"enabled": {"type": "boolean"}}},
        ]

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
            tools=tools,
        )

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "toggle_wifi" in prompt or "Toggle WiFi" in prompt

    def test_prompt_includes_early_stopping_instructions(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
            stop_token="</complete>",
            early_stopping_condition="problem is solved",
        )

        simulator(conversation_history=[])

        prompt = simulator.logs[0].get("prompt", "")
        assert "</complete>" in prompt or "problem is solved" in prompt

    def test_prompt_includes_conversation_history(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
        )

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
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "Success", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
        )

        simulator(conversation_history=[])

        assert len(simulator.logs) == 1
        assert simulator.logs[0]["status"] == SimulatorCallStatus.Successful.value

    def test_logs_failed_calls(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["bad json", "still bad"])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
            max_try=2,
        )

        with pytest.raises(UserSimulatorError):
            simulator(conversation_history=[])

        # With instructor-based retries, the simulator logs 1 entry
        # (retries happen inside _structured_chat)
        assert len(simulator.logs) == 1
        assert simulator.logs[0]["status"] == SimulatorCallStatus.ModelCallError.value

    def test_gather_traces_returns_complete_info(self, dummy_model):
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"text": "test", "tool_calls": []}'])

        simulator = AgenticUserLLMSimulator(
            model=model,
            user_profile={"name": "Test"},
            scenario="Test",
        )

        simulator(conversation_history=[])

        traces = simulator.gather_traces()

        assert "simulator_type" in traces
        assert traces["simulator_type"] == "AgenticUserLLMSimulator"
        assert "total_calls" in traces
        assert traces["total_calls"] == 1
        assert "successful_calls" in traces
        assert traces["successful_calls"] == 1
