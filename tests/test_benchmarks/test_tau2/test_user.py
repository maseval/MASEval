"""Unit tests for Tau2User."""

import pytest
from unittest.mock import MagicMock
from maseval.benchmark.tau2 import Tau2User


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.mark.benchmark
def test_init_basic(mock_model):
    """Test basic Tau2User initialization."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    assert user.scenario == "test"
    assert user.tools == {}
    assert user.tool_definitions is None
    assert user.llm_args == {}
    assert user.max_turns == 50


@pytest.mark.benchmark
def test_init_passes_tools(mock_model):
    """Test that tools are correctly stored."""
    tools = {"test_tool": lambda x: x}

    user = Tau2User(model=mock_model, scenario="test", initial_query="hi", tools=tools)

    assert user.tools == tools


@pytest.mark.benchmark
def test_user_has_initial_query(mock_model):
    """Test that user has initial query method."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="Hello!")

    assert user.get_initial_query() == "Hello!"


@pytest.mark.benchmark
def test_user_has_scenario(mock_model):
    """Test that user scenario is set."""
    scenario = "You are a customer who needs help."

    user = Tau2User(model=mock_model, scenario=scenario, initial_query="hi")

    assert user.scenario == scenario


@pytest.mark.benchmark
def test_user_empty_tools(mock_model):
    """Test that user works with empty tools dict."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi", tools={})

    assert user.tools == {}


@pytest.mark.benchmark
def test_gather_traces(mock_model):
    """Test that gather_traces returns expected structure."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    traces = user.gather_traces()

    assert isinstance(traces, dict)
    assert "type" in traces
    assert "messages" in traces
    assert "max_turns" in traces
    assert "turns_used" in traces
    assert "stopped_by_user" in traces
    assert traces["type"] == "Tau2User"
    assert traces["turns_used"] == 0
    assert traces["stopped_by_user"] is False


@pytest.mark.benchmark
def test_is_done_initially_false(mock_model):
    """Test that user is not done initially."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    assert user.is_done() is False


@pytest.mark.benchmark
def test_inject_greeting(mock_model):
    """Test that inject_greeting prepends an assistant message."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    user.inject_greeting("Hi! How can I help?")

    assert len(user._messages) == 1
    assert user._messages[0]["role"] == "assistant"
    assert user._messages[0]["content"] == "Hi! How can I help?"


@pytest.mark.benchmark
def test_stop_tokens_exact_case(mock_model):
    """Test that stop tokens use exact case matching."""
    assert Tau2User.STOP == "###STOP###"
    assert Tau2User.TRANSFER == "###TRANSFER###"
    assert Tau2User.OUT_OF_SCOPE == "###OUT-OF-SCOPE###"


@pytest.mark.benchmark
def test_system_prompt_contains_scenario(mock_model):
    """Test that system prompt includes scenario in <scenario> tags."""
    user = Tau2User(model=mock_model, scenario="My test scenario", initial_query="hi")

    assert "<scenario>" in user._system_prompt
    assert "My test scenario" in user._system_prompt
    assert "</scenario>" in user._system_prompt


@pytest.mark.benchmark
def test_guidelines_loaded_no_tools(mock_model):
    """Test that simulation_guidelines.md is loaded when no tools."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    # Should load the no-tools guidelines
    assert "User Simulation Guidelines" in user._system_prompt


@pytest.mark.benchmark
def test_guidelines_loaded_with_tools(mock_model):
    """Test that simulation_guidelines_tools.md is loaded when tools provided."""
    user = Tau2User(
        model=mock_model,
        scenario="test",
        initial_query="hi",
        tools={"my_tool": lambda: None},
    )

    # Should load the tools guidelines
    assert "User Simulation Guidelines" in user._system_prompt


@pytest.mark.benchmark
def test_llm_args_passed(mock_model):
    """Test that llm_args are stored."""
    user = Tau2User(
        model=mock_model,
        scenario="test",
        initial_query="hi",
        llm_args={"temperature": 0.0},
    )

    assert user.llm_args == {"temperature": 0.0}


@pytest.mark.benchmark
def test_max_turns_custom(mock_model):
    """Test custom max_turns."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi", max_turns=10)

    assert user.max_turns == 10


@pytest.mark.benchmark
class TestTau2UserScenarios:
    """Tests for various Tau2User scenario handling."""

    def test_scenario_stored_as_is(self, mock_model):
        """Test that scenario is stored verbatim."""
        scenario = "Persona:\n\tFrustrated customer\nInstructions:\n\tGet a refund."

        user = Tau2User(model=mock_model, scenario=scenario, initial_query="hi")

        assert user.scenario == scenario

    def test_scenario_in_system_prompt(self, mock_model):
        """Test scenario appears in system prompt."""
        scenario = "Instructions:\n\tTest instructions here"

        user = Tau2User(model=mock_model, scenario=scenario, initial_query="hi")

        assert scenario in user._system_prompt


# =============================================================================
# Tau2User.respond() Early Exit Tests — Lines 186-188
# =============================================================================


@pytest.mark.benchmark
class TestTau2UserRespondEarlyExit:
    """Tests for Tau2User.respond() early exit conditions."""

    def test_respond_after_stopped_raises(self, mock_model):
        """respond() raises UserExhaustedError when already stopped."""
        from maseval.core.exceptions import UserExhaustedError

        user = Tau2User(model=mock_model, scenario="test", initial_query="hi")
        user._stopped = True
        with pytest.raises(UserExhaustedError):
            user.respond("anything")
        assert user._last_respond_steps == 0

    def test_respond_over_max_turns_raises(self, mock_model):
        """respond() raises UserExhaustedError when max_turns exceeded."""
        from maseval.core.exceptions import UserExhaustedError

        user = Tau2User(model=mock_model, scenario="test", initial_query="hi", max_turns=1)
        user._turn_count = 2
        with pytest.raises(UserExhaustedError):
            user.respond("anything")
        assert user._stopped is True

    def test_respond_after_stopped_returns_exhausted_response(self, mock_model):
        """respond() returns exhausted_response when stopped and configured."""
        user = Tau2User(
            model=mock_model,
            scenario="test",
            initial_query="hi",
            exhausted_response="User done.",
        )
        user._stopped = True
        result = user.respond("anything")
        assert result == "User done."
        assert user._last_respond_steps == 0


# =============================================================================
# Tau2User._generate_response() Tool Call Flow — Lines 229-250
# =============================================================================


@pytest.mark.benchmark
class TestTau2UserToolCallFlow:
    """Tests for Tau2User tool call execution in _generate_response()."""

    def test_tool_call_executed_and_result_stored(self):
        """Tool calls execute, results stored, then text generated."""
        mock_model = MagicMock()
        # First call returns tool call, second returns text
        mock_model.chat.side_effect = [
            MagicMock(content="", tool_calls=[{"id": "tc1", "function": {"name": "pay_bill", "arguments": '{"bill_id": "b1"}'}}]),
            MagicMock(content="Payment done", tool_calls=[]),
        ]
        user = Tau2User(
            model=mock_model,
            scenario="test",
            initial_query="pay",
            tools={"pay_bill": lambda bill_id: "paid"},
            tool_definitions=[{"type": "function", "function": {"name": "pay_bill"}}],
        )
        result = user.respond("How can I help?")
        assert result == "Payment done"
        # Verify tool call and result in messages
        has_tc = any(m.get("tool_calls") for m in user._messages)
        has_tool_result = any(m.get("role") == "tool" for m in user._messages)
        assert has_tc
        assert has_tool_result

    def test_stop_token_sets_stopped(self):
        """Stop token in response sets _stopped flag."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(content="###STOP###", tool_calls=[])
        user = Tau2User(model=mock_model, scenario="test", initial_query="hi")
        user.respond("Hello")
        assert user.is_done()
        assert user._stopped

    def test_transfer_token_sets_stopped(self):
        """Transfer token in response sets _stopped flag."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(content="###TRANSFER###", tool_calls=[])
        user = Tau2User(model=mock_model, scenario="test", initial_query="hi")
        user.respond("Hello")
        assert user.is_done()

    def test_out_of_scope_token_sets_stopped(self):
        """Out-of-scope token in response sets _stopped flag."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(content="###OUT-OF-SCOPE###", tool_calls=[])
        user = Tau2User(model=mock_model, scenario="test", initial_query="hi")
        user.respond("Hello")
        assert user.is_done()

    def test_tool_not_found_returns_error(self):
        """Tool not found in tools dict returns error message."""
        mock_model = MagicMock()
        mock_model.chat.side_effect = [
            MagicMock(content="", tool_calls=[{"id": "tc1", "function": {"name": "missing_tool", "arguments": "{}"}}]),
            MagicMock(content="ok", tool_calls=[]),
        ]
        user = Tau2User(
            model=mock_model,
            scenario="test",
            initial_query="hi",
            tools={"other_tool": lambda: None},
            tool_definitions=[{"type": "function", "function": {"name": "missing_tool"}}],
        )
        user.respond("go")
        # Should not crash; tool result should contain error
        tool_msgs = [m for m in user._messages if m.get("role") == "tool"]
        assert any("Error" in m.get("content", "") for m in tool_msgs)

    def test_tool_execution_error_captured(self):
        """Tool execution exception captured as error message."""
        mock_model = MagicMock()
        mock_model.chat.side_effect = [
            MagicMock(content="", tool_calls=[{"id": "tc1", "function": {"name": "bad_tool", "arguments": "{}"}}]),
            MagicMock(content="ok", tool_calls=[]),
        ]

        def bad_tool():
            raise RuntimeError("tool broke")

        user = Tau2User(
            model=mock_model,
            scenario="test",
            initial_query="hi",
            tools={"bad_tool": bad_tool},
            tool_definitions=[{"type": "function", "function": {"name": "bad_tool"}}],
        )
        user.respond("go")
        tool_msgs = [m for m in user._messages if m.get("role") == "tool"]
        assert any("Error" in m.get("content", "") for m in tool_msgs)

    def test_step_counting(self):
        """_last_respond_steps tracks messages added during respond()."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(content="Hello user", tool_calls=[])
        user = Tau2User(model=mock_model, scenario="test", initial_query="hi")
        user.respond("agent message")
        # Should have added: assistant msg (from respond) + user msg (from _generate_response) = 2
        assert user._last_respond_steps >= 1


# =============================================================================
# Tau2User._flip_roles() Tests — Lines 282, 294-297
# =============================================================================


@pytest.mark.benchmark
class TestFlipRoles:
    """Tests for Tau2User._flip_roles() message role transformation."""

    def test_user_becomes_assistant(self):
        """User messages become assistant messages (role flip)."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "user", "content": "I need help"}]
        flipped = user._flip_roles()
        assert flipped[0]["role"] == "assistant"
        assert flipped[0]["content"] == "I need help"

    def test_assistant_becomes_user(self):
        """Assistant messages become user messages (role flip)."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "assistant", "content": "How can I help?"}]
        flipped = user._flip_roles()
        assert flipped[0]["role"] == "user"
        assert flipped[0]["content"] == "How can I help?"

    def test_user_with_tool_calls_becomes_assistant_with_tools(self):
        """User messages with tool_calls flip to assistant with tool_calls preserved."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "user", "content": "", "tool_calls": [{"name": "f", "id": "tc1"}]}]
        flipped = user._flip_roles()
        assert flipped[0]["role"] == "assistant"
        assert "tool_calls" in flipped[0]

    def test_assistant_with_tool_calls_skipped(self):
        """Assistant messages with tool_calls are skipped (original behavior)."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "assistant", "content": "x", "tool_calls": [{"name": "f"}]}]
        flipped = user._flip_roles()
        assert len(flipped) == 0

    def test_tool_message_user_requestor_kept(self):
        """Tool messages with requestor=user are kept."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "tool", "requestor": "user", "tool_call_id": "tc1", "content": "result"}]
        flipped = user._flip_roles()
        assert len(flipped) == 1
        assert flipped[0]["role"] == "tool"

    def test_tool_message_assistant_requestor_skipped(self):
        """Tool messages with requestor=assistant are skipped."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [{"role": "tool", "requestor": "assistant", "content": "result"}]
        flipped = user._flip_roles()
        assert len(flipped) == 0

    def test_mixed_messages(self):
        """Complex mixed message sequence flips correctly."""
        user = Tau2User(model=MagicMock(), scenario="test", initial_query="hi")
        user._messages = [
            {"role": "assistant", "content": "Hi!"},  # → user
            {"role": "user", "content": "Help"},  # → assistant
            {"role": "assistant", "content": "", "tool_calls": [{"name": "f"}]},  # → skipped
            {"role": "tool", "requestor": "assistant", "content": "r"},  # → skipped
            {"role": "user", "content": "", "tool_calls": [{"name": "g"}]},  # → assistant with tc
            {"role": "tool", "requestor": "user", "tool_call_id": "tc2", "content": "r2"},  # → kept
        ]
        flipped = user._flip_roles()
        assert len(flipped) == 4
        assert flipped[0]["role"] == "user"
        assert flipped[1]["role"] == "assistant"
        assert flipped[2]["role"] == "assistant"
        assert "tool_calls" in flipped[2]
        assert flipped[3]["role"] == "tool"
