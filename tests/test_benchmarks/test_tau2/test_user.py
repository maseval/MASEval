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
