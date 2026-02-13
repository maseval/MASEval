"""Unit tests for Tau2User."""

import pytest
from unittest.mock import MagicMock
from maseval.benchmark.tau2 import Tau2User


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.mark.benchmark
def test_extract_user_profile():
    """Test extracting persona and profile from scenario."""
    scenario = "Persona: Angry Customer\n\nTask: Return item."

    profile = Tau2User._extract_user_profile(scenario)

    assert profile["persona"] == "Angry Customer"
    assert profile["full_scenario"] == scenario


@pytest.mark.benchmark
def test_extract_user_profile_no_persona():
    """Test extracting profile when no persona is present."""
    scenario = "Just a task."

    profile = Tau2User._extract_user_profile(scenario)

    assert "persona" not in profile
    assert profile["full_scenario"] == scenario


@pytest.mark.benchmark
def test_get_tool_raises(mock_model):
    """Test that base Tau2User raises NotImplementedError for get_tool."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="hi")

    with pytest.raises(NotImplementedError):
        user.get_tool()


@pytest.mark.benchmark
def test_init_passes_tools(mock_model):
    """Test that tools are correctly passed to parent AgenticUser."""
    tools = {"test_tool": lambda x: x}

    user = Tau2User(model=mock_model, scenario="test", initial_query="hi", tools=tools)

    assert user.tools == tools


@pytest.mark.benchmark
def test_user_has_initial_query(mock_model):
    """Test that user has initial query method."""
    user = Tau2User(model=mock_model, scenario="test", initial_query="Hello!")

    # Check that get_initial_query method returns the query
    assert user.get_initial_query() == "Hello!"


@pytest.mark.benchmark
def test_user_has_scenario(mock_model):
    """Test that user scenario is set."""
    scenario = "You are a customer who needs help."

    user = Tau2User(model=mock_model, scenario=scenario, initial_query="hi")

    # Check that scenario is accessible
    assert hasattr(user, "scenario") or hasattr(user, "_scenario")


@pytest.mark.benchmark
def test_extract_profile_with_task():
    """Test extracting profile with Task field."""
    scenario = "Persona: Helpful Customer\n\nTask: Buy a new phone."

    profile = Tau2User._extract_user_profile(scenario)

    assert profile["persona"] == "Helpful Customer"
    assert "full_scenario" in profile


@pytest.mark.benchmark
def test_user_profile_initialization(mock_model):
    """Test that user profile is initialized correctly."""
    scenario = "Persona: Test User\n\nTask: Test task."

    user = Tau2User(model=mock_model, scenario=scenario, initial_query="hi")

    # User profile should be extracted from scenario
    assert user.user_profile is not None
    assert isinstance(user.user_profile, dict)


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
    # Should have expected trace fields
    assert "type" in traces
    assert "messages" in traces
    assert "message_count" in traces
    assert traces["type"] == "Tau2User"
    assert traces["message_count"] == 1


@pytest.mark.benchmark
class TestTau2UserScenarios:
    """Tests for various Tau2User scenario handling."""

    def test_multiline_persona(self, mock_model):
        """Test extracting persona from multiline scenario."""
        scenario = """Persona: Frustrated customer who has been waiting
for a long time

Task: Get a refund."""

        profile = Tau2User._extract_user_profile(scenario)

        # Should capture the first line after Persona:
        assert "Frustrated" in profile.get("persona", "")

    def test_scenario_with_context(self, mock_model):
        """Test scenario with additional context."""
        scenario = """Persona: Regular customer

Context: Customer has been with the company for 5 years.

Task: Upgrade their plan."""

        profile = Tau2User._extract_user_profile(scenario)

        assert profile["full_scenario"] == scenario

    def test_complex_scenario(self, mock_model):
        """Test complex real-world scenario."""
        scenario = """Persona: Jane, a busy professional

Background: Jane is a software engineer at a startup. She travels frequently
for work and needs reliable phone service.

Task: Jane wants to enable international roaming on her line before
her upcoming trip to Europe."""

        profile = Tau2User._extract_user_profile(scenario)

        assert "Jane" in profile.get("persona", "")
        assert profile["full_scenario"] == scenario
