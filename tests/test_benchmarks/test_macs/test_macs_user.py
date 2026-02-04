"""Unit tests for MACSUser."""

import pytest
from unittest.mock import MagicMock

from maseval.benchmark.macs import MACSUser

from conftest import DummyModelAdapter


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.benchmark
class TestMACSUserInit:
    """Tests for MACSUser initialization."""

    def test_init_with_defaults(self, macs_model, sample_scenario, initial_query):
        """Initialization with required args uses proper defaults."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        assert user.model == macs_model
        assert user.scenario == sample_scenario
        assert user.name == "Simulated User"
        assert user.max_turns == 5
        assert user._turn_count == 1  # Initial query counts as first turn
        assert not user._stopped
        assert "full_scenario" in user.user_profile

    def test_macs_default_max_turns_is_five(self, macs_model, sample_scenario, initial_query):
        """MACS benchmark defaults to max_turns=5 per MACS paper.

        This is a MACS-specific default that differs from the base class default of 1.
        If the base class default changes, this test ensures MACS maintains its value.
        """
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        assert user.max_turns == MACSUser.DEFAULT_MAX_TURNS
        assert user.max_turns == 5

    def test_macs_default_stop_tokens(self, macs_model, sample_scenario, initial_query):
        """MACS uses ['</stop>'] as stop tokens per MACS paper.

        This is a MACS-specific default. If the base class default changes,
        this test ensures MACS maintains its value.
        """
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        assert user.stop_tokens == MACSUser.DEFAULT_STOP_TOKENS
        assert user.stop_tokens == ["</stop>"]

    def test_init_with_custom_params(self, macs_model, sample_scenario, initial_query):
        """Custom name and max_turns are respected."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
            name="Test User",
            max_turns=10,
        )

        assert user.name == "Test User"
        assert user.max_turns == 10

    def test_init_uses_base_template_with_early_stopping(self, macs_model, sample_scenario, initial_query):
        """Uses base UserLLMSimulator template with early stopping configured."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )
        # Verify early stopping is configured
        assert user.stop_tokens == MACSUser.DEFAULT_STOP_TOKENS
        assert user.early_stopping_condition == MACSUser.DEFAULT_EARLY_STOPPING_CONDITION
        # Simulator should have the same config (uses first token from list)
        assert user.simulator.stop_token == "</stop>"
        assert user.simulator.early_stopping_condition is not None
        assert "goals have been satisfactorily addressed" in user.simulator.early_stopping_condition


# =============================================================================
# Unit Tests: User Profile Extraction
# =============================================================================


@pytest.mark.benchmark
class TestUserProfileExtraction:
    """Tests for _extract_user_profile static method."""

    def test_extract_profile_with_background(self, sample_scenario):
        """Parses Background: section."""
        profile = MACSUser._extract_user_profile(sample_scenario)

        assert "full_scenario" in profile
        assert profile["full_scenario"] == sample_scenario

    def test_extract_profile_is_statements(self):
        """Parses 'User's X is Y' statements."""
        scenario = """Background:
* User's name is Alice
* User's age is 30
* User's company is TechCorp"""

        profile = MACSUser._extract_user_profile(scenario)

        assert profile.get("name") == "Alice"
        assert profile.get("age") == "30"
        assert profile.get("company") == "TechCorp"

    def test_extract_profile_has_statements(self):
        """Parses 'User has X' statements."""
        scenario = """Background:
* User has a pet dog
* User has premium membership"""

        profile = MACSUser._extract_user_profile(scenario)

        # These should be captured with some key
        assert "full_scenario" in profile  # At minimum, full scenario is always there

    def test_extract_profile_no_background(self, minimal_scenario):
        """Handles missing Background section."""
        profile = MACSUser._extract_user_profile(minimal_scenario)

        # Should still have full_scenario
        assert profile["full_scenario"] == minimal_scenario

    def test_extract_profile_includes_full_scenario(self, sample_scenario):
        """Full scenario is always in profile."""
        profile = MACSUser._extract_user_profile(sample_scenario)

        assert "full_scenario" in profile
        assert sample_scenario in profile["full_scenario"]


# =============================================================================
# Unit Tests: Conversation State
# =============================================================================


@pytest.mark.benchmark
class TestConversationState:
    """Tests for conversation state management."""

    def test_is_done_false_initially_without_assistant_message(self, macs_model, sample_scenario, initial_query):
        """is_done() returns False when no assistant message to evaluate.

        When there's no assistant message yet (only the initial user message),
        there's nothing to evaluate for satisfaction. The loop should continue
        to get an agent response first.
        """
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        # No assistant message yet, so is_done() returns False
        # (nothing to evaluate, need to get agent response first)
        assert not user.is_done()

    def test_is_done_after_max_turns(self, macs_model, sample_scenario, initial_query):
        """is_done() returns True after max turns."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=2,
        )

        # Manually increment turn count
        user._turn_count = 2

        assert user.is_done()

    def test_is_done_after_stop_token(self, macs_model, sample_scenario, initial_query):
        """is_done() returns True after </stop> detected."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        # Manually set stopped flag
        user._stopped = True

        assert user.is_done()

    def test_is_done_returns_false_when_not_satisfied(self, macs_model, sample_scenario, initial_query):
        """is_done() returns False when user is not satisfied with response."""
        from unittest.mock import MagicMock

        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=5,
        )

        # Mock the simulator to return a response without stop token
        user.simulator = MagicMock(return_value="I need more information.")

        # respond() calls simulator, increments turn count, and checks for stop token
        response = user.respond("Here is your flight info.")

        # The user's response should be added to messages
        # initial_query is turn 1, this respond is turn 2
        assert user._turn_count == 2
        assert "I need more information" in response

        # is_done() is a cheap state check - no </stop> token was found
        assert not user.is_done()


# =============================================================================
# Unit Tests: Reset
# =============================================================================


@pytest.mark.benchmark
class TestReset:
    """Tests for reset method."""

    def test_reset_clears_turn_count(self, macs_model, sample_scenario, initial_query):
        """reset() resets turn count to initial value."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )
        user._turn_count = 3

        user.reset()

        assert user._turn_count == 1  # Back to initial (counts initial query)

    def test_reset_clears_stopped(self, macs_model, sample_scenario, initial_query):
        """reset() clears stopped flag."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )
        user._stopped = True

        user.reset()

        assert not user._stopped


# =============================================================================
# Unit Tests: Response Simulation
# =============================================================================


@pytest.mark.benchmark
class TestResponseSimulation:
    """Tests for respond method."""

    def test_respond_increments_turn(self, sample_scenario, initial_query):
        """Turn count increments on respond call."""
        model = DummyModelAdapter(responses=['{"text": "Yes, confirmed.", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        initial_count = user._turn_count

        # Replace the simulator with a mock that returns a controlled response
        user.simulator = MagicMock(return_value="Yes, confirmed.")
        user.respond("When would you like to travel?")

        assert user._turn_count == initial_count + 1

    def test_respond_detects_stop(self, sample_scenario, initial_query):
        """Detects </stop> token."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        # Replace the simulator with a mock that returns a response with stop token
        user.simulator = MagicMock(return_value="Thanks! </stop>")
        user.respond("Your flight is booked!")

        assert user._stopped
        assert user.is_done()

    def test_respond_cleans_stop_token(self, sample_scenario, initial_query):
        """Removes </stop> from response."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        # Replace the simulator with a mock that returns a response with stop token
        user.simulator = MagicMock(return_value="Perfect, thanks! </stop>")
        response = user.respond("Booking confirmed!")

        assert "</stop>" not in response
        assert "Perfect, thanks!" in response

    def test_respond_returns_empty_when_done(self, sample_scenario, initial_query):
        """Returns empty string when is_done is True."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )
        user._stopped = True  # Already done

        response = user.respond("Any follow-up?")

        assert response == ""

    def test_respond_returns_empty_at_max_turns(self, sample_scenario, initial_query):
        """Returns empty string when max turns reached."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=3,
        )
        user._turn_count = 3  # At max

        response = user.respond("One more question?")

        assert response == ""

    def test_respond_fallback_message(self, sample_scenario, initial_query):
        """Provides fallback when response is only stop token."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        # Replace the simulator with a mock that returns only the stop token
        user.simulator = MagicMock(return_value="</stop>")
        response = user.respond("Booking complete!")

        # When response is only stop token, base class provides fallback message
        assert response == "Thank you, that's all I needed!"
        assert user._stopped


# =============================================================================
# Unit Tests: Tool Interface
# =============================================================================


@pytest.mark.benchmark
class TestToolInterface:
    """Tests for get_tool method."""

    def test_get_tool_raises_not_implemented(self, macs_model, sample_scenario, initial_query):
        """Base MACSUser.get_tool() raises NotImplementedError."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            user.get_tool()

        assert "get_tool" in str(exc_info.value)
        assert "subclass" in str(exc_info.value).lower()


# =============================================================================
# Unit Tests: Tracing
# =============================================================================


@pytest.mark.benchmark
class TestTracing:
    """Tests for gather_traces method."""

    def test_gather_traces_includes_macs_fields(self, macs_model, sample_scenario, initial_query):
        """Traces include MACS-specific fields."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=7,
        )
        user._turn_count = 3
        user._stopped = True

        traces = user.gather_traces()

        assert traces["max_turns"] == 7
        assert traces["turns_used"] == 3
        assert traces["stopped_by_user"] is True

    def test_gather_traces_inherits_base_fields(self, macs_model, sample_scenario, initial_query):
        """Traces include base User fields."""
        user = MACSUser(
            model=macs_model,
            scenario=sample_scenario,
            initial_query=initial_query,
        )

        traces = user.gather_traces()

        assert "gathered_at" in traces
        assert "name" in traces
        assert traces["name"] == "Simulated User"


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSUserIntegration:
    """Integration tests for MACSUser."""

    def test_conversation_lifecycle(self, sample_scenario, initial_query):
        """Test complete conversation lifecycle with is_done() method."""
        responses = [
            "Yes, Monday works.",
            "I prefer Delta.",
            "Aisle seat please.",
            "Book it! </stop>",
        ]
        model = DummyModelAdapter(responses=responses)
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=5,
        )

        # Replace the simulator with a mock that cycles through responses
        user.simulator = MagicMock(side_effect=responses)

        # Simulate multi-turn conversation using respond
        questions = [
            "When would you like to travel?",
            "Any airline preference?",
            "Window or aisle?",
            "I'll book the flight. Confirmation?",
        ]

        for i, question in enumerate(questions):
            if user._stopped or user._turn_count >= user.max_turns:
                break
            response = user.respond(question)
            if i < len(questions) - 1:
                assert response != ""

        # After stop token, should be done
        # initial_query counts as turn 1, so with 4 responds we'd have 5 turns
        # But stop token was hit on 4th response, so turn_count is 5
        assert user.is_done()
        assert user._turn_count == 5

    def test_max_turns_enforcement(self, sample_scenario, initial_query):
        """Test that max turns is enforced."""
        model = DummyModelAdapter(responses=["Response"] * 10)
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=3,
        )

        # Replace the simulator with a mock that returns a controlled response
        user.simulator = MagicMock(return_value="Response")

        # Simulate 3 turns
        for i in range(3):
            user.respond(f"Question {i}")

        # Should be done after 3 turns
        assert user.is_done()
        assert user._turn_count == 3

        # Additional calls should return empty
        response = user.respond("One more?")
        assert response == ""

    def test_reset_allows_new_conversation(self, sample_scenario, initial_query):
        """Test that reset allows starting new conversation."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        user = MACSUser(
            model=model,
            scenario=sample_scenario,
            initial_query=initial_query,
            max_turns=2,
        )

        # Max out turns
        user._turn_count = 2
        user._stopped = True
        assert user.is_done()

        # Reset
        user.reset()

        # After reset, turn count goes back to 1 (initial query counts)
        # and stopped flag is cleared
        assert user._turn_count == 1
        assert not user._stopped
