"""Test LLMUser class functionality.

These tests verify the LLMUser class (maseval.core.user.LLMUser) behavior:
- Conversation history management (MessageHistory)
- Multi-turn interaction (max_turns, turn counting)
- Early stopping via stop tokens
- Optional initial prompts and LLM-generated initial queries

Note: This tests the LLMUser class, NOT the UserLLMSimulator class.
UserLLMSimulator is tested in test_llm_simulator.py.
"""

import pytest


@pytest.mark.core
class TestUser:
    """Tests for LLMUser basics."""

    def test_user_respond_updates_messages(self, dummy_user):
        """Test that respond adds to message history."""
        initial_len = len(dummy_user.messages)

        # respond adds assistant message, then user response
        dummy_user.respond("How can I help?")

        # Should have added 2 messages: assistant question + user response
        assert len(dummy_user.messages) == initial_len + 2

    def test_user_messages_includes_both_sides(self, dummy_user):
        """Test that messages includes both user and assistant messages."""
        # Simulate a response (adds assistant + user messages)
        dummy_user.respond("Question for user")

        messages = list(dummy_user.messages)
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_user_gather_traces_includes_interactions(self, dummy_user):
        """Test that gather_traces() includes conversation history."""
        traces = dummy_user.gather_traces()

        assert "type" in traces
        assert "gathered_at" in traces
        assert "name" in traces
        assert "message_count" in traces
        assert "messages" in traces

        assert traces["name"] == "test_user"
        assert isinstance(traces["messages"], list)
        assert traces["message_count"] == len(traces["messages"])

    def test_user_gather_config_includes_profile(self, dummy_user):
        """Test that gather_config() includes user profile."""
        config = dummy_user.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert "name" in config

        assert config["name"] == "test_user"

    def test_user_initialization(self, dummy_model):
        """Test that User can be initialized with required parameters."""
        from conftest import DummyUser

        user = DummyUser(
            name="test_user",
            model=dummy_model,
            user_profile={"role": "customer"},
            scenario="test scenario",
            initial_query="Hello",
        )

        assert user.name == "test_user"
        assert user.user_profile == {"role": "customer"}
        assert user.scenario == "test scenario"
        assert len(user.messages) == 1


# =============================================================================
# Multi-Turn Configuration Tests
# =============================================================================


@pytest.mark.core
class TestUserMultiTurn:
    """Tests for max_turns behavior."""

    def test_default_max_turns_is_one(self, dummy_model):
        """Default single-turn mode."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user.max_turns == 1

    def test_custom_max_turns(self, dummy_model):
        """Custom max_turns is stored correctly."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=5)
        assert user.max_turns == 5

    def test_is_done_after_max_turns(self, dummy_model):
        """is_done() returns True when turn count >= max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=2)
        user._turn_count = 2

        assert user.is_done()

    def test_is_done_before_max_turns(self, dummy_model):
        """is_done() returns False when turn count < max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user._turn_count = 1

        assert not user.is_done()

    def test_respond_increments_turn_count(self, dummy_model):
        """Each respond() call increments _turn_count."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=5)
        initial_count = user._turn_count

        user.respond("Question 1")
        assert user._turn_count == initial_count + 1

        user.respond("Question 2")
        assert user._turn_count == initial_count + 2

    def test_respond_returns_empty_when_done(self, dummy_model):
        """Returns empty string when is_done() is True."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=1)
        user._turn_count = 1  # Already at max

        response = user.respond("More questions?")
        assert response == ""

    def test_turn_count_starts_at_zero_without_initial_query(self, dummy_model):
        """Turn count starts at 0 when no initial_query provided."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user._turn_count == 0

    def test_turn_count_starts_at_one_with_initial_query(self, dummy_model):
        """Turn count starts at 1 when initial_query is provided."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, initial_query="Hello")
        assert user._turn_count == 1


# =============================================================================
# Stop Token Tests
# =============================================================================


@pytest.mark.core
class TestUserStopTokens:
    """Tests for stop_tokens early termination."""

    def test_no_stop_tokens_by_default(self, dummy_model):
        """stop_tokens is empty by default."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        assert user.stop_tokens == []
        assert user.early_stopping_condition is None

    def test_stop_tokens_without_condition_raises(self, dummy_model):
        """ValueError raised when stop_tokens set but early_stopping_condition is None."""
        from conftest import DummyUser

        with pytest.raises(ValueError, match="must both be set or both be None"):
            DummyUser(name="test", model=dummy_model, stop_tokens=["</stop>"])

    def test_condition_without_stop_tokens_raises(self, dummy_model):
        """ValueError raised when early_stopping_condition set but stop_tokens is empty."""
        from conftest import DummyUser

        with pytest.raises(ValueError, match="must both be set or both be None"):
            DummyUser(
                name="test",
                model=dummy_model,
                early_stopping_condition="goals are met",
            )

    def test_custom_stop_tokens(self, dummy_model):
        """Custom stop_tokens is stored."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</done>"],
            early_stopping_condition="goals are met",
        )
        assert user.stop_tokens == ["</done>"]
        assert user.early_stopping_condition == "goals are met"

    def test_stop_token_detection_sets_stopped(self, dummy_model):
        """Detecting stop token sets _stopped = True."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "Thanks! </stop>"  # type: ignore[union-attr]  # mock

        user.respond("Here's your answer")

        assert user._stopped

    def test_stop_token_removed_from_response(self, dummy_model):
        """Stop token is stripped from returned response."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "Perfect, thanks! </stop>"  # type: ignore[union-attr]  # mock

        response = user.respond("Booking confirmed!")

        assert "</stop>" not in response
        assert "Perfect, thanks!" in response

    def test_is_done_true_after_stop_token(self, dummy_model):
        """is_done() returns True after stop token detected."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "Done </stop>"  # type: ignore[union-attr]  # mock

        user.respond("Result")

        assert user.is_done()

    def test_stop_token_case_insensitive(self, dummy_model):
        """Stop token detection is case-insensitive."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</STOP>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "Thanks! </stop>"  # type: ignore[union-attr]  # mock (lowercase)

        user.respond("Answer")

        assert user._stopped

    def test_fallback_message_when_only_stop_token(self, dummy_model):
        """Provides fallback when response is only stop token."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "</stop>"  # type: ignore[union-attr]  # mock

        response = user.respond("Done!")

        assert response == "Thank you, that's all I needed!"
        assert user._stopped

    def test_stop_token_response_counts_as_turn(self, dummy_model):
        """The response containing stop token still counts as a turn."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "Thank you, all is clear </stop>"  # type: ignore[union-attr]  # mock

        initial_turn_count = user._turn_count
        user.respond("Here is your result")

        # Turn count should increment even though stop token was detected
        assert user._turn_count == initial_turn_count + 1
        assert user._stopped
        assert user.is_done()

    def test_multiple_stop_tokens(self, dummy_model):
        """Multiple stop tokens can be configured."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>", "</done>", "</end>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        assert user.stop_tokens == ["</stop>", "</done>", "</end>"]

    def test_second_stop_token_triggers_stop(self, dummy_model):
        """Any stop token in the list triggers termination."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</stop>", "</done>", "</end>"],
            early_stopping_condition="goals are met",
            max_turns=5,
        )
        user.simulator.return_value = "All finished </done>"  # type: ignore[assignment]

        user.respond("Here's your answer")

        assert user._stopped
        assert user._stop_reason == "</done>"


# =============================================================================
# Optional Initial Prompt Tests
# =============================================================================


@pytest.mark.core
class TestUserInitialQuery:
    """Tests for optional initial_query behavior."""

    def test_with_initial_query_adds_message(self, dummy_model):
        """Providing initial_query adds it to messages."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_query="I need help booking a flight",
        )

        assert len(user.messages) == 1
        assert user.messages[0]["role"] == "user"
        assert user.messages[0]["content"] == "I need help booking a flight"

    def test_without_initial_query_empty_messages(self, dummy_model):
        """No initial_query means empty message history."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        # No initial_query provided

        assert len(user.messages) == 0

    def test_get_initial_query_generates_message(self, dummy_model):
        """get_initial_query() uses LLM to generate first message."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        user.simulator.return_value = "I want to book a hotel"  # type: ignore[union-attr]  # mock

        query = user.get_initial_query()

        assert query == "I want to book a hotel"
        user.simulator.assert_called_once()  # type: ignore[union-attr]  # mock

    def test_get_initial_query_adds_to_messages(self, dummy_model):
        """Generated query is added to message history."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)
        user.simulator.return_value = "Help me please"  # type: ignore[union-attr]  # mock

        user.get_initial_query()

        assert len(user.messages) == 1
        assert user.messages[0]["role"] == "user"
        assert user.messages[0]["content"] == "Help me please"

    def test_get_initial_query_returns_existing_query(self, dummy_model):
        """get_initial_query() returns existing initial query if present."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_query="Already have a message",
        )

        query = user.get_initial_query()
        assert query == "Already have a message"

    def test_get_initial_query_counts_as_turn(self, dummy_model):
        """Initial query increments turn count."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "Initial query"  # type: ignore[union-attr]  # mock

        user.get_initial_query()

        assert user._turn_count == 1  # Counts as first turn


# =============================================================================
# Message History Completeness Tests
# =============================================================================


@pytest.mark.core
class TestUserMessageHistory:
    """Tests for complete message tracing."""

    def test_initial_message_in_history(self, dummy_model):
        """Initial query is in message history."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_query="Hello agent",
        )

        assert len(user.messages) == 1
        assert user.messages[0]["content"] == "Hello agent"

    def test_assistant_message_recorded(self, dummy_model):
        """respond() records assistant message before responding."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "User reply"  # type: ignore[union-attr]  # mock

        user.respond("Agent says hello")

        # Should have: assistant message + user response
        assert len(user.messages) == 2
        assert user.messages[0]["role"] == "assistant"
        assert user.messages[0]["content"] == "Agent says hello"

    def test_user_response_recorded(self, dummy_model):
        """respond() records user response."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=3)
        user.simulator.return_value = "Thanks for the help"  # type: ignore[union-attr]  # mock

        user.respond("Here's your answer")

        assert user.messages[-1]["role"] == "user"
        assert user.messages[-1]["content"] == "Thanks for the help"

    def test_full_conversation_tracked(self, dummy_model):
        """Multiple exchanges create complete conversation trace."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_query="I need a flight",
            max_turns=3,
        )
        user.simulator.side_effect = ["Monday works", "Yes, book it"]  # type: ignore[union-attr]  # mock

        # Two agent-user exchanges
        user.respond("When do you want to travel?")
        user.respond("Shall I book it?")

        messages = list(user.messages)
        assert len(messages) == 5  # initial + 2*(assistant + user)

        # Verify order
        assert messages[0]["role"] == "user"  # initial
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    def test_gather_traces_includes_all_messages(self, dummy_model):
        """gather_traces() includes complete conversation."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            initial_query="Hello",
            max_turns=2,
        )
        user.simulator.return_value = "Got it"  # type: ignore[union-attr]  # mock

        user.respond("Agent response")

        traces = user.gather_traces()

        assert traces["message_count"] == 3
        assert len(traces["messages"]) == 3


# =============================================================================
# Config Tests
# =============================================================================


@pytest.mark.core
class TestUserConfig:
    """Tests for gather_config updates."""

    def test_config_includes_max_turns(self, dummy_model):
        """gather_config() includes max_turns."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model, max_turns=7)

        config = user.gather_config()

        assert config["max_turns"] == 7

    def test_config_includes_stop_tokens(self, dummy_model):
        """gather_config() includes stop_tokens."""
        from conftest import DummyUser

        user = DummyUser(
            name="test",
            model=dummy_model,
            stop_tokens=["</end>"],
            early_stopping_condition="goals are met",
        )

        config = user.gather_config()

        assert config["stop_tokens"] == ["</end>"]

    def test_config_includes_empty_stop_tokens(self, dummy_model):
        """gather_config() includes stop_tokens even when empty."""
        from conftest import DummyUser

        user = DummyUser(name="test", model=dummy_model)

        config = user.gather_config()

        assert "stop_tokens" in config
        assert config["stop_tokens"] == []
