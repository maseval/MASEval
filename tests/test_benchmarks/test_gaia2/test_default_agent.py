"""Tests for DefaultGaia2Agent and related components.

Tests the text-based ReAct agent implementation that matches ARE's reference agent.
"""

import pytest

from conftest import DummyModelAdapter


# =============================================================================
# Test Helper Functions
# =============================================================================


@pytest.mark.benchmark
class TestParseActionFromText:
    """Tests for _parse_action_from_text helper function."""

    def test_parses_valid_thought_and_action(self):
        """Test parsing valid Thought/Action format."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Thought: I need to check the calendar for events.

Action:
{"action": "Calendar__get_events", "action_input": {"date": "2024-01-15"}}<end_action>"""

        result = _parse_action_from_text(text)

        assert result is not None
        thought, tool_name, tool_args = result
        assert "check the calendar" in thought
        assert tool_name == "Calendar__get_events"
        assert tool_args == {"date": "2024-01-15"}

    def test_parses_action_without_thought(self):
        """Test parsing action when Thought is missing."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Action:
{"action": "Email__send", "action_input": {"to": "user@example.com"}}<end_action>"""

        result = _parse_action_from_text(text)

        assert result is not None
        thought, tool_name, tool_args = result
        assert thought == ""
        assert tool_name == "Email__send"

    def test_parses_empty_action_input(self):
        """Test parsing action with empty input."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Thought: Just getting the time.

Action:
{"action": "SystemApp__get_current_time", "action_input": {}}<end_action>"""

        result = _parse_action_from_text(text)

        assert result is not None
        _, tool_name, tool_args = result
        assert tool_name == "SystemApp__get_current_time"
        assert tool_args == {}

    def test_returns_none_for_invalid_format(self):
        """Test that invalid format returns None."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        invalid_texts = [
            "Just some plain text without any action.",
            "Thought: Thinking...\n\nBut no action here.",
            "Action: not valid json",
            "Action:\n{broken json",
        ]

        for text in invalid_texts:
            result = _parse_action_from_text(text)
            assert result is None, f"Expected None for: {text[:50]}..."

    def test_rejects_trailing_comma_in_json(self):
        """Test that trailing commas in JSON are rejected (matching ARE)."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Thought: Testing.

Action:
{"action": "test_tool", "action_input": {"key": "value",}}<end_action>"""

        # ARE json_action_executor.py:33-57 does not fix trailing commas.
        # Trailing commas are invalid JSON, so parsing should fail.
        result = _parse_action_from_text(text)

        assert result is None

    def test_handles_dict_action_input(self):
        """Test that nested dict action_input is handled."""
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Thought: Testing.

Action:
{"action": "test_tool", "action_input": {"nested": "value"}}<end_action>"""

        result = _parse_action_from_text(text)

        assert result is not None
        _, _, tool_args = result
        assert tool_args == {"nested": "value"}


@pytest.mark.benchmark
class TestApplyStopTruncation:
    """Tests for _apply_stop_truncation helper function.

    Matches ARE's LiteLLMEngine client-side truncation (litellm_engine.py:126-127).
    """

    def test_truncates_on_end_action(self):
        """Test truncation at `<end_action>` stop token."""
        from maseval.benchmark.gaia2.gaia2 import _apply_stop_truncation

        text = 'Thought: checking.\n\nAction:\n{"action": "test", "action_input": {}}<end_action>extra stuff'
        result = _apply_stop_truncation(text, ["<end_action>", "Observation:"])

        assert result == 'Thought: checking.\n\nAction:\n{"action": "test", "action_input": {}}'
        assert "<end_action>" not in result
        assert "extra stuff" not in result

    def test_truncates_on_observation(self):
        """Test truncation at ``Observation:`` stop token."""
        from maseval.benchmark.gaia2.gaia2 import _apply_stop_truncation

        text = 'Thought: checking.\n\nAction:\n{"action": "test", "action_input": {}}Observation: some output'
        result = _apply_stop_truncation(text, ["<end_action>", "Observation:"])

        assert "Observation:" not in result
        assert "some output" not in result

    def test_no_stop_token_returns_unchanged(self):
        """Test that text without stop tokens passes through unchanged."""
        from maseval.benchmark.gaia2.gaia2 import _apply_stop_truncation

        text = 'Thought: checking.\n\nAction:\n{"action": "test", "action_input": {}}'
        result = _apply_stop_truncation(text, ["<end_action>", "Observation:"])

        assert result == text

    def test_truncates_on_first_occurrence(self):
        """Test that only content before the first stop token is kept."""
        from maseval.benchmark.gaia2.gaia2 import _apply_stop_truncation

        text = "before<end_action>middle<end_action>after"
        result = _apply_stop_truncation(text, ["<end_action>"])

        assert result == "before"

    def test_empty_stop_sequences(self):
        """Test with empty stop sequences list."""
        from maseval.benchmark.gaia2.gaia2 import _apply_stop_truncation

        text = "some text<end_action>more"
        result = _apply_stop_truncation(text, [])

        assert result == text


@pytest.mark.benchmark
class TestBuildToolDescriptions:
    """Tests for _build_tool_descriptions helper function."""

    def test_builds_descriptions_from_tools(self, sample_tools_dict):
        """Test building tool descriptions matching ARE's Jinja2 format."""
        from maseval.benchmark.gaia2.gaia2 import _build_tool_descriptions

        # Create mock tools with proper attributes
        class MockTool:
            def __init__(self, name, desc, inputs, output_type="string"):
                self.name = name
                self.description = desc
                self.inputs = inputs
                self.output_type = output_type

            def __call__(self, **kwargs):
                return "result"

        tools = {
            "TestTool": MockTool(
                "TestTool",
                "A test tool",
                {
                    "properties": {"arg1": {"type": "string", "description": "First arg"}},
                    "required": ["arg1"],
                },
            )
        }

        result = _build_tool_descriptions(tools)

        # ARE format: "- {name}: {desc}\n    Takes inputs: {inputs}\n    Returns an output of type: {output_type}"
        assert "- TestTool: A test tool" in result
        assert "Takes inputs:" in result
        assert "arg1" in result
        assert "Returns an output of type: string" in result

    def test_handles_tools_without_attributes(self):
        """Test handling tools without description/inputs attributes."""
        from maseval.benchmark.gaia2.gaia2 import _build_tool_descriptions

        def plain_function(**kwargs):
            return "result"

        tools = {"plain_tool": plain_function}

        result = _build_tool_descriptions(tools)

        # ARE format: "- {name}: {desc}\n    Takes inputs: {inputs}\n    Returns an output of type: {output_type}"
        assert "- plain_tool:" in result
        assert "Takes inputs:" in result
        assert "Returns an output of type: string" in result


@pytest.mark.benchmark
class TestBuildSystemPrompt:
    """Tests for _build_system_prompt helper function."""

    def test_builds_complete_system_prompt(self, sample_tools_dict):
        """Test building complete system prompt with tools."""
        from maseval.benchmark.gaia2.gaia2 import _build_system_prompt

        # Create mock tools with attributes
        class MockTool:
            def __init__(self, desc):
                self.description = desc
                self.inputs = {}

            def __call__(self, **kwargs):
                return "result"

        tools = {name: MockTool(f"Description for {name}") for name in sample_tools_dict}

        result = _build_system_prompt(tools)

        # Check structure
        assert "<general_instructions>" in result
        assert "<agent_instructions>" in result
        assert "<environment_instructions>" in result

        # Check content from templates
        assert "MetaOSSAgent" in result
        assert "Thought:" in result
        assert "Action:" in result
        assert "<end_action>" in result

    def test_includes_tool_descriptions_in_prompt(self):
        """Test that tool descriptions are included in prompt."""
        from maseval.benchmark.gaia2.gaia2 import _build_system_prompt

        class MockTool:
            description = "Send an email to recipient"
            inputs = {"properties": {"to": {"type": "string", "description": "Recipient"}}, "required": ["to"]}

            def __call__(self, **kwargs):
                return "sent"

        tools = {"Email__send": MockTool()}

        result = _build_system_prompt(tools)

        assert "Email__send" in result
        assert "Send an email" in result


# =============================================================================
# Test DefaultGaia2Agent
# =============================================================================


@pytest.mark.benchmark
class TestDefaultGaia2AgentInit:
    """Tests for DefaultGaia2Agent initialization."""

    def test_initializes_with_defaults(self, sample_tools_dict, gaia2_model_react):
        """Test agent initializes with default parameters."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
        )

        assert agent.max_iterations == 80
        assert agent.invalid_format_retries == 10
        assert agent.llm_args["temperature"] == 0.5
        assert agent.llm_args["max_tokens"] == 16384
        stop_sequences = agent.llm_args["stop"]
        assert isinstance(stop_sequences, list)
        assert "<end_action>" in stop_sequences

    def test_allows_custom_parameters(self, sample_tools_dict, gaia2_model_react):
        """Test agent accepts custom parameters."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
            max_iterations=50,
            invalid_format_retries=5,
            llm_args={"temperature": 0.7},
            verbose=2,
        )

        assert agent.max_iterations == 50
        assert agent.invalid_format_retries == 5
        assert agent.llm_args["temperature"] == 0.7
        assert agent.verbose == 2

    def test_none_llm_args_override_defaults(self, sample_tools_dict, gaia2_model_react):
        """Test that llm_args with None values override defaults.

        Reasoning models (o1, o3, GPT-5) don't support stop, temperature, etc.
        Setting them to None omits them from the API call while client-side
        stop-token truncation still works.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
            llm_args={"stop": None, "temperature": None},
        )

        # None values stored in llm_args (filtered out at call time)
        assert agent.llm_args["stop"] is None
        assert agent.llm_args["temperature"] is None
        # max_tokens retains default
        assert agent.llm_args["max_tokens"] == 16384

    def test_builds_system_prompt_with_tools(self, sample_tools_dict, gaia2_model_react):
        """Test that system prompt includes tool descriptions."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
        )

        # System prompt should contain tool names
        for tool_name in sample_tools_dict:
            assert tool_name in agent.system_prompt


@pytest.mark.benchmark
class TestDefaultGaia2AgentRun:
    """Tests for DefaultGaia2Agent.run() method."""

    def test_executes_react_loop(self, sample_tools_dict, gaia2_model_react):
        """Test agent executes ReAct loop correctly."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
        )

        result = agent.run("Check my calendar")

        # Should have executed and terminated
        assert agent._terminated
        assert agent.iteration_count >= 1
        assert "calendar is empty" in result.lower() or "done" in result.lower() or len(result) > 0

    def test_terminates_on_send_message_to_user(self, sample_tools_dict, gaia2_model_termination):
        """Test agent terminates when calling send_message_to_user."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_termination,
        )

        result = agent.run("Hello")

        assert agent._terminated
        assert "ready to help" in result.lower()

    def test_continues_after_wait_for_notification(self, sample_tools_dict, gaia2_model_wait_notification):
        """Test agent does NOT terminate on wait_for_notification.

        wait_for_notification is a pause, not a termination signal.  The agent
        must continue its loop and eventually terminate via send_message_to_user.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_wait_notification,
        )

        result = agent.run("Wait for updates")

        assert agent._terminated
        # Agent continued past wait_for_notification and terminated via send_message_to_user
        assert "finished waiting" in result.lower()
        # At least 2 iterations: one for wait, one for send_message_to_user
        assert agent.iteration_count >= 2

    def test_retries_on_invalid_format(self, sample_tools_dict, gaia2_model_invalid_format):
        """Test agent retries when format is invalid then hits max iterations."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_invalid_format,
            max_iterations=3,
            invalid_format_retries=3,
        )

        result = agent.run("Do something")

        # Agent should exhaust iterations; ARE base_agent.py:849 increments on every iteration
        assert "Max iterations (3) reached" in result
        assert agent.iteration_count == 3

    def test_respects_max_iterations(self, sample_tools_dict):
        """Test agent stops at max_iterations."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        # Model that never terminates (always calls a non-termination tool)
        model = DummyModelAdapter(
            responses=['Thought: Checking.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {}}<end_action>']
        )

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=model,
            max_iterations=3,
        )

        result = agent.run("Loop forever")

        assert "Max iterations (3) reached" in result
        assert agent.iteration_count == 3

    def test_handles_tool_not_found(self, sample_tools_dict, gaia2_model_react):
        """Test agent handles unknown tool gracefully."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        model = DummyModelAdapter(
            responses=[
                'Thought: Trying unknown tool.\n\nAction:\n{"action": "Unknown__tool", "action_input": {}}<end_action>',
                'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=model,
        )

        agent.run("Try unknown tool")

        # Should have continued after error
        messages = agent.get_messages()
        error_found = any("not found" in str(m.get("content", "")).lower() for m in messages)
        assert error_found or agent._terminated

    def test_handles_tool_execution_error(self):
        """Test agent handles tool execution errors gracefully."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        def failing_tool(**kwargs):
            raise ValueError("Tool failed!")

        tools = {
            "Failing__tool": failing_tool,
            "AgentUserInterface__send_message_to_user": lambda **kwargs: "sent",
        }

        model = DummyModelAdapter(
            responses=[
                'Thought: Try failing.\n\nAction:\n{"action": "Failing__tool", "action_input": {}}<end_action>',
                'Thought: Report error.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Error occurred"}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(tools=tools, model=model)
        agent.run("Test error handling")

        # Should have error in observation
        messages = agent.get_messages()
        error_observed = any("error" in str(m.get("content", "")).lower() for m in messages)
        assert error_observed

    def test_wait_for_notification_executes_tool_and_continues(self):
        """Test wait_for_notification executes the tool and records observation.

        Verifies the multi-turn notification loop: the agent calls
        wait_for_notification, the tool executes (advancing simulation
        time), the observation is added to messages, and the agent
        continues to call more tools before terminating.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        wait_called = []

        def mock_wait(**kwargs):
            wait_called.append(kwargs)
            return "No notifications"

        tools = {
            "Calendar__get_events": lambda **kwargs: "[]",
            "SystemApp__wait_for_notification": mock_wait,
            "AgentUserInterface__send_message_to_user": lambda **kwargs: "sent",
        }

        model = DummyModelAdapter(
            responses=[
                # Step 1: wait for notification
                'Thought: Need to wait.\n\nAction:\n{"action": "SystemApp__wait_for_notification", "action_input": {"timeout": 240}}<end_action>',
                # Step 2: check calendar (agent continued!)
                'Thought: Check calendar after wait.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {}}<end_action>',
                # Step 3: terminate
                'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "All done."}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(tools=tools, model=model)
        result = agent.run("Multi-turn task")

        # wait_for_notification was actually executed as a tool
        assert len(wait_called) == 1
        # Agent continued past wait and executed more tools
        assert agent.iteration_count == 3
        assert result == "All done."
        # Observation from wait is in messages
        messages = agent.get_messages()
        wait_observation = [m for m in messages if "No notifications" in str(m.get("content", ""))]
        assert len(wait_observation) > 0


@pytest.mark.benchmark
class TestDefaultGaia2AgentState:
    """Tests for DefaultGaia2Agent state management."""

    def test_reset_clears_state(self, sample_tools_dict, gaia2_model_react):
        """Test reset() clears all state."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
        )

        # Run once
        agent.run("First task")

        # Reset
        agent.reset()

        assert agent._messages == []
        assert agent._iteration_count == 0
        assert agent._format_retry_count == 0
        assert agent._terminated is False
        assert agent._final_message is None

    def test_get_messages_returns_history(self, sample_tools_dict, gaia2_model_termination):
        """Test get_messages() returns conversation history."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_termination,
        )

        agent.run("Hello")
        messages = agent.get_messages()

        assert len(messages) > 0
        assert messages[0]["role"] == "user"
        assert "Hello" in messages[0]["content"]

    def test_iteration_count_property(self, sample_tools_dict, gaia2_model_react):
        """Test iteration_count property."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_react,
        )

        assert agent.iteration_count == 0

        agent.run("Do task")

        assert agent.iteration_count > 0


# =============================================================================
# Test DefaultGaia2AgentAdapter
# =============================================================================


@pytest.mark.benchmark
class TestDefaultGaia2AgentAdapter:
    """Tests for DefaultGaia2AgentAdapter."""

    def test_wraps_agent_correctly(self, sample_tools_dict, gaia2_model_termination):
        """Test adapter wraps agent and runs correctly."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent, DefaultGaia2AgentAdapter

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_termination,
        )
        adapter = DefaultGaia2AgentAdapter(agent, name="test_adapter")

        result = adapter.run("Hello")

        assert "ready to help" in result.lower()
        assert adapter.name == "test_adapter"

    def test_gather_traces_includes_agent_state(self, sample_tools_dict, gaia2_model_termination):
        """Test gather_traces includes agent execution state."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent, DefaultGaia2AgentAdapter

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_termination,
        )
        adapter = DefaultGaia2AgentAdapter(agent, name="test_adapter")

        adapter.run("Hello")
        traces = adapter.gather_traces()

        assert "messages" in traces
        assert "iteration_count" in traces
        assert "terminated" in traces
        assert traces["terminated"] is True
        assert traces["iteration_count"] >= 1

    def test_get_messages_delegates_to_agent(self, sample_tools_dict, gaia2_model_termination):
        """Test get_messages delegates to underlying agent."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent, DefaultGaia2AgentAdapter

        agent = DefaultGaia2Agent(
            tools=sample_tools_dict,
            model=gaia2_model_termination,
        )
        adapter = DefaultGaia2AgentAdapter(agent)

        adapter.run("Test")

        adapter_messages = adapter.get_messages()
        agent_messages = agent.get_messages()

        # Adapter returns MessageHistory, agent returns list - compare contents
        assert adapter_messages.to_list() == agent_messages


# =============================================================================
# Test Default Constants
# =============================================================================


@pytest.mark.benchmark
class TestDefaultConstants:
    """Tests for default constants matching ARE."""

    def test_default_max_iterations_matches_are(self):
        """Test DEFAULT_MAX_ITERATIONS matches ARE's default."""
        from maseval.benchmark.gaia2.gaia2 import _DEFAULT_MAX_ITERATIONS

        assert _DEFAULT_MAX_ITERATIONS == 80

    def test_default_temperature_matches_are(self):
        """Test DEFAULT_TEMPERATURE matches ARE's default."""
        from maseval.benchmark.gaia2.gaia2 import _DEFAULT_TEMPERATURE

        assert _DEFAULT_TEMPERATURE == 0.5

    def test_default_max_tokens_matches_are(self):
        """Test DEFAULT_MAX_TOKENS matches ARE's default."""
        from maseval.benchmark.gaia2.gaia2 import _DEFAULT_MAX_TOKENS

        assert _DEFAULT_MAX_TOKENS == 16384

    def test_default_invalid_format_retries_matches_are(self):
        """Test DEFAULT_INVALID_FORMAT_RETRIES matches ARE's default."""
        from maseval.benchmark.gaia2.gaia2 import _DEFAULT_INVALID_FORMAT_RETRIES

        assert _DEFAULT_INVALID_FORMAT_RETRIES == 10

    def test_stop_sequences_match_are(self):
        """Test STOP_SEQUENCES match ARE's tokens."""
        from maseval.benchmark.gaia2.gaia2 import _STOP_SEQUENCES

        assert "<end_action>" in _STOP_SEQUENCES
        assert "Observation:" in _STOP_SEQUENCES

    def test_termination_tools_include_expected(self):
        """Test TERMINATION_TOOLS contains only send_message_to_user.

        wait_for_notification is NOT a termination tool — it pauses the
        agent while ARE processes events, then the agent resumes.
        """
        from maseval.benchmark.gaia2.gaia2 import _TERMINATION_TOOLS

        assert "AgentUserInterface__send_message_to_user" in _TERMINATION_TOOLS
        assert "SystemApp__wait_for_notification" not in _TERMINATION_TOOLS
