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
        """Test parsing action with empty input.

        ARE's ``parse_json_tool_call`` normalizes falsy action_input (including
        empty dict ``{}``) to empty string ``""`` via ``action_input or ""``.
        """
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = """Thought: Just getting the time.

Action:
{"action": "SystemApp__get_current_time", "action_input": {}}<end_action>"""

        result = _parse_action_from_text(text)

        assert result is not None
        _, tool_name, tool_args = result
        assert tool_name == "SystemApp__get_current_time"
        # ARE normalizes empty/falsy action_input to "" (parse_json_tool_call: `action_input or ""`)
        assert tool_args == ""

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
        """Test agent pauses on wait_for_notification and resumes.

        wait_for_notification pauses the inner step loop (PAUSED state). The
        outer turn loop continues — eventually the agent terminates via
        send_message_to_user. Matches ARE's two-level loop architecture.
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
        """Test agent handles unknown tool gracefully.

        ARE json_action_executor.py:210-212: raises UnavailableToolAgentError
        with message "Error: unknown tool {name}, should be instead one of ...".
        Error appears as ERROR: message (not Observation:) in agent context.
        """
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

        # Should have continued after error and terminated
        assert agent._terminated
        messages = agent.get_messages()
        # Error is formatted as ERROR: (not Observation:) matching ARE
        error_msgs = [m for m in messages if "ERROR:" in str(m.get("content", ""))]
        assert len(error_msgs) >= 1
        # ARE error format: "Error: unknown tool {name}, should be instead one of ..."
        error_content = str(error_msgs[0].get("content", ""))
        assert "unknown tool" in error_content.lower()
        assert "should be instead one of" in error_content.lower()

    def test_handles_tool_execution_error(self):
        """Test agent handles tool execution errors as ERROR: messages.

        ARE json_action_executor.py:224-227: raises JsonExecutionAgentError with
        error details and tool description reminder. Errors appear as ERROR:
        messages (not Observation:) in agent context.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        class FailingTool:
            description = "A tool that fails"
            inputs = {"param": {"type": "string", "description": "A parameter"}}
            output_type = "string"

            def __call__(self, **kwargs):
                raise ValueError("Tool failed!")

        tools = {
            "Failing__tool": FailingTool(),
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

        # Error appears as ERROR: message (not Observation:)
        messages = agent.get_messages()
        error_msgs = [m for m in messages if "ERROR:" in str(m.get("content", ""))]
        assert len(error_msgs) >= 1
        error_content = str(error_msgs[0].get("content", ""))
        # ARE format: includes "Error in tool call execution:" and tool description reminder
        assert "Error in tool call execution" in error_content
        assert "As a reminder, this tool's description is the following" in error_content
        # No Observation: message for the error
        error_observations = [m for m in messages if "Observation:" in str(m.get("content", "")) and "Tool failed" in str(m.get("content", ""))]
        assert len(error_observations) == 0

    def test_step_counter_increments_on_errors(self):
        """Test step counter increments for errors, not just observations.

        ARE base_agent.py:450-451: id_output_step incremented for BOTH
        "observation" and "error" roles. Each output gets a unique step number.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        def failing_tool(**kwargs):
            raise ValueError("Tool failed!")

        tools = {
            "Failing__tool": failing_tool,
            "Calendar__get_events": lambda **kwargs: "[]",
            "AgentUserInterface__send_message_to_user": lambda **kwargs: "sent",
        }

        model = DummyModelAdapter(
            responses=[
                # Step 1: call failing tool -> ERROR at step 1
                'Thought: Try failing.\n\nAction:\n{"action": "Failing__tool", "action_input": {}}<end_action>',
                # Step 2: call working tool -> Observation at step 2
                'Thought: Try calendar.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {}}<end_action>',
                # Step 3: terminate
                'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(tools=tools, model=model)
        agent.run("Test step counting")

        messages = agent.get_messages()
        # Find all OUTPUT OF STEP messages
        step_msgs = [m for m in messages if "[OUTPUT OF STEP" in str(m.get("content", ""))]
        assert len(step_msgs) >= 2

        # Extract step numbers
        import re

        step_numbers = []
        for m in step_msgs:
            match = re.search(r"\[OUTPUT OF STEP (\d+)\]", str(m.get("content", "")))
            if match:
                step_numbers.append(int(match.group(1)))

        # All step numbers should be unique (no duplicates from error path)
        assert len(step_numbers) == len(set(step_numbers)), f"Duplicate step numbers: {step_numbers}"
        # Steps should be sequential
        assert step_numbers == sorted(step_numbers), f"Steps not sequential: {step_numbers}"

    def test_unknown_tool_step_counter_increments(self):
        """Test that calling an unknown tool increments the step counter.

        Ensures no duplicate step numbers when error is followed by success.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        tools = {
            "Calendar__get_events": lambda **kwargs: "[]",
            "AgentUserInterface__send_message_to_user": lambda **kwargs: "sent",
        }

        model = DummyModelAdapter(
            responses=[
                # Step 1: unknown tool -> ERROR at step 1
                'Thought: Try unknown.\n\nAction:\n{"action": "NonExistent__tool", "action_input": {}}<end_action>',
                # Step 2: valid tool -> Observation at step 2
                'Thought: Try calendar.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {}}<end_action>',
                # Step 3: terminate
                'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(tools=tools, model=model)
        agent.run("Test unknown tool step counting")

        messages = agent.get_messages()
        import re

        step_numbers = []
        for m in messages:
            match = re.search(r"\[OUTPUT OF STEP (\d+)\]", str(m.get("content", "")))
            if match:
                step_numbers.append(int(match.group(1)))

        assert len(step_numbers) == len(set(step_numbers)), f"Duplicate step numbers: {step_numbers}"

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

    def test_execute_tool_normalizes_empty_action_input(self):
        """Test that empty action_input {} doesn't crash kwargs-only tools.

        ARE's parse_json_tool_call normalizes empty dict {} to empty string ""
        (via ``action_input or ""``). _execute_tool must normalize this back
        to {} before calling the tool, matching ARE's execute_tool_call
        (json_action_executor.py:204). Without this, tools with **kwargs-only
        signatures crash with TypeError on the positional string argument.
        """
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        call_log = []

        def no_arg_tool(**kwargs):
            call_log.append(kwargs)
            return "2024-01-15T10:00:00Z"

        tools = {
            "SystemApp__get_current_time": no_arg_tool,
            "AgentUserInterface__send_message_to_user": lambda **kwargs: "sent",
        }

        model = DummyModelAdapter(
            responses=[
                'Thought: Get the time.\n\nAction:\n{"action": "SystemApp__get_current_time", "action_input": {}}<end_action>',
                'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>',
            ]
        )

        agent = DefaultGaia2Agent(tools=tools, model=model)
        agent.run("What time is it?")

        # Tool was called successfully with no arguments (not with positional "")
        assert len(call_log) == 1
        assert call_log[0] == {}
        assert agent._terminated


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


# =============================================================================
# Test ARE Import Delegation
# =============================================================================


@pytest.mark.benchmark
class TestAREImportDelegation:
    """Verify that functions delegate to ARE's implementations."""

    def test_parse_action_uses_are_parse_json_tool_call(self):
        """_parse_action_from_text delegates JSON parsing to ARE's parse_json_tool_call."""
        from are.simulation.agents.default_agent.tools.json_action_executor import parse_json_tool_call

        # If the import works, ARE is installed and the function is available.
        # Verify _parse_action_from_text produces the same result as calling
        # parse_json_tool_call on the action blob directly.
        from maseval.benchmark.gaia2.gaia2 import _parse_action_from_text

        text = 'Thought: checking.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {"date": "2024-01-15"}}<end_action>'
        result = _parse_action_from_text(text)
        assert result is not None
        _, tool_name, tool_args = result

        # Compare with ARE's direct output
        are_name, are_args = parse_json_tool_call('{"action": "Calendar__get_events", "action_input": {"date": "2024-01-15"}}')
        assert tool_name == are_name
        assert tool_args == are_args

    def test_get_offset_uses_are_function(self):
        """_react_loop uses ARE's get_offset_from_time_config_mode at runtime."""
        from are.simulation.agents.default_agent.base_agent import (
            SimulatedGenerationTimeConfig,
            get_offset_from_time_config_mode,
        )

        # Verify the ARE function exists and works with ARE's own type
        config = SimulatedGenerationTimeConfig(mode="measured")
        assert get_offset_from_time_config_mode(config, 2.5) == 2.5

        config_fixed = SimulatedGenerationTimeConfig(mode="fixed", seconds=1.0)
        assert get_offset_from_time_config_mode(config_fixed, 2.5) == 1.0


# =============================================================================
# Test _check_environment_stop
# =============================================================================


@pytest.mark.benchmark
class TestCheckEnvironmentStop:
    """Tests for DefaultGaia2Agent._check_environment_stop()."""

    def test_returns_false_when_no_environment(self, sample_tools_dict):
        """_check_environment_stop returns False when environment is None."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=None)

        assert agent._check_environment_stop() is False

    def test_returns_true_when_stop_message_present(self, sample_tools_dict):
        """_check_environment_stop returns True when has_environment_stop_message is True."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_env = MagicMock()
        mock_ns = MagicMock()
        mock_ns.message_queue.has_environment_stop_message.return_value = True
        mock_env.get_notification_system.return_value = mock_ns

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        assert agent._check_environment_stop() is True

    def test_returns_false_when_no_stop_message(self, sample_tools_dict):
        """_check_environment_stop returns False when no stop message."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_env = MagicMock()
        mock_ns = MagicMock()
        mock_ns.message_queue.has_environment_stop_message.return_value = False
        mock_env.get_notification_system.return_value = mock_ns

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        assert agent._check_environment_stop() is False

    def test_returns_false_on_exception(self, sample_tools_dict):
        """_check_environment_stop returns False when exception occurs."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_env = MagicMock()
        mock_ns = MagicMock()
        mock_ns.message_queue.has_environment_stop_message.side_effect = RuntimeError("broken")
        mock_env.get_notification_system.return_value = mock_ns

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        assert agent._check_environment_stop() is False


# =============================================================================
# Test _pull_notifications
# =============================================================================


@pytest.mark.benchmark
class TestPullNotifications:
    """Tests for DefaultGaia2Agent._pull_notifications()."""

    def test_noop_when_no_environment(self, sample_tools_dict):
        """_pull_notifications does nothing when environment is None."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=None)

        agent._pull_notifications()

        assert agent._messages == []

    def test_injects_user_messages(self, sample_tools_dict):
        """_pull_notifications adds user messages to history."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_env = MagicMock()
        mock_env.poll_notifications.return_value = (["Hello from user"], [], False)

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        agent._pull_notifications()

        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "user"
        assert "Hello from user" in agent._messages[0]["content"]
        assert "User messages updates:" in agent._messages[0]["content"]

    def test_injects_env_notifications(self, sample_tools_dict):
        """_pull_notifications adds env notifications to history."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_env = MagicMock()
        mock_env.poll_notifications.return_value = ([], ["[08:01] New email arrived"], False)

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        agent._pull_notifications()

        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "user"
        assert "New email arrived" in agent._messages[0]["content"]
        assert "Environment notifications updates:" in agent._messages[0]["content"]


# =============================================================================
# Test _get_max_turns
# =============================================================================


@pytest.mark.benchmark
class TestGetMaxTurns:
    """Tests for DefaultGaia2Agent._get_max_turns()."""

    def test_returns_none_when_no_environment(self, sample_tools_dict):
        """_get_max_turns returns None when environment is None."""
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=None)

        assert agent._get_max_turns() is None

    def test_returns_nb_turns_from_scenario(self, sample_tools_dict):
        """_get_max_turns returns scenario.nb_turns."""
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_scenario = MagicMock()
        mock_scenario.nb_turns = 3
        mock_env = MagicMock()
        mock_env.get_scenario.return_value = mock_scenario

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        assert agent._get_max_turns() == 3

    def test_returns_none_when_scenario_has_no_nb_turns(self, sample_tools_dict):
        """_get_max_turns returns None when scenario has no nb_turns attribute."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        mock_scenario = SimpleNamespace()  # No nb_turns attribute
        mock_env = MagicMock()
        mock_env.get_scenario.return_value = mock_scenario

        model = DummyModelAdapter(responses=["dummy"])
        agent = DefaultGaia2Agent(tools=sample_tools_dict, model=model, environment=mock_env)

        assert agent._get_max_turns() is None
