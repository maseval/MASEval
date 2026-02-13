"""Unit tests for MACSEvaluator."""

import json
import pytest

from maseval import MessageHistory, Task
from maseval.benchmark.macs import MACSEvaluator

from conftest import DummyModelAdapter


# =============================================================================
# Unit Tests: Initialization
# =============================================================================


@pytest.mark.benchmark
class TestMACSEvaluatorInit:
    """Tests for MACSEvaluator initialization."""

    def test_init_user_type_with_template(self, macs_model, sample_task):
        """Initializes with gsr_type='user' and loads appropriate template."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        assert evaluator.gsr_type == "user"
        assert evaluator.model == macs_model
        assert evaluator.task == sample_task
        assert "{{scenario}}" in evaluator.template
        assert "{{history}}" in evaluator.template
        assert "{{assertions}}" in evaluator.template

    def test_init_system_type_with_template(self, macs_model, sample_task):
        """Initializes with gsr_type='system' and includes invocations placeholder."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        assert evaluator.gsr_type == "system"
        assert "{{invocations}}" in evaluator.template

    def test_init_custom_template(self, macs_model, sample_task):
        """Custom template overrides default."""
        custom = "Custom template: {{scenario}} {{history}} {{assertions}}"
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user", template=custom)

        assert evaluator.template == custom


# =============================================================================
# Unit Tests: Assertion Parsing
# =============================================================================


@pytest.mark.benchmark
class TestAssertionParsing:
    """Tests for _parse_assertions method."""

    def test_parse_user_assertions(self, macs_model, sample_task):
        """Filters to user assertions only."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        assertions = [
            "user: This is a user assertion",
            "agent: This is an agent assertion",
            "user: Another user assertion",
        ]
        parsed = evaluator._parse_assertions(assertions)

        assert len(parsed) == 2
        assert "This is a user assertion" in parsed
        assert "Another user assertion" in parsed
        assert "This is an agent assertion" not in parsed

    def test_parse_system_assertions(self, macs_model, sample_task):
        """Filters to system (agent:) assertions."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        assertions = [
            "user: User assertion",
            "agent: Agent assertion 1",
            "agent: Agent assertion 2",
        ]
        parsed = evaluator._parse_assertions(assertions)

        assert len(parsed) == 2
        assert "Agent assertion 1" in parsed
        assert "Agent assertion 2" in parsed
        assert "User assertion" not in parsed

    def test_parse_no_prefix_is_user(self, macs_model, sample_task):
        """Unprefixed assertions are user type (AWS default)."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        assertions = [
            "No prefix assertion",
            "user: Explicit user assertion",
        ]
        parsed = evaluator._parse_assertions(assertions)

        assert len(parsed) == 2
        assert "No prefix assertion" in parsed
        assert "Explicit user assertion" in parsed

    def test_parse_mixed_assertions(self, macs_model, sample_task):
        """Correctly splits mixed assertions."""
        evaluator_user = MACSEvaluator(macs_model, sample_task, gsr_type="user")
        evaluator_system = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        assertions = [
            "user: User only",
            "agent: Agent only",
            "Unprefixed becomes user",
        ]

        user_parsed = evaluator_user._parse_assertions(assertions)
        system_parsed = evaluator_system._parse_assertions(assertions)

        assert len(user_parsed) == 2  # user: + unprefixed
        assert len(system_parsed) == 1  # agent: only

    def test_parse_empty_assertions(self, macs_model, sample_task):
        """Empty list returns empty."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        parsed = evaluator._parse_assertions([])

        assert parsed == []

    def test_parse_case_insensitive(self, macs_model, sample_task):
        """Prefix matching is case-insensitive."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        assertions = [
            "USER: Uppercase user",
            "User: Mixed case user",
            "AGENT: Should be excluded",
        ]
        parsed = evaluator._parse_assertions(assertions)

        assert len(parsed) == 2
        assert "Uppercase user" in parsed
        assert "Mixed case user" in parsed


# =============================================================================
# Unit Tests: Trace Filtering
# =============================================================================


@pytest.mark.benchmark
class TestTraceFiltering:
    """Tests for filter_traces method."""

    def test_filter_traces_user_type(self, macs_model, sample_task, sample_trace):
        """User type gets user messages only."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        # User trace uses 'messages' key (consistent with how User.gather_traces works)
        traces = {"user": {"messages": sample_trace.to_list()}, "tools": {"tool1": {}}}
        filtered = evaluator.filter_traces(traces)

        assert "messages" in filtered
        assert isinstance(filtered["messages"], list)
        assert len(filtered["messages"]) == len(sample_trace)
        # Should not have tools in user evaluation
        assert "tools" not in filtered or filtered.get("tools") is None

    def test_filter_traces_system_type(self, macs_model, sample_task, sample_trace, sample_tool_traces):
        """System type gets messages and tool_traces."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        # Create traces with agent structure as expected by filter_traces
        traces = {
            "agents": {"test_agent": {"messages": sample_trace.to_list()}},
            "tools": sample_tool_traces,
        }
        filtered = evaluator.filter_traces(traces)

        # System should get messages and tool_traces
        assert "messages" in filtered
        assert "tool_traces" in filtered
        assert filtered["tool_traces"] == sample_tool_traces


# =============================================================================
# Unit Tests: Conversation Formatting
# =============================================================================


@pytest.mark.benchmark
class TestConversationFormatting:
    """Tests for _format_conversation_history method."""

    def test_format_conversation_history(self, macs_model, sample_task, sample_trace):
        """Formats MessageHistory to string."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        formatted = evaluator._format_conversation_history(sample_trace)

        assert "user: I need to book a flight to New York" in formatted
        assert "assistant: Sure! When would you like to travel?" in formatted
        assert "user: Next Monday" in formatted

    def test_format_conversation_list_content(self, macs_model, sample_task):
        """Handles list content (multi-modal)."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        trace = MessageHistory(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
                }
            ]
        )
        formatted = evaluator._format_conversation_history(trace)

        assert "user: Hello World" in formatted

    def test_format_tool_invocations(self, macs_model, sample_task, sample_tool_traces):
        """Formats tool traces for system eval."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        formatted = evaluator._format_tool_invocations(sample_tool_traces)

        assert "Tool: search_flights" in formatted
        assert "Inputs:" in formatted
        assert "origin" in formatted
        assert "Outputs:" in formatted
        assert "Status:" in formatted

    def test_format_tool_invocations_empty(self, macs_model, sample_task):
        """Handles empty tool traces."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="system")

        formatted = evaluator._format_tool_invocations({})

        assert "No tool invocations recorded" in formatted


# =============================================================================
# Unit Tests: GSR Computation
# =============================================================================


@pytest.mark.benchmark
class TestGSRComputation:
    """Tests for _compute_gsr method."""

    def test_compute_gsr_all_true(self, macs_model, sample_task):
        """All true → gsr=1.0, partial=1.0."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        report = [
            {"assertion": "Test 1", "answer": "TRUE"},
            {"assertion": "Test 2", "answer": "TRUE"},
            {"assertion": "Test 3", "answer": "TRUE"},
        ]
        gsr, partial = evaluator._compute_gsr(report)

        assert gsr == 1.0
        assert partial == 1.0

    def test_compute_gsr_all_false(self, macs_model, sample_task):
        """All false → gsr=0.0, partial=0.0."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        report = [
            {"assertion": "Test 1", "answer": "FALSE"},
            {"assertion": "Test 2", "answer": "FALSE"},
        ]
        gsr, partial = evaluator._compute_gsr(report)

        assert gsr == 0.0
        assert partial == 0.0

    def test_compute_gsr_mixed(self, macs_model, sample_task):
        """Mixed → gsr=0.0, partial=fraction."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        report = [
            {"assertion": "Test 1", "answer": "TRUE"},
            {"assertion": "Test 2", "answer": "FALSE"},
            {"assertion": "Test 3", "answer": "TRUE"},
        ]
        gsr, partial = evaluator._compute_gsr(report)

        assert gsr == 0.0  # Not all true
        assert partial == pytest.approx(2 / 3)

    def test_compute_gsr_empty(self, macs_model, sample_task):
        """Empty report → gsr=1.0, partial=1.0."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        gsr, partial = evaluator._compute_gsr([])

        assert gsr == 1.0
        assert partial == 1.0

    def test_compute_gsr_case_insensitive(self, macs_model, sample_task):
        """Answer matching is case-insensitive."""
        evaluator = MACSEvaluator(macs_model, sample_task, gsr_type="user")

        report = [
            {"assertion": "Test 1", "answer": "true"},
            {"assertion": "Test 2", "answer": "True"},
            {"assertion": "Test 3", "answer": "TRUE"},
        ]
        gsr, partial = evaluator._compute_gsr(report)

        assert gsr == 1.0
        assert partial == 1.0


# =============================================================================
# Unit Tests: Evaluation Call
# =============================================================================


@pytest.mark.benchmark
class TestEvaluationCall:
    """Tests for __call__ method."""

    def test_call_returns_expected_format(self, sample_task, sample_trace):
        """Returns gsr, partial_gsr, report."""
        response = json.dumps(
            [
                {"assertion": "Flight booking was confirmed", "answer": "TRUE", "evidence": "Confirmation ABC123"},
                {"assertion": "User received confirmation number", "answer": "TRUE", "evidence": "ABC123 mentioned"},
            ]
        )
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert "gsr" in result
        assert "partial_gsr" in result
        assert "report" in result
        assert result["gsr"] == 1.0
        assert len(result["report"]) == 2

    def test_call_handles_json_error(self, sample_task, sample_trace):
        """Graceful handling of JSON parse error."""
        model = DummyModelAdapter(responses=["This is not valid JSON"])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert result["gsr"] == 0.0
        assert result["partial_gsr"] == 0.0
        assert "error" in result
        assert "JSON decode error" in result["error"]
        assert "raw_response" in result

    def test_call_handles_wrapped_response(self, sample_task, sample_trace):
        """Handles {'assertions': [...]} wrapper."""
        response = json.dumps({"assertions": [{"assertion": "Test", "answer": "TRUE", "evidence": "Found"}]})
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert result["gsr"] == 1.0
        assert len(result["report"]) == 1

    def test_call_handles_results_wrapper(self, sample_task, sample_trace):
        """Handles {'results': [...]} wrapper."""
        response = json.dumps({"results": [{"assertion": "Test", "answer": "FALSE", "evidence": "Not found"}]})
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert result["gsr"] == 0.0
        assert len(result["report"]) == 1

    def test_call_handles_single_dict_response(self, sample_task, sample_trace):
        """Handles single dict instead of list."""
        response = json.dumps({"assertion": "Test", "answer": "TRUE", "evidence": "Found"})
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert len(result["report"]) == 1

    def test_call_missing_scenario_raises(self, sample_trace):
        """Missing scenario raises ValueError."""
        task = Task(
            query="Test query",
            environment_data={},
            evaluation_data={"assertions": ["user: Test assertion"]},
            metadata={},  # No scenario!
        )
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        evaluator = MACSEvaluator(model, task, gsr_type="user")

        traces = {"messages": sample_trace}
        with pytest.raises(ValueError, match="scenario"):
            evaluator(traces)

    def test_call_no_assertions_returns_perfect(self, sample_task_no_assertions, sample_trace):
        """No assertions → perfect score."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        evaluator = MACSEvaluator(model, sample_task_no_assertions, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert result["gsr"] == 1.0
        assert result["partial_gsr"] == 1.0
        assert result["report"] == []

    def test_call_adds_assertion_type(self, sample_task, sample_trace):
        """Report items include assertion_type."""
        response = json.dumps([{"assertion": "Test", "answer": "TRUE", "evidence": "Found"}])
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="user")

        traces = {"messages": sample_trace}
        result = evaluator(traces)

        assert result["report"][0]["assertion_type"] == "user"

    def test_call_system_includes_tool_invocations(self, sample_task, sample_trace, sample_tool_traces):
        """System evaluation includes tool invocations in prompt."""
        from unittest.mock import patch

        response = json.dumps([{"assertion": "Test", "answer": "TRUE", "evidence": "Found"}])
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, sample_task, gsr_type="system")

        traces = {"messages": sample_trace, "tool_traces": sample_tool_traces}

        # Capture the messages sent to the model
        captured_messages = []
        original_chat = model._chat_impl

        def capture_messages(messages, *args, **kwargs):
            captured_messages.append(messages)
            return original_chat(messages, *args, **kwargs)

        with patch.object(model, "_chat_impl", side_effect=capture_messages):
            evaluator(traces)

        # Check that tool invocations were included in the prompt
        assert len(captured_messages) > 0
        # The prompt is in the first user message content
        prompt = captured_messages[0][0]["content"]
        assert "search_flights" in prompt or "book_flight" in prompt


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSEvaluatorIntegration:
    """Integration tests for MACSEvaluator."""

    def test_full_user_evaluation(self):
        """Complete user-side evaluation flow."""
        task = Task(
            query="Book a flight",
            environment_data={},
            evaluation_data={
                "assertions": [
                    "user: Booking was successful",
                    "agent: Internal record created",  # Should be ignored for user eval
                ]
            },
            metadata={"scenario": "Business traveler booking a flight"},
        )

        response = json.dumps([{"assertion": "Booking was successful", "answer": "TRUE", "evidence": "Confirmed"}])
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, task, gsr_type="user")

        trace = MessageHistory(
            [
                {"role": "user", "content": "Book a flight"},
                {"role": "assistant", "content": "Your flight is booked!"},
            ]
        )

        result = evaluator({"messages": trace})

        assert result["gsr"] == 1.0
        assert len(result["report"]) == 1
        assert result["report"][0]["assertion_type"] == "user"

    def test_full_system_evaluation(self):
        """Complete system-side evaluation flow."""
        task = Task(
            query="Book a flight",
            environment_data={},
            evaluation_data={
                "assertions": [
                    "user: Should be ignored for system eval",
                    "agent: Database was updated",
                ]
            },
            metadata={"scenario": "System checking internal operations"},
        )

        response = json.dumps([{"assertion": "Database was updated", "answer": "TRUE", "evidence": "DB log shows update"}])
        model = DummyModelAdapter(responses=[response])
        evaluator = MACSEvaluator(model, task, gsr_type="system")

        trace = MessageHistory([{"role": "assistant", "content": "Done"}])
        tool_traces = {"update_db": {"invocations": [{"inputs": {}, "outputs": "OK", "status": "success"}]}}

        result = evaluator({"messages": trace, "tool_traces": tool_traces})

        assert result["gsr"] == 1.0
        assert result["report"][0]["assertion_type"] == "system"
