"""Tests for MessageHistory and ToolInvocationHistory."""

import pytest

from maseval.core.history import MessageHistory


@pytest.mark.core
class TestMessageHistory:
    """Tests for MessageHistory conversation container."""

    def test_add_tool_call_with_metadata(self):
        """Tool call messages store optional metadata."""
        history = MessageHistory()
        tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        history.add_tool_call(tc, content="Calling f", metadata={"key": "val"})

        msg = history[0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == tc
        assert msg["content"] == "Calling f"
        assert msg["metadata"] == {"key": "val"}
        assert "timestamp" in msg

    def test_add_tool_call_without_content(self):
        """Tool call messages work without optional text content."""
        history = MessageHistory()
        tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        history.add_tool_call(tc)

        msg = history[0]
        assert "content" not in msg

    def test_add_tool_response_with_name_and_metadata(self):
        """Tool response messages store name and metadata."""
        history = MessageHistory()
        history.add_tool_response(
            tool_call_id="call_1",
            content="result",
            name="my_tool",
            metadata={"took_ms": 42},
        )

        msg = history[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == "result"
        assert msg["name"] == "my_tool"
        assert msg["metadata"] == {"took_ms": 42}
        assert "timestamp" in msg

    def test_add_tool_response_minimal(self):
        """Tool response works with only required fields."""
        history = MessageHistory()
        history.add_tool_response(tool_call_id="call_1", content="ok")

        msg = history[0]
        assert msg["role"] == "tool"
        assert "name" not in msg
        assert "metadata" not in msg

    def test_clear(self):
        """Clear removes all messages."""
        history = MessageHistory()
        history.add_message("user", "hello")
        history.add_message("assistant", "hi")
        assert len(history) == 2

        history.clear()
        assert len(history) == 0
        assert not history  # bool check

    def test_filter_by_role(self):
        """filter_by_role returns only messages with matching role."""
        history = MessageHistory()
        history.add_message("user", "q1")
        history.add_message("assistant", "a1")
        history.add_message("user", "q2")
        history.add_message("system", "sys")

        user_msgs = history.filter_by_role("user")
        assert len(user_msgs) == 2
        assert all(m["role"] == "user" for m in user_msgs)

        system_msgs = history.filter_by_role("system")
        assert len(system_msgs) == 1

    def test_get_last_message(self):
        """get_last_message returns last or None if empty."""
        history = MessageHistory()
        assert history.get_last_message() is None

        history.add_message("user", "first")
        history.add_message("assistant", "second")
        last = history.get_last_message()
        assert last is not None
        assert last["content"] == "second"

    def test_to_openai_format_strips_metadata_and_timestamps(self):
        """to_openai_format returns only OpenAI-compatible fields."""
        history = MessageHistory()
        history.add_message("user", "hello", metadata={"key": "val"})
        history.add_message("assistant", "hi")

        tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        history.add_tool_call(tc, content="calling")
        history.add_tool_response(tool_call_id="call_1", content="done", name="f")

        openai_msgs = history.to_openai_format()

        # All should have role
        assert all("role" in m for m in openai_msgs)
        # None should have metadata or timestamp
        assert all("metadata" not in m for m in openai_msgs)
        assert all("timestamp" not in m for m in openai_msgs)

        # Check tool call message preserves tool_calls
        tc_msg = openai_msgs[2]
        assert "tool_calls" in tc_msg

        # Check tool response preserves tool_call_id and name
        tr_msg = openai_msgs[3]
        assert tr_msg["tool_call_id"] == "call_1"
        assert tr_msg["name"] == "f"

    def test_explicit_timestamp_preserved(self):
        """Explicit timestamps are used instead of auto-generated ones."""
        history = MessageHistory()
        history.add_message("user", "hello", timestamp="2024-01-01T00:00:00")
        assert history[0]["timestamp"] == "2024-01-01T00:00:00"

        history.add_tool_call(
            [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            timestamp="2024-01-01T00:00:01",
        )
        assert history[1]["timestamp"] == "2024-01-01T00:00:01"

        history.add_tool_response(tool_call_id="c1", content="ok", timestamp="2024-01-01T00:00:02")
        assert history[2]["timestamp"] == "2024-01-01T00:00:02"
