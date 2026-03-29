"""Test MessageHistory class behavior.

These tests verify that MessageHistory behaves like a list and correctly
handles multi-modal content, tool calls, and metadata.
"""

import pytest
from maseval import MessageHistory


@pytest.mark.core
class TestMessageHistory:
    """Tests for MessageHistory list-like interface and functionality."""

    def test_message_history_iterable_interface(self):
        """Test that MessageHistory can be iterated like a list."""
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        history.add_message("user", "How are you?")

        messages = list(history)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_message_history_indexing_and_slicing(self):
        """Test that MessageHistory supports indexing and slicing."""
        history = MessageHistory()
        history.add_message("user", "Message 1")
        history.add_message("assistant", "Message 2")
        history.add_message("user", "Message 3")
        history.add_message("assistant", "Message 4")

        # Test indexing
        assert history[0]["content"] == "Message 1"
        assert history[1]["content"] == "Message 2"
        assert history[-1]["content"] == "Message 4"
        assert history[-2]["content"] == "Message 3"

        # Test slicing
        slice_result = history[1:3]
        assert len(slice_result) == 2
        assert slice_result[0]["content"] == "Message 2"
        assert slice_result[1]["content"] == "Message 3"

        # Test slice from start
        assert len(history[:2]) == 2
        # Test slice to end
        assert len(history[2:]) == 2

    def test_message_history_boolean_context(self):
        """Test that MessageHistory works in boolean contexts."""
        history = MessageHistory()

        # Empty history should be falsy
        assert not history
        assert len(history) == 0

        # Non-empty history should be truthy
        history.add_message("user", "Hello")
        assert history
        assert len(history) == 1

    def test_message_history_tool_calls(self):
        """Test that MessageHistory correctly handles tool calls."""
        history = MessageHistory()

        # Add regular message
        history.add_message("user", "What's the weather?")

        # Add tool call message
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        history.add_tool_call(tool_calls=tool_calls, content="Let me check the weather")

        # Add tool response
        history.add_tool_response(tool_call_id="call_123", content="Temperature: 72°F", name="get_weather")

        # Verify structure
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "tool_calls" in history[1]
        assert history[1]["tool_calls"][0]["id"] == "call_123"
        assert history[2]["role"] == "tool"
        assert history[2]["tool_call_id"] == "call_123"
        assert history[2]["content"] == "Temperature: 72°F"

    def test_message_history_multi_modal_content(self):
        """Test that MessageHistory handles multi-modal content correctly."""
        history = MessageHistory()

        # Add text message
        history.add_message("user", "Hello")

        # Add multi-modal message (text + image)
        multi_modal_content = [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"},
            },
        ]
        history.add_message("user", multi_modal_content)

        # Verify
        assert len(history) == 2
        assert isinstance(history[0]["content"], str)
        assert isinstance(history[1]["content"], list)
        assert len(history[1]["content"]) == 2
        assert history[1]["content"][0]["type"] == "text"
        assert history[1]["content"][1]["type"] == "image_url"

    def test_message_history_metadata_preservation(self):
        """Test that MessageHistory preserves metadata and timestamps."""
        history = MessageHistory()

        # Add message with metadata
        history.add_message(
            role="user",
            content="Test message",
            name="TestAgent",
            metadata={"custom_field": "custom_value", "importance": "high"},
        )

        message = history[0]
        assert message["role"] == "user"
        assert message["content"] == "Test message"
        assert message["name"] == "TestAgent"
        assert "timestamp" in message
        assert message["metadata"]["custom_field"] == "custom_value"
        assert message["metadata"]["importance"] == "high"

    def test_message_history_to_list_conversion(self):
        """Test that MessageHistory can be converted to a list."""
        history = MessageHistory()
        history.add_message("user", "Message 1")
        history.add_message("assistant", "Message 2")
        history.add_message("user", "Message 3")

        # Test to_list() method
        messages_list = history.to_list()
        assert isinstance(messages_list, list)
        assert len(messages_list) == 3
        assert messages_list[0]["content"] == "Message 1"

        # Test list() conversion
        messages_list2 = list(history)
        assert messages_list2 == messages_list

    def test_message_history_initialization_with_messages(self):
        """Test that MessageHistory can be initialized with existing messages."""
        initial_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        history = MessageHistory(initial_messages)
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there"

    def test_message_history_empty_initialization(self):
        """Test that MessageHistory can be initialized empty."""
        history = MessageHistory()
        assert len(history) == 0
        assert not history
        assert list(history) == []

    def test_message_history_repr(self):
        """Test that MessageHistory has a useful string representation."""
        history = MessageHistory()
        history.add_message("user", "Test")

        repr_str = repr(history)
        assert "MessageHistory" in repr_str
        assert "1" in repr_str  # Should mention number of messages

    def test_message_history_system_messages(self):
        """Test that MessageHistory handles system messages correctly."""
        history = MessageHistory()
        history.add_message("system", "You are a helpful assistant")
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")

        assert len(history) == 3
        assert history[0]["role"] == "system"
        assert history[0]["content"] == "You are a helpful assistant"

    def test_message_history_custom_timestamp(self):
        """Test that custom timestamps can be provided."""
        history = MessageHistory()

        custom_timestamp = "2024-01-01T12:00:00"
        history.add_message("user", "Test", timestamp=custom_timestamp)

        assert history[0]["timestamp"] == custom_timestamp

    def test_message_history_multiple_tool_calls(self):
        """Test that multiple tool calls can be added to a single message."""
        history = MessageHistory()

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"timezone": "EST"}'},
            },
        ]

        history.add_tool_call(tool_calls=tool_calls)

        assert len(history) == 1
        assert len(history[0]["tool_calls"]) == 2
        assert history[0]["tool_calls"][0]["id"] == "call_1"
        assert history[0]["tool_calls"][1]["id"] == "call_2"

    def test_message_history_iteration_doesnt_modify(self):
        """Test that iterating over history doesn't modify it."""
        history = MessageHistory()
        history.add_message("user", "Message 1")
        history.add_message("assistant", "Message 2")

        # Iterate multiple times
        for _ in range(3):
            messages = list(history)
            assert len(messages) == 2

        # Verify still has 2 messages
        assert len(history) == 2

    def test_clear(self):
        """Test that clear removes all messages."""
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        assert len(history) == 2
        history.clear()
        assert len(history) == 0
        assert not history

    def test_filter_by_role(self):
        """Test filtering messages by role."""
        history = MessageHistory()
        history.add_message("user", "U1")
        history.add_message("assistant", "A1")
        history.add_message("user", "U2")
        history.add_message("system", "S1")
        user_msgs = history.filter_by_role("user")
        assert len(user_msgs) == 2
        assert all(m["role"] == "user" for m in user_msgs)
        assert history.filter_by_role("system") == [history[3]]
        assert history.filter_by_role("tool") == []

    def test_get_last_message(self):
        """Test get_last_message returns last message or None."""
        history = MessageHistory()
        assert history.get_last_message() is None
        history.add_message("user", "First")
        history.add_message("assistant", "Second")
        last = history.get_last_message()
        assert last is not None
        assert last["content"] == "Second"

    def test_to_openai_format_strips_metadata_and_timestamps(self):
        """Test to_openai_format strips metadata and timestamps."""
        history = MessageHistory()
        history.add_message("user", "Hello", metadata={"key": "val"})
        history.add_message("assistant", "Hi", name="bot")
        openai_msgs = history.to_openai_format()
        assert len(openai_msgs) == 2
        assert "metadata" not in openai_msgs[0]
        assert "timestamp" not in openai_msgs[0]
        assert openai_msgs[0] == {"role": "user", "content": "Hello"}
        assert openai_msgs[1] == {"role": "assistant", "content": "Hi", "name": "bot"}

    def test_to_openai_format_preserves_tool_calls(self):
        """Test to_openai_format preserves tool_calls and tool_call_id."""
        history = MessageHistory()
        tc = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        history.add_tool_call(tool_calls=tc)
        history.add_tool_response(tool_call_id="c1", content="result")
        openai_msgs = history.to_openai_format()
        assert openai_msgs[0]["tool_calls"] == tc
        assert openai_msgs[1]["tool_call_id"] == "c1"

    def test_add_tool_call_without_content(self):
        """Test add_tool_call without content omits content key."""
        history = MessageHistory()
        history.add_tool_call(tool_calls=[{"id": "c1"}])
        assert "content" not in history[0]

    def test_add_tool_call_with_content(self):
        """Test add_tool_call with content includes it."""
        history = MessageHistory()
        history.add_tool_call(tool_calls=[{"id": "c1"}], content="thinking...")
        assert history[0]["content"] == "thinking..."


@pytest.mark.core
class TestToolInvocationHistory:
    """Tests for ToolInvocationHistory."""

    def test_add_and_len(self):
        from maseval.core.history import ToolInvocationHistory

        h = ToolInvocationHistory()
        assert len(h) == 0
        h.add_invocation(inputs={"x": 1}, outputs={"y": 2}, status="success")
        assert len(h) == 1

    def test_to_list(self):
        from maseval.core.history import ToolInvocationHistory

        h = ToolInvocationHistory()
        h.add_invocation(inputs="in", outputs="out", status="ok")
        records = h.to_list()
        assert len(records) == 1
        assert records[0]["inputs"] == "in"
        assert records[0]["outputs"] == "out"
        assert records[0]["status"] == "ok"
        assert "id" in records[0]
        assert "timestamp" in records[0]

    def test_init_with_records(self):
        from maseval.core.history import ToolInvocationHistory

        h = ToolInvocationHistory(records=[{"id": "x", "inputs": 1, "outputs": 2, "status": "ok"}])
        assert len(h) == 1
