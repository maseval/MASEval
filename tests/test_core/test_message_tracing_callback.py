"""Tests for MessageTracingAgentCallback.

These tests verify that MessageTracingAgentCallback correctly captures agent
conversations to memory, including message content, metadata, tool calls, and
multi-agent interactions. This callback enables detailed analysis of agent
behavior without modifying core execution logic.
"""

import pytest

from maseval.core.callbacks import MessageTracingAgentCallback
from maseval.core.agent import AgentAdapter
from maseval.core.history import MessageHistory
from conftest import DummyAgent


class TracingTestAgentAdapter(AgentAdapter):
    """Test adapter implementation that populates message history for testing.

    This adapter simulates realistic agent behavior by creating proper message
    histories with user queries, assistant responses, and optional tool calls.
    """

    def _run_agent(self, query: str) -> str:
        history = MessageHistory()
        history.add_message(role="user", content=query)

        response = self.agent.run(query)

        # Add tool call if query mentions "tool"
        if "tool" in query.lower():
            # Add assistant message with tool call
            history.add_tool_call(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ]
            )
            # Add tool response
            history.add_tool_response(tool_call_id="call_1", content="Tool result", name="test_tool")
            # Add final assistant response after tool use
            history.add_message(role="assistant", content=response)
        else:
            # Normal response without tools
            history.add_message(role="assistant", content=response)

        # Store history directly
        self.messages = history

        return response


@pytest.mark.core
class TestMessageTracingCallback:
    """Tests for MessageTracingAgentCallback."""

    def test_initialization(self):
        """Test callback initialization with default parameters.

        Verifies that callback starts with empty conversation history and
        correct default settings for metadata inclusion and verbosity.
        """
        callback = MessageTracingAgentCallback()
        assert callback.include_metadata is True
        assert callback.verbose is False
        assert len(callback.get_all_conversations()) == 0

    def test_basic_tracing(self):
        """Test basic message tracing captures conversation to memory.

        Verifies that a single agent run creates a conversation record with
        query, agent name, message count, and complete message history.
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="test_agent", callbacks=[callback])

        # Run agent - should trigger callbacks
        agent_adapter.run("Test query")

        # Check traced conversation
        conversations = callback.get_all_conversations()
        assert len(conversations) == 1

        conv = conversations[0]
        assert conv["agent_name"] == "test_agent"
        assert conv["query"] == "Test query"
        assert conv["message_count"] == 2
        assert len(conv["messages"]) == 2
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][1]["role"] == "assistant"

    def test_multiple_conversations(self):
        """Test tracing multiple conversations from same agent.

        Verifies that callback correctly accumulates separate conversation
        records for multiple sequential agent runs, maintaining query order.
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="agent1", callbacks=[callback])

        queries = ["Query 1", "Query 2", "Query 3"]
        for query in queries:
            agent_adapter.run(query)

        # Check all conversations traced
        conversations = callback.get_all_conversations()
        assert len(conversations) == 3

        for i, conv in enumerate(conversations):
            assert conv["query"] == queries[i]
            assert conv["agent_name"] == "agent1"

    def test_metadata_included(self):
        """Test that conversation metadata is captured when enabled.

        Verifies that metadata fields (timestamp, tool call indicators, role
        counts) are included in conversation records when metadata flag is True.
        """
        callback = MessageTracingAgentCallback(include_metadata=True)
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="agent", callbacks=[callback])

        agent_adapter.run("Test query with tool")

        conv = callback.get_all_conversations()[0]
        assert "metadata" in conv
        assert "timestamp" in conv["metadata"]
        assert "has_tool_calls" in conv["metadata"]
        assert conv["metadata"]["has_tool_calls"] is True
        assert "roles" in conv["metadata"]

    def test_metadata_excluded(self):
        """Test that metadata is excluded when disabled.

        Verifies that conversation records contain only core fields (query,
        agent name, messages) when include_metadata is set to False.
        """
        callback = MessageTracingAgentCallback(include_metadata=False)
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="agent", callbacks=[callback])

        agent_adapter.run("Test query")

        conv = callback.get_all_conversations()[0]
        assert "metadata" not in conv

        # Check messages don't have timestamps
        for msg in conv["messages"]:
            assert "timestamp" not in msg

    def test_multi_agent_tracing(self):
        """Test tracing conversations from multiple different agents.

        Verifies that a single callback instance can track conversations from
        multiple agents simultaneously and retrieve them per-agent or globally.
        """
        callback = MessageTracingAgentCallback()

        # Create two agents
        agent1 = DummyAgent()
        adapter1 = TracingTestAgentAdapter(agent1, name="agent1", callbacks=[callback])

        agent2 = DummyAgent()
        adapter2 = TracingTestAgentAdapter(agent2, name="agent2", callbacks=[callback])

        # Run both agents
        adapter1.run("Query for agent1")
        adapter2.run("Query for agent2")
        adapter1.run("Another query for agent1")

        # Check all conversations traced
        conversations = callback.get_all_conversations()
        assert len(conversations) == 3

        # Check per-agent retrieval
        agent1_convs = callback.get_conversations_by_agent("agent1")
        agent2_convs = callback.get_conversations_by_agent("agent2")

        assert len(agent1_convs) == 2
        assert len(agent2_convs) == 1

    def test_statistics(self):
        """Test conversation statistics generation.

        Verifies that callback correctly aggregates statistics including total
        conversations, message counts, averages, and per-agent breakdowns.
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="test_agent", callbacks=[callback])

        # Run multiple times
        agent_adapter.run("Query 1")
        agent_adapter.run("Query 2 with tool")

        stats = callback.get_statistics()

        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 6  # 2 + 4 (with tool calls)
        assert stats["avg_messages_per_conversation"] == 3.0
        assert "test_agent" in stats["agents"]

        agent_stats = stats["agent_statistics"]["test_agent"]
        assert agent_stats["conversations"] == 2
        assert agent_stats["total_messages"] == 6

    def test_clear(self):
        """Test clearing all stored conversations.

        Verifies that clear() removes all conversation records from memory,
        useful for resetting state between benchmark runs.
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="agent", callbacks=[callback])

        # Trace some conversations
        agent_adapter.run("Query 1")
        agent_adapter.run("Query 2")
        assert len(callback.get_all_conversations()) == 2

        # Clear
        callback.clear()
        assert len(callback.get_all_conversations()) == 0

    def test_tool_call_tracing(self):
        """Test that tool calls and responses are captured in conversation trace.

        Verifies that conversations with tool usage include the full sequence:
        user message, assistant message with tool_calls, tool message, and
        final assistant response.
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()
        agent_adapter = TracingTestAgentAdapter(agent, name="agent", callbacks=[callback])

        # Run query that triggers tool call
        agent_adapter.run("Query with tool")

        conv = callback.get_all_conversations()[0]

        # Should have: user, assistant (with tool_calls), tool, assistant (final)
        assert conv["message_count"] == 4

        messages = conv["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "assistant"

    def test_no_history_handling(self):
        """Test graceful handling when agent returns empty message history.

        Verifies that callback creates valid conversation records even when
        agent adapters return empty histories (edge case for minimal agents).
        """
        callback = MessageTracingAgentCallback()
        agent = DummyAgent()

        class NoHistoryAdapter(AgentAdapter):
            def get_messages(self):
                return MessageHistory()

            def _run_agent(self, query: str):
                return MessageHistory()

        agent_adapter = NoHistoryAdapter(agent, name="agent", callbacks=[callback])

        agent_adapter.run("Test")

        # Should still trace, but with empty messages
        conversations = callback.get_all_conversations()
        assert len(conversations) == 1
        assert conversations[0]["message_count"] == 0

    def test_repr(self):
        """Test string representation shows conversation count.

        Verifies that callback's string representation includes the current
        number of traced conversations for debugging and logging.
        """
        callback = MessageTracingAgentCallback()

        repr_str = repr(callback)
        assert "MessageTracingAgentCallback" in repr_str
        assert "conversations=0" in repr_str

    def test_verbose_mode_prints(self, capsys):
        """Test that verbose=True produces output."""
        callback = MessageTracingAgentCallback(verbose=True)
        agent = DummyAgent()
        adapter = TracingTestAgentAdapter(agent_instance=agent, name="verbose_agent", callbacks=[callback])
        adapter.run("Hello")
        captured = capsys.readouterr().out
        assert len(captured) > 0

    def test_get_conversations_by_agent_nonexistent(self):
        """Test that querying a nonexistent agent returns empty list."""
        callback = MessageTracingAgentCallback()
        assert callback.get_conversations_by_agent("nonexistent") == []

    def test_get_statistics_empty(self):
        """Test statistics on empty callback."""
        callback = MessageTracingAgentCallback()
        stats = callback.get_statistics()
        assert stats["total_conversations"] == 0
        assert stats["total_messages"] == 0
        assert stats["agents"] == []
