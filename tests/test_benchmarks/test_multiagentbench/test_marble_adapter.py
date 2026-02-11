"""Tests for MarbleAgentAdapter."""

import pytest
from unittest.mock import MagicMock

from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
    MarbleAgentAdapter,
)
from maseval import AgentError

pytestmark = pytest.mark.benchmark


class TestMarbleAgentAdapter:
    """Tests for MarbleAgentAdapter class."""

    @pytest.fixture
    def mock_marble_agent(self):
        """Create a mock MARBLE agent."""
        agent = MagicMock()
        agent.profile = "Test researcher profile"
        agent.strategy = "cot"
        agent.llm = "gpt-4o"
        agent.relationships = {"agent2": "collaborator"}
        agent.task_history = ["task1", "task2"]
        agent.session_id = "session_123"
        agent.act.return_value = ("Test result", "Communication message")
        agent.get_token_usage.return_value = 500
        agent.seralize_message.return_value = "Serialized messages"
        return agent

    @pytest.fixture
    def adapter(self, mock_marble_agent):
        """Create a MarbleAgentAdapter."""
        return MarbleAgentAdapter(
            marble_agent=mock_marble_agent,
            agent_id="test_agent",
        )

    def test_init(self, adapter, mock_marble_agent):
        """Adapter should initialize correctly."""
        assert adapter.agent_id == "test_agent"
        assert adapter.profile == "Test researcher profile"
        assert adapter.marble_agent is mock_marble_agent

    def test_agent_id_property(self, adapter):
        """agent_id property should return correct value."""
        assert adapter.agent_id == "test_agent"

    def test_profile_property(self, adapter):
        """profile property should return correct value."""
        assert adapter.profile == "Test researcher profile"

    def test_marble_agent_property(self, adapter, mock_marble_agent):
        """marble_agent property should return the underlying agent."""
        assert adapter.marble_agent is mock_marble_agent

    def test_run_agent_success(self, adapter, mock_marble_agent):
        """_run_agent should execute MARBLE agent's act method."""
        result = adapter._run_agent("Test query")

        mock_marble_agent.act.assert_called_once_with("Test query")
        assert result == "Test result"

    def test_run_agent_logs_action(self, adapter, mock_marble_agent):
        """_run_agent should log actions."""
        adapter._run_agent("Test query")

        assert len(adapter._action_log) == 1
        assert adapter._action_log[0]["task"] == "Test query"
        assert adapter._action_log[0]["result"] == "Test result"
        assert adapter._action_log[0]["has_communication"] is True

    def test_run_agent_logs_communication(self, adapter, mock_marble_agent):
        """_run_agent should log communications when present."""
        adapter._run_agent("Test query")

        assert len(adapter._communication_log) == 1
        assert adapter._communication_log[0]["communication"] == "Communication message"

    def test_run_agent_no_communication(self, adapter, mock_marble_agent):
        """_run_agent should handle None communication."""
        mock_marble_agent.act.return_value = ("Result", None)
        adapter._run_agent("Test query")

        assert len(adapter._communication_log) == 0
        assert adapter._action_log[0]["has_communication"] is False

    def test_run_agent_updates_messages(self, adapter, mock_marble_agent):
        """_run_agent should update message history."""
        adapter._run_agent("Test query")

        messages = adapter.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test query"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Test result"

    def test_run_agent_error(self, adapter, mock_marble_agent):
        """_run_agent should wrap exceptions in AgentError."""
        mock_marble_agent.act.side_effect = RuntimeError("Agent failed")

        with pytest.raises(AgentError, match="MARBLE agent 'test_agent' failed"):
            adapter._run_agent("Test query")

    def test_get_token_usage(self, adapter, mock_marble_agent):
        """get_token_usage should return token count from MARBLE agent."""
        assert adapter.get_token_usage() == 500
        mock_marble_agent.get_token_usage.assert_called_once()

    def test_get_token_usage_no_method(self, adapter, mock_marble_agent):
        """get_token_usage should return 0 if method not available."""
        del mock_marble_agent.get_token_usage
        assert adapter.get_token_usage() == 0

    def test_get_memory_str(self, adapter, mock_marble_agent):
        """get_memory_str should return memory string."""
        mock_memory = MagicMock()
        mock_memory.get_memory_str.return_value = "Memory contents"
        mock_marble_agent.memory = mock_memory

        assert adapter.get_memory_str() == "Memory contents"

    def test_get_memory_str_no_memory(self, adapter, mock_marble_agent):
        """get_memory_str should return empty string if no memory."""
        mock_marble_agent.memory = None
        assert adapter.get_memory_str() == ""

    def test_get_memory_str_no_method(self, adapter, mock_marble_agent):
        """get_memory_str should return empty string if no get_memory_str method."""
        mock_marble_agent.memory = MagicMock(spec=[])  # No methods
        assert adapter.get_memory_str() == ""

    def test_get_serialized_messages(self, adapter, mock_marble_agent):
        """get_serialized_messages should return serialized messages."""
        result = adapter.get_serialized_messages("session_1")

        mock_marble_agent.seralize_message.assert_called_once_with("session_1")
        assert result == "Serialized messages"

    def test_get_serialized_messages_no_method(self, adapter, mock_marble_agent):
        """get_serialized_messages should return empty string if no method."""
        del mock_marble_agent.seralize_message
        assert adapter.get_serialized_messages() == ""

    def test_gather_traces(self, adapter, mock_marble_agent):
        """gather_traces should include MARBLE-specific data."""
        # Run agent to generate some data
        adapter._run_agent("Test query")

        traces = adapter.gather_traces()

        assert traces["agent_id"] == "test_agent"
        assert traces["profile"] == "Test researcher profile"
        assert traces["token_usage"] == 500
        assert len(traces["action_log"]) == 1
        assert len(traces["communication_log"]) == 1
        assert traces["relationships"] == {"agent2": "collaborator"}
        assert traces["task_history"] == ["task1", "task2"]

    def test_gather_traces_no_relationships(self, adapter, mock_marble_agent):
        """gather_traces should handle missing relationships."""
        del mock_marble_agent.relationships
        traces = adapter.gather_traces()
        assert traces["relationships"] == {}

    def test_gather_traces_no_task_history(self, adapter, mock_marble_agent):
        """gather_traces should handle missing task_history."""
        del mock_marble_agent.task_history
        traces = adapter.gather_traces()
        assert traces["task_history"] == []

    def test_gather_config(self, adapter, mock_marble_agent):
        """gather_config should include MARBLE configuration."""
        config = adapter.gather_config()

        assert config["agent_id"] == "test_agent"
        assert config["profile"] == "Test researcher profile"
        assert config["strategy"] == "cot"
        assert config["llm"] == "gpt-4o"

    def test_gather_config_missing_attrs(self, adapter, mock_marble_agent):
        """gather_config should handle missing attributes."""
        del mock_marble_agent.strategy
        del mock_marble_agent.llm

        config = adapter.gather_config()

        assert config["strategy"] == "default"
        assert config["llm"] == "unknown"


class TestCreateMarbleAgentsImportError:
    """Tests for create_marble_agents when MARBLE is not available."""

    def test_create_marble_agents_import_error(self):
        """create_marble_agents should raise ImportError when MARBLE not available."""
        import sys
        from unittest.mock import patch

        from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
            create_marble_agents,
        )

        # Temporarily remove marble modules to simulate MARBLE not being available
        marble_modules = {k: v for k, v in sys.modules.items() if "marble" in k}
        for module_name in marble_modules:
            sys.modules.pop(module_name, None)

        try:
            with patch.dict("sys.modules", {"marble.agent.base_agent": None}):
                with pytest.raises(ImportError, match="MARBLE is not available"):
                    create_marble_agents(
                        agent_configs=[{"agent_id": "test"}],
                        marble_env=MagicMock(),
                        model="gpt-4o",
                    )
        finally:
            sys.modules.update(marble_modules)
