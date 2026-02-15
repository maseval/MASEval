"""Tests for MultiAgentBench environment."""

import sys
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest

from maseval.benchmark.multiagentbench.environment import (
    MultiAgentBenchEnvironment,
    INFRASTRUCTURE_DOMAINS,
)
from maseval import EnvironmentError

pytestmark = pytest.mark.benchmark


class TestInfrastructureDomains:
    """Tests for infrastructure domain constants."""

    def test_infrastructure_domains_contains_expected(self):
        """INFRASTRUCTURE_DOMAINS should contain expected domains."""
        assert "database" in INFRASTRUCTURE_DOMAINS

    def test_infrastructure_domains_excludes_simple(self):
        """INFRASTRUCTURE_DOMAINS should not include simple domains."""
        assert "research" not in INFRASTRUCTURE_DOMAINS
        assert "bargaining" not in INFRASTRUCTURE_DOMAINS
        assert "minecraft" not in INFRASTRUCTURE_DOMAINS


class TestMultiAgentBenchEnvironment:
    """Tests for MultiAgentBenchEnvironment class."""

    def test_init_with_research_task(self, sample_research_task_data: Dict[str, Any]):
        """Environment should initialize for research domain."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.domain == "research"
        assert env.state is not None

    def test_init_with_bargaining_task(self, sample_bargaining_task_data: Dict[str, Any]):
        """Environment should initialize for bargaining domain."""
        env = MultiAgentBenchEnvironment(task_data=sample_bargaining_task_data)

        assert env.domain == "bargaining"

    def test_setup_state_extracts_domain(self, sample_research_task_data: Dict[str, Any]):
        """setup_state should extract domain from task data."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        state = env.state

        assert state["domain"] == "research"

    def test_setup_state_extracts_max_iterations(self, sample_research_task_data: Dict[str, Any]):
        """setup_state should extract max_iterations."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        state = env.state

        assert state["max_iterations"] == 10

    def test_is_done_initially_false(self, sample_research_task_data: Dict[str, Any]):
        """is_done should return False initially."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.is_done() is False

    def test_is_task_completed_initially_false(self, sample_research_task_data: Dict[str, Any]):
        """is_task_completed should return False initially."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.is_task_completed() is False

    def test_gather_traces_includes_domain(self, sample_research_task_data: Dict[str, Any]):
        """gather_traces should include domain information."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        traces = env.gather_traces()

        assert traces["domain"] == "research"
        assert "tool_invocations" in traces
        assert "marble_state" in traces
        assert "is_done" in traces

    def test_gather_config_includes_domain(self, sample_research_task_data: Dict[str, Any]):
        """gather_config should include domain information."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        config = env.gather_config()

        assert config["domain"] == "research"
        assert "tool_descriptions" in config


class TestMultiAgentBenchEnvironmentRealMarble:
    """Tests that need the real _create_marble_environment (no mock)."""

    @pytest.fixture(autouse=True)
    def _mock_marble_environment(self):
        """Override: let these tests use real marble imports."""
        yield

    def test_init_raises_without_marble(self, sample_research_task_data: Dict[str, Any]):
        """Constructor should raise ImportError when MARBLE is not available."""
        marble_modules = {k: v for k, v in sys.modules.items() if "marble" in k}
        for module_name in marble_modules:
            sys.modules.pop(module_name, None)

        try:
            with patch.dict("sys.modules", {"marble.environments.base_env": None}):
                with pytest.raises(ImportError, match="MARBLE is not available"):
                    MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        finally:
            sys.modules.update(marble_modules)

    @pytest.mark.live
    @pytest.mark.slow
    def test_marble_env_type_in_state(self, sample_research_task_data: Dict[str, Any]):
        """setup_state should include MARBLE env type (needs real marble)."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        assert env.state["marble_env_type"] == "ResearchEnvironment"


class TestInfrastructureCheck:
    """Tests for infrastructure checking."""

    def test_database_without_docker_raises(self):
        """Environment should raise for database without Docker."""
        task_data = {
            "scenario": "database",
            "environment": {"type": "DB"},
            "task": {"content": "Query database"},
            "agents": [{"agent_id": "agent1"}],
        }

        with patch("shutil.which", return_value=None):
            with pytest.raises(EnvironmentError, match="requires external infrastructure"):
                MultiAgentBenchEnvironment(task_data=task_data)

    def test_database_with_docker_passes_infra_check(self):
        """Database domain should pass infrastructure check when Docker is available."""
        task_data = {
            "scenario": "database",
            "environment": {"type": "DB"},
            "task": {"content": "Query database"},
            "agents": [{"agent_id": "agent1"}],
        }

        with patch("shutil.which", return_value="/usr/bin/docker"):
            # Mock _create_marble_environment since DBEnvironment.__init__ starts Docker + DB
            with patch.object(MultiAgentBenchEnvironment, "_create_marble_environment", return_value=MagicMock()):
                env = MultiAgentBenchEnvironment(task_data=task_data)
                assert env.domain == "database"

    def test_minecraft_no_infrastructure_check(self):
        """Minecraft should not require infrastructure check (fails at runtime instead)."""
        task_data = {
            "scenario": "minecraft",
            "environment": {"type": "Minecraft"},
            "task": {"content": "Build something"},
            "agents": [{"agent_id": "agent1"}],
        }

        env = MultiAgentBenchEnvironment(task_data=task_data)
        assert env.domain == "minecraft"


class TestWithMockedMarbleEnv:
    """Tests with mocked MARBLE environment."""

    def test_is_done_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """is_done should delegate to MARBLE env when available."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        # Mock MARBLE env
        mock_marble_env = MagicMock()
        mock_marble_env.is_done.return_value = True
        env._marble_env = mock_marble_env

        assert env.is_done() is True
        mock_marble_env.is_done.assert_called_once()

    def test_is_task_completed_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """is_task_completed should delegate to MARBLE env when available."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.is_task_completed.return_value = True
        env._marble_env = mock_marble_env

        assert env.is_task_completed() is True
        mock_marble_env.is_task_completed.assert_called_once()

    def test_get_marble_state_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """get_marble_state should return state from MARBLE env."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.get_state.return_value = {"key": "value"}
        env._marble_env = mock_marble_env

        assert env.get_marble_state() == {"key": "value"}
        mock_marble_env.get_state.assert_called_once()

    def test_get_tool_descriptions_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """get_tool_descriptions should return descriptions from MARBLE env."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.action_handler_descriptions = {"tool1": {"desc": "Tool 1"}}
        env._marble_env = mock_marble_env

        assert env.get_tool_descriptions() == {"tool1": {"desc": "Tool 1"}}

    def test_apply_action_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """apply_action should delegate to MARBLE env when available."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.apply_action.return_value = {"result": "success"}
        env._marble_env = mock_marble_env

        result = env.apply_action("agent1", "action1", {"arg": "value"})

        assert result == {"result": "success"}
        mock_marble_env.apply_action.assert_called_once_with("agent1", "action1", {"arg": "value"})

    def test_gather_traces_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """gather_traces should include MARBLE state when available."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.get_state.return_value = {"state_key": "state_value"}
        mock_marble_env.is_done.return_value = False
        mock_marble_env.is_task_completed.return_value = False
        env._marble_env = mock_marble_env

        traces = env.gather_traces()

        assert traces["marble_state"] == {"state_key": "state_value"}
        assert traces["is_done"] is False
        assert traces["is_task_completed"] is False
        assert traces["marble_env_type"] == "MagicMock"

    def test_create_tools_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """create_tools should wrap MARBLE action handlers."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        # Create mock MARBLE env with action handlers
        mock_handler = MagicMock(return_value="handler_result")
        mock_marble_env = MagicMock()
        mock_marble_env._action_handlers = {"test_action": mock_handler}
        env._marble_env = mock_marble_env

        tools = env.create_tools()

        assert "test_action" in tools
        # Test the wrapped handler
        result = tools["test_action"](arg1="value1")
        assert result == "handler_result"
        mock_handler.assert_called_once_with(arg1="value1")

    def test_create_tools_traces_invocations(self, sample_research_task_data: Dict[str, Any]):
        """create_tools should trace tool invocations."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_handler = MagicMock(return_value="result")
        mock_marble_env = MagicMock()
        mock_marble_env._action_handlers = {"test_action": mock_handler}
        env._marble_env = mock_marble_env

        tools = env.create_tools()
        tools["test_action"](input="test")

        # Check that invocation was traced
        assert "test_action" in env._tool_histories
        history = env._tool_histories["test_action"]
        invocations = history.to_list()
        assert len(invocations) == 1

    def test_create_tools_traces_errors(self, sample_research_task_data: Dict[str, Any]):
        """create_tools should trace tool errors."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_handler = MagicMock(side_effect=RuntimeError("Handler error"))
        mock_marble_env = MagicMock()
        mock_marble_env._action_handlers = {"test_action": mock_handler}
        env._marble_env = mock_marble_env

        tools = env.create_tools()

        with pytest.raises(RuntimeError):
            tools["test_action"](input="test")

        # Check that error was traced
        history = env._tool_histories["test_action"]
        invocations = history.to_list()
        assert len(invocations) == 1
        assert invocations[0]["status"] == "error"


class TestGetTool:
    """Tests for get_tool method."""

    def test_get_tool_returns_none_if_not_found(self, sample_research_task_data: Dict[str, Any]):
        """get_tool should return None for unknown tool."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)
        assert env.get_tool("unknown_tool") is None

    def test_get_tool_returns_tool_if_found(self, sample_research_task_data: Dict[str, Any]):
        """get_tool should return tool if it exists."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_handler = MagicMock()
        mock_marble_env = MagicMock()
        mock_marble_env._action_handlers = {"test_tool": mock_handler}
        env._marble_env = mock_marble_env

        # Create tools and update env.tools (which is normally set during __init__)
        env.tools = env.create_tools()

        tool = env.get_tool("test_tool")
        assert tool is not None


class TestGatherConfig:
    """Tests for gather_config method."""

    def test_gather_config_with_marble_env(self, sample_research_task_data: Dict[str, Any]):
        """gather_config should include tool descriptions."""
        env = MultiAgentBenchEnvironment(task_data=sample_research_task_data)

        mock_marble_env = MagicMock()
        mock_marble_env.action_handler_descriptions = {"tool1": {"name": "Tool 1"}}
        env._marble_env = mock_marble_env

        config = env.gather_config()

        assert config["tool_descriptions"] == {"tool1": {"name": "Tool 1"}}
        assert config["marble_env_type"] == "MagicMock"


class TestSetupStateEdgeCases:
    """Tests for setup_state edge cases."""

    def test_setup_state_with_string_task(self):
        """setup_state should handle task as string."""
        task_data = {
            "scenario": "research",
            "environment": {},
            "task": "String task content",
            "agents": [{"agent_id": "agent1"}],
        }

        env = MultiAgentBenchEnvironment(task_data=task_data)
        assert env.state is not None

    def test_setup_state_default_max_iterations(self):
        """setup_state should default max_iterations to 10."""
        task_data = {
            "scenario": "research",
            "environment": {},
            "task": {"content": "Task"},
            "agents": [{"agent_id": "agent1"}],
        }

        env = MultiAgentBenchEnvironment(task_data=task_data)
        assert env.state["max_iterations"] == 10

    def test_setup_state_uses_env_max_iterations(self):
        """setup_state should use environment max_iterations if provided."""
        task_data = {
            "scenario": "research",
            "environment": {"max_iterations": 25},
            "task": {"content": "Task"},
            "agents": [{"agent_id": "agent1"}],
        }

        env = MultiAgentBenchEnvironment(task_data=task_data)
        assert env.state["max_iterations"] == 25
