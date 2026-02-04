"""Tests for Gaia2Environment.

Tests the environment wrapper that integrates with ARE simulation.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Gaia2Environment Class Structure
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentClassStructure:
    """Tests for Gaia2Environment class structure (no ARE needed)."""

    def test_class_inherits_from_environment(self):
        """Test Gaia2Environment inherits from Environment base class."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.core.environment import Environment

        assert issubclass(Gaia2Environment, Environment)

    def test_has_required_methods(self):
        """Test Gaia2Environment has required methods."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        assert hasattr(Gaia2Environment, "setup_state")
        assert hasattr(Gaia2Environment, "create_tools")
        assert hasattr(Gaia2Environment, "cleanup")
        assert hasattr(Gaia2Environment, "get_are_environment")
        assert hasattr(Gaia2Environment, "get_scenario")
        assert hasattr(Gaia2Environment, "get_simulation_time")


# =============================================================================
# Test Gaia2Environment Initialization (with mocked ARE)
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentInit:
    """Tests for Gaia2Environment initialization with mocked ARE."""

    def test_stores_scenario_from_task_data(self):
        """Test environment stores scenario from task data."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        # Create mock ARE modules
        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.scenario_id = "test_scenario"

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            task_data = {"scenario": mock_scenario}
            env = Gaia2Environment(task_data=task_data)

            assert env._scenario is mock_scenario

    def test_raises_without_scenario(self):
        """Test raises error when scenario is missing."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            with pytest.raises(ValueError, match="scenario"):
                Gaia2Environment(task_data={})


# =============================================================================
# Test Gaia2Environment.create_tools
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentCreateTools:
    """Tests for Gaia2Environment.create_tools()."""

    def test_create_tools_wraps_are_tools(self):
        """Test create_tools returns wrapped ARE tools."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper

        # Create mock ARE modules
        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance

        # Create mock app and tool
        mock_tool = MagicMock()
        mock_tool.name = "TestTool__do_something"
        mock_tool.description = "Test tool"
        mock_tool.inputs = {}

        mock_app = MagicMock()
        mock_app.get_tools.return_value = [mock_tool]
        mock_env_instance.apps = {"TestApp": mock_app}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.scenario_id = "test"

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            tools = env.create_tools()

            assert "TestTool__do_something" in tools
            assert isinstance(tools["TestTool__do_something"], AREToolWrapper)

    def test_create_tools_returns_empty_when_no_are_env(self):
        """Test create_tools returns empty dict when ARE env is None."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            # Manually set _are_env to None
            env._are_env = None
            tools = env.create_tools()

            assert tools == {}


# =============================================================================
# Test Gaia2Environment.cleanup
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentCleanup:
    """Tests for Gaia2Environment.cleanup()."""

    def test_cleanup_stops_are_environment(self):
        """Test cleanup calls stop on ARE environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            env.cleanup()

            mock_env_instance.stop.assert_called_once()

    def test_cleanup_handles_no_are_environment(self):
        """Test cleanup handles case when no ARE environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            env._are_env = None

            # Should not raise
            env.cleanup()

    def test_cleanup_handles_stop_error(self):
        """Test cleanup handles error during stop gracefully."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.stop.side_effect = Exception("Stop failed")

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            # Should not raise
            env.cleanup()


# =============================================================================
# Test Gaia2Environment Accessors
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentAccessors:
    """Tests for Gaia2Environment accessor methods."""

    def test_get_are_environment_returns_are_env(self):
        """Test get_are_environment returns ARE environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_are_environment() is mock_env_instance

    def test_get_scenario_returns_scenario(self):
        """Test get_scenario returns scenario."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_scenario() is mock_scenario

    def test_get_simulation_time_returns_time(self):
        """Test get_simulation_time returns current time."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.time_manager.current_time = 123.5

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_simulation_time() == 123.5

    def test_get_simulation_time_returns_zero_when_no_env(self):
        """Test get_simulation_time returns 0 when no environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            env._are_env = None

            assert env.get_simulation_time() == 0.0


# =============================================================================
# Test Gaia2Environment Tracing
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentTracing:
    """Tests for Gaia2Environment tracing methods."""

    def test_gather_traces_includes_type(self):
        """Test gather_traces includes type information."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.time_manager.current_time = 0.0

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            traces = env.gather_traces()

            assert "type" in traces
            assert traces["type"] == "Gaia2Environment"

    def test_gather_config_includes_environment_info(self):
        """Test gather_config includes environment information."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are = MagicMock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.scenario_id = "test_scenario"

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            env = Gaia2Environment(
                task_data={
                    "scenario": mock_scenario,
                    "capability": "execution",
                }
            )
            config = env.gather_config()

            assert "type" in config
            assert config["type"] == "Gaia2Environment"
