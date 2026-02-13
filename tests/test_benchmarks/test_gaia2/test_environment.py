"""Tests for Gaia2Environment.

Tests the environment wrapper that integrates with ARE simulation.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch


def _make_are_mock():
    """Create a fully-mocked ARE module structure for sys.modules patching.

    Returns a (mock_are, modules_dict) tuple where modules_dict can be
    passed directly to ``patch.dict(sys.modules, modules_dict)``.
    """
    mock_are = MagicMock()

    # Mock preprocess_scenario as a no-op
    def _preprocess_scenario(scenario, judge_config, max_scenario_duration):
        scenario.duration = max_scenario_duration
        # In real ARE, start_time and time_increment_in_seconds are set from
        # JSON data before preprocess_scenario runs. Ensure real values so
        # environment.py guards (e.g. start_time > 0) don't fail on MagicMock.
        if not isinstance(getattr(scenario, "start_time", None), (int, float)):
            scenario.start_time = 1728975600.0  # 2024-10-15 07:00:00 UTC
        if not isinstance(getattr(scenario, "time_increment_in_seconds", None), (int, float)):
            scenario.time_increment_in_seconds = 1

    mock_are.simulation.scenarios.scenario_imported_from_json.utils.preprocess_scenario = _preprocess_scenario

    # Mock get_scenario_duration to return a sensible default
    def _get_scenario_duration(scenario, max_time_duration, max_duration):
        return max_duration

    mock_are.simulation.scenarios.scenario_imported_from_json.utils.get_scenario_duration = _get_scenario_duration

    # Mock scenario config constants
    mock_are.simulation.scenarios.config.MAX_SCENARIO_DURATION = 1800
    mock_are.simulation.scenarios.config.MAX_TIME_SCENARIO_DURATION = 420

    modules = {
        "are": mock_are,
        "are.simulation": mock_are.simulation,
        "are.simulation.environment": mock_are.simulation.environment,
        "are.simulation.types": mock_are.simulation.types,
        "are.simulation.validation": mock_are.simulation.validation,
        "are.simulation.scenarios": mock_are.simulation.scenarios,
        "are.simulation.scenarios.config": mock_are.simulation.scenarios.config,
        "are.simulation.scenarios.scenario_imported_from_json": (mock_are.simulation.scenarios.scenario_imported_from_json),
        "are.simulation.scenarios.scenario_imported_from_json.utils": (mock_are.simulation.scenarios.scenario_imported_from_json.utils),
    }
    return mock_are, modules


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

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario
        mock_scenario.scenario_id = "test_scenario"

        with patch.dict(sys.modules, modules):
            task_data = {"scenario": mock_scenario}
            env = Gaia2Environment(task_data=task_data)

            assert env._scenario is mock_scenario

    def test_raises_without_scenario(self):
        """Test raises error when scenario is missing."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        _, modules = _make_are_mock()

        with patch.dict(sys.modules, modules):
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
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool
        from types import SimpleNamespace

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance

        # Create mock tool matching ARE's AppTool interface (required by AppToolAdapter)
        mock_tool = SimpleNamespace(
            name="TestTool__do_something",
            _public_name="TestTool__do_something",
            _public_description="Test tool",
            function_description="Test tool",
            app_name="TestTool",
            return_type=str,
            args=[SimpleNamespace(name="arg1", arg_type="str", description="An argument", has_default=False)],
        )
        mock_tool.__call__ = lambda **kw: "result"

        mock_app = MagicMock()
        mock_app.get_tools.return_value = [mock_tool]
        mock_env_instance.apps = {"TestApp": mock_app}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario
        mock_scenario.scenario_id = "test"

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            tools = env.create_tools()

            assert "TestTool__do_something" in tools
            assert isinstance(tools["TestTool__do_something"], Gaia2GenericTool)

    def test_create_tools_returns_empty_when_no_are_env(self):
        """Test create_tools returns empty dict when ARE env is None."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
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

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            env.cleanup()

            mock_env_instance.stop.assert_called_once()

    def test_cleanup_handles_no_are_environment(self):
        """Test cleanup handles case when no ARE environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            env._are_env = None

            # Should not raise
            env.cleanup()

    def test_cleanup_handles_stop_error(self):
        """Test cleanup handles error during stop gracefully."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.stop.side_effect = Exception("Stop failed")

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
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

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_are_environment() is mock_env_instance

    def test_get_scenario_returns_scenario(self):
        """Test get_scenario returns scenario."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_scenario() is mock_scenario

    def test_get_simulation_time_returns_time(self):
        """Test get_simulation_time returns current time."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.current_time = 123.5

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})

            assert env.get_simulation_time() == 123.5

    def test_get_simulation_time_returns_zero_when_no_env(self):
        """Test get_simulation_time returns 0 when no environment."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
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

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}
        mock_env_instance.current_time = 0.0

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(task_data={"scenario": mock_scenario})
            traces = env.gather_traces()

            assert "type" in traces
            assert traces["type"] == "Gaia2Environment"

    def test_gather_config_includes_environment_info(self):
        """Test gather_config includes environment information."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []  # No oracle events in mock scenario
        mock_scenario.scenario_id = "test_scenario"

        with patch.dict(sys.modules, modules):
            env = Gaia2Environment(
                task_data={
                    "scenario": mock_scenario,
                    "capability": "execution",
                }
            )
            config = env.gather_config()

            assert "type" in config
            assert config["type"] == "Gaia2Environment"


# =============================================================================
# Test Gaia2Environment Judge Engine Config
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EnvironmentJudgeEngineConfig:
    """Tests for judge engine configuration in Gaia2Environment."""

    def test_default_judge_config_when_no_engine_config(self):
        """Test default GraphPerEventJudgeConfig is used when no judge_engine_config."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        # Track calls to GraphPerEventJudgeConfig
        mock_judge_config_cls = MagicMock()
        mock_judge_config_instance = MagicMock()
        mock_judge_config_cls.return_value = mock_judge_config_instance
        mock_are.simulation.validation.GraphPerEventJudgeConfig = mock_judge_config_cls

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []

        with patch.dict(sys.modules, modules):
            Gaia2Environment(task_data={"scenario": mock_scenario})

            # Default: GraphPerEventJudgeConfig() called with no arguments
            mock_judge_config_cls.assert_called_once_with()

    def test_custom_judge_engine_config_creates_engine(self):
        """Test custom judge_engine_config creates engine via create_judge_engine."""
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_are, modules = _make_are_mock()
        mock_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_env_instance
        mock_env_instance.apps = {}

        # Mock LLMEngineConfig
        mock_llm_config_cls = MagicMock()
        mock_llm_config_instance = MagicMock()
        mock_llm_config_cls.return_value = mock_llm_config_instance

        # Mock create_judge_engine
        mock_create_engine = MagicMock()
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock GraphPerEventJudgeConfig
        mock_judge_config_cls = MagicMock()
        mock_judge_config_instance = MagicMock()
        mock_judge_config_cls.return_value = mock_judge_config_instance

        mock_are.simulation.validation.GraphPerEventJudgeConfig = mock_judge_config_cls

        # Add the extra ARE modules that get imported when judge_engine_config is set
        modules["are.simulation.agents"] = mock_are.simulation.agents
        modules["are.simulation.agents.are_simulation_agent_config"] = MagicMock(LLMEngineConfig=mock_llm_config_cls)
        modules["are.simulation.validation.configs"] = MagicMock(create_judge_engine=mock_create_engine)

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400
        mock_scenario.events = []

        judge_engine_config = Gaia2JudgeEngineConfig(
            model_name="openai/gpt-4o",
            provider="openrouter",
            endpoint="https://openrouter.ai/api/v1",
        )

        with patch.dict(sys.modules, modules):
            Gaia2Environment(
                task_data={"scenario": mock_scenario},
                judge_engine_config=judge_engine_config,
            )

            # LLMEngineConfig should be created with the custom values
            mock_llm_config_cls.assert_called_once_with(
                model_name="openai/gpt-4o",
                provider="openrouter",
                endpoint="https://openrouter.ai/api/v1",
            )

            # create_judge_engine should be called with the LLMEngineConfig
            mock_create_engine.assert_called_once_with(mock_llm_config_instance)

            # GraphPerEventJudgeConfig should be created with the custom engine
            mock_judge_config_cls.assert_called_once_with(engine=mock_engine)


# =============================================================================
# Test poll_notifications
# =============================================================================


@pytest.mark.benchmark
class TestPollNotifications:
    """Tests for Gaia2Environment.poll_notifications()."""

    def test_returns_empty_when_no_are_env(self):
        """poll_notifications returns empty when ARE environment is not set up."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        # Create env without triggering setup_state (no ARE)
        env = Gaia2Environment.__new__(Gaia2Environment)
        env._are_env = None
        env._scenario = None
        env._judge_engine_config = None
        env._tool_wrappers = {}
        env.state = {}

        user_msgs, env_notifs, has_stop = env.poll_notifications()

        assert user_msgs == []
        assert env_notifs == []
        assert has_stop is False

    def test_returns_empty_when_no_notification_system(self):
        """poll_notifications returns empty when notification_system is None."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        # Use a bare mock without notification_system attribute
        mock_env = MagicMock(spec=[])

        env = Gaia2Environment.__new__(Gaia2Environment)
        env._are_env = mock_env
        env._scenario = None
        env._judge_engine_config = None
        env._tool_wrappers = {}
        env.state = {}

        user_msgs, env_notifs, has_stop = env.poll_notifications()

        assert user_msgs == []
        assert env_notifs == []
        assert has_stop is False

    def test_returns_empty_when_queue_empty(self):
        """poll_notifications returns empty when queue has no messages."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        mock_ns = MagicMock()
        mock_ns.message_queue.get_by_timestamp.return_value = []

        mock_env = MagicMock()
        mock_env.notification_system = mock_ns

        env = Gaia2Environment.__new__(Gaia2Environment)
        env._are_env = mock_env
        env._scenario = None
        env._judge_engine_config = None
        env._tool_wrappers = {}
        env.state = {}

        user_msgs, env_notifs, has_stop = env.poll_notifications()

        assert user_msgs == []
        assert env_notifs == []
        assert has_stop is False
