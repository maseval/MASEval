"""Tests for AREEnvironment."""

from unittest.mock import MagicMock, patch, PropertyMock
import sys
import pytest

from maseval.interface.environments.are import AREEnvironment


def _make_mock_scenario(scenario_id="test-001", duration=600, seed=42, start_time=0):
    """Create a mock ARE Scenario."""
    scenario = MagicMock()
    scenario.scenario_id = scenario_id
    scenario.duration = duration
    scenario.seed = seed
    scenario.start_time = start_time
    scenario.time_increment_in_seconds = 1
    scenario.apps = [MagicMock(name="EmailClient"), MagicMock(name="Calendar")]
    return scenario


def _make_mock_are_env(apps=None):
    """Create a mock ARE Environment."""
    env = MagicMock()
    if apps is None:
        # Create mock apps with mock tools
        email_app = MagicMock()
        email_tool = MagicMock()
        email_tool.name = "EmailClient__send_email"
        email_tool.description = "Send an email"
        email_tool.inputs = {"to": {"type": "string", "description": "Recipient"}}
        email_tool.output_type = "string"
        email_app.get_tools.return_value = [email_tool]
        email_app.name = "EmailClient"

        calendar_app = MagicMock()
        cal_tool = MagicMock()
        cal_tool.name = "Calendar__create_event"
        cal_tool.description = "Create event"
        cal_tool.inputs = {"title": {"type": "string", "description": "Title"}}
        cal_tool.output_type = "string"
        calendar_app.get_tools.return_value = [cal_tool]
        calendar_app.name = "Calendar"

        env.apps = {"EmailClient": email_app, "Calendar": calendar_app}
    else:
        env.apps = apps
    env.current_time = 0.0
    return env


class TestAREEnvironmentScenarioPath:
    """Tests for AREEnvironment with Scenario objects."""

    @patch("maseval.interface.environments.are._import_are")
    def test_setup_state_with_scenario(self, mock_import):
        """setup_state initialises from an ARE Scenario and returns state dict."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        task_data = {"scenario": scenario}

        env = AREEnvironment(task_data)

        assert env.state["scenario_id"] == "test-001"
        assert env.state["duration"] == 600
        assert env.state["seed"] == 42
        assert env._are_env is mock_are_env

    @patch("maseval.interface.environments.are._import_are")
    def test_setup_state_requires_scenario_or_apps(self, mock_import):
        """setup_state raises ValueError if neither scenario nor apps provided."""
        mock_import.return_value = MagicMock()

        with pytest.raises(ValueError, match="must contain either"):
            AREEnvironment(task_data={})

    @patch("maseval.interface.environments.are._import_are")
    def test_create_tools_wraps_are_tools(self, mock_import):
        """create_tools wraps all ARE app tools in AREToolWrapper."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        tools = env.get_tools()
        assert "EmailClient__send_email" in tools
        assert "Calendar__create_event" in tools
        assert len(tools) == 2

        # Check wrapper metadata
        email_tool = tools["EmailClient__send_email"]
        assert email_tool.name == "EmailClient__send_email"
        assert email_tool.description == "Send an email"

    @patch("maseval.interface.environments.are._import_are")
    def test_start_runs_scenario(self, mock_import):
        """start() calls are_env.run() with the scenario."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.start()

        mock_are_env.run.assert_called_once()
        call_kwargs = mock_are_env.run.call_args
        assert call_kwargs[1].get("wait_for_end") is False

    @patch("maseval.interface.environments.are._import_are")
    def test_stop_stops_env(self, mock_import):
        """stop() calls are_env.stop()."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.stop()

        mock_are_env.stop.assert_called_once()

    @patch("maseval.interface.environments.are._import_are")
    def test_pause_and_resume(self, mock_import):
        """pause() and resume_with_offset() delegate to ARE env."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.pause()
        mock_are_env.pause.assert_called_once()

        env.resume_with_offset(5.0)
        mock_are_env.resume_with_offset.assert_called_once_with(5.0)

    @patch("maseval.interface.environments.are._import_are")
    def test_get_simulation_time(self, mock_import):
        """get_simulation_time() returns ARE env's current_time."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_env.current_time = 42.5
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        assert env.get_simulation_time() == 42.5

    @patch("maseval.interface.environments.are._import_are")
    def test_cleanup_stops_env(self, mock_import):
        """cleanup() stops the ARE environment."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.cleanup()

        mock_are_env.stop.assert_called_once()

    @patch("maseval.interface.environments.are._import_are")
    def test_gather_traces(self, mock_import):
        """gather_traces returns structured trace data."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_env.current_time = 100.0
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        traces = env.gather_traces()
        assert traces["scenario_id"] == "test-001"
        assert traces["tool_count"] == 2
        assert "tools" in traces
        assert traces["final_simulation_time"] == 100.0

    @patch("maseval.interface.environments.are._import_are")
    def test_gather_config(self, mock_import):
        """gather_config returns environment configuration."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        config = env.gather_config()
        assert config["scenario_id"] == "test-001"
        assert config["duration"] == 600
        assert config["notification_verbosity"] == "medium"
        assert "tool_names" in config


class TestAREEnvironmentShorthandPath:
    """Tests for AREEnvironment with apps+events shorthand."""

    @patch("maseval.interface.environments.are._import_are")
    @patch("maseval.interface.environments.are.AREEnvironment._build_scenario_from_shorthand")
    def test_shorthand_builds_scenario(self, mock_build, mock_import):
        """Shorthand task_data with 'apps' key triggers scenario construction."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_scenario = _make_mock_scenario()
        mock_build.return_value = mock_scenario

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        task_data = {
            "apps": [MagicMock(), MagicMock()],
            "events": [MagicMock()],
            "duration": 300,
            "seed": 99,
        }
        env = AREEnvironment(task_data=task_data)

        mock_build.assert_called_once_with(task_data)
        assert env._scenario is mock_scenario

    @patch("maseval.interface.environments.are._import_are")
    def test_shorthand_passes_config_to_scenario(self, mock_import):
        """Shorthand config values are passed through to Scenario construction."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        # Patch Scenario import inside _build_scenario_from_shorthand
        mock_scenario_cls = MagicMock()
        mock_scenario_instance = _make_mock_scenario()
        mock_scenario_cls.return_value = mock_scenario_instance

        with patch.dict("sys.modules", {
            "are": MagicMock(),
            "are.simulation": MagicMock(),
            "are.simulation.scenarios": MagicMock(),
            "are.simulation.scenarios.scenario": MagicMock(Scenario=mock_scenario_cls),
        }):
            apps = [MagicMock(), MagicMock()]
            task_data = {
                "apps": apps,
                "duration": 300,
                "seed": 99,
                "start_time": 100,
                "time_increment_in_seconds": 5,
            }
            env = AREEnvironment(task_data=task_data)

            mock_scenario_cls.assert_called_once()
            call_kwargs = mock_scenario_cls.call_args[1]
            assert call_kwargs["duration"] == 300
            assert call_kwargs["seed"] == 99
            assert call_kwargs["start_time"] == 100
            assert call_kwargs["time_increment_in_seconds"] == 5
            mock_scenario_instance.initialize.assert_called_once()


class TestAREToolWrapper:
    """Tests for AREToolWrapper simulation time tracking."""

    def _make_wrapper(self, mock_env):
        """Create an AREToolWrapper with a mock tool and environment."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

        mock_tool = MagicMock()
        mock_tool.name = "TestTool__do_thing"
        mock_tool.description = "Does a thing"
        mock_tool.inputs = {"x": {"type": "string", "description": "input"}}
        mock_tool.output_type = "string"
        mock_tool.args = []
        mock_tool.return_value = "result"

        wrapper = AREToolWrapper(mock_tool, mock_env)
        return wrapper

    def test_invocation_records_simulation_time(self):
        """Wrapper records simulation time before/after tool call in meta."""
        mock_env = MagicMock()
        mock_env.get_simulation_time = MagicMock(side_effect=[100.0, 105.0])

        wrapper = self._make_wrapper(mock_env)
        wrapper(x="hello")

        assert len(wrapper.history.logs) == 1
        meta = wrapper.history.logs[0]["meta"]
        assert meta["simulation_time_before"] == 100.0
        assert meta["simulation_time_after"] == 105.0
        assert meta["simulation_time_elapsed"] == 5.0
        assert "wall_time" in meta

    def test_invocation_records_none_when_sim_time_unavailable(self):
        """Wrapper records None values when get_simulation_time raises."""
        mock_env = MagicMock()
        mock_env.get_simulation_time = MagicMock(side_effect=AttributeError)

        wrapper = self._make_wrapper(mock_env)
        wrapper(x="hello")

        assert len(wrapper.history.logs) == 1
        meta = wrapper.history.logs[0]["meta"]
        assert meta["simulation_time_before"] is None
        assert meta["simulation_time_after"] is None
        assert meta["simulation_time_elapsed"] is None
