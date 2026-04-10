"""Integration tests for AREEnvironment with real ARE simulation stack.

These tests exercise AREEnvironment against real ARE apps and scenarios —
no mocks. They validate that the maseval wrapper correctly integrates with
ARE's simulation infrastructure: tool wrapping, lifecycle, tracing, oracle
mode, and the shorthand construction path.

Marked ``interface`` + ``are``. Runs in the default test suite (no network
or API keys needed — ARE is a local dependency).
"""

import pytest

try:
    from are.simulation.apps import CalendarApp, ContactsApp, SystemApp
    from are.simulation.scenarios.scenario import Scenario

    HAS_ARE = True
except ImportError:
    HAS_ARE = False

pytestmark = [pytest.mark.interface, pytest.mark.are]

skip_no_are = pytest.mark.skipif(not HAS_ARE, reason="ARE not installed")


def _make_scenario(duration=60, seed=42, start_time=0):
    """Create a minimal ARE scenario with Calendar + Contacts + SystemApp."""
    apps = [CalendarApp(), ContactsApp(), SystemApp()]
    scenario = Scenario(
        scenario_id="test-integration",  # ty: ignore[unknown-argument]
        apps=apps,  # ty: ignore[unknown-argument]
        events=[],  # ty: ignore[unknown-argument]
        duration=duration,  # ty: ignore[unknown-argument]
        seed=seed,  # ty: ignore[unknown-argument]
        start_time=start_time,  # ty: ignore[unknown-argument]
        time_increment_in_seconds=1,  # ty: ignore[unknown-argument]
    )
    scenario.initialize()
    return scenario


# =============================================================================
# Environment Lifecycle
# =============================================================================


@skip_no_are
class TestAREEnvironmentLifecycle:
    """Test the full AREEnvironment lifecycle with real ARE."""

    def test_scenario_path_creates_environment(self):
        """AREEnvironment initializes from a real ARE Scenario."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            assert env.state["scenario_id"] == "test-integration"
            assert env.state["duration"] == 60
            assert env.state["seed"] == 42
            assert "CalendarApp" in env.state["app_names"]
            assert "ContactsApp" in env.state["app_names"]
            assert "SystemApp" in env.state["app_names"]
        finally:
            env.cleanup()

    def test_shorthand_path_creates_environment(self):
        """AREEnvironment initializes from shorthand apps + config."""
        from maseval.interface.environments.are import AREEnvironment

        apps = [CalendarApp(), ContactsApp(), SystemApp()]
        env = AREEnvironment(
            environment_data={
                "apps": apps,
                "duration": 30,
                "seed": 99,
            }
        )
        try:
            assert env.state["duration"] == 30
            assert env.state["seed"] == 99
            assert len(env.tools) > 0
        finally:
            env.cleanup()

    def test_start_stop_lifecycle(self):
        """start() begins simulation, stop() ends it without error."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario(duration=10)
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            assert env.get_simulation_time() >= 0
            env.stop()
        finally:
            env.cleanup()

    def test_pause_resume(self):
        """pause() and resume_with_offset() control simulation time."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario(duration=60)
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            env.pause()
            time_at_pause = env.get_simulation_time()
            env.resume_with_offset(10.0)
            time_after_resume = env.get_simulation_time()
            assert time_after_resume >= time_at_pause + 10.0
        finally:
            env.cleanup()

    def test_from_apps_classmethod(self):
        """from_apps() creates a working environment with real ARE apps."""
        from maseval.interface.environments.are import AREEnvironment

        apps = [CalendarApp(), ContactsApp(), SystemApp()]
        env = AREEnvironment.from_apps(
            apps=apps,
            duration=30,
            seed=99,
        )
        try:
            assert env.state["duration"] == 30
            assert env.state["seed"] == 99
            assert len(env.tools) > 0
        finally:
            env.cleanup()

    def test_cleanup_is_idempotent(self):
        """cleanup() can be called multiple times without error."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario(duration=10)
        env = AREEnvironment(environment_data={"scenario": scenario})
        env.start()
        env.cleanup()
        env.cleanup()  # second call should not raise


# =============================================================================
# Tool Wrapping
# =============================================================================


@skip_no_are
class TestAREToolWrapping:
    """Test that real ARE tools are correctly wrapped and callable."""

    def test_tools_are_created_from_real_apps(self):
        """create_tools() produces AREToolWrapper instances for all ARE app tools."""
        from maseval.interface.environments.are import AREEnvironment
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            tools = env.get_tools()
            assert len(tools) > 0
            for name, tool in tools.items():
                assert isinstance(tool, AREToolWrapper), f"{name} is {type(tool).__name__}"
        finally:
            env.cleanup()

    def test_tool_metadata_from_real_apps(self):
        """Wrapped tools expose name, description, inputs, output_type from ARE."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            for name, tool in env.get_tools().items():
                assert isinstance(tool.name, str) and tool.name
                assert isinstance(tool.description, str) and tool.description
                assert isinstance(tool.inputs, dict)
                assert isinstance(tool.output_type, str)
        finally:
            env.cleanup()

    def test_tool_call_returns_result_and_traces(self):
        """Calling a real ARE tool returns a result and records a traced invocation."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            get_time = env.get_tool("SystemApp__get_current_time")
            assert get_time is not None

            result = get_time()
            assert isinstance(result, dict)
            assert "current_timestamp" in result

            # Invocation was traced
            assert len(get_time.history) == 1
            record = get_time.history.to_list()[0]
            assert record["status"] == "success"
            assert record["outputs"] == result

            # Simulation time was captured in meta
            meta = record["meta"]
            assert meta["simulation_time_before"] is not None
            assert meta["simulation_time_after"] is not None
            assert isinstance(meta["simulation_time_elapsed"], (int, float))
        finally:
            env.cleanup()

    def test_tool_error_is_traced_and_reraised(self):
        """A tool call that fails records the error and re-raises."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            # get_calendar_event with a nonexistent ID should fail
            get_event = env.get_tool("CalendarApp__get_calendar_event")
            assert get_event is not None

            with pytest.raises(Exception):
                get_event(event_id="nonexistent-id-12345")

            assert len(get_event.history) == 1
            assert get_event.history.to_list()[0]["status"] == "error"
        finally:
            env.cleanup()

    def test_multiple_tool_calls_accumulate_history(self):
        """Multiple calls to the same tool accumulate in history."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            get_time = env.get_tool("SystemApp__get_current_time")
            assert get_time is not None
            get_time()
            get_time()
            get_time()

            assert len(get_time.history) == 3
            for record in get_time.history.to_list():
                assert record["status"] == "success"
        finally:
            env.cleanup()


# =============================================================================
# AUI Tool Filtering
# =============================================================================


@skip_no_are
class TestAUIToolFiltering:
    """Test AUI tool filtering with real ARE apps."""

    def test_filter_aui_tools_excludes_removal_set(self):
        """filter_aui_tools=True removes tools in _AUI_TOOLS_TO_REMOVE."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(
            environment_data={"scenario": scenario},
            filter_aui_tools=True,
        )
        try:
            for name in env.tools:
                assert name not in AREEnvironment._AUI_TOOLS_TO_REMOVE
        finally:
            env.cleanup()

    def test_unfiltered_has_at_least_as_many_tools(self):
        """Unfiltered environment has >= tools as filtered."""
        from maseval.interface.environments.are import AREEnvironment

        env_unfiltered = AREEnvironment(environment_data={"scenario": _make_scenario()})
        env_filtered = AREEnvironment(
            environment_data={"scenario": _make_scenario()},
            filter_aui_tools=True,
        )
        try:
            assert len(env_filtered.tools) <= len(env_unfiltered.tools)
        finally:
            env_unfiltered.cleanup()
            env_filtered.cleanup()


# =============================================================================
# Oracle Mode
# =============================================================================


@skip_no_are
class TestAREOracleMode:
    """Test oracle mode with real ARE simulation."""

    def test_oracle_mode_produces_traces(self):
        """run_oracle=True produces oracle_traces with apps_state and world_logs."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario(duration=10)
        env = AREEnvironment(
            environment_data={"scenario": scenario},
            run_oracle=True,
        )
        try:
            traces = env.get_oracle_traces()
            assert traces is not None
            assert "apps_state" in traces
            assert "world_logs" in traces
            assert isinstance(traces["apps_state"], dict)
            assert isinstance(traces["world_logs"], list)

            # Oracle traces appear in gather_traces too
            full_traces = env.gather_traces()
            assert full_traces["oracle_traces"] is not None
        finally:
            env.cleanup()

    def test_no_oracle_by_default(self):
        """Oracle mode is off by default."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            assert env.get_oracle_traces() is None
        finally:
            env.cleanup()


# =============================================================================
# Tracing & Config
# =============================================================================


@skip_no_are
class TestARETracingAndConfig:
    """Test that tracing and config capture real environment state."""

    def test_gather_traces_after_tool_calls(self):
        """gather_traces() captures tool invocation history from real calls."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            get_time = env.get_tool("SystemApp__get_current_time")
            assert get_time is not None
            get_time()
            get_time()

            traces = env.gather_traces()
            assert traces["tool_count"] == len(env.tools)
            assert traces["scenario_id"] == "test-integration"

            # The tool we called should have 2 invocations in traces
            tool_traces = traces["tools"]["SystemApp__get_current_time"]
            assert tool_traces["total_invocations"] == 2
        finally:
            env.cleanup()

    def test_gather_config_captures_settings(self):
        """gather_config() records environment settings for reproducibility."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(
            environment_data={"scenario": scenario},
            notification_verbosity="high",
            run_oracle=False,
        )
        try:
            config = env.gather_config()
            assert config["scenario_id"] == "test-integration"
            assert config["duration"] == 60
            assert config["seed"] == 42
            assert config["notification_verbosity"] == "high"
            assert config["run_oracle"] is False
            assert config["tool_count"] == len(env.tools)
            assert "tool_names" in config
            assert "SystemApp__get_current_time" in config["tool_names"]
        finally:
            env.cleanup()

    def test_simulation_time_advances_with_wait(self):
        """Simulation time advances when wait_for_notification is called."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            env.start()
            t0 = env.get_simulation_time()

            wait_tool = env.get_tool("SystemApp__wait_for_notification")
            if wait_tool:
                wait_tool(timeout=2)
                t1 = env.get_simulation_time()
                assert t1 > t0, f"Simulation time should advance: {t0} -> {t1}"
        finally:
            env.cleanup()


# =============================================================================
# Accessors
# =============================================================================


@skip_no_are
class TestAREAccessors:
    """Test convenience accessors return real ARE objects."""

    def test_get_are_environment_returns_real_env(self):
        """get_are_environment() returns the actual ARE Environment instance."""
        from are.simulation.environment import Environment as RealAREEnv

        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            are_env = env.get_are_environment()
            assert isinstance(are_env, RealAREEnv)
        finally:
            env.cleanup()

    def test_get_scenario_returns_real_scenario(self):
        """get_scenario() returns the ARE Scenario that was passed in."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            assert env.get_scenario() is scenario
        finally:
            env.cleanup()

    def test_get_notification_system_returns_real_system(self):
        """get_notification_system() returns a real ARE notification system."""
        from maseval.interface.environments.are import AREEnvironment

        scenario = _make_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        try:
            ns = env.get_notification_system()
            assert ns is not None
            assert hasattr(ns, "message_queue")
        finally:
            env.cleanup()


# =============================================================================
# Error Handling
# =============================================================================


@skip_no_are
class TestAREErrorHandling:
    """Test that errors propagate correctly (not swallowed)."""

    def test_invalid_environment_data_raises(self):
        """Missing scenario and apps raises ValueError."""
        from maseval.interface.environments.are import AREEnvironment

        with pytest.raises(ValueError, match="must contain either"):
            AREEnvironment(environment_data={})
