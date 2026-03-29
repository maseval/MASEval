"""AREEnvironment — generic maseval Environment wrapping ARE simulation.

Provides a reusable building block for interactive agent environments
using Meta's ARE (Agents Research Environments) infrastructure.

Original Repository: https://github.com/facebookresearch/meta-agents-research-environments
Code License: MIT
"""

from typing import Any, Dict, List, Optional, Tuple

from maseval.core.environment import Environment
from maseval.core.callback import EnvironmentCallback
from maseval.interface.environments.are_tool_wrapper import AREToolWrapper


def _check_are_installed() -> None:
    """Check if ARE is installed and raise a helpful error if not."""
    try:
        import are  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ARE (Agent Research Environments) is required for AREEnvironment.\n"
            "Install with: pip install maseval[are]\n"
            "Or: uv add meta-agents-research-environments"
        ) from e


def _import_are() -> Any:
    """Lazily import and return the ARE simulation module.

    Returns:
        The ``are.simulation`` module namespace with Environment,
        EnvironmentConfig, Scenario, etc.

    Raises:
        ImportError: If ARE is not installed.
    """
    _check_are_installed()
    from types import SimpleNamespace
    from are.simulation.environment import Environment as AREEnv  # type: ignore[import-not-found]
    from are.simulation.environment import EnvironmentConfig  # type: ignore[import-not-found]

    return SimpleNamespace(
        Environment=AREEnv,
        EnvironmentConfig=EnvironmentConfig,
    )


class AREEnvironment(Environment):
    """Generic maseval Environment wrapping ARE's simulation infrastructure.

    Supports two construction paths via ``task_data`` (= ``task.environment_data``):

    1. **Scenario path:** ``task_data = {"scenario": <ARE Scenario>}``
    2. **Shorthand path:** ``task_data = {"apps": [...], "events": [...], "duration": 1800, ...}``

    Lifecycle is user-controlled: call ``start()`` before ``run_agents()``,
    ``stop()`` after. ``pause()``/``resume_with_offset()`` control simulation time.
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[EnvironmentCallback]] = None,
        run_oracle: bool = False,
        notification_verbosity: str = "medium",
    ):
        """Initialize AREEnvironment.

        Args:
            task_data: ``task.environment_data`` dict. Must contain either:
                - ``"scenario"``: ARE Scenario object, OR
                - ``"apps"``: list of ARE App instances, plus optional ``"events"``,
                  ``"duration"``, ``"seed"``, ``"start_time"``, ``"time_increment_in_seconds"``
            callbacks: Optional maseval EnvironmentCallbacks.
            run_oracle: If True, run ARE oracle mode during setup to generate
                expected event log. Stored in traces for evaluation.
            notification_verbosity: ARE notification verbosity level.
                ``"low"`` = no environment notifications,
                ``"medium"`` = standard notifications,
                ``"high"`` = all notifications.
        """
        self._run_oracle = run_oracle
        self._notification_verbosity = notification_verbosity
        self._are_env: Any = None
        self._scenario: Any = None
        self._oracle_traces: Optional[Dict[str, Any]] = None
        self._tool_wrappers: Dict[str, AREToolWrapper] = {}

        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ARE environment from task data.

        Args:
            task_data: Dict with ``"scenario"`` or ``"apps"`` key.

        Returns:
            State dict with scenario metadata.

        Raises:
            ValueError: If task_data contains neither ``"scenario"`` nor ``"apps"``.
        """
        are_mod = _import_are()

        scenario = task_data.get("scenario")

        if scenario is None and "apps" not in task_data:
            raise ValueError(
                "task_data must contain either 'scenario' (ARE Scenario object) "
                "or 'apps' (list of ARE App instances)."
            )

        if scenario is None:
            scenario = self._build_scenario_from_shorthand(task_data)

        self._scenario = scenario

        # Run oracle mode if requested
        if self._run_oracle:
            self._oracle_traces = self._run_oracle_mode(are_mod, scenario)

        # Create ARE Environment (but don't start the event loop yet)
        config = are_mod.EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
            time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
        )
        if getattr(scenario, "start_time", None) and scenario.start_time > 0:
            config.start_time = scenario.start_time

        # Create notification system based on verbosity
        notification_system = self._create_notification_system()
        self._are_env = are_mod.Environment(config, notification_system=notification_system)

        # Register apps from scenario so tools are available before start()
        self._are_env.register_apps(scenario.apps)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": scenario.duration,
            "seed": getattr(scenario, "seed", None),
            "start_time": getattr(scenario, "start_time", None),
            "app_names": [getattr(app, "name", str(app)) for app in scenario.apps],
            "oracle_traces": self._oracle_traces,
        }

    def _build_scenario_from_shorthand(self, task_data: Dict[str, Any]) -> Any:
        """Build an ARE Scenario from shorthand task_data.

        Args:
            task_data: Dict with ``"apps"``, and optional ``"events"``,
                ``"duration"``, ``"seed"``, ``"start_time"``,
                ``"time_increment_in_seconds"``.

        Returns:
            ARE Scenario instance.
        """
        from are.simulation.scenarios.scenario import Scenario  # type: ignore[import-not-found]

        apps = task_data["apps"]
        events = task_data.get("events", [])
        duration = task_data.get("duration", 1800)
        seed = task_data.get("seed", 0)
        start_time = task_data.get("start_time", 0)
        time_increment = task_data.get("time_increment_in_seconds", 1)

        scenario = Scenario(
            scenario_id=task_data.get("scenario_id", "custom"),
            apps=apps,
            events=events,
            duration=duration,
            seed=seed,
            start_time=start_time,
            time_increment_in_seconds=time_increment,
        )
        scenario.initialize()
        return scenario

    def _run_oracle_mode(self, are_mod: Any, scenario: Any) -> Dict[str, Any]:
        """Run ARE oracle mode to generate expected event log.

        Args:
            are_mod: ARE module namespace.
            scenario: ARE Scenario instance.

        Returns:
            Dict with oracle event log.

        Raises:
            AttributeError: If the ARE Environment API changed and expected
                methods (get_apps_state, get_world_logs) are missing.
        """
        oracle_config = are_mod.EnvironmentConfig(
            oracle_mode=True,
            duration=scenario.duration,
            time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
        )
        oracle_env = are_mod.Environment(oracle_config)
        oracle_env.run(scenario, wait_for_end=True, schedule_events=True)

        # Capture oracle state
        oracle_traces = {
            "apps_state": oracle_env.get_apps_state(),
            "world_logs": oracle_env.get_world_logs(),
        }

        # Soft-reset so app state is clean for agent run
        scenario.soft_reset()

        return oracle_traces

    def _create_notification_system(self) -> Any:
        """Create ARE notification system based on verbosity setting.

        Returns:
            ARE NotificationSystem instance.
        """
        try:
            from are.simulation.notification_system import (  # type: ignore[import-not-found]
                VerboseNotificationSystem,
                VerbosityLevel,
            )

            level_map = {
                "low": VerbosityLevel.LOW,
                "medium": VerbosityLevel.MEDIUM,
                "high": VerbosityLevel.HIGH,
            }
            level = level_map.get(self._notification_verbosity, VerbosityLevel.MEDIUM)
            return VerboseNotificationSystem(verbosity_level=level)
        except ImportError:
            return None

    def create_tools(self) -> Dict[str, AREToolWrapper]:
        """Wrap all ARE app tools in AREToolWrapper.

        Returns:
            Dict mapping tool names to AREToolWrapper instances.
        """
        tools: Dict[str, AREToolWrapper] = {}

        if self._are_env is None:
            return tools

        for app in self._are_env.apps.values():
            for are_tool in app.get_tools():
                wrapper = AREToolWrapper(are_tool, self)
                tools[are_tool.name] = wrapper
                self._tool_wrappers[are_tool.name] = wrapper

        return tools

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the ARE simulation event loop.

        Call this after environment setup and before running agents.
        Runs the scenario with ``wait_for_end=False`` so control returns
        immediately for agent interaction.
        """
        if self._are_env is not None and self._scenario is not None:
            self._are_env.run(self._scenario, wait_for_end=False, schedule_events=True)

    def stop(self) -> None:
        """Stop the ARE simulation event loop."""
        if self._are_env is not None:
            self._are_env.stop()

    def pause(self) -> None:
        """Pause simulation time progression."""
        if self._are_env is not None:
            self._are_env.pause()

    def resume_with_offset(self, offset: float) -> None:
        """Resume simulation with a time offset.

        Args:
            offset: Seconds to advance simulation clock before resuming.
        """
        if self._are_env is not None:
            self._are_env.resume_with_offset(offset)

    # ── Notification Polling ──────────────────────────────────────────

    def poll_notifications(self) -> Tuple[List[str], List[str], bool]:
        """Drain pending notifications from ARE's notification queue.

        Returns:
            Tuple of ``(user_messages, env_notifications, has_stop_signal)``.
            ``user_messages``: Messages from simulated users.
            ``env_notifications``: System events (new email, calendar reminder, etc.).
            ``has_stop_signal``: True when simulation has ended.

        Agent adapters should call this between agent steps and inject
        the messages into the agent's context.

        Raises:
            Any exception from the underlying ARE notification system is
            propagated so the benchmark runner can classify it via
            ``fail_on_task_error`` / ``fail_on_setup_error``.
        """
        if self._are_env is None:
            return [], [], False

        notification_system = getattr(self._are_env, "notification_system", None)
        if notification_system is None:
            return [], [], False

        from datetime import datetime, timezone
        from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

        sim_time = self.get_simulation_time()
        timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
        unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

        if not unhandled:
            return [], [], False

        user_messages: List[str] = []
        env_notifications: List[str] = []
        has_stop = False

        for notif in unhandled:
            msg_type = getattr(notif, "message_type", None)
            if msg_type == MessageType.USER_MESSAGE:
                user_messages.append(notif.message)
            elif msg_type == MessageType.ENVIRONMENT_NOTIFICATION:
                ts = notif.timestamp.strftime("%Y-%m-%d %H:%M:%S") if notif.timestamp else ""
                env_notifications.append(f"[{ts}] {notif.message}")
            elif msg_type == MessageType.ENVIRONMENT_STOP:
                has_stop = True

        return user_messages, env_notifications, has_stop

    # ── Data Access ───────────────────────────────────────────────────

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds since scenario start."""
        if self._are_env is None:
            return 0.0
        try:
            return self._are_env.current_time
        except AttributeError:
            return 0.0

    def get_are_environment(self) -> Any:
        """Get the underlying ARE Environment instance."""
        return self._are_env

    def get_oracle_traces(self) -> Optional[Dict[str, Any]]:
        """Get oracle event log if oracle mode was enabled.

        Returns:
            Oracle traces dict, or None if oracle was not run.
        """
        return self._oracle_traces

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop ARE simulation. Called by maseval after task completes."""
        if self._are_env is not None:
            try:
                self._are_env.stop()
            except Exception:
                pass

    # ── Tracing & Config ──────────────────────────────────────────────

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces from environment and all tools."""
        tool_traces = {}
        for name, wrapper in self._tool_wrappers.items():
            tool_traces[name] = wrapper.gather_traces()

        return {
            **super().gather_traces(),
            "scenario_id": self.state.get("scenario_id"),
            "duration": self.state.get("duration"),
            "seed": self.state.get("seed"),
            "app_names": self.state.get("app_names", []),
            "oracle_traces": self._oracle_traces,
            "final_simulation_time": self.get_simulation_time(),
            "tool_count": len(self._tool_wrappers),
            "tools": tool_traces,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration for reproducibility."""
        return {
            **super().gather_config(),
            "scenario_id": self.state.get("scenario_id"),
            "duration": self.state.get("duration"),
            "seed": self.state.get("seed"),
            "start_time": self.state.get("start_time"),
            "notification_verbosity": self._notification_verbosity,
            "run_oracle": self._run_oracle,
        }
