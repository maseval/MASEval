"""Gaia2 Benchmark - Environment.

MASEval Environment wrapping ARE's simulation.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

from typing import Any, Dict, List, Optional, Tuple

from maseval import Environment

from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool


class Gaia2Environment(Environment):
    """MASEval Environment wrapping ARE's simulation.

    The ARE simulation runs its own internal event loop. Agent interaction
    happens purely through tool calls - including time control via
    SystemApp.wait_for_notification(). No special execution loop needed.

    Exposes all ARE app tools (Calendar, Email, Messaging, Contacts, Shopping,
    Cab, City, FileSystem, Browser, ChatsApp, SystemApp, Timer) to agents.

    Key Features:
        - Wraps ARE's simulation environment
        - Provides MASEval-compatible tool wrappers with tracing
        - Exposes simulation time for temporal reasoning tasks
        - Handles proper cleanup of ARE resources
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        judge_engine_config: Optional[Any] = None,
    ):
        """Initialize Gaia2 environment.

        Args:
            task_data: Task data containing:
                - scenario: ARE BenchmarkScenario object
                - capability: Capability type (execution, search, etc.)
                - universe_id: Universe identifier
            callbacks: Optional callbacks
            judge_engine_config: Optional :class:`Gaia2JudgeEngineConfig` controlling
                which LLM model and provider the ARE judge uses for semantic comparison.
                Passed explicitly from ``setup_environment()`` (lives in ``evaluation_data``).
        """
        self._scenario = task_data.get("scenario")
        self._judge_engine_config = judge_engine_config
        self._are_env: Any = None
        self._tool_wrappers: Dict[str, Gaia2GenericTool] = {}

        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ARE scenario and start simulation.

        Delegates to ARE's ``preprocess_scenario()`` for faithful preprocessing:

        1. Ensure SystemApp is present.
        2. Set scenario duration from ARE defaults (1800s standard, 420s for Time).
        3. Initialize the scenario (populates apps, events).
        4. Run oracle mode to generate expected event log.
        5. Soft-reset so app state is clean for agent run.
        6. Create judge and initialize turns with trigger conditions.
        7. Start the agent-mode simulation.

        Args:
            task_data: Task data with scenario, capability, universe_id

        Returns:
            State dictionary with scenario metadata
        """
        # Import ARE modules (optional dependency)
        try:
            from are.simulation.environment import Environment as AREEnvironment  # type: ignore[import-not-found]
            from are.simulation.environment import EnvironmentConfig  # type: ignore[import-not-found]
            from are.simulation.scenarios.scenario_imported_from_json.utils import (  # type: ignore[import-not-found]
                get_scenario_duration,
                preprocess_scenario,
            )
            from are.simulation.validation import GraphPerEventJudgeConfig  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ARE (Agent Research Environments) is required for Gaia2 benchmark.\n"
                "Install with: pip install meta-agents-research-environments\n"
                "Or: uv add --optional gaia2 meta-agents-research-environments"
            ) from e

        # Scenario duration defaults from ARE scenarios/config.py:18-19
        from are.simulation.scenarios.config import (  # type: ignore[import-not-found]
            MAX_SCENARIO_DURATION,
            MAX_TIME_SCENARIO_DURATION,
        )

        scenario = task_data.get("scenario")
        if scenario is None:
            raise ValueError("Task data must contain 'scenario' with ARE BenchmarkScenario")

        # Determine scenario duration (matching ARE's get_scenario_duration)
        # ARE scenarios/config.py:18: MAX_SCENARIO_DURATION = 1800 (30 min)
        # ARE scenarios/config.py:19: MAX_TIME_SCENARIO_DURATION = 420 (7 min)
        max_duration = get_scenario_duration(scenario, MAX_TIME_SCENARIO_DURATION, MAX_SCENARIO_DURATION)

        # Use ARE's preprocess_scenario() for faithful preprocessing.
        # This handles: SystemApp insertion, duration setting, scenario initialization,
        # oracle run, soft reset, judge creation, turn initialization with trigger
        # conditions, and judge state initialization.
        # GraphPerEventJudge uses an LLM for semantic comparison of tool arguments
        # (email content, calendar descriptions, etc.) via soft checkers.
        # ARE scenarios/scenario_imported_from_json/utils.py:43-157
        if self._judge_engine_config is not None:
            # User provided custom judge engine config — create engine explicitly
            # ARE validation/configs.py:32-59
            from are.simulation.agents.are_simulation_agent_config import (  # type: ignore[import-not-found]
                LLMEngineConfig,
            )
            from are.simulation.validation.configs import create_judge_engine  # type: ignore[import-not-found]

            llm_engine_config = LLMEngineConfig(
                model_name=self._judge_engine_config.model_name,
                provider=self._judge_engine_config.provider,
                endpoint=self._judge_engine_config.endpoint,
            )
            engine = create_judge_engine(llm_engine_config)
            judge_config = GraphPerEventJudgeConfig(engine=engine)
        else:
            # Default: use ARE's built-in defaults (Llama 3.3 70B via HuggingFace)
            # ARE validation/configs.py:28-29, 149
            judge_config = GraphPerEventJudgeConfig()

        preprocess_scenario(
            scenario=scenario,
            judge_config=judge_config,
            max_scenario_duration=max_duration,
        )

        # Create ARE environment for the agent run
        # Match ARE scenario_runner.py:267-282
        from are.simulation.notification_system import VerboseNotificationSystem  # type: ignore[import-not-found]

        config = EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
            time_increment_in_seconds=scenario.time_increment_in_seconds,
        )
        if scenario.start_time and scenario.start_time > 0:
            config.start_time = scenario.start_time
        # Match ARE scenario_runner.py:281: VerboseNotificationSystem() defaults
        # to VerbosityLevel.MEDIUM, which includes environment notifications
        # (email, messaging, shopping, cab, calendar). Without this, the default
        # is VerbosityLevel.LOW (no environment notifications).
        self._are_env = AREEnvironment(config, notification_system=VerboseNotificationSystem())

        # Run scenario (registers apps, schedules events, starts event loop)
        # wait_for_end=False so control returns immediately for agent interaction
        self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": scenario.duration,
            "capability": task_data.get("capability"),
            "universe_id": task_data.get("universe_id"),
            "start_time": getattr(scenario, "start_time", None),
        }

    # Tools removed by ARE's remove_aui_irrelevant_tools()
    # ARE agents/default_agent/are_simulation_main.py:206-228
    # User messages are delivered via the notification system, not via these tools.
    _AUI_TOOLS_TO_REMOVE = {
        "AgentUserInterface__get_last_message_from_user",
        "AgentUserInterface__get_last_message_from_agent",
        "AgentUserInterface__get_last_unread_messages",
        "AgentUserInterface__get_all_messages",
    }

    def create_tools(self) -> Dict[str, Gaia2GenericTool]:
        """Wrap ARE app tools for MASEval tracing.

        Creates framework-agnostic Gaia2GenericTool instances that provide
        clean API with built-in tracing.

        Filters out AgentUserInterface message-retrieval tools that ARE removes
        in ``remove_aui_irrelevant_tools()``, and sets ``wait_for_user_response``
        to ``False`` so the AUI does not block waiting for a response when the
        agent sends a message. User messages are delivered via the notification
        system instead.

        ARE agents/default_agent/are_simulation_main.py:206-228

        Returns:
            Dict mapping tool names to Gaia2GenericTool instances
        """
        tools: Dict[str, Gaia2GenericTool] = {}

        if self._are_env is None:
            return tools

        # Get all tools from all apps, filtering out AUI message-retrieval tools
        # ARE agents/default_agent/are_simulation_main.py:221-227
        for app in self._are_env.apps.values():
            # Set wait_for_user_response=False on AUI so it doesn't block
            # ARE agents/default_agent/are_simulation_main.py:216
            if hasattr(app, "wait_for_user_response"):
                app.wait_for_user_response = False

            for tool in app.get_tools():
                if tool.name in self._AUI_TOOLS_TO_REMOVE:
                    continue
                wrapper = Gaia2GenericTool(tool, self)
                tools[tool.name] = wrapper
                self._tool_wrappers[tool.name] = wrapper

        return tools

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds.

        Returns:
            Current simulation time in seconds since scenario start
        """
        if self._are_env is None:
            return 0.0

        try:
            return self._are_env.current_time
        except AttributeError:
            return 0.0

    def get_scenario(self) -> Any:
        """Get the ARE scenario object.

        Returns:
            ARE BenchmarkScenario object
        """
        return self._scenario

    def get_are_environment(self) -> Any:
        """Get the underlying ARE Environment.

        Used by the evaluator to access completed events and judge.

        Returns:
            ARE Environment instance
        """
        return self._are_env

    def get_notification_system(self) -> Any:
        """Get the ARE notification system.

        Used by agents that need to poll for messages between iterations,
        matching ARE's pre-step notification polling behavior.

        Returns:
            ARE NotificationSystem instance, or None if not available
        """
        if self._are_env is None:
            return None
        return getattr(self._are_env, "notification_system", None)

    def poll_notifications(self) -> Tuple[List[str], List[str], bool]:
        """Poll pending notifications from the ARE notification system.

        Drains all pending messages from the notification queue and returns
        them as pre-formatted strings. Call this between agent steps to
        receive messages that arrived during ``wait_for_notification()`` or
        from background simulation events.

        GAIA2 uses an event-driven multi-turn architecture.  When the agent
        calls ``SystemApp__wait_for_notification``, the ARE environment
        processes scheduled events, advances simulation time, and queues
        notifications.  After the tool returns, call this method to retrieve
        those notifications and inject them into the agent's context before
        the next LLM call.

        ARE agents/default_agent/steps/are_simulation.py:26-62

        Returns:
            Tuple of ``(user_messages, env_notifications, has_stop_message)``.
            ``user_messages`` and ``env_notifications`` contain pre-formatted
            strings ready to inject into agent context. ``has_stop_message``
            is True when the environment has signalled the simulation is over.
        """
        notification_system = self.get_notification_system()
        if notification_system is None:
            return [], [], False

        try:
            from datetime import datetime, timezone

            from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

            # Use simulation time, not wall-clock time. Notifications are timestamped
            # with simulation time (via TimeManager), so querying with wall-clock would
            # drain all messages prematurely. Matches ARE agents/default_agent/steps/are_simulation.py:30-32.
            sim_time = self.get_simulation_time()
            timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
            unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

            if not unhandled:
                return [], [], False

            # Separate by message type, matching ARE steps/are_simulation.py:34-61
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

        except Exception:
            return [], [], False

    def get_turn_notifications(self) -> Tuple[List[str], bool, bool]:
        """Get notifications for turn transitions, re-queuing env notifications.

        Matches ARE's ``get_notifications()`` in ``are_simulation_main.py:331-359``:
        drains the notification queue, separates by type, re-queues environment
        notifications (so the inner loop's pre-step picks them up), and returns
        user messages and status flags.

        Returns:
            Tuple of ``(user_messages, has_env_notifications, has_stop)``.
            ``user_messages`` are raw message strings for ``[TASK]`` formatting.
            ``has_env_notifications`` is True when env notifications were re-queued.
            ``has_stop`` is True when the environment signalled stop.
        """
        notification_system = self.get_notification_system()
        if notification_system is None:
            return [], False, False

        try:
            from datetime import datetime, timezone

            from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

            sim_time = self.get_simulation_time()
            timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
            unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

            if not unhandled:
                return [], False, False

            user_messages: List[str] = []
            has_env = False
            has_stop = False

            for notif in unhandled:
                msg_type = getattr(notif, "message_type", None)
                if msg_type == MessageType.USER_MESSAGE:
                    user_messages.append(notif.message)
                elif msg_type == MessageType.ENVIRONMENT_NOTIFICATION:
                    # Re-queue for inner loop's pre-step to pick up
                    # ARE are_simulation_main.py:349-352
                    notification_system.message_queue.put(notif)
                    has_env = True
                elif msg_type == MessageType.ENVIRONMENT_STOP:
                    has_stop = True

            return user_messages, has_env, has_stop

        except Exception:
            return [], False, False

    def get_start_time(self) -> Optional[float]:
        """Get the scenario start time.

        Returns:
            Start time as Unix timestamp, or None if not available
        """
        return self.state.get("start_time")

    def pause(self) -> None:
        """Pause the ARE simulation environment.

        Stops time progression during LLM generation, matching ARE's
        simulated generation time behavior.
        ARE simulation/environment.py:262-272

        No-op if environment is not available or not running.
        """
        if self._are_env is not None:
            try:
                self._are_env.pause()
            except Exception:
                pass

    def resume_with_offset(self, offset: float) -> None:
        """Resume the ARE simulation environment with a time offset.

        Advances simulation time by the given offset and resumes the event loop.
        ARE simulation/environment.py:286-298

        Args:
            offset: Time in seconds to advance the simulation clock
        """
        if self._are_env is not None:
            try:
                self._are_env.resume_with_offset(offset)
            except Exception:
                pass

    def cleanup(self) -> None:
        """Stop ARE simulation when task completes.

        Ensures proper resource cleanup and stops any running simulation.
        """
        if self._are_env is not None:
            try:
                self._are_env.stop()
            except Exception:
                pass  # Ignore cleanup errors

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces from environment and all tools.

        Returns:
            Trace dictionary with scenario info and all tool traces
        """
        tool_traces = {}
        for name, wrapper in self._tool_wrappers.items():
            tool_traces[name] = wrapper.gather_traces()

        return {
            **super().gather_traces(),
            "scenario_id": self.state.get("scenario_id"),
            "capability": self.state.get("capability"),
            "universe_id": self.state.get("universe_id"),
            "final_simulation_time": self.get_simulation_time(),
            "tool_count": len(self._tool_wrappers),
            "tools": tool_traces,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration.

        Returns:
            Configuration dictionary
        """
        config = super().gather_config()
        config.update(
            {
                "scenario_id": self.state.get("scenario_id"),
                "capability": self.state.get("capability"),
                "universe_id": self.state.get("universe_id"),
                "duration": self.state.get("duration"),
            }
        )
        return config
