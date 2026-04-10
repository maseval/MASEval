"""Gaia2 Benchmark - Environment.

MASEval Environment wrapping ARE's simulation.

Original Repository: https://github.com/facebookresearch/meta-agents-research-environments
Code License: MIT

Citation:
    Froger, R., Benhalloum, A., Rusakov, A., et al. (2026). Gaia2: Benchmarking LLM
    Agents on Dynamic and Asynchronous Environments. ICLR 2026.
    https://openreview.net/forum?id=9gw03JpKK4
"""

from typing import Any, Dict, List, Optional

from maseval.interface.environments.are import AREEnvironment


class Gaia2Environment(AREEnvironment):
    """GAIA2 benchmark environment built on AREEnvironment.

    Extends AREEnvironment with GAIA2-specific setup:
    - Delegates to ARE's ``preprocess_scenario()`` for oracle run, judge
      creation, and turn initialization
    - Configures custom judge engine for semantic comparison
    - Filters AUI tools (notification-based message delivery)

    Inherits from AREEnvironment:
    - Tool wrapping with simulation time tracking (AREToolWrapper)
    - Notification polling (poll_notifications, get_turn_notifications)
    - Lifecycle control (pause, resume_with_offset, cleanup)
    - Tracing and configuration gathering
    """

    def __init__(
        self,
        environment_data: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        judge_engine_config: Optional[Any] = None,
    ):
        """Initialize Gaia2 environment.

        Args:
            environment_data: Environment data containing:
                - scenario: ARE BenchmarkScenario object
                - capability: Capability type (execution, search, etc.)
                - universe_id: Universe identifier
            callbacks: Optional callbacks
            judge_engine_config: Optional :class:`Gaia2JudgeEngineConfig` controlling
                which LLM model and provider the ARE judge uses for semantic comparison.
                Passed explicitly from ``setup_environment()`` (lives in ``evaluation_data``).
        """
        self._judge_engine_config = judge_engine_config
        # Gaia2 always uses notification-based delivery, so filter AUI tools
        super().__init__(
            environment_data,
            callbacks=callbacks,
            filter_aui_tools=True,
            notification_verbosity="medium",
        )

    def setup_state(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
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
            environment_data: Environment data with scenario, capability, universe_id

        Returns:
            State dictionary with scenario metadata
        """
        # Import ARE modules (optional dependency)
        try:
            from are.simulation.environment import Environment as AREEnv  # type: ignore[import-not-found]
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

        scenario = environment_data.get("scenario")
        if scenario is None:
            raise ValueError("Environment data must contain 'scenario' with ARE BenchmarkScenario")

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
        self._are_env = AREEnv(config, notification_system=VerboseNotificationSystem())

        # Store scenario for lifecycle methods and accessors
        self._scenario = scenario

        # Run scenario (registers apps, schedules events, starts event loop)
        # wait_for_end=False so control returns immediately for agent interaction
        self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": scenario.duration,
            "capability": environment_data.get("capability"),
            "universe_id": environment_data.get("universe_id"),
            "start_time": getattr(scenario, "start_time", None),
        }

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces with GAIA2-specific fields."""
        traces = super().gather_traces()
        traces["capability"] = self.state.get("capability")
        traces["universe_id"] = self.state.get("universe_id")
        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather config with GAIA2-specific fields."""
        config = super().gather_config()
        config["capability"] = self.state.get("capability")
        config["universe_id"] = self.state.get("universe_id")
        return config
