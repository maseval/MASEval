"""Gaia2 Benchmark - Environment.

MASEval Environment wrapping ARE's simulation.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

from typing import Any, Dict, List, Optional

from maseval import Environment

from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper


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
    ):
        """Initialize Gaia2 environment.

        Args:
            task_data: Task data containing:
                - scenario: ARE BenchmarkScenario object
                - capability: Capability type (execution, search, etc.)
                - universe_id: Universe identifier
                - duration: Scenario duration in seconds
            callbacks: Optional callbacks
        """
        self._scenario = task_data.get("scenario")
        self._are_env: Any = None
        self._tool_wrappers: Dict[str, AREToolWrapper] = {}

        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ARE scenario and start simulation.

        Args:
            task_data: Task data with scenario, capability, universe_id

        Returns:
            State dictionary with scenario metadata
        """
        # Import ARE modules (optional dependency)
        try:
            from are.simulation.environment import Environment as AREEnvironment  # type: ignore[import-not-found]
            from are.simulation.environment import EnvironmentConfig  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ARE (Agent Research Environments) is required for Gaia2 benchmark.\n"
                "Install with: pip install meta-agents-research-environments\n"
                "Or: uv add --optional gaia2 meta-agents-research-environments"
            ) from e

        scenario = task_data.get("scenario")
        if scenario is None:
            raise ValueError("Task data must contain 'scenario' with ARE BenchmarkScenario")

        # Create ARE environment with config
        config = EnvironmentConfig(
            oracle_mode=False,
            duration=getattr(scenario, "duration", 86400),  # Default 24 hours
        )
        self._are_env = AREEnvironment(config)

        # Initialize scenario (loads apps, events, state)
        self._are_env.initialize_scenario(scenario)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": getattr(scenario, "duration", None),
            "capability": task_data.get("capability"),
            "universe_id": task_data.get("universe_id"),
        }

    def create_tools(self) -> Dict[str, AREToolWrapper]:
        """Wrap all ARE app tools for MASEval tracing.

        Includes critical tools:
            - SystemApp.get_current_time(): Query simulation time
            - SystemApp.wait_for_notification(timeout): Advance simulation time
            - All domain app tools (calendar, email, messaging, etc.)

        Returns:
            Dict mapping tool names to AREToolWrapper instances
        """
        tools: Dict[str, AREToolWrapper] = {}

        if self._are_env is None:
            return tools

        # Get all tools from all apps in the ARE environment
        for app in self._are_env.apps.values():
            for tool in app.get_tools():
                wrapper = AREToolWrapper(tool, self)
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
            return self._are_env.time_manager.current_time
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
