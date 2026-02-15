"""MultiAgentBench Environment implementation.

This module provides the MASEval Environment wrapper for MARBLE environments.
"""

import importlib
import shutil
from typing import Any, Callable, Dict, Optional

from maseval import Environment, EnvironmentError, ToolInvocationHistory
from maseval.benchmark.multiagentbench._constants import MARBLE_IMPORT_ERROR, ensure_marble_on_path


# Domains requiring external infrastructure
INFRASTRUCTURE_DOMAINS = frozenset({"database"})


class MultiAgentBenchEnvironment(Environment):
    """MASEval Environment wrapper for MARBLE environments.

    This environment wraps MARBLE's domain-specific environments (Research,
    Bargaining, Coding, etc.) and exposes their tools through MASEval's
    tracing infrastructure.

    Attributes:
        domain: The domain name (e.g., "research", "bargaining")
        marble_env: The underlying MARBLE environment instance
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
    ):
        """Initialize the environment.

        Args:
            task_data: Task data containing environment configuration

        Raises:
            EnvironmentError: If required infrastructure is unavailable
            ImportError: If MARBLE is not available
        """
        self.domain = task_data.get("scenario", "")
        self._marble_env: Any = None
        self._tool_histories: Dict[str, ToolInvocationHistory] = {}
        super().__init__(task_data)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state and optionally create MARBLE environment.

        Args:
            task_data: Task data containing environment configuration

        Returns:
            Initial state dictionary

        Raises:
            EnvironmentError: If required infrastructure is unavailable
        """
        domain = task_data.get("scenario", "")
        env_config = task_data.get("environment", {})
        task_config = task_data.get("task", {})

        # Check infrastructure requirements
        if domain.lower() in INFRASTRUCTURE_DOMAINS:
            if not self._check_infrastructure(domain):
                raise EnvironmentError(
                    f"Domain '{domain}' requires external infrastructure. See README.md for setup instructions.",
                    component="MultiAgentBenchEnvironment",
                )

        # Build config for MARBLE environment
        marble_config = {
            "description": env_config.get("description", f"{domain} environment"),
            "task_description": task_config.get("content", "") if isinstance(task_config, dict) else str(task_config),
            "ground_truth": env_config.get("ground_truth", ""),
            "max_iterations": env_config.get("max_iterations") or task_data.get("max_iterations", 10),
        }

        # Pass werewolf config path for WerewolfEnv (different constructor)
        if domain.lower() == "werewolf":
            marble_config["werewolf_config_path"] = task_data.get("werewolf_config_path", "")

        self._marble_env = self._create_marble_environment(domain, marble_config)

        return {
            "domain": domain,
            "env_config": env_config,
            "task_config": task_config,
            "marble_env_type": type(self._marble_env).__name__,
            "max_iterations": marble_config["max_iterations"],
        }

    def _check_infrastructure(self, domain: str) -> bool:
        """Check if required infrastructure is available.

        Args:
            domain: Domain name

        Returns:
            True if infrastructure is available, False otherwise
        """
        domain_lower = domain.lower()

        if domain_lower == "database":
            # Check Docker availability
            return shutil.which("docker") is not None

        return True

    def _create_marble_environment(
        self,
        domain: str,
        config: Dict[str, Any],
    ) -> Any:
        """Create the appropriate MARBLE environment.

        Args:
            domain: Domain name
            config: Environment configuration

        Returns:
            MARBLE environment instance

        Raises:
            ImportError: If MARBLE is not available
        """
        domain_lower = domain.lower()
        env_name = config.get("name", domain_lower)

        ensure_marble_on_path()

        # Import MARBLE environments
        try:
            from marble.environments.base_env import BaseEnvironment  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(MARBLE_IMPORT_ERROR.format(error=e)) from e

        # Map domains to environment classes
        env_mapping: Dict[str, str] = {
            "coding": "marble.environments.coding_env.CodingEnvironment",
            "database": "marble.environments.db_env.DBEnvironment",
            "research": "marble.environments.research_env.ResearchEnvironment",
            "bargaining": "marble.environments.world_env.WorldSimulationEnvironment",
            "minecraft": "marble.environments.minecraft_env.MinecraftEnvironment",
            "werewolf": "marble.environments.werewolf_env.WerewolfEnv",
        }

        env_class_path = env_mapping.get(domain_lower)

        if env_class_path is None:
            # Use base environment for unknown domains
            return BaseEnvironment(env_name, config)

        # Dynamic import of domain-specific environment
        module_path, class_name = env_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        env_class = getattr(module, class_name)

        # WerewolfEnv has a different constructor: (name, config_path, log_dir)
        if domain_lower == "werewolf":
            config_path = config.get("werewolf_config_path", "")
            return env_class(env_name, config_path)

        # MARBLE domain envs take (config, name) while BaseEnvironment takes (name, config)
        return env_class(config=config, name=env_name)

    def create_tools(self) -> Dict[str, Callable]:
        """Create tools from MARBLE environment for MASEval tracing.

        MARBLE environments expose tools via action_handler_descriptions.
        This method wraps them for MASEval's tracing infrastructure.

        Returns:
            Dict mapping tool names to wrapped callables
        """
        tools: Dict[str, Callable] = {}

        # Get action handlers from MARBLE environment
        action_handlers = getattr(self._marble_env, "_action_handlers", {})

        for action_name, handler in action_handlers.items():
            # Create tool history for this action
            history = ToolInvocationHistory()
            self._tool_histories[action_name] = history

            # Wrap handler for tracing
            wrapped = self._wrap_tool_for_tracing(action_name, handler, history)
            tools[action_name] = wrapped

        return tools

    def _wrap_tool_for_tracing(
        self,
        name: str,
        handler: Callable,
        history: ToolInvocationHistory,
    ) -> Callable:
        """Wrap a MARBLE action handler for MASEval tracing.

        Args:
            name: Tool name
            handler: Original handler callable
            history: ToolInvocationHistory to record invocations

        Returns:
            Wrapped callable that records invocations
        """

        def traced_handler(**kwargs: Any) -> Any:
            try:
                result = handler(**kwargs)
                history.add_invocation(
                    inputs=kwargs,
                    outputs=result,
                    status="success",
                )
                return result
            except Exception as e:
                history.add_invocation(
                    inputs=kwargs,
                    outputs=str(e),
                    status="error",
                )
                raise

        # Attach metadata for inspection
        traced_handler._original_name = name  # type: ignore[attr-defined]
        traced_handler._history = history  # type: ignore[attr-defined]

        return traced_handler

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a specific tool by name.

        Args:
            name: Tool name

        Returns:
            Tool callable if found, None otherwise
        """
        return self.tools.get(name)

    def get_tool_descriptions(self) -> Dict[str, Any]:
        """Get tool descriptions in OpenAI function format.

        Returns:
            Dict mapping tool names to their OpenAI-format descriptions
        """
        return getattr(self._marble_env, "action_handler_descriptions", {})

    def apply_action(
        self,
        agent_id: Optional[str],
        action_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an action in the MARBLE environment.

        Args:
            agent_id: ID of the agent performing the action
            action_name: Name of the action to execute
            arguments: Arguments for the action

        Returns:
            Action result dictionary

        """
        return self._marble_env.apply_action(agent_id, action_name, arguments)

    def is_done(self) -> bool:
        """Check if the environment has reached a terminal state.

        Returns:
            True if done, False otherwise
        """
        return self._marble_env.is_done()

    def is_task_completed(self) -> bool:
        """Check if the task has been completed successfully.

        Returns:
            True if task completed, False otherwise
        """
        return self._marble_env.is_task_completed()

    def get_marble_state(self) -> Dict[str, Any]:
        """Get the current MARBLE environment state.

        Returns:
            State dictionary from MARBLE environment
        """
        return self._marble_env.get_state()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces including tool invocations.

        Returns:
            Dict with environment traces
        """
        traces = super().gather_traces()

        # Add domain-specific info
        traces["domain"] = self.domain
        traces["marble_env_type"] = type(self._marble_env).__name__
        traces["marble_state"] = self.get_marble_state()
        traces["is_done"] = self.is_done()
        traces["is_task_completed"] = self.is_task_completed()

        # Collect tool invocation histories
        tool_traces = {}
        for name, history in self._tool_histories.items():
            invocations = history.to_list()
            tool_traces[name] = {
                "invocations": invocations,
                "invocation_count": len(invocations),
            }
        traces["tool_invocations"] = tool_traces

        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration.

        Returns:
            Dict with environment configuration
        """
        config = super().gather_config()

        config["domain"] = self.domain
        config["marble_env_type"] = type(self._marble_env).__name__

        # Add tool descriptions
        config["tool_descriptions"] = self.get_tool_descriptions()

        return config
