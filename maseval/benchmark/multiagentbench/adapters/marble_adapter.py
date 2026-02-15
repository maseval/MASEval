"""MARBLE agent adapter for MASEval.

This module provides an adapter that wraps MARBLE's BaseAgent for use with
MASEval's tracing and evaluation infrastructure.
"""

from typing import Any, Dict, List, Sequence, Tuple

from maseval import AgentAdapter, AgentError, MessageHistory
from maseval.benchmark.multiagentbench._constants import MARBLE_IMPORT_ERROR, ensure_marble_on_path


class MarbleAgentAdapter(AgentAdapter):
    """Adapter wrapping a MARBLE BaseAgent for MASEval tracing.

    This adapter provides a unified interface to MARBLE agents while
    capturing all relevant traces for evaluation.

    Attributes:
        agent_id: Unique identifier for the agent
        marble_agent: The underlying MARBLE BaseAgent instance
        profile: Agent's role profile from MARBLE config
    """

    def __init__(
        self,
        marble_agent: Any,
        agent_id: str,
    ):
        """Initialize the adapter.

        Args:
            marble_agent: MARBLE BaseAgent instance
            agent_id: Unique identifier for this agent
        """
        self._marble_agent = marble_agent
        self._agent_id = agent_id
        self._profile = getattr(marble_agent, "profile", "")
        self._communication_log: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []
        super().__init__(agent_instance=marble_agent, name=agent_id)
        # Initialize message history
        self.messages: MessageHistory = MessageHistory()

    @property
    def agent_id(self) -> str:
        """Return the agent's unique identifier."""
        return self._agent_id

    @property
    def profile(self) -> str:
        """Return the agent's profile."""
        return self._profile

    @property
    def marble_agent(self) -> Any:
        """Return the underlying MARBLE agent."""
        return self._marble_agent

    def _run_agent(self, query: str) -> str:
        """Execute the MARBLE agent's act() method.

        Args:
            query: The task/query to pass to the agent

        Returns:
            String result from the agent

        Raises:
            AgentError: If agent execution fails
        """
        try:
            # Call MARBLE agent's act method
            result, communication = self._marble_agent.act(query)

            # Log action
            self._action_log.append(
                {
                    "task": query,
                    "result": result,
                    "has_communication": communication is not None,
                }
            )

            # Log communication if present
            if communication is not None:
                self._communication_log.append(
                    {
                        "session_id": str(getattr(self._marble_agent, "session_id", "")),
                        "communication": communication,
                    }
                )

            # Update message history
            self.messages.add_message(role="user", content=query)
            self.messages.add_message(role="assistant", content=result)

            return result

        except Exception as e:
            raise AgentError(
                f"MARBLE agent '{self._agent_id}' failed: {e}",
                component=f"MarbleAgentAdapter:{self._agent_id}",
            ) from e

    def get_token_usage(self) -> int:
        """Get the total token usage from the MARBLE agent.

        Returns:
            Total tokens used by the agent
        """
        if hasattr(self._marble_agent, "get_token_usage"):
            return self._marble_agent.get_token_usage()
        return 0

    def get_memory_str(self) -> str:
        """Get the agent's memory as a string.

        Returns:
            Serialized memory string
        """
        memory = getattr(self._marble_agent, "memory", None)
        if memory is not None and hasattr(memory, "get_memory_str"):
            return memory.get_memory_str()
        return ""

    def get_serialized_messages(self, session_id: str = "") -> str:
        """Get serialized inter-agent messages.

        Args:
            session_id: Optional session ID filter

        Returns:
            Serialized message string
        """
        # Note: "seralize_message" is misspelled in MARBLE's API (should be "serialize")
        if hasattr(self._marble_agent, "seralize_message"):
            return self._marble_agent.seralize_message(session_id)
        return ""

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces including agent-specific data.

        Returns:
            Dict with all agent traces
        """
        traces = super().gather_traces()

        # Add MARBLE-specific traces
        traces["agent_id"] = self._agent_id
        traces["profile"] = self._profile
        traces["token_usage"] = self.get_token_usage()
        traces["action_log"] = self._action_log
        traces["communication_log"] = self._communication_log
        traces["memory"] = self.get_memory_str()
        traces["relationships"] = getattr(self._marble_agent, "relationships", {})

        # Add task history if available
        traces["task_history"] = getattr(self._marble_agent, "task_history", [])

        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather agent configuration.

        Returns:
            Dict with agent configuration
        """
        config = super().gather_config()

        config["agent_id"] = self._agent_id
        config["profile"] = self._profile
        config["strategy"] = getattr(self._marble_agent, "strategy", "default")
        config["llm"] = getattr(self._marble_agent, "llm", "unknown")

        return config


def create_marble_agents(
    agent_configs: Sequence[Dict[str, Any]],
    marble_env: Any,
    model: str,
) -> Tuple[List[MarbleAgentAdapter], Dict[str, MarbleAgentAdapter]]:
    """Create MarbleAgentAdapters from agent configurations.

    This is a factory function that creates MARBLE BaseAgent instances
    and wraps them in MarbleAgentAdapters.

    Args:
        agent_configs: List of agent configuration dicts from task data
        marble_env: MARBLE environment instance
        model: Model ID to use for agents

    Returns:
        Tuple of (agents_list, agents_dict)

    Raises:
        ImportError: If MARBLE is not available
    """
    ensure_marble_on_path()
    try:
        from marble.agent.base_agent import BaseAgent  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(MARBLE_IMPORT_ERROR.format(error=e)) from e

    agents_list: List[MarbleAgentAdapter] = []
    agents_dict: Dict[str, MarbleAgentAdapter] = {}

    for config in agent_configs:
        agent_id = config.get("agent_id", f"agent_{len(agents_list)}")

        # Create MARBLE agent
        marble_agent = BaseAgent(config=config, env=marble_env, model=model)

        # Wrap in adapter
        adapter = MarbleAgentAdapter(marble_agent=marble_agent, agent_id=agent_id)

        agents_list.append(adapter)
        agents_dict[agent_id] = adapter

    return agents_list, agents_dict
