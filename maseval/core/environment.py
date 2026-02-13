from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .callback import EnvironmentCallback
from .tracing import TraceableMixin
from .config import ConfigurableMixin


class Environment(ABC, TraceableMixin, ConfigurableMixin):
    """Manages the state and tools available during a task execution.

    Subclasses must implement:
    - setup_state(task_data) -> Any: Initialize environment state from task data
    - create_tools() -> Dict[str, Any]: Create tools keyed by name
    """

    def __init__(self, task_data: Dict[str, Any], callbacks: Optional[List[EnvironmentCallback]] = None):
        super().__init__()
        self.callbacks = callbacks or []
        for cb in self.callbacks:
            cb.on_setup_start(self)
        self.state = self.setup_state(task_data)
        self.tools = self.create_tools()
        for cb in self.callbacks:
            cb.on_setup_end(self)

    @abstractmethod
    def setup_state(self, task_data: dict) -> Any:
        """Initializes the environment's state from task data."""
        pass

    @abstractmethod
    def create_tools(self) -> Dict[str, Any]:
        """Creates tools that can interact with the environment's state.

        Returns:
            Dict mapping tool names to tool instances
        """
        pass

    def get_tools(self) -> Dict[str, Any]:
        """Get all tools as a dict."""
        return self.tools

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            The tool, or None if not found
        """
        return self.tools.get(name)

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this environment and its tools.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `tool_count` - Number of tools in environment
        - `tools` - Dictionary of tool traces keyed by tool name

        Returns:
            Dictionary containing environment execution traces.
        """
        tool_traces = {}

        for tool_name, tool in self.tools.items():
            # Try to gather traces from the tool
            if hasattr(tool, "gather_traces"):
                tool_traces[tool_name] = tool.gather_traces()
            # Check for ToolInvocationHistory
            elif hasattr(tool, "history") and hasattr(tool.history, "to_list"):
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "invocations": tool.history.to_list(),
                    "total_invocations": len(tool.history.to_list()),
                }
            # Check if it has a simulator (common pattern in examples)
            elif hasattr(tool, "simulator") and hasattr(tool.simulator, "gather_traces"):
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "simulator_traces": tool.simulator.gather_traces(),
                }
            else:
                # Basic info if no tracing capability
                tool_traces[tool_name] = {
                    "type": type(tool).__name__,
                    "has_tracing": False,
                }

        return {
            **super().gather_traces(),
            "tool_count": len(self.tools),
            "tools": tool_traces,
        }

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this environment.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `tool_count` - Number of tools
        - `tool_names` - List of tool names

        Returns:
            Dictionary containing environment configuration.
        """
        return {
            **super().gather_config(),
            "tool_count": len(self.tools),
            "tool_names": list(self.tools.keys()),
        }
