"""Gaia2 Benchmark - Tool Wrapper.

Wraps ARE AppTool instances for MASEval compatibility and tracing.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin
from maseval.core.history import ToolInvocationHistory

if TYPE_CHECKING:
    from maseval.benchmark.gaia2.environment import Gaia2Environment


class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    """Wraps ARE AppTool for MASEval tracing and compatibility.

    Records all tool invocations with inputs, outputs, timestamps,
    and simulation time for post-hoc analysis.

    This wrapper preserves ARE's native return types while adding
    MASEval tracing capabilities.
    """

    def __init__(self, are_tool: Any, environment: "Gaia2Environment"):
        """Initialize the tool wrapper.

        Args:
            are_tool: ARE AppTool instance to wrap
            environment: The Gaia2Environment this tool belongs to
        """
        super().__init__()
        self.are_tool = are_tool
        self.environment = environment

        # Extract tool metadata from ARE tool
        self.name: str = getattr(are_tool, "name", str(are_tool))
        self.description: str = getattr(are_tool, "description", "")
        self.inputs: Dict[str, Any] = self._extract_schema(are_tool)

        # Initialize invocation history
        self.history = ToolInvocationHistory()

    def __call__(self, **kwargs: Any) -> Any:
        """Execute tool and record invocation.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result (preserves ARE's native return type)
        """
        start_time = datetime.now()
        sim_time_before = self._get_simulation_time()

        # Execute the ARE tool
        status = "success"
        result = None
        error_message = None

        try:
            result = self.are_tool(**kwargs)
        except Exception as e:
            status = "error"
            error_message = str(e)
            raise
        finally:
            sim_time_after = self._get_simulation_time()

            # Record invocation with timing metadata
            self.history.add_invocation(
                inputs=kwargs,
                outputs=result if status == "success" else error_message,
                status=status,
                timestamp=start_time.isoformat(),
                meta={
                    "wall_time": start_time.isoformat(),
                    "simulation_time_before": sim_time_before,
                    "simulation_time_after": sim_time_after,
                    "simulation_time_elapsed": sim_time_after - sim_time_before if sim_time_after and sim_time_before else None,
                },
            )

        return result

    def _get_simulation_time(self) -> Optional[float]:
        """Get current simulation time from ARE environment.

        Returns:
            Simulation time in seconds, or None if not available
        """
        try:
            return self.environment.get_simulation_time()
        except Exception:
            return None

    def _extract_schema(self, are_tool: Any) -> Dict[str, Any]:
        """Convert ARE tool schema to MASEval input format.

        Args:
            are_tool: ARE AppTool instance

        Returns:
            Dictionary describing tool inputs
        """
        schema: Dict[str, Any] = {}

        # Try to extract schema from ARE tool
        # ARE tools typically have an 'inputs' or 'parameters' attribute
        if hasattr(are_tool, "inputs"):
            schema = dict(are_tool.inputs) if are_tool.inputs else {}
        elif hasattr(are_tool, "parameters"):
            schema = dict(are_tool.parameters) if are_tool.parameters else {}
        elif hasattr(are_tool, "args_schema"):
            # Pydantic schema format
            try:
                schema = are_tool.args_schema.model_json_schema() if are_tool.args_schema else {}
            except Exception:
                schema = {}

        return schema

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this tool.

        Returns:
            Dictionary containing tool traces with invocation history
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "description": self.description,
            "invocations": self.history.to_list(),
            "total_invocations": len(self.history),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this tool.

        Returns:
            Dictionary containing tool configuration
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
            "inputs_schema": self.inputs,
        }


def wrap_are_tools(
    are_tools: List[Any],
    environment: "Gaia2Environment",
) -> Dict[str, AREToolWrapper]:
    """Wrap multiple ARE tools for MASEval.

    Args:
        are_tools: List of ARE AppTool instances
        environment: The Gaia2Environment these tools belong to

    Returns:
        Dictionary mapping tool names to wrapped tools
    """
    wrapped: Dict[str, AREToolWrapper] = {}

    for tool in are_tools:
        wrapper = AREToolWrapper(tool, environment)
        wrapped[wrapper.name] = wrapper

    return wrapped
