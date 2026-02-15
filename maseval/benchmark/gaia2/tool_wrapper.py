"""Gaia2 Benchmark - Tool Wrapper.

Framework-agnostic wrapper for ARE AppTool instances, following MACSGenericTool pattern.
Provides clean API with built-in tracing for MASEval compatibility.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin
from maseval.core.history import ToolInvocationHistory

if TYPE_CHECKING:
    from maseval.benchmark.gaia2.environment import Gaia2Environment


class Gaia2GenericTool(TraceableMixin, ConfigurableMixin):
    """Framework-agnostic wrapper for ARE tools.

    Similar to MACSGenericTool - provides clean API with built-in tracing.
    Developers wrap this for their framework using composition.

    Example for smolagents:

        class MySmolagentsTool(smolagents.Tool):
            skip_forward_signature_validation = True

            def __init__(self, generic_tool: Gaia2GenericTool):
                self.generic_tool = generic_tool
                self.name = generic_tool.name
                self.description = generic_tool.description
                self.inputs = generic_tool.inputs
                self.output_type = generic_tool.output_type
                super().__init__()

            def forward(self, **kwargs) -> str:
                return self.generic_tool(**kwargs)

            def gather_traces(self):
                return self.generic_tool.gather_traces()

    This wrapper preserves ARE's native return types while adding
    MASEval tracing capabilities and providing a framework-agnostic interface.
    """

    def __init__(self, are_tool: Any, environment: "Gaia2Environment"):
        """Initialize the tool wrapper.

        Args:
            are_tool: ARE AppTool instance to wrap
            environment: The Gaia2Environment this tool belongs to
        """
        super().__init__()
        self._are_tool = are_tool
        self._environment = environment

        # Delegate metadata extraction to ARE's AppToolAdapter (tool_utils.py:544-584).
        # This is the source of truth for tool name, description, inputs, and output_type.
        from are.simulation.tool_utils import AppToolAdapter  # type: ignore[import-not-found]

        adapter = AppToolAdapter(are_tool)
        self.name: str = adapter.name
        self.description: str = adapter.description
        self.inputs: Dict[str, Any] = adapter.inputs
        self.output_type: str = adapter.output_type
        self.actual_return_type: Optional[str] = adapter.actual_return_type
        self.input_schema: Dict[str, Any] = self._extract_schema(are_tool)

        # Initialize invocation history
        self.history = ToolInvocationHistory()

    @staticmethod
    def _extract_schema(are_tool: Any) -> Dict[str, Any]:
        """Convert ARE's args list to JSON schema format for tracing/config.

        Args:
            are_tool: ARE AppTool instance

        Returns:
            JSON schema dictionary with properties and required fields
        """
        args = getattr(are_tool, "args", None)
        if not args:
            return {}

        properties = {}
        required = []

        for arg in args:
            param_name = getattr(arg, "name", None)
            if not param_name:
                continue

            properties[param_name] = {
                "type": arg.arg_type,
                "description": getattr(arg, "description", ""),
            }

            if not arg.has_default:
                required.append(param_name)

        return {"properties": properties, "required": required}

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
            result = self._are_tool(**kwargs)
        except Exception as e:
            status = "error"
            error_message = str(e)
            raise
        finally:
            sim_time_after = self._get_simulation_time()

            # Record invocation with timing metadata (same structure as before)
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
            return self._environment.get_simulation_time()
        except Exception:
            return None

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
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        """String representation of the tool."""
        args = ", ".join(f"{k}: {v['type']}" for k, v in self.inputs.items())
        return f"{self.__class__.__name__}({self.name}({args}) -> {self.output_type})"


def wrap_are_tools(
    are_tools: List[Any],
    environment: "Gaia2Environment",
) -> Dict[str, Gaia2GenericTool]:
    """Wrap multiple ARE tools for MASEval.

    Args:
        are_tools: List of ARE AppTool instances
        environment: The Gaia2Environment these tools belong to

    Returns:
        Dictionary mapping tool names to wrapped tools
    """
    wrapped: Dict[str, Gaia2GenericTool] = {}

    for tool in are_tools:
        wrapper = Gaia2GenericTool(tool, environment)
        wrapped[wrapper.name] = wrapper

    return wrapped
