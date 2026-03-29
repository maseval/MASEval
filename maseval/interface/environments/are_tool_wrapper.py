"""ARE Tool Wrapper for MASEval.

Framework-agnostic wrapper for ARE Tool instances. Provides a callable
interface with ToolInvocationHistory tracing and metadata exposure for
framework adapters (smolagents, LangGraph, etc.) to build framework-native
tools from.

This is the layer 1->2 wrapper:
- Layer 1: ARE Tool (forward(), inputs, output_type)
- Layer 2: maseval generic (callable, ToolInvocationHistory, metadata)
- Layer 3: framework-specific -- NOT handled here.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin
from maseval.core.history import ToolInvocationHistory

try:
    from are.simulation.tool_utils import AppToolAdapter  # type: ignore[import-not-found]
except ImportError:
    AppToolAdapter = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from maseval.interface.environments.are import AREEnvironment


class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    """Framework-agnostic wrapper for ARE tools with maseval tracing.

    Wraps an ARE Tool and exposes its metadata (name, description, inputs,
    output_type) so that agent adapters can construct framework-native tools.

    Example for smolagents::

        class MySmolagentsTool(smolagents.Tool):
            skip_forward_signature_validation = True

            def __init__(self, wrapper: AREToolWrapper):
                self.wrapper = wrapper
                self.name = wrapper.name
                self.description = wrapper.description
                self.inputs = wrapper.inputs
                self.output_type = wrapper.output_type
                super().__init__()

            def forward(self, **kwargs) -> str:
                return self.wrapper(**kwargs)
    """

    def __init__(self, are_tool: Any, environment: "AREEnvironment"):
        """Initialize the tool wrapper.

        Args:
            are_tool: ARE Tool instance to wrap.
            environment: The AREEnvironment this tool belongs to.
        """
        super().__init__()
        self._are_tool = are_tool
        self._environment = environment
        self.history = ToolInvocationHistory()

        # Expose ARE tool metadata via AppToolAdapter (canonical source of truth)
        if AppToolAdapter is None:
            raise ImportError("ARE (Agent Research Environments) is required for AREToolWrapper.\nInstall with: pip install maseval[are]")
        adapter = AppToolAdapter(are_tool)
        self.name: str = adapter.name
        self.description: str = adapter.description
        self.inputs: Dict[str, Any] = adapter.inputs
        self.output_type: str = adapter.output_type
        self.actual_return_type: Optional[str] = adapter.actual_return_type

        # Extract JSON schema from ARE tool args (if available)
        self.input_schema: Dict[str, Any] = self._extract_schema(are_tool)

    @staticmethod
    def _extract_schema(are_tool: Any) -> Dict[str, Any]:
        """Convert ARE's args list to JSON schema format.

        Args:
            are_tool: ARE Tool instance.

        Returns:
            JSON schema dict with properties and required fields.
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

    def _get_simulation_time(self) -> Optional[float]:
        """Get the current simulation time from the ARE environment.

        Returns:
            The current simulation time, or None if unavailable.
        """
        try:
            return self._environment.get_simulation_time()
        except Exception:
            return None

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the ARE tool with tracing.

        Args:
            **kwargs: Tool arguments matching the inputs schema.

        Returns:
            Tool output (type varies per tool).

        Raises:
            Any exception from the underlying ARE tool is re-raised.
        """
        start_time = datetime.now()
        sim_time_before = self._get_simulation_time()
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
            if sim_time_before is not None and sim_time_after is not None:
                sim_time_elapsed = sim_time_after - sim_time_before
            else:
                sim_time_elapsed = None

            meta = {
                "wall_time": start_time.isoformat(),
                "simulation_time_before": sim_time_before,
                "simulation_time_after": sim_time_after,
                "simulation_time_elapsed": sim_time_elapsed,
            }

            self.history.add_invocation(
                inputs=kwargs,
                outputs=result if status == "success" else error_message,
                status=status,
                timestamp=start_time.isoformat(),
                meta=meta,
            )

        return result

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this tool.

        Returns:
            Dictionary with tool name, invocation history, and counts.
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "invocations": self.history.to_list(),
            "total_invocations": len(self.history),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this tool.

        Returns:
            Dictionary with tool name, description, and schema.
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in self.inputs.items())
        return f"{self.__class__.__name__}({self.name}({args}) -> {self.output_type})"
