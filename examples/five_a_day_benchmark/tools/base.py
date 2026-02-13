"""Base tool interface for the 5-a-day benchmark.

Provides framework-agnostic tool base class with adapters for:
- smolagents
- langgraph
- llamaindex
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin


# Common argument schemas for tools
ARG_SCHEMAS = {
    "action": {"type": "string", "description": "Action to perform", "nullable": False},
    "to": {"type": "string", "description": "Recipient email address", "nullable": True},
    "subject": {"type": "string", "description": "Email subject", "nullable": True},
    "body": {"type": "string", "description": "Email body or message content", "nullable": True},
    "email_id": {"type": "string", "description": "Email ID", "nullable": True},
    "transaction_id": {"type": "string", "description": "Transaction ID", "nullable": True},
    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)", "nullable": True},
    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)", "nullable": True},
    "expression": {"type": "string", "description": "Mathematical expression", "nullable": True},
    "code": {"type": "string", "description": "Python code to execute", "nullable": True},
    "person": {"type": "string", "description": "Person name", "nullable": True},
    "relationship": {"type": "string", "description": "Relationship type", "nullable": True},
    "asset_name": {"type": "string", "description": "Asset ticker symbol (e.g., AAPL, GOOGL)", "nullable": True},
    "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)", "nullable": True},
    "ticker": {"type": "string", "description": "Stock ticker symbol", "nullable": True},
    "date": {"type": "string", "description": "Date (YYYY-MM-DD)", "nullable": True},
    "time": {"type": "string", "description": "Time (HH:MM)", "nullable": True},
    "duration": {"type": "integer", "description": "Duration in minutes", "nullable": True},
    "activity_type": {"type": "string", "description": "Type of activity", "nullable": True},
    "location": {"type": "string", "description": "Location", "nullable": True},
    "budget": {"type": "number", "description": "Budget amount", "nullable": True},
    "checkin": {"type": "string", "description": "Check-in date", "nullable": True},
    "checkout": {"type": "string", "description": "Check-out date", "nullable": True},
    "activity_id": {"type": "string", "description": "Activity ID", "nullable": True},
    "workout_id": {"type": "string", "description": "Workout ID", "nullable": True},
    "max_price": {"type": "number", "description": "Maximum price", "nullable": True},
    "max_distance": {"type": "number", "description": "Maximum distance", "nullable": True},
    "min_wifi": {"type": "integer", "description": "Minimum wifi rating", "nullable": True},
    "hotel_id": {"type": "string", "description": "Hotel ID", "nullable": True},
    "start_time": {"type": "string", "description": "Start time (HH:MM)", "nullable": True},
    "end_time": {"type": "string", "description": "End time (HH:MM)", "nullable": True},
    "title": {"type": "string", "description": "Event title", "nullable": True},
    "description": {"type": "string", "description": "Event description", "nullable": True},
}


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""

    tool_name: str
    inputs: dict[str, Any]
    outputs: Any
    timestamp: datetime
    status: str  # "success" or "error"
    error_message: str | None = None


@dataclass
class ToolResult:
    """Standardized tool result."""

    success: bool
    data: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC, TraceableMixin, ConfigurableMixin):
    """Framework-agnostic base class for all benchmark tools.

    This class provides:
    - Unified execution interface via execute()
    - Invocation history tracking
    - Tracing and configuration support
    - Conversion methods to framework-specific tool types
    """

    def __init__(self, name: str, description: str, tool_args: list[str] | None = None):
        super().__init__()
        self.name = name
        self.description = description
        self.history: list[ToolInvocation] = []

        # Build tool arguments schema
        self.tool_args = {}
        if tool_args:
            for arg in tool_args:
                if arg in ARG_SCHEMAS:
                    self.tool_args[arg] = ARG_SCHEMAS[arg]
                else:
                    # Default fallback for unknown args
                    self.tool_args[arg] = {"type": "string", "description": arg, "nullable": True}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given inputs."""
        pass

    def __call__(self, **kwargs) -> ToolResult:
        """Allow tool to be called directly."""
        timestamp = datetime.now()
        try:
            result = self.execute(**kwargs)
            self.history.append(
                ToolInvocation(
                    tool_name=self.name,
                    inputs=kwargs,
                    outputs=result.data,
                    timestamp=timestamp,
                    status="success" if result.success else "error",
                    error_message=result.error,
                )
            )
            return result
        except Exception as e:
            self.history.append(
                ToolInvocation(
                    tool_name=self.name,
                    inputs=kwargs,
                    outputs=None,
                    timestamp=timestamp,
                    status="error",
                    error_message=str(e),
                )
            )
            return ToolResult(success=False, data=None, error=str(e))

    def get_history(self) -> list[ToolInvocation]:
        """Get tool invocation history."""
        return self.history

    def clear_history(self) -> None:
        """Clear tool invocation history."""
        self.history = []

    def gather_traces(self) -> dict[str, Any]:
        """Gather execution traces from this tool."""
        return {
            **super().gather_traces(),
            "name": self.name,
            "invocations": [
                {
                    "tool_name": inv.tool_name,
                    "inputs": inv.inputs,
                    "outputs": inv.outputs,
                    "timestamp": inv.timestamp.isoformat() if inv.timestamp else None,
                    "status": inv.status,
                    "error_message": inv.error_message,
                }
                for inv in self.history
            ],
            "total_invocations": len(self.history),
        }

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this tool."""
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
        }

    # Framework conversion methods

    def to_smolagents(self) -> Any:
        """Convert to smolagents Tool."""
        try:
            from smolagents import Tool as SmolagentsTool  # noqa: F401
        except ImportError as e:
            raise ImportError("smolagents is not installed. Install it with: pip install maseval[smolagents]") from e

        return SmolagentsToolAdapter(self)

    def to_langgraph(self) -> Any:
        """Convert to LangChain/LangGraph StructuredTool."""
        try:
            from langchain_core.tools import StructuredTool  # noqa: F401
        except ImportError as e:
            raise ImportError("langchain is not installed. Install it with: pip install maseval[langgraph]") from e

        return LangGraphToolAdapter(self)

    def to_llamaindex(self) -> Any:
        """Convert to LlamaIndex FunctionTool."""
        try:
            from llama_index.core.tools import FunctionTool  # noqa: F401  # type: ignore
        except ImportError as e:
            raise ImportError("llamaindex is not installed. Install it with: pip install llama-index-core") from e

        return LlamaIndexToolAdapter(self)


class SmolagentsToolAdapter(TraceableMixin, ConfigurableMixin):
    """Adapter that wraps BaseTool for smolagents framework.

    Uses composition to avoid __call__ conflicts between smolagents.Tool and BaseTool.
    """

    def __init__(self, base_tool: BaseTool):
        super().__init__()
        try:
            from smolagents import Tool as SmolagentsTool
        except ImportError as e:
            raise ImportError("smolagents is not installed") from e

        self.base_tool = base_tool

        # Tool names already use underscores (email_send, banking_get_balance)
        # No sanitization needed - names are framework-compatible by design

        # Create a dynamic smolagents Tool class
        class DynamicSmolagentsTool(SmolagentsTool):
            name = base_tool.name
            description = base_tool.description
            # Use specific tool arguments (excluding 'action' which is smolagents internal)
            inputs = base_tool.tool_args
            output_type = "string"
            skip_forward_signature_validation = True

            def __init__(self):  # noqa: N807
                super().__init__()

            def forward(self, **kwargs) -> str:  # noqa: N807
                """Execute the base tool and return string result.

                No action parameter - each tool does one specific thing.
                Smolagents passes all tool arguments directly to forward().
                """
                # Filter out None values from kwargs
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                result = base_tool(**filtered_kwargs)
                if result.success:
                    return str(result.data)
                else:
                    raise RuntimeError(result.error or "Tool execution failed")

        # Set the class name to the tool name for better tracing (CamelCase)
        class_name = "".join(x.title() for x in base_tool.name.split("_")) + "Tool"
        DynamicSmolagentsTool.__name__ = class_name
        DynamicSmolagentsTool.__qualname__ = class_name

        self.tool = DynamicSmolagentsTool()
        self.name = base_tool.name  # Expose for Environment's _tools_dict

    def gather_traces(self) -> dict[str, Any]:
        """Gather traces from wrapped tool."""
        return self.base_tool.gather_traces()

    def gather_config(self) -> dict[str, Any]:
        """Gather config from wrapped tool."""
        return self.base_tool.gather_config()


class LangGraphToolAdapter(TraceableMixin, ConfigurableMixin):
    """Adapter that wraps BaseTool for LangGraph/LangChain framework."""

    def __init__(self, base_tool: BaseTool):
        super().__init__()
        try:
            from langchain_core.tools import StructuredTool
        except ImportError as e:
            raise ImportError("langchain is not installed") from e

        self.base_tool = base_tool
        self.name = base_tool.name  # Expose for Environment's _tools_dict

        def tool_function(**kwargs) -> str:
            """Wrapper function for LangChain StructuredTool."""
            result = base_tool(**kwargs)
            if result.success:
                return str(result.data)
            else:
                raise RuntimeError(result.error or "Tool execution failed")

        self.tool = StructuredTool.from_function(
            func=tool_function,
            name=base_tool.name,
            description=base_tool.description,
        )

    def gather_traces(self) -> dict[str, Any]:
        """Gather traces from wrapped tool."""
        return self.base_tool.gather_traces()

    def gather_config(self) -> dict[str, Any]:
        """Gather config from wrapped tool."""
        return self.base_tool.gather_config()


class LlamaIndexToolAdapter(TraceableMixin, ConfigurableMixin):
    """Adapter that wraps BaseTool for LlamaIndex framework."""

    def __init__(self, base_tool: BaseTool):
        super().__init__()
        try:
            from llama_index.core.tools import FunctionTool  # type: ignore
        except ImportError as e:
            raise ImportError("llamaindex is not installed") from e

        self.base_tool = base_tool
        self.name = base_tool.name  # Expose for Environment's _tools_dict

        def tool_function(**kwargs) -> str:
            """Wrapper function for LlamaIndex FunctionTool."""
            result = base_tool(**kwargs)
            if result.success:
                return str(result.data)
            else:
                raise RuntimeError(result.error or "Tool execution failed")

        self.tool = FunctionTool.from_defaults(
            fn=tool_function,
            name=base_tool.name,
            description=base_tool.description,
        )

    def gather_traces(self) -> dict[str, Any]:
        """Gather traces from wrapped tool."""
        return self.base_tool.gather_traces()

    def gather_config(self) -> dict[str, Any]:
        """Gather config from wrapped tool."""
        return self.base_tool.gather_config()
