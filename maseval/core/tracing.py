"""Core tracing infrastructure for comprehensive execution tracking.

This module provides the base `TraceableMixin` class that enables any component
(agents, models, tools, simulators, callbacks) to provide structured traces of
their execution for later evaluation and analysis.
"""

from typing import Any, Dict
from datetime import datetime


class TraceableMixin:
    """Mixin that provides tracing capability to any component.

    Classes that inherit from TraceableMixin can be registered with a Benchmark
    instance and will have their traces automatically collected before evaluation.

    The `gather_traces()` method provides a default implementation that returns
    basic metadata (component type and timestamp). Subclasses can override or
    extend this method to include component-specific execution information
    (messages, invocations, errors, etc.). The returned dictionary must be
    JSON-serializable.

    How to use:
        All core MASEval components (AgentAdapter, ModelAdapter, Environment,
        User, LLMSimulator, BenchmarkCallback, etc.) inherit from TraceableMixin
        by default and provide comprehensive tracing out of the box.

        For custom components, simply inherit from TraceableMixin and optionally
        extend the `gather_traces()` method to add your own tracing data:

        ```python
        class MyCustomTool(TraceableMixin):
            def __init__(self):
                self.logs = []

            def execute(self, *args, **kwargs):
                result = self._do_work(*args, **kwargs)
                self.logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "args": args,
                    "kwargs": kwargs,
                    "result": result
                })
                return result

            def gather_traces(self) -> Dict[str, Any]:
                return {
                    **super().gather_traces(),
                    "total_calls": len(self.logs),
                    "logs": self.logs
                }
        ```

        Then register it with your benchmark:

        ```python
        benchmark = MyBenchmark(tasks, agent_data)
        tool = MyCustomTool()
        benchmark.register("tool", "my_tool", tool)
        ```

    Thread Safety:
        Trace collection happens synchronously in the main thread after all
        task execution completes. Individual components should use appropriate
        thread-safe data structures (e.g., threading.Lock) when accumulating
        traces during concurrent execution, but the `gather_traces()` method
        itself is called sequentially.

    Attributes:
        Components can store traces in any internal data structure. Common patterns:
        - `self.logs = []` for invocation histories
        - `self._messages = MessageHistory()` for conversations
        - `self.logs = []` for simulator attempts
    """

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this component.

        Provides a default implementation that returns basic metadata about
        the component (type and collection timestamp). Subclasses should
        extend this method to include their own execution data.

        This method is called by the Benchmark before evaluation to collect
        all execution data. The returned dictionary must be JSON-serializable.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp of when traces were collected

        Subclasses typically add additional component-specific data.

        Returns:
            Dictionary containing traces with standardized structure.

        How to use:
            Override this method and call `super().gather_traces()` to extend
            the base implementation with your own data:

            ```python
            def gather_traces(self) -> Dict[str, Any]:
                return {
                    **super().gather_traces(),
                    "my_field": self._my_data,
                    "execution_count": len(self._history)
                }
            ```

            If you don't need custom tracing, you can use the default
            implementation without overriding (it will still return basic
            metadata about your component).
        """
        return {
            "type": self.__class__.__name__,
            "gathered_at": datetime.now().isoformat(),
        }
