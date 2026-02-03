"""Core configuration infrastructure for comprehensive configuration tracking.

This module provides the base `ConfigurableMixin` class that enables any component
(agents, models, tools, simulators, callbacks) to provide structured configuration
information for later analysis and reproducibility.
"""

from typing import Any, Dict
from datetime import datetime


class ConfigurableMixin:
    """Mixin that provides configuration gathering capability to any component.

    Classes that inherit from ConfigurableMixin can be registered with a Benchmark
    instance and will have their configurations automatically collected before evaluation.

    The `gather_config()` method provides a default implementation that returns
    basic metadata (component type and timestamp). Subclasses can override or
    extend this method to include component-specific configuration information
    (model parameters, agent settings, tool specifications, etc.). The returned
    dictionary must be JSON-serializable.

    How to use:
        All core MASEval components (AgentAdapter, ModelAdapter, Environment,
        User, LLMSimulator, BenchmarkCallback, etc.) inherit from ConfigurableMixin
        by default and provide comprehensive configuration out of the box.

        For custom components, simply inherit from ConfigurableMixin and optionally
        extend the `gather_config()` method to add your own configuration data:

        ```python
        class MyCustomTool(ConfigurableMixin):
            def __init__(self, temperature: float = 0.7, max_retries: int = 3):
                self.temperature = temperature
                self.max_retries = max_retries

            def gather_config(self) -> Dict[str, Any]:
                return {
                    **super().gather_config(),
                    "temperature": self.temperature,
                    "max_retries": self.max_retries,
                    "version": "1.0.0"
                }
        ```

        Then register it with your benchmark:

        ```python
        benchmark = MyBenchmark(tasks, agent_data)
        tool = MyCustomTool(temperature=0.9)
        benchmark.register("tool", "my_tool", tool)
        ```

    Thread Safety:
        Configuration collection happens synchronously in the main thread after all
        task execution completes. The `gather_config()` method is called sequentially
        and should return static configuration data (not runtime state).

    Attributes:
        Components should expose their configuration through instance variables or
        properties that can be accessed during configuration gathering.
    """

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this component.

        Provides a default implementation that returns basic metadata about
        the component (type and collection timestamp). Subclasses should
        extend this method to include their own configuration data.

        This method is called by the Benchmark before evaluation to collect
        all configuration information. The returned dictionary must be JSON-serializable.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp of when config was collected

        Subclasses typically add additional component-specific configuration.

        Returns:
            Dictionary containing configuration with standardized structure.

        How to use:
            Override this method and call `super().gather_config()` to extend
            the base implementation with your own data:

            ```python
            def gather_config(self) -> Dict[str, Any]:
                return {
                    **super().gather_config(),
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            ```

            If you don't need custom configuration tracking, you can use the default
            implementation without overriding (it will still return basic
            metadata about your component).
        """
        return {
            "type": self.__class__.__name__,
            "gathered_at": datetime.now().isoformat(),
        }
