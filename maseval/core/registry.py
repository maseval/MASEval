"""Thread-safe component registry for task execution.

This module provides the ComponentRegistry class that tracks components
(agents, models, tools, etc.) during task execution. It uses thread-local
storage to enable parallel task execution without cross-contamination.
"""

import threading
from typing import Any, Dict, Optional, Union
from datetime import datetime

from .tracing import TraceableMixin
from .config import ConfigurableMixin
from .usage import Usage, UsageTrackableMixin

# Type alias for components that can be registered
RegisterableComponent = Union[TraceableMixin, ConfigurableMixin, UsageTrackableMixin]


class ComponentRegistry:
    """Thread-safe registry for tracking components during task execution.

    Each thread gets its own isolated registry state, enabling parallel
    task execution without cross-contamination. The registry tracks both
    Traceable and Configurable components for comprehensive data collection.

    Usage:
        registry = ComponentRegistry()

        # Register components (thread-local)
        registry.register("agents", "orchestrator", agent_adapter)
        registry.register("environment", "env", environment)

        # Collect data
        traces = registry.collect_traces()
        configs = registry.collect_configs()

        # Clear for next task
        registry.clear()
    """

    def __init__(self, benchmark_config: Optional[Dict[str, Any]] = None):
        """Initialize the registry.

        Args:
            benchmark_config: Benchmark-level configuration to include in
                collect_configs() output. This is shared (not thread-local).
        """
        self._local = threading.local()
        self._benchmark_config = benchmark_config or {}

        # Persistent usage aggregates (NOT cleared between repetitions).
        # Protected by a lock since multiple threads may call collect_usage().
        self._usage_lock = threading.Lock()
        self._usage_total: Usage = Usage()
        self._usage_by_component: Dict[str, Usage] = {}

    # --- Thread-local state properties ---

    @property
    def _trace_registry(self) -> Dict[str, TraceableMixin]:
        if not hasattr(self._local, "trace_registry"):
            self._local.trace_registry = {}
        return self._local.trace_registry

    @property
    def _component_id_map(self) -> Dict[int, str]:
        if not hasattr(self._local, "component_id_map"):
            self._local.component_id_map = {}
        return self._local.component_id_map

    @property
    def _config_registry(self) -> Dict[str, ConfigurableMixin]:
        if not hasattr(self._local, "config_registry"):
            self._local.config_registry = {}
        return self._local.config_registry

    @property
    def _config_component_id_map(self) -> Dict[int, str]:
        if not hasattr(self._local, "config_component_id_map"):
            self._local.config_component_id_map = {}
        return self._local.config_component_id_map

    @property
    def _usage_registry(self) -> Dict[str, UsageTrackableMixin]:
        if not hasattr(self._local, "usage_registry"):
            self._local.usage_registry = {}
        return self._local.usage_registry

    @property
    def _usage_component_id_map(self) -> Dict[int, str]:
        if not hasattr(self._local, "usage_component_id_map"):
            self._local.usage_component_id_map = {}
        return self._local.usage_component_id_map

    # --- Public API ---

    def register(self, category: str, name: str, component: RegisterableComponent) -> RegisterableComponent:
        """Register a component for trace and config collection.

        Args:
            category: Component category (e.g., "agents", "models", "environment", "seeding")
            name: Unique identifier within the category
            component: Component instance (TraceableMixin and/or ConfigurableMixin)

        Returns:
            The component (for chaining)

        Raises:
            ValueError: If component already registered under a different key
        """
        component_id = id(component)
        key = f"{category}:{name}"

        # Check for duplicate registration under different key
        existing_key = (
            self._component_id_map.get(component_id)
            or self._config_component_id_map.get(component_id)
            or self._usage_component_id_map.get(component_id)
        )
        if existing_key and existing_key != key:
            raise ValueError(
                f"Component is already registered as '{existing_key}' and cannot be "
                f"re-registered as '{key}'. Note: Environments, users, and agents "
                f"returned from setup methods are automatically registered."
            )
        if existing_key == key:
            return component  # Idempotent

        # Register for tracing if supported
        if isinstance(component, TraceableMixin):
            self._trace_registry[key] = component
            self._component_id_map[component_id] = key

        # Register for config if supported
        if isinstance(component, ConfigurableMixin):
            self._config_registry[key] = component
            self._config_component_id_map[component_id] = key

        # Register for usage tracking if supported
        if isinstance(component, UsageTrackableMixin):
            self._usage_registry[key] = component
            self._usage_component_id_map[component_id] = key

        return component

    def clear(self) -> None:
        """Clear per-repetition registrations for the current thread.

        Does NOT clear persistent usage aggregates (``total_usage``,
        ``usage_by_component``), which accumulate across all repetitions.
        """
        self._trace_registry.clear()
        self._component_id_map.clear()
        self._config_registry.clear()
        self._config_component_id_map.clear()
        self._usage_registry.clear()
        self._usage_component_id_map.clear()

    def collect_traces(self) -> Dict[str, Any]:
        """Collect execution traces from all registered components."""
        traces: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._trace_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
        }

        for key, component in self._trace_registry.items():
            category, comp_name = key.split(":", 1)

            try:
                component_traces = component.gather_traces()

                # Inject name from registry if component doesn't have it
                if "name" not in component_traces:
                    component_traces["name"] = comp_name

                # Handle environment and user as direct values (not nested in dict)
                if category == "environment":
                    traces["environment"] = component_traces
                elif category == "user":
                    traces["user"] = component_traces
                else:
                    # Ensure category exists in traces
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = component_traces
            except Exception as e:
                # Gracefully handle tracing errors
                error_info = {
                    "error": f"Failed to gather traces: {e}",
                    "error_type": type(e).__name__,
                    "component_type": type(component).__name__,
                }

                if category == "environment":
                    traces["environment"] = error_info
                elif category == "user":
                    traces["user"] = error_info
                else:
                    if category not in traces:
                        traces[category] = {}
                    traces[category][comp_name] = error_info

        return traces

    def collect_configs(self) -> Dict[str, Any]:
        """Collect configuration from all registered components."""
        configs: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._config_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
            "benchmark": self._benchmark_config,
        }

        for key, component in self._config_registry.items():
            category, comp_name = key.split(":", 1)

            try:
                component_config = component.gather_config()

                # Inject name from registry if component doesn't have it
                if "name" not in component_config:
                    component_config["name"] = comp_name

                # Handle environment and user as direct values (not nested in dict)
                if category == "environment":
                    configs["environment"] = component_config
                elif category == "user":
                    configs["user"] = component_config
                else:
                    # Ensure category exists in configs
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = component_config
            except Exception as e:
                # Gracefully handle config gathering errors
                error_info = {
                    "error": f"Failed to gather config: {e}",
                    "error_type": type(e).__name__,
                    "component_type": type(component).__name__,
                }

                if category == "environment":
                    configs["environment"] = error_info
                elif category == "user":
                    configs["user"] = error_info
                else:
                    if category not in configs:
                        configs[category] = {}
                    configs[category][comp_name] = error_info

        return configs

    def collect_usage(self) -> Dict[str, Any]:
        """Collect usage from all registered UsageTrackableMixin components.

        Returns a structured dict (same shape as ``collect_traces()`` and
        ``collect_configs()``). Also accumulates into persistent aggregates
        (``total_usage``, ``usage_by_component``) that survive ``clear()``.
        """
        usage: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "thread_id": threading.current_thread().ident,
                "total_components": len(self._usage_registry),
            },
            "agents": {},
            "models": {},
            "tools": {},
            "simulators": {},
            "callbacks": {},
            "environment": None,
            "user": None,
            "other": {},
        }

        for key, component in self._usage_registry.items():
            category, comp_name = key.split(":", 1)

            try:
                component_usage = component.gather_usage()

                # Inject grouping fields from registry context if not set
                if component_usage.category is None:
                    component_usage.category = category
                if component_usage.component_name is None:
                    component_usage.component_name = comp_name

                usage_dict = component_usage.to_dict()

                # Handle environment and user as direct values (not nested in dict)
                if category == "environment":
                    usage["environment"] = usage_dict
                elif category == "user":
                    usage["user"] = usage_dict
                else:
                    if category not in usage:
                        usage[category] = {}
                    usage[category][comp_name] = usage_dict

                # Accumulate into persistent aggregates (thread-safe).
                with self._usage_lock:
                    self._usage_total = self._usage_total + component_usage
                    if key in self._usage_by_component:
                        self._usage_by_component[key] = self._usage_by_component[key] + component_usage
                    else:
                        self._usage_by_component[key] = component_usage

            except Exception as e:
                error_info = {
                    "error": f"Failed to gather usage: {e}",
                    "error_type": type(e).__name__,
                    "component_type": type(component).__name__,
                }

                if category == "environment":
                    usage["environment"] = error_info
                elif category == "user":
                    usage["user"] = error_info
                else:
                    if category not in usage:
                        usage[category] = {}
                    usage[category][comp_name] = error_info

        return usage

    @property
    def total_usage(self) -> Usage:
        """Running usage total across all repetitions. Queryable at any time."""
        with self._usage_lock:
            return self._usage_total

    @property
    def usage_by_component(self) -> Dict[str, Usage]:
        """Per-component running totals across all repetitions."""
        with self._usage_lock:
            return dict(self._usage_by_component)

    def update_benchmark_config(self, benchmark_config: Dict[str, Any]) -> None:
        """Update the benchmark-level configuration.

        Args:
            benchmark_config: New benchmark configuration dict.
        """
        self._benchmark_config = benchmark_config
