"""Tests for ComponentRegistry thread safety and functionality.

These tests verify that ComponentRegistry correctly isolates state between
threads and provides proper trace/config collection.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from maseval.core.registry import ComponentRegistry
from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin


# ==================== Test Components ====================


class MockTraceableComponent(TraceableMixin):
    """Component that implements TraceableMixin for testing."""

    def __init__(self, name: str, trace_data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._name = name
        self._trace_data = trace_data or {"component": name}

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            **self._trace_data,
        }


class MockConfigurableComponent(TraceableMixin, ConfigurableMixin):
    """Component that implements both TraceableMixin and ConfigurableMixin."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        TraceableMixin.__init__(self)
        ConfigurableMixin.__init__(self)
        self._name = name
        self._config = config or {"setting": "default"}

    def gather_traces(self) -> Dict[str, Any]:
        return {"name": self._name, "traced": True}

    def gather_config(self) -> Dict[str, Any]:
        return {"name": self._name, **self._config}


# ==================== Basic Functionality Tests ====================


@pytest.mark.core
class TestComponentRegistryBasics:
    """Tests for basic ComponentRegistry functionality."""

    def test_register_traceable_component(self):
        """Verify component registered for tracing."""
        registry = ComponentRegistry()
        component = MockTraceableComponent("test")

        result = registry.register("agents", "my_agent", component)

        assert result is component
        assert "agents:my_agent" in registry._trace_registry
        assert registry._trace_registry["agents:my_agent"] is component

    def test_register_configurable_component(self):
        """Verify configurable component registered in both registries."""
        registry = ComponentRegistry()
        component = MockConfigurableComponent("test")

        registry.register("models", "my_model", component)

        assert "models:my_model" in registry._trace_registry
        assert "models:my_model" in registry._config_registry

    def test_duplicate_key_idempotent(self):
        """Same component, same key should be idempotent."""
        registry = ComponentRegistry()
        component = MockTraceableComponent("test")

        registry.register("agents", "agent1", component)
        registry.register("agents", "agent1", component)  # Same key, no error

        assert len(registry._trace_registry) == 1

    def test_duplicate_component_different_key_raises(self):
        """Same component with different key should raise ValueError."""
        registry = ComponentRegistry()
        component = MockTraceableComponent("test")

        registry.register("agents", "name1", component)

        with pytest.raises(ValueError) as exc_info:
            registry.register("agents", "name2", component)

        assert "already registered" in str(exc_info.value)
        assert "agents:name1" in str(exc_info.value)

    def test_clear_removes_all_registrations(self):
        """Clear should remove all registrations."""
        registry = ComponentRegistry()
        registry.register("agents", "a1", MockTraceableComponent("a1"))
        registry.register("models", "m1", MockConfigurableComponent("m1"))

        registry.clear()

        assert len(registry._trace_registry) == 0
        assert len(registry._config_registry) == 0
        assert len(registry._component_id_map) == 0

    def test_collect_traces_structure(self):
        """Verify trace output has expected structure."""
        registry = ComponentRegistry()
        agent = MockTraceableComponent("agent1", {"steps": 5})
        registry.register("agents", "agent1", agent)

        traces = registry.collect_traces()

        assert "metadata" in traces
        assert "agents" in traces
        assert "agent1" in traces["agents"]
        assert traces["agents"]["agent1"]["steps"] == 5

    def test_collect_configs_structure(self):
        """Verify config output has expected structure."""
        registry = ComponentRegistry(benchmark_config={"name": "test_benchmark"})
        model = MockConfigurableComponent("model1", {"temperature": 0.7})
        registry.register("models", "model1", model)

        configs = registry.collect_configs()

        assert "metadata" in configs
        assert "benchmark" in configs
        assert configs["benchmark"]["name"] == "test_benchmark"
        assert "models" in configs
        assert "model1" in configs["models"]


# ==================== Thread Safety Tests ====================


@pytest.mark.core
class TestComponentRegistryThreadSafety:
    """Tests for ComponentRegistry thread isolation."""

    def test_registry_thread_isolation(self):
        """Verify registrations in one thread don't appear in another."""
        registry = ComponentRegistry()
        results = {}
        barrier = threading.Barrier(2)

        def worker(worker_id: int):
            barrier.wait()  # Synchronize start

            # Register unique component
            component = MockTraceableComponent(f"comp_{worker_id}")
            registry.register("agents", f"agent_{worker_id}", component)

            # Record what this thread sees
            results[worker_id] = list(registry._trace_registry.keys())

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should only see its own component
        assert results[0] == ["agents:agent_0"]
        assert results[1] == ["agents:agent_1"]

    def test_clear_only_affects_current_thread(self):
        """Clearing in one thread shouldn't affect another."""
        registry = ComponentRegistry()
        thread1_sees_after_clear = []
        thread2_sees_after_clear = []
        barrier = threading.Barrier(2)
        sync_point = threading.Barrier(2)

        def thread1_worker():
            registry.register("agents", "t1_agent", MockTraceableComponent("t1"))
            barrier.wait()  # Both threads have registered
            sync_point.wait()  # Wait for thread 2 to check

            registry.clear()
            thread1_sees_after_clear.extend(list(registry._trace_registry.keys()))

        def thread2_worker():
            registry.register("agents", "t2_agent", MockTraceableComponent("t2"))
            barrier.wait()  # Both threads have registered
            sync_point.wait()  # Sync before thread 1 clears

            # Wait a bit for thread 1 to clear
            time.sleep(0.05)
            thread2_sees_after_clear.extend(list(registry._trace_registry.keys()))

        t1 = threading.Thread(target=thread1_worker)
        t2 = threading.Thread(target=thread2_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Thread 1 cleared its own registry
        assert thread1_sees_after_clear == []
        # Thread 2 still has its component
        assert thread2_sees_after_clear == ["agents:t2_agent"]

    def test_concurrent_registration_no_race(self):
        """Multiple threads registering simultaneously without races."""
        registry = ComponentRegistry()
        errors = []
        barrier = threading.Barrier(4)

        def worker(worker_id: int):
            try:
                barrier.wait()
                for i in range(10):
                    component = MockTraceableComponent(f"comp_{worker_id}_{i}")
                    registry.register("agents", f"agent_{worker_id}_{i}", component)
            except Exception as e:
                errors.append((worker_id, e))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_collect_traces(self):
        """Multiple threads collecting traces simultaneously."""
        registry = ComponentRegistry()
        results = {}
        barrier = threading.Barrier(4)

        def worker(worker_id: int):
            # Each thread registers its own component
            registry.register("agents", f"agent_{worker_id}", MockTraceableComponent(f"agent_{worker_id}"))
            barrier.wait()

            # All threads collect simultaneously
            traces = registry.collect_traces()
            results[worker_id] = traces

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        # Each thread should see only its own agent
        for worker_id, traces in results.items():
            assert f"agent_{worker_id}" in traces["agents"]
            assert len(traces["agents"]) == 1
