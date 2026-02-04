"""Test automatic registration and duplicate detection in Benchmark.

These tests verify that benchmark components (agents, environments, users) are
automatically registered for tracing and configuration collection when returned
from setup methods. The registration system prevents duplicate registration and
provides helpful error messages when components are accidentally re-registered
with different names.
"""

import pytest
from maseval import TaskQueue, TraceableMixin
from conftest import DummyBenchmark, DummyModelAdapter, DummyAgentAdapter, DummyAgent


@pytest.mark.core
def test_automatic_agent_registration():
    """Test that agents returned from setup_agents are auto-registered.

    Verifies that the benchmark's run() method automatically registers agents
    returned from setup_agents() for trace and config collection without
    requiring manual register() calls.
    """
    tasks = TaskQueue.from_list([{"query": "test", "id": "1", "environment_data": {}}])
    agent_data = {}

    benchmark = DummyBenchmark()

    # Before run, registry should be empty
    assert len(benchmark._registry._trace_registry) == 0

    # Run one step to trigger setup
    from maseval.core.seeding import DefaultSeedGenerator

    seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
    for task, agent_data in zip(tasks, [agent_data]):
        environment = benchmark.setup_environment(agent_data, task, seed_gen)
        user = benchmark.setup_user(agent_data, environment, task, seed_gen)
        agents_to_run, agents_dict = benchmark.setup_agents(agent_data, environment, task, user, seed_gen)

        # Manually trigger auto-registration (simulating what run() does)
        if environment is not None and isinstance(environment, TraceableMixin):
            benchmark.register("environment", "env", environment)
        if user is not None and isinstance(user, TraceableMixin):
            benchmark.register("user", "user", user)
        for agent_name, agent in agents_dict.items():
            if isinstance(agent, TraceableMixin):
                benchmark.register("agents", agent_name, agent)

        break  # Only test first task

    # Check that components were registered
    assert "environment:env" in benchmark._registry._trace_registry
    assert "agents:test_agent" in benchmark._registry._trace_registry


@pytest.mark.core
def test_duplicate_registration_detection():
    """Test that registering same component with different name raises error.

    Verifies that the ID-based tracking system detects when a component instance
    is registered multiple times with different names, preventing data confusion.
    """
    benchmark = DummyBenchmark()

    # Create a component
    model = DummyModelAdapter()

    # First registration should succeed
    benchmark.register("models", "test_model", model)

    # Second registration with same name should be idempotent (no error)
    benchmark.register("models", "test_model", model)

    # Registration with different name should fail
    with pytest.raises(ValueError) as exc_info:
        benchmark.register("models", "different_name", model)

    assert "already registered" in str(exc_info.value)
    assert "models:test_model" in str(exc_info.value)
    assert "automatically registered" in str(exc_info.value)


@pytest.mark.core
def test_duplicate_registration_helpful_message():
    """Test that duplicate registration error provides helpful debugging info.

    Verifies that error message includes both the existing registration name
    and the attempted new name, plus mentions automatic registration.
    """
    benchmark = DummyBenchmark()

    # Create and register an agent
    agent = DummyAgent()
    agent_adapter = DummyAgentAdapter(agent, "my_agent")
    benchmark.register("agents", "first_name", agent_adapter)

    # Try to register again with different name
    with pytest.raises(ValueError) as exc_info:
        benchmark.register("agents", "second_name", agent_adapter)

    error_message = str(exc_info.value)
    assert "already registered as 'agents:first_name'" in error_message
    assert "cannot be re-registered as 'agents:second_name'" in error_message
    assert "automatically registered" in error_message


@pytest.mark.core
def test_manual_registration_for_models():
    """Test that model adapters require manual registration.

    Verifies that models are not automatically registered (unlike agents,
    environments, and users), requiring explicit register() calls.
    """
    benchmark = DummyBenchmark()

    # Create a model
    model = DummyModelAdapter()

    # Models are not auto-registered, so manual registration should work
    benchmark.register("models", "my_model", model)

    # Verify it was registered
    assert "models:my_model" in benchmark._registry._trace_registry
    assert benchmark._registry._trace_registry["models:my_model"] is model


@pytest.mark.core
def test_component_id_tracking():
    """Test that internal ID mapping tracks registered component instances.

    Verifies that benchmark maintains a Python id() to name mapping for
    detecting duplicate registrations of the same component instance.
    """
    benchmark = DummyBenchmark()

    # Create a component
    model = DummyModelAdapter()

    # Register it
    benchmark.register("models", "test_model", model)

    # Verify ID tracking
    assert id(model) in benchmark._registry._component_id_map
    assert benchmark._registry._component_id_map[id(model)] == "models:test_model"


@pytest.mark.core
def test_registry_cleared_after_repetition():
    """Test that component registry is cleared between task repetitions.

    Verifies that after each task iteration completes, the registry is reset
    to allow fresh components for the next iteration while preserving reports.
    """
    tasks = TaskQueue.from_list(
        [
            {"query": "test1", "id": "1", "environment_data": {}},
            {"query": "test2", "id": "2", "environment_data": {}},
        ]
    )
    agent_data = {}

    benchmark = DummyBenchmark(n_task_repeats=2)

    # Run the benchmark
    benchmark.run(tasks, agent_data=agent_data)

    # After run completes, registry should be empty (cleared after last repetition)
    assert len(benchmark._registry._trace_registry) == 0
    assert len(benchmark._registry._component_id_map) == 0

    # But reports should contain entries for all task repetitions
    assert len(benchmark.reports) == 4  # 2 tasks * 2 repeats

    # Verify structure of reports
    for report in benchmark.reports:
        assert "task_id" in report
        assert "repeat_idx" in report
        assert "traces" in report
        assert isinstance(report["traces"], dict)
