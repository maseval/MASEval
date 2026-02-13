"""Integration tests for GAIA2 benchmark using real HuggingFace data.

These tests validate that GAIA2 components work correctly with actual
downloaded data, not synthetic fixtures.  They are marked ``live`` + ``slow``
+ ``benchmark`` + ``gaia2`` because they download real data from HuggingFace
and exercise the ARE simulation stack.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_gaia2/test_integration.py -v
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark, pytest.mark.gaia2]


# =============================================================================
# Fixture: load a small set of real tasks
# =============================================================================


@pytest.fixture(scope="module")
def real_gaia2_tasks():
    """Load a small set of GAIA2 tasks from HuggingFace.

    Loads 5 tasks to keep download and runtime manageable while still
    exercising real data paths.
    """
    from maseval.benchmark.gaia2.data_loader import load_tasks

    tasks = load_tasks(capability="execution", split="validation", limit=5)
    return list(tasks)


@pytest.fixture(scope="module")
def first_real_task(real_gaia2_tasks):
    """Return the first real GAIA2 task."""
    assert len(real_gaia2_tasks) > 0, "No GAIA2 tasks loaded"
    return real_gaia2_tasks[0]


# =============================================================================
# Environment Tests with Real Data
# =============================================================================


class TestGaia2EnvironmentWithRealData:
    """Test that real downloaded GAIA2 tasks work with Gaia2Environment."""

    def test_environment_creates_from_real_task(self, first_real_task):
        """Gaia2Environment can be created from a real task."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        assert env is not None

    def test_environment_setup_state(self, first_real_task):
        """setup_state() succeeds with a real ARE scenario.

        This exercises ARE's preprocess_scenario() with real scenario data.
        """
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        try:
            state = env.setup_state(first_real_task.environment_data)

            assert isinstance(state, dict)
            assert "capability" in state
            assert "duration" in state
            assert state["duration"] > 0, "Scenario duration should be positive"
        finally:
            env.cleanup()

    def test_real_tools_are_created(self, first_real_task):
        """Tools created from a real scenario are non-empty Gaia2GenericTool instances."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        try:
            env.setup_state(first_real_task.environment_data)
            tools = env.create_tools()

            assert len(tools) > 0, "No tools created from real scenario. ARE environment should expose app tools (Calendar, Email, etc.)."

            for name, tool in tools.items():
                assert isinstance(tool, Gaia2GenericTool), f"Tool '{name}' is {type(tool).__name__}, expected Gaia2GenericTool"
                assert tool.name, "Tool has empty name"
        finally:
            env.cleanup()

    def test_real_tools_have_descriptions(self, first_real_task):
        """Tools from real scenarios have descriptions and inputs schema."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        try:
            env.setup_state(first_real_task.environment_data)
            tools = env.create_tools()

            for name, tool in tools.items():
                # Every real ARE tool should have a description
                assert tool.description, (
                    f"Tool '{name}' has empty description. ARE tools should provide _public_description or function_description."
                )
                # inputs should be a dict (possibly empty for tools with no args)
                assert isinstance(tool.inputs, dict), f"Tool '{name}' inputs is {type(tool.inputs).__name__}, expected dict"
        finally:
            env.cleanup()

    def test_environment_traces(self, first_real_task):
        """gather_traces() returns expected keys after real scenario setup."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        try:
            env.setup_state(first_real_task.environment_data)
            env.create_tools()
            traces = env.gather_traces()

            assert isinstance(traces, dict)
            assert "capability" in traces
            assert "tool_count" in traces
            assert traces["tool_count"] > 0
        finally:
            env.cleanup()


# =============================================================================
# Default Agent Tests with Real Tools
# =============================================================================


class TestDefaultAgentWithRealTools:
    """Test DefaultGaia2Agent construction with real ARE tools."""

    def test_agent_builds_system_prompt_with_real_tools(self, first_real_task):
        """DefaultGaia2Agent system prompt includes real tool names."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.benchmark.gaia2.gaia2 import DefaultGaia2Agent

        from conftest import DummyModelAdapter

        env = Gaia2Environment(task_data=first_real_task.environment_data)
        try:
            env.setup_state(first_real_task.environment_data)
            tools = env.create_tools()

            model = DummyModelAdapter(
                model_id="test-model",
                responses=[
                    'Thought: Done.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>'
                ],
            )

            agent = DefaultGaia2Agent(
                tools=tools,  # type: ignore[arg-type]  # Gaia2GenericTool has __call__
                model=model,
                environment=env,
                max_iterations=1,
            )

            # System prompt should mention real tool names
            assert "AgentUserInterface__send_message_to_user" in agent.system_prompt, "System prompt should include the AgentUserInterface tool"
            # Check at least one domain tool is mentioned
            tool_names = list(tools.keys())
            mentioned = any(name in agent.system_prompt for name in tool_names)
            assert mentioned, f"System prompt should mention at least one tool. Tool names: {tool_names[:5]}..."
        finally:
            env.cleanup()


# =============================================================================
# Evaluator Tests with Real Oracle Events
# =============================================================================


class TestGaia2EvaluatorWithRealData:
    """Test Gaia2Evaluator with real oracle events."""

    def test_evaluator_creates_from_real_task(self, first_real_task):
        """Gaia2Evaluator can be created with real oracle events."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        env = Gaia2Environment(task_data=first_real_task.environment_data)

        evaluator = Gaia2Evaluator(
            task=first_real_task,
            environment=env,
            use_llm_judge=False,
        )
        assert evaluator is not None

    def test_evaluator_filter_traces_with_real_data(self, first_real_task):
        """filter_traces() works with a synthetic trace structure."""
        from maseval.benchmark.gaia2.environment import Gaia2Environment
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        env = Gaia2Environment(task_data=first_real_task.environment_data)

        evaluator = Gaia2Evaluator(
            task=first_real_task,
            environment=env,
            use_llm_judge=False,
        )

        # Provide a minimal synthetic trace
        traces = {
            "agents": {
                "test_agent": {
                    "messages": [],
                    "iteration_count": 1,
                }
            },
            "tools": {},
            "environment": {"final_simulation_time": 0.0},
        }

        filtered = evaluator.filter_traces(traces)
        assert isinstance(filtered, dict)


# =============================================================================
# Pipeline Smoke Test
# =============================================================================


class TestGaia2PipelineSmoke:
    """Smoke test for the full GAIA2 pipeline with real data."""

    def test_full_pipeline_single_task(self, first_real_task):
        """Gaia2Benchmark.run() on one real task produces a result.

        Uses DummyModelAdapter (no API keys needed) and the ConcreteGaia2Benchmark
        from conftest. The agent immediately sends a message to terminate.
        """
        from maseval import TaskQueue

        from .conftest import ConcreteGaia2Benchmark

        benchmark = ConcreteGaia2Benchmark.create(progress_bar=False)

        task_queue = TaskQueue([first_real_task])
        results = benchmark.run(task_queue, agent_data={})

        assert len(results) == 1, f"Expected 1 result, got {len(results)}. Check test_data_integrity tests first if this fails."

        result = results[0]
        assert "status" in result, "Result missing 'status' key"
        # All possible TaskExecutionStatus values
        known_statuses = {
            "success",
            "agent_error",
            "environment_error",
            "user_error",
            "task_timeout",
            "unknown_execution_error",
            "evaluation_failed",
            "setup_failed",
        }
        assert result["status"] in known_statuses, f"Unexpected status '{result['status']}'. Known: {known_statuses}"
