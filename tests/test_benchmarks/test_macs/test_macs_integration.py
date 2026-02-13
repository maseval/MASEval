"""Integration tests for MACS benchmark components.

These tests verify the complete MACS benchmark pipeline end-to-end.
Component-specific unit tests are in their respective test files.
"""

import json
import pytest

from maseval import Task
from maseval.benchmark.macs import (
    MACSEnvironment,
    MACSEvaluator,
    MACSUser,
)

from .conftest import ConcreteMACSBenchmark
from conftest import DummyModelAdapter


# =============================================================================
# Full Benchmark Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestFullBenchmarkIntegration:
    """End-to-end integration tests for MACS benchmark.

    These tests verify the complete pipeline works correctly when all
    components are used together. Component-specific behavior is tested
    in the individual test files (test_macs_*.py).
    """

    def test_complete_task_lifecycle(self, sample_agent_data, travel_task):
        """Test complete task: setup → run → evaluate."""
        # Model responses for various stages
        responses = [
            # Tool simulation (if tools are called)
            '{"flights": [{"id": "DL123", "time": "8:15am"}]}',
            # User evaluation
            json.dumps([{"assertion": "User acknowledged", "answer": "TRUE", "evidence": "OK"}]),
            # System evaluation
            json.dumps([{"assertion": "Tool called", "answer": "TRUE", "evidence": "OK"}]),
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)

        # Setup phase
        from maseval.core.seeding import DefaultSeedGenerator

        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
        env = benchmark.setup_environment(sample_agent_data, travel_task, seed_gen)
        user = benchmark.setup_user(sample_agent_data, env, travel_task, seed_gen)
        agents_list, agents_dict = benchmark.setup_agents(sample_agent_data, env, travel_task, user, seed_gen)
        evaluators = benchmark.setup_evaluators(env, travel_task, agents_list, user, seed_gen)

        # Verify setup
        assert isinstance(env, MACSEnvironment)
        assert isinstance(user, MACSUser)
        assert len(agents_list) == 1
        assert len(evaluators) == 2

        # Run phase
        final_answer = benchmark.run_agents(agents_list, travel_task, env)
        assert final_answer is not None

        # Evaluate phase
        traces = {
            "agents": {
                "integration_agent": {
                    "messages": [
                        {"role": "user", "content": travel_task.query},
                        {"role": "assistant", "content": final_answer},
                    ]
                }
            },
            "tools": {},
        }
        results = benchmark.evaluate(evaluators, agents_dict, final_answer, traces)

        # Verify results
        assert len(results) == 1
        assert "user_gsr" in results[0]
        assert "system_gsr" in results[0]
        assert "overall_gsr" in results[0]


# =============================================================================
# Data Loading Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestDataLoadingIntegration:
    """Integration tests for data loading with benchmark components."""

    def test_loaded_task_works_with_environment(self, macs_model, sample_agent_data):
        """Tasks loaded from data work with MACSEnvironment."""
        # Create a mock task that simulates loaded data
        task = Task(
            query="Book a flight",
            environment_data={
                "model_id": "test-model",
                "tools": [
                    {
                        "tool_name": "flight_search",
                        "actions": [{"name": "search", "description": "Search flights"}],
                    }
                ],
            },
            user_data={"model_id": "test-model"},
            evaluation_data={"model_id": "test-model", "assertions": ["user: Booking done"]},
            id="task-000001",
            metadata={"scenario": "Travel booking scenario"},
        )

        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = ConcreteMACSBenchmark(macs_model)
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
        env = benchmark.setup_environment(sample_agent_data, task, seed_gen)

        assert "search" in env.tools

    def test_loaded_agent_config_works_with_environment(self, macs_model_factory):
        """Agent config works with tool assignment."""
        # Simulate loaded agent config
        agent_config = {
            "agents": [
                {"agent_id": "main", "agent_name": "Main Agent", "tools": ["tool_group"]},
            ],
            "primary_agent_id": "main",
        }

        task_data = {
            "environment_data": {
                "tools": [
                    {
                        "tool_name": "tool_group",
                        "actions": [{"name": "action1", "description": "Action 1"}],
                    }
                ]
            }
        }

        env = MACSEnvironment(task_data, macs_model_factory)

        # Get tools for agent from config
        agent_spec = agent_config["agents"][0]
        agent_tools = env.get_tools_for_agent(agent_spec)  # type: ignore[arg-type]

        assert "action1" in agent_tools


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_evaluator_handles_malformed_llm_response(self, travel_task, sample_conversation):
        """Evaluator gracefully handles malformed LLM responses."""
        model = DummyModelAdapter(responses=["This is not valid JSON at all"])
        evaluator = MACSEvaluator(model, travel_task, gsr_type="user")

        traces = {"messages": sample_conversation}
        result = evaluator(traces)

        # Should return error result, not crash
        assert result["gsr"] == 0.0
        assert "error" in result

    def test_environment_handles_empty_tool_specs(self, macs_model_factory):
        """Environment handles tasks with no tools."""
        task_data = {"environment_data": {"tools": []}}
        env = MACSEnvironment(task_data, macs_model_factory)

        assert env.tools == {}


# =============================================================================
# End-to-End Pipeline Tests (benchmark.run())
# =============================================================================


@pytest.mark.benchmark
class TestEndToEndPipeline:
    """End-to-end tests that call benchmark.run() with TaskQueue.

    These tests verify the complete MACS benchmark pipeline by actually
    calling benchmark.run(). More granular integration tests are in
    TestFullBenchmarkIntegration above.
    """

    def test_run_single_task_complete_pipeline(self, sample_agent_data, travel_task):
        """Full end-to-end test: single task through benchmark.run()."""
        model = DummyModelAdapter(
            responses=[
                '{"text": "Yes, that flight works.", "details": {}}',
                '[{"assertion": "User request acknowledged", "answer": "TRUE", "evidence": "OK"}]',
                '[{"assertion": "Tool was called", "answer": "TRUE", "evidence": "OK"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(model)
        reports = benchmark.run([travel_task], agent_data=sample_agent_data)

        # Verify complete report structure
        assert len(reports) == 1
        report = reports[0]
        assert report["task_id"] == str(travel_task.id)
        assert report["repeat_idx"] == 0
        assert report["status"] == "success"
        assert "traces" in report
        assert "config" in report
        assert "eval" in report

    def test_run_multiple_tasks(self, sample_agent_data, macs_task_queue):
        """Run benchmark with multiple tasks via TaskQueue."""
        model = DummyModelAdapter(
            responses=[
                '{"text": "User response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(model)
        reports = benchmark.run(macs_task_queue, agent_data=sample_agent_data)

        assert len(reports) == len(macs_task_queue)
        for report in reports:
            assert report["status"] == "success"
            assert "eval" in report

    def test_run_with_task_repeats(self, sample_agent_data, sample_task):
        """Run benchmark with multiple task repetitions."""
        model = DummyModelAdapter(
            responses=[
                '{"text": "response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        n_repeats = 3
        benchmark = ConcreteMACSBenchmark(model, n_task_repeats=n_repeats)
        reports = benchmark.run([sample_task], agent_data=sample_agent_data)

        assert len(reports) == n_repeats
        for i, report in enumerate(reports):
            assert report["repeat_idx"] == i
            assert report["task_id"] == str(sample_task.id)

    def test_run_with_callbacks(self, sample_agent_data, sample_task):
        """Benchmark triggers callbacks during run."""
        from maseval import BenchmarkCallback

        class TrackingCallback(BenchmarkCallback):
            def __init__(self):
                self.events = []

            def on_run_start(self, benchmark):
                self.events.append("run_start")

            def on_task_start(self, benchmark, task):
                self.events.append("task_start")

            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                self.events.append(f"repeat_start_{repeat_idx}")

            def on_task_repeat_end(self, benchmark, report):
                self.events.append(f"repeat_end_{report['repeat_idx']}")

            def on_task_end(self, benchmark, task, result):
                self.events.append("task_end")

            def on_run_end(self, benchmark, results):
                self.events.append("run_end")

        model = DummyModelAdapter(
            responses=[
                '{"text": "response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        callback = TrackingCallback()
        benchmark = ConcreteMACSBenchmark(model, callbacks=[callback])
        benchmark.run([sample_task], agent_data=sample_agent_data)

        # Verify callback sequence
        expected_order = ["run_start", "task_start", "repeat_start_0", "repeat_end_0", "task_end", "run_end"]
        for event in expected_order:
            assert event in callback.events
        for i in range(len(expected_order) - 1):
            assert callback.events.index(expected_order[i]) < callback.events.index(expected_order[i + 1])
