"""Test benchmark lifecycle execution.

These tests verify that the complete benchmark orchestration works correctly,
including the three-stage lifecycle (Setup → Run → Evaluate) and component
registration/cleanup between task repetitions.
"""

import pytest
from maseval import TaskQueue


@pytest.mark.core
class TestBenchmarkLifecycle:
    """Tests for complete benchmark execution lifecycle."""

    def test_benchmark_complete_run_single_task(self, simple_benchmark):
        """Test that a benchmark completes successfully with a single task."""
        benchmark, tasks, agent_data = simple_benchmark

        # Run benchmark
        reports = benchmark.run(tasks, agent_data=agent_data)

        # Verify we got one report
        assert len(reports) == 3  # 3 tasks in dummy_task_queue

        # Verify report structure
        report = reports[0]
        assert "task_id" in report
        assert "repeat_idx" in report
        assert "traces" in report
        assert "config" in report
        assert "eval" in report

        # Verify all stages were called
        assert len(benchmark.setup_environment_calls) == 3
        assert len(benchmark.setup_user_calls) == 3
        assert len(benchmark.setup_agents_calls) == 3
        assert len(benchmark.setup_evaluators_calls) == 3
        assert len(benchmark.run_agents_calls) == 3
        assert len(benchmark.evaluate_calls) == 3

    def test_benchmark_complete_run_multiple_tasks(self):
        """Test that a benchmark handles multiple tasks correctly."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
                {"query": "Task 3", "environment_data": {}},
            ]
        )
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # Should have 3 reports (one per task)
        assert len(reports) == 3

        # Each report should have unique task_id
        task_ids = [r["task_id"] for r in reports]
        assert len(set(task_ids)) == 3

        # Verify queries match - call format is (agents, task, environment, query)
        queries = [call[3] for call in benchmark.run_agents_calls]
        assert queries == ["Task 1", "Task 2", "Task 3"]

    def test_benchmark_task_repetitions(self):
        """Test that task repetitions work correctly."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = DummyBenchmark(n_task_repeats=3)

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # Should have 3 reports (one per repetition)
        assert len(reports) == 3

        # All should have same task_id but different repeat_idx
        task_ids = [r["task_id"] for r in reports]
        assert len(set(task_ids)) == 1  # All same task_id

        repeat_indices = [r["repeat_idx"] for r in reports]
        assert repeat_indices == [0, 1, 2]

        # Setup should be called 3 times (once per repetition)
        assert len(benchmark.setup_environment_calls) == 3
        assert len(benchmark.setup_agents_calls) == 3
        assert len(benchmark.run_agents_calls) == 3

    def test_benchmark_lifecycle_hooks_order(self):
        """Test that lifecycle hooks are called in the correct order."""
        from conftest import DummyBenchmark
        from maseval import BenchmarkCallback

        # Track callback invocations
        invocations = []

        class OrderTrackingCallback(BenchmarkCallback):
            def on_run_start(self, benchmark):
                invocations.append("on_run_start")

            def on_task_start(self, benchmark, task):
                invocations.append(f"on_task_start:{task.query}")

            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                invocations.append(f"on_task_repeat_start:{task.query}:{repeat_idx}")

            def on_task_repeat_end(self, benchmark, report):
                invocations.append(f"on_task_repeat_end:{report['repeat_idx']}")

            def on_task_end(self, benchmark, task, result):
                invocations.append(f"on_task_end:{task.query}")

            def on_run_end(self, benchmark, results):
                invocations.append("on_run_end")

        tasks = TaskQueue.from_list(
            [
                {"query": "Task1", "environment_data": {}},
                {"query": "Task2", "environment_data": {}},
            ]
        )
        benchmark = DummyBenchmark(
            n_task_repeats=2,
            callbacks=[OrderTrackingCallback()],
        )

        benchmark.run(tasks, agent_data={"model": "test"})

        # Verify order
        expected = [
            "on_run_start",
            # Task 1
            "on_task_start:Task1",
            "on_task_repeat_start:Task1:0",
            "on_task_repeat_end:0",
            "on_task_repeat_start:Task1:1",
            "on_task_repeat_end:1",
            "on_task_end:Task1",
            # Task 2
            "on_task_start:Task2",
            "on_task_repeat_start:Task2:0",
            "on_task_repeat_end:0",
            "on_task_repeat_start:Task2:1",
            "on_task_repeat_end:1",
            "on_task_end:Task2",
            "on_run_end",
        ]
        assert invocations == expected

    def test_benchmark_component_cleanup_between_repeats(self):
        """Test that component registry is cleared between task repetitions."""
        from conftest import DummyBenchmark
        from maseval import BenchmarkCallback

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])

        # Track registry size after each repetition
        registry_sizes = []

        class RegistryTracker(BenchmarkCallback):
            def on_task_repeat_end(self, benchmark, report):
                # Registry should still have components right after repeat ends
                # but before it's cleared
                pass

            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                # At start of new repeat, registry should be empty (except for callbacks)
                if repeat_idx > 0:
                    # After first repeat, registry should have been cleared
                    registry_sizes.append(len(benchmark._registry._trace_registry))

        benchmark = DummyBenchmark(
            n_task_repeats=2,
            callbacks=[RegistryTracker()],
        )
        benchmark.run(tasks, agent_data={"model": "test"})

        # After first repetition completes and second starts, registry should be cleared
        # Note: This test verifies cleanup happens between repeats

    def test_benchmark_registry_cleared_after_task(self):
        """Test that registry is properly cleared after each task repetition."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark(n_task_repeats=1)

        # Before run, registry should be empty
        assert len(benchmark._registry._trace_registry) == 0
        assert len(benchmark._registry._config_registry) == 0

        benchmark.run(tasks, agent_data={"model": "test"})

        # After run completes, registry should be cleared
        assert len(benchmark._registry._trace_registry) == 0
        assert len(benchmark._registry._config_registry) == 0

    def test_benchmark_reports_structure(self):
        """Test that benchmark reports have the correct structure."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        report = reports[0]

        # Verify required keys
        assert "task_id" in report
        assert "repeat_idx" in report
        assert "traces" in report
        assert "config" in report
        assert "eval" in report

        # Verify traces structure
        traces = report["traces"]
        assert "metadata" in traces
        assert "agents" in traces
        assert "environment" in traces

        # Verify config structure
        config = report["config"]
        assert "metadata" in config
        assert "agents" in config
        assert "environment" in config
        assert "benchmark" in config

        # Verify eval is a list
        assert isinstance(report["eval"], list)

    def test_benchmark_agent_data_per_task(self):
        """Test that different agent_data can be provided per task."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list(
            [
                {"query": "Task1", "environment_data": {}},
                {"query": "Task2", "environment_data": {}},
            ]
        )

        # Provide different agent_data for each task
        agent_data_list = [
            {"model": "model-1", "temp": 0.5},
            {"model": "model-2", "temp": 0.9},
        ]

        benchmark = DummyBenchmark()

        benchmark.run(tasks, agent_data=agent_data_list)

        # Verify each task received its specific agent_data
        assert len(benchmark.setup_agents_calls) == 2
        agent_data_received = [call[0] for call in benchmark.setup_agents_calls]

        assert agent_data_received[0] == {"model": "model-1", "temp": 0.5}
        assert agent_data_received[1] == {"model": "model-2", "temp": 0.9}

    def test_benchmark_invalid_agent_data_length(self):
        """Test that providing wrong number of agent_data items raises error."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list(
            [
                {"query": "Task1", "environment_data": {}},
                {"query": "Task2", "environment_data": {}},
            ]
        )

        # Provide mismatched agent_data
        agent_data_list = [{"model": "model-1"}]  # Only 1 item for 2 tasks

        with pytest.raises(
            ValueError,
            match="must either be a single dict or an iterable matching the number of tasks",
        ):
            benchmark = DummyBenchmark()
            benchmark.run(tasks, agent_data=agent_data_list)

    def test_benchmark_n_task_repeats_validation(self):
        """Test that n_task_repeats must be at least 1."""
        from conftest import DummyBenchmark

        with pytest.raises(ValueError, match="n_task_repeats must be at least 1"):
            DummyBenchmark(n_task_repeats=0)


@pytest.mark.core
class TestFailureSafeExecution:
    """Tests for failure-safe execution with graceful error handling.

    These tests verify that benchmarks can gracefully handle failures in task execution
    and evaluation, with configurable fail-fast behavior.
    """

    def test_task_execution_failure_graceful(self):
        """Test that task execution failures are caught and recorded when fail_on_task_error=False."""
        from maseval import TaskExecutionStatus
        from conftest import DummyBenchmark

        class FailingAgent:
            def run(self, query: str) -> str:
                raise RuntimeError("Agent execution failed!")

        from maseval import AgentAdapter

        class FailingAgentAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class TaskFailureBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = FailingAgent()
                agent_adapter = FailingAgentAdapter(agent, "failing_agent")
                return [agent_adapter], {"failing_agent": agent_adapter}

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = TaskFailureBenchmark(
            fail_on_task_error=False,
        )

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        report = reports[0]
        # RuntimeError from agent framework is classified as UNKNOWN_EXECUTION_ERROR
        # (we can't determine if it's agent's fault or infrastructure failure)
        assert report["status"] == TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value
        assert "error" in report
        assert report["error"]["error_type"] == "RuntimeError"
        assert "Agent execution failed!" in report["error"]["error_message"]
        assert report["eval"] is None

    def test_task_execution_failure_strict(self):
        """Test that task execution failures raise when fail_on_task_error=True."""
        from conftest import DummyBenchmark

        class FailingAgent:
            def run(self, query: str) -> str:
                raise RuntimeError("Agent execution failed!")

        from maseval import AgentAdapter

        class FailingAgentAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class TaskFailureBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = FailingAgent()
                agent_adapter = FailingAgentAdapter(agent, "failing_agent")
                return [agent_adapter], {"failing_agent": agent_adapter}

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = TaskFailureBenchmark(
            fail_on_task_error=True,
        )

        with pytest.raises(RuntimeError, match="Agent execution failed!"):
            benchmark.run(tasks, agent_data={"model": "test"})

    def test_evaluation_failure_graceful(self):
        """Test that evaluation failures are caught and recorded when fail_on_evaluation_error=False."""
        from maseval import TaskExecutionStatus, Evaluator
        from conftest import DummyBenchmark

        class FailingEvaluator(Evaluator):
            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                raise ValueError("Evaluation failed!")

        class EvaluationFailureBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                return [FailingEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = EvaluationFailureBenchmark(
            fail_on_evaluation_error=False,
        )

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        report = reports[0]
        assert report["status"] == TaskExecutionStatus.EVALUATION_FAILED.value
        assert "error" in report
        assert report["error"]["error_type"] == "ValueError"
        assert "Evaluation failed!" in report["error"]["error_message"]
        assert report["eval"] is None

    def test_evaluation_failure_strict(self):
        """Test that evaluation failures raise when fail_on_evaluation_error=True."""
        from maseval import Evaluator
        from conftest import DummyBenchmark

        class FailingEvaluator(Evaluator):
            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                raise ValueError("Evaluation failed!")

        class EvaluationFailureBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                return [FailingEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = EvaluationFailureBenchmark(
            fail_on_evaluation_error=True,
        )

        with pytest.raises(ValueError, match="Evaluation failed!"):
            benchmark.run(tasks, agent_data={"model": "test"})

    def test_setup_failure_graceful(self):
        """Test that setup failures are caught and recorded when fail_on_setup_error=False."""
        from maseval import TaskExecutionStatus
        from conftest import DummyBenchmark

        class SetupFailureBenchmark(DummyBenchmark):
            def setup_environment(self, agent_data, task, seed_generator):
                raise RuntimeError("Environment setup failed!")

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = SetupFailureBenchmark(
            fail_on_setup_error=False,
        )

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        report = reports[0]
        assert report["status"] == TaskExecutionStatus.SETUP_FAILED.value
        assert "error" in report
        assert report["error"]["error_type"] == "RuntimeError"
        assert "Environment setup failed!" in report["error"]["error_message"]
        assert report["eval"] is None

    def test_setup_failure_strict(self):
        """Test that setup failures raise when fail_on_setup_error=True."""
        from conftest import DummyBenchmark

        class SetupFailureBenchmark(DummyBenchmark):
            def setup_environment(self, agent_data, task, seed_generator):
                raise RuntimeError("Environment setup failed!")

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = SetupFailureBenchmark(
            fail_on_setup_error=True,
        )

        with pytest.raises(RuntimeError, match="Environment setup failed!"):
            benchmark.run(tasks, agent_data={"model": "test"})

    def test_get_failed_tasks(self):
        """Test get_failed_tasks() method."""
        from maseval import TaskExecutionStatus
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        class FailingAgent:
            def run(self, query: str) -> str:
                raise RuntimeError("Failed!")

        from maseval import AgentAdapter

        class FailingAgentAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class MixedBenchmark(DummyBenchmark):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.task_counter = 0

            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                if self.task_counter == 1:  # Fail second task
                    agent = FailingAgent()
                    agent_adapter = FailingAgentAdapter(agent, "failing")
                else:
                    agent = DummyAgent()
                    agent_adapter = DummyAgentAdapter(agent, "test_agent")

                self.task_counter += 1
                return [agent_adapter], {agent_adapter.name: agent_adapter}

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
                {"query": "Task 3", "environment_data": {}},
            ]
        )
        benchmark = MixedBenchmark()
        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # Test using internal state
        failed = benchmark.get_failed_tasks()
        assert len(failed) == 1

        # Test using provided reports (should give same result)
        failed_external = benchmark.get_failed_tasks(reports=reports)
        assert len(failed_external) == 1
        assert [t.id for t in failed] == [t.id for t in failed_external]

        # RuntimeError from agent framework is classified as UNKNOWN_EXECUTION_ERROR
        # (we can't determine if it's agent's fault or infrastructure failure)
        exec_failed = benchmark.get_failed_tasks(TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR)
        assert len(exec_failed) == 1

        # External reports version
        exec_failed_external = benchmark.get_failed_tasks(TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR, reports=reports)
        assert len(exec_failed_external) == 1

        # No evaluation failures in this test
        eval_failed = benchmark.get_failed_tasks(TaskExecutionStatus.EVALUATION_FAILED)
        assert len(eval_failed) == 0

        # Modifying returned reports shouldn't affect internal state
        reports_copy = reports.copy()
        reports_copy.append(
            {
                "task_id": "fake-task",
                "status": TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value,
                "error": "Fake error",
            }
        )

        # Internal state query should be unaffected
        failed_after_modify = benchmark.get_failed_tasks()
        assert len(failed_after_modify) == 1

        # But querying the modified list should reflect changes (though task won't be found)
        failed_from_modified = benchmark.get_failed_tasks(reports=reports_copy)
        # Still 1 because fake-task doesn't exist in benchmark.tasks
        assert len(failed_from_modified) == 1

    def test_get_failed_tasks_before_run_raises(self):
        """Test that get_failed_tasks() raises if called before run()."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark()

        with pytest.raises(RuntimeError, match="must be called after run"):
            benchmark.get_failed_tasks()

    def test_successful_task_has_success_status(self):
        """Test that successful tasks have SUCCESS status."""
        from maseval import TaskExecutionStatus
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert len(reports) == 1
        assert reports[0]["status"] == TaskExecutionStatus.SUCCESS.value
        assert "error" not in reports[0]
        assert reports[0]["eval"] is not None

    def test_default_failure_flags(self):
        """Test that failure flags default to False (graceful handling)."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark()

        assert benchmark.fail_on_setup_error is False
        assert benchmark.fail_on_task_error is False
        assert benchmark.fail_on_evaluation_error is False

    def test_multiple_run_calls_no_side_effects(self):
        """Test that multiple run() calls work correctly without side effects.

        This verifies that:
        - self.reports is cleared at start of each run()
        - self.tasks is overwritten with new tasks each run()
        - No accumulation or cross-contamination between runs
        """
        from conftest import DummyBenchmark

        # Create benchmark
        benchmark = DummyBenchmark()

        # First run with 3 tasks
        tasks1 = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
                {"query": "Task 3", "environment_data": {}},
            ]
        )

        reports1 = benchmark.run(tasks=tasks1, agent_data={"model": "test"})
        assert len(reports1) == 3
        assert len(benchmark.reports) == 3

        # Second run with 2 different tasks
        tasks2 = TaskQueue.from_list(
            [
                {"query": "Task A", "environment_data": {}},
                {"query": "Task B", "environment_data": {}},
            ]
        )

        reports2 = benchmark.run(tasks=tasks2, agent_data={"model": "test"})
        assert len(reports2) == 2
        # Verify reports were cleared from first run
        assert len(benchmark.reports) == 2

        # Verify no cross-contamination
        task_ids_1 = set(r["task_id"] for r in reports1)
        task_ids_2 = set(r["task_id"] for r in reports2)
        assert task_ids_1.isdisjoint(task_ids_2), "Task IDs from different runs should not overlap"

        # Third run - retry pattern (simulating failed tasks)
        # Use one task from tasks1
        retry_tasks = TaskQueue([list(tasks1)[0]])
        reports3 = benchmark.run(tasks=retry_tasks, agent_data={"model": "test"})
        assert len(reports3) == 1
        assert len(benchmark.reports) == 1

    def test_retry_failed_tasks_pattern(self):
        """Test the intended use case: benchmark.run(benchmark.get_failed_tasks(, agent_data={"model": "test"})).

        This verifies that failed tasks can be retried by passing them back to run().
        This includes returning tasks that failed using the correct format that run() expects.
        """
        from maseval import TaskExecutionStatus
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        class FailingAgent:
            def run(self, query: str) -> str:
                raise RuntimeError("Failed!")

        from maseval import AgentAdapter

        class FailingAgentAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class ConditionalFailureBenchmark(DummyBenchmark):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.task_counter = 0
                self.fail_on_first_run = True

            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                # Fail second task on first run only
                if self.task_counter == 1 and self.fail_on_first_run:
                    agent = FailingAgent()
                    agent_adapter = FailingAgentAdapter(agent, "failing")
                else:
                    agent = DummyAgent()
                    agent_adapter = DummyAgentAdapter(agent, "test_agent")

                self.task_counter += 1
                return [agent_adapter], {agent_adapter.name: agent_adapter}

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
                {"query": "Task 3", "environment_data": {}},
            ]
        )

        benchmark = ConditionalFailureBenchmark()

        # First run - one task will fail
        reports = benchmark.run(tasks=tasks, agent_data={"model": "test"})
        assert len(reports) == 3

        # Get failed tasks - should have 1 failure
        failed = benchmark.get_failed_tasks()
        assert len(failed) == 1
        assert list(failed)[0].query == "Task 2"

        # Retry the failed tasks (simulate fixing the issue)
        benchmark.fail_on_first_run = False
        benchmark.task_counter = 0  # Reset counter
        retry_reports = benchmark.run(tasks=failed, agent_data={"model": "test"})

        # Should have 1 report for the retried task
        assert len(retry_reports) == 1
        assert retry_reports[0]["status"] == TaskExecutionStatus.SUCCESS.value

        # Verify the retry pattern works end-to-end
        failed_after_retry = benchmark.get_failed_tasks()
        assert len(failed_after_retry) == 0
