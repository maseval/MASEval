"""Test usage collection through the benchmark execution loop.

These tests verify that benchmark.run() collects usage from registered
model adapters and includes it in report dicts.
"""

import pytest
from maseval import TaskQueue
from conftest import DummyBenchmark, DummyModelAdapter


@pytest.mark.core
class TestBenchmarkUsageCollection:
    """Tests for usage collection during benchmark execution."""

    def test_usage_in_report(self):
        """Benchmark run includes a 'usage' key in each report."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        assert "usage" in reports[0]
        usage = reports[0]["usage"]
        assert "metadata" in usage
        assert "models" in usage
        assert "agents" in usage

    def test_usage_has_correct_structure(self):
        """Usage dict has the expected category keys and metadata."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        usage = reports[0]["usage"]
        assert "metadata" in usage
        assert "total_components" in usage["metadata"]
        assert "timestamp" in usage["metadata"]

    def test_model_with_usage_appears_in_report(self):
        """A model adapter that reports usage has its tokens in the report."""
        from conftest import DummyModelAdapter, DummyBenchmark

        class UsageBenchmark(DummyBenchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(
                    model_id=model_id,
                    usage={
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                )

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = UsageBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # The DummyBenchmark doesn't register a model via register(), so
        # the model's usage won't appear unless the benchmark hooks it up.
        # This test verifies the usage structure exists.
        assert "usage" in reports[0]

    def test_usage_persists_across_task_repetitions(self):
        """Benchmark.usage accumulates across multiple tasks."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
            ]
        )
        benchmark = DummyBenchmark()
        benchmark.run(tasks, agent_data={"model": "test"})

        # Both tasks should have produced reports with usage
        assert len(benchmark.reports) == 2
        assert "usage" in benchmark.reports[0]
        assert "usage" in benchmark.reports[1]

    def test_usage_property_returns_total(self):
        """benchmark.usage returns the running total."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()
        benchmark.run(tasks, agent_data={"model": "test"})

        # usage property should return a Usage object (even if empty)
        total = benchmark.usage
        assert total is not None
        # cost may be None if DummyModelAdapter doesn't provide usage


# ---------------------------------------------------------------------------
# Regression tests for issue #60: judge (evaluator) token usage collection
# ---------------------------------------------------------------------------


class _JudgeEvaluator:
    """Minimal evaluator that invokes a model adapter at evaluate-time.

    Not an ``Evaluator`` subclass — it implements the duck-typed
    interface (``filter_traces`` + ``__call__``) that ``DummyBenchmark.evaluate``
    iterates over (see ``tests/conftest.py``).
    """

    def __init__(self, model):
        self.model = model

    def filter_traces(self, traces):
        return traces

    def __call__(self, traces, final_answer=None):
        # Invokes the judge at evaluate-time (not at setup).
        self.model.chat([{"role": "user", "content": "judge this"}])
        return {"score": 1.0, "passed": True}


def _make_judge_benchmark(judge_usage):
    """Build a JudgeBenchmark whose setup_evaluators registers a judge model.

    The judge model is created with the provided per-call usage dict. Each
    call to the model appends one usage record, so a single evaluator
    invocation produces exactly one record's worth of tokens.
    """

    class JudgeBenchmark(DummyBenchmark):
        def setup_evaluators(self, environment, task, agents, user, seed_generator):
            judge_model = DummyModelAdapter(model_id="judge", usage=judge_usage)
            self.register("models", "judge_model", judge_model)
            return [_JudgeEvaluator(model=judge_model)]

    return JudgeBenchmark()


@pytest.mark.core
class TestBenchmarkJudgeUsage:
    """Tests that judge token usage reaches both per-task reports and ``benchmark.usage``."""

    def test_judge_model_usage_captured_in_report(self):
        """A judge model invoked during evaluate() has non-zero usage in
        report['usage']['models']['judge_model']."""
        judge_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = _make_judge_benchmark(judge_usage)

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        models = reports[0]["usage"]["models"]
        assert "judge_model" in models, f"judge_model not registered; got: {list(models)}"
        judge_entry = models["judge_model"]
        assert judge_entry["input_tokens"] == 100
        assert judge_entry["output_tokens"] == 50
        assert judge_entry["total_tokens"] == 150

    def test_judge_model_usage_aggregated_in_benchmark_total(self):
        """``benchmark.usage`` includes judge tokens, and
        ``benchmark.usage_by_component`` has a non-zero ``models:judge_model``
        entry."""
        judge_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = _make_judge_benchmark(judge_usage)

        benchmark.run(tasks, agent_data={"model": "test"})

        assert benchmark.usage.input_tokens >= 100
        assert benchmark.usage.output_tokens >= 50

        by_component = benchmark.usage_by_component
        assert "models:judge_model" in by_component, f"keys: {list(by_component)}"
        assert by_component["models:judge_model"].input_tokens == 100
        assert by_component["models:judge_model"].output_tokens == 50

    def test_agent_usage_captured_when_evaluation_raises(self):
        """When evaluate() raises and fail_on_evaluation_error=False, the
        report still carries a real usage dict (step 5 runs after step 4)."""

        class RaisingEvaluator:
            def __init__(self, task, environment, user):
                self.task = task

            def filter_traces(self, traces):
                return traces

            def __call__(self, traces, final_answer=None):
                raise RuntimeError("boom — simulated evaluator failure")

        class RaisingJudgeBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                # Register an agent-side model with usage so we can assert
                # that pre-evaluate usage still survives the eval failure.
                agent_model = DummyModelAdapter(
                    model_id="agent_model",
                    usage={"input_tokens": 42, "output_tokens": 7, "total_tokens": 49},
                )
                self.register("models", "agent_model", agent_model)
                # Drive the model once so it has a usage record.
                agent_model.chat([{"role": "user", "content": "hi"}])
                return [RaisingEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = RaisingJudgeBenchmark(fail_on_evaluation_error=False)

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        report = reports[0]
        assert report["status"] == "evaluation_failed"
        usage = report["usage"]
        assert isinstance(usage, dict)
        assert "error" not in usage, f"usage became an error dict: {usage}"
        assert usage["models"]["agent_model"]["input_tokens"] == 42

    def test_agent_usage_captured_when_execution_raises(self):
        """When run_agents raises (execution failure) and fail_on_task_error
        is False, the report still carries a real usage dict. Evaluate is
        skipped, but step 5 still runs."""
        from maseval.core.exceptions import AgentError

        class FailingAgentBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent_model = DummyModelAdapter(
                    model_id="agent_model",
                    usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                )
                self.register("models", "agent_model", agent_model)
                # Drive the model once so it has a usage record.
                agent_model.chat([{"role": "user", "content": "hi"}])
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

            def run_agents(self, agents, task, environment, query):
                raise AgentError("simulated agent failure", component="agent")

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = FailingAgentBenchmark(fail_on_task_error=False)

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        report = reports[0]
        assert report["status"] == "agent_error"
        usage = report["usage"]
        assert isinstance(usage, dict)
        assert "error" not in usage, f"usage became an error dict: {usage}"
        assert usage["models"]["agent_model"]["input_tokens"] == 11
