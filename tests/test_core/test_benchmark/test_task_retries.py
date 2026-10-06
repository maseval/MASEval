"""Tests for task retries configured via ``TaskProtocol``.

These tests verify that ``max_retries`` retries failures outside the agent's
control, that ``timeout_action`` retries timeouts, and that every attempt is
recorded in the report's ``attempts`` field.
"""

import time

import pytest

from maseval import Task, TaskQueue
from maseval.core.exceptions import AgentError, EnvironmentError, UserError
from maseval.core.task import TaskProtocol, TimeoutAction
from conftest import DummyBenchmark, DummyModelAdapter

MODEL_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FlakyBenchmark(DummyBenchmark):
    """Benchmark whose attempts fail as scripted.

    Each attempt pops one delay (slept during setup) and one error (raised by
    ``run_agents``). Each attempt also records its agent seed and calls a model once.
    """

    def __init__(self, errors=(), delays=(), **kwargs):
        super().__init__(**kwargs)
        self.errors = list(errors)
        self.delays = list(delays)
        self.agent_seeds = []

    def setup_environment(self, agent_data, task, seed_generator):
        if self.delays:
            time.sleep(self.delays.pop(0))
        return super().setup_environment(agent_data, task, seed_generator)

    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        self.agent_seeds.append(seed_generator.derive_seed("agent"))
        return super().setup_agents(agent_data, environment, task, user, seed_generator)

    def run_agents(self, agents, task, environment, query):
        model = DummyModelAdapter(model_id="agent_model", usage=MODEL_USAGE)
        self.register("models", "agent_model", model)
        model.chat([{"role": "user", "content": query}])
        if self.errors:
            raise self.errors.pop(0)
        return super().run_agents(agents, task, environment, query)


class FlakySetupBenchmark(DummyBenchmark):
    """Fails ``setup_environment`` for the first ``n_failures`` attempts."""

    def __init__(self, n_failures, **kwargs):
        super().__init__(**kwargs)
        self.n_failures = n_failures

    def setup_environment(self, agent_data, task, seed_generator):
        if self.n_failures > 0:
            self.n_failures -= 1
            raise RuntimeError("setup boom")
        return super().setup_environment(agent_data, task, seed_generator)


class _FailingEvaluator:
    def filter_traces(self, traces):
        return traces

    def __call__(self, traces, final_answer=None):
        raise ValueError("eval boom")


class EvaluationFailureBenchmark(DummyBenchmark):
    def setup_evaluators(self, environment, task, agents, user, seed_generator):
        return [_FailingEvaluator()]


def _task(**protocol_kwargs):
    return TaskQueue([Task(query="q", environment_data={}, protocol=TaskProtocol(**protocol_kwargs))])


def _statuses(report):
    return [attempt["status"] for attempt in report["attempts"]]


# --------------------------------------------------------------------------
# max_retries
# --------------------------------------------------------------------------


@pytest.mark.core
class TestFailureRetries:
    """Tests for retries of failures outside the agent's control."""

    @pytest.mark.parametrize("num_workers", [1, 2], ids=["sequential", "parallel"])
    def test_retries_until_success(self, num_workers):
        """A failed attempt is retried and the report describes the successful one."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")] * 2, num_workers=num_workers)

        reports = benchmark.run(_task(max_retries=2), agent_data={"model": "test"})

        assert len(reports) == 1
        assert reports[0]["status"] == "success"
        assert reports[0]["error"] is None
        assert _statuses(reports[0]) == ["environment_error", "environment_error", "success"]

    @pytest.mark.parametrize(
        "error, status",
        [
            (EnvironmentError("boom"), "environment_error"),
            (UserError("boom"), "user_error"),
            (RuntimeError("boom"), "unknown_execution_error"),
        ],
    )
    def test_infrastructure_failures_are_retried(self, error, status):
        """Environment, user, and unclassified execution errors are retried."""
        benchmark = FlakyBenchmark(errors=[error])

        reports = benchmark.run(_task(max_retries=1), agent_data={"model": "test"})

        assert _statuses(reports[0]) == [status, "success"]

    def test_setup_failure_is_retried(self):
        """Setup failures are retried."""
        benchmark = FlakySetupBenchmark(n_failures=1)

        reports = benchmark.run(_task(max_retries=1), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["setup_failed", "success"]

    def test_agent_error_is_not_retried(self):
        """Agent errors count against the agent and are never retried."""
        benchmark = FlakyBenchmark(errors=[AgentError("bad tool call")])

        reports = benchmark.run(_task(max_retries=3), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["agent_error"]
        assert len(benchmark.setup_environment_calls) == 1

    def test_evaluation_failure_is_not_retried(self):
        """Evaluation failures are never retried."""
        benchmark = EvaluationFailureBenchmark()

        reports = benchmark.run(_task(max_retries=3), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["evaluation_failed"]
        assert len(benchmark.setup_environment_calls) == 1

    def test_no_retries_by_default(self):
        """With the default protocol a failed task runs once."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")])

        reports = benchmark.run(_task(), agent_data={"model": "test"})

        assert reports[0]["status"] == "environment_error"
        assert _statuses(reports[0]) == ["environment_error"]

    def test_gives_up_after_max_retries(self):
        """Once retries are used up, the report describes the last failed attempt."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")] * 3)

        reports = benchmark.run(_task(max_retries=1), agent_data={"model": "test"})

        assert reports[0]["status"] == "environment_error"
        assert "env down" in reports[0]["error"]["error_message"]
        assert _statuses(reports[0]) == ["environment_error", "environment_error"]

    def test_fail_on_task_error_raises_without_retry(self):
        """Fail-fast flags raise on the first failure."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")], fail_on_task_error=True)

        with pytest.raises(EnvironmentError, match="env down"):
            benchmark.run(_task(max_retries=3), agent_data={"model": "test"})

        assert len(benchmark.setup_environment_calls) == 1

    def test_attempts_reuse_seeds(self):
        """Every attempt of a repetition derives the same seeds."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")], seed=42)

        benchmark.run(_task(max_retries=1), agent_data={"model": "test"})

        assert len(benchmark.agent_seeds) == 2
        assert benchmark.agent_seeds[0] is not None
        assert benchmark.agent_seeds[0] == benchmark.agent_seeds[1]

    def test_attempts_record_usage(self):
        """Each attempt carries its own usage, so failed attempts are not lost from cost totals."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")])

        reports = benchmark.run(_task(max_retries=1), agent_data={"model": "test"})

        for attempt in reports[0]["attempts"]:
            assert attempt["usage"]["models"]["agent_model"]["input_tokens"] == 10
        assert benchmark.usage.to_dict()["input_tokens"] == 20


# --------------------------------------------------------------------------
# timeout_action
# --------------------------------------------------------------------------


@pytest.mark.core
class TestTimeoutRetries:
    """Tests for retries of timed-out tasks."""

    def test_skip_does_not_retry(self):
        """SKIP records the timeout without retrying."""
        benchmark = FlakyBenchmark(delays=[0.05])

        reports = benchmark.run(_task(timeout_seconds=0.01, timeout_action=TimeoutAction.SKIP), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["task_timeout"]

    def test_retry_retries_once_with_same_timeout(self):
        """RETRY reruns a timed-out task once with the same timeout."""
        benchmark = FlakyBenchmark(delays=[0.05, 0.05])

        reports = benchmark.run(_task(timeout_seconds=0.01, timeout_action=TimeoutAction.RETRY), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["task_timeout", "task_timeout"]
        assert [a["timeout_seconds"] for a in reports[0]["attempts"]] == [0.01, 0.01]

    def test_extend_retries_once_with_double_timeout(self):
        """EXTEND reruns a timed-out task once with double the timeout."""
        benchmark = FlakyBenchmark(delays=[0.05, 0.05])

        reports = benchmark.run(_task(timeout_seconds=0.01, timeout_action=TimeoutAction.EXTEND), agent_data={"model": "test"})

        assert _statuses(reports[0]) == ["task_timeout", "task_timeout"]
        assert [a["timeout_seconds"] for a in reports[0]["attempts"]] == [0.01, 0.02]

    def test_timeout_retry_does_not_use_max_retries(self):
        """Timeout retries and failure retries have separate budgets."""
        benchmark = FlakyBenchmark(errors=[EnvironmentError("env down")], delays=[0.2])

        reports = benchmark.run(
            _task(timeout_seconds=0.05, timeout_action=TimeoutAction.RETRY, max_retries=1),
            agent_data={"model": "test"},
        )

        assert _statuses(reports[0]) == ["task_timeout", "environment_error", "success"]
