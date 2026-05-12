"""Tests for report-schema consistency across the whole task lifecycle.

Every report a benchmark produces — whether the task succeeded or failed in
setup, execution, or evaluation, and whether it ran sequentially or in
parallel — must carry the same set of top-level keys so downstream consumers
can rely on a stable schema. When a task did not succeed, ``report["error"]``
must always be populated. In addition, when a ``fail_on_*`` flag is set a
parallel run must abort just like a sequential one (it must not silently
swallow the failure and keep going).
"""

import time

import pytest

from maseval import (
    AgentAdapter,
    Evaluator,
    Task,
    TaskExecutionStatus,
    TaskQueue,
)
from maseval.core.task import TaskProtocol
from conftest import DummyBenchmark


# Canonical top-level keys every report must carry.
REPORT_KEYS = {"task_id", "repeat_idx", "status", "error", "traces", "config", "usage", "eval", "task"}
ERROR_KEYS = {"error_type", "error_message", "traceback"}


def assert_canonical_schema(report):
    """Assert a single report carries the canonical schema."""
    missing = REPORT_KEYS - set(report.keys())
    assert not missing, f"report is missing keys: {missing}"

    # The ``task`` block is always present and structured.
    assert isinstance(report["task"], dict)
    assert {"query", "metadata", "protocol"} <= set(report["task"].keys())

    # ``traces`` / ``config`` are always dicts (never absent / None).
    assert isinstance(report["traces"], dict)
    assert isinstance(report["config"], dict)

    # ``error`` is None exactly when the task succeeded; otherwise it is populated.
    if report["status"] == TaskExecutionStatus.SUCCESS.value:
        assert report["error"] is None
    else:
        assert report["error"] is not None, f"status={report['status']!r} but error is None"
        assert ERROR_KEYS <= set(report["error"].keys())


# --------------------------------------------------------------------------
# Failure-injection benchmarks
# --------------------------------------------------------------------------


class _FailingAgent:
    def run(self, query: str) -> str:
        raise RuntimeError("agent boom")


class _FailingAgentAdapter(AgentAdapter):
    def _run_agent(self, query: str) -> str:
        return self.agent.run(query)


class SetupFailureBenchmark(DummyBenchmark):
    def setup_environment(self, agent_data, task, seed_generator):
        raise RuntimeError("setup boom")


class SelectiveSetupFailureBenchmark(DummyBenchmark):
    """Fails ``setup_environment`` only for tasks whose query starts with ``fail``."""

    def setup_environment(self, agent_data, task, seed_generator):
        if task.query.startswith("fail"):
            raise RuntimeError("setup boom")
        return super().setup_environment(agent_data, task, seed_generator)


class SetupTimeoutBenchmark(DummyBenchmark):
    def setup_environment(self, agent_data, task, seed_generator):
        time.sleep(0.05)
        return super().setup_environment(agent_data, task, seed_generator)


class ExecutionFailureBenchmark(DummyBenchmark):
    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        adapter = _FailingAgentAdapter(_FailingAgent(), "failing_agent")
        return [adapter], {"failing_agent": adapter}


class _FailingEvaluator(Evaluator):
    def filter_traces(self, traces):
        return traces

    def __call__(self, traces, final_answer=None):
        raise ValueError("eval boom")


class EvaluationFailureBenchmark(DummyBenchmark):
    def setup_evaluators(self, environment, task, agents, user, seed_generator):
        return [_FailingEvaluator(task, environment, user)]


class UnexpectedWorkerFailureBenchmark(DummyBenchmark):
    """Simulates a failure escaping ``_execute_task_repetition``'s own handling.

    This is the only case ``_run_parallel`` should turn into a degraded report
    on its own (and only when no ``fail_on_*`` flag is set).
    """

    def _execute_task_repetition(self, task, agent_data, repeat_idx):
        if task.query == "boom":
            raise RuntimeError("worker exploded")
        return super()._execute_task_repetition(task, agent_data, repeat_idx)


# name -> (benchmark class, expected status, task timeout_seconds or None)
SCENARIOS = {
    "success": (DummyBenchmark, TaskExecutionStatus.SUCCESS, None),
    "setup_failure": (SetupFailureBenchmark, TaskExecutionStatus.SETUP_FAILED, None),
    "setup_timeout": (SetupTimeoutBenchmark, TaskExecutionStatus.TASK_TIMEOUT, 0.001),
    "execution_failure": (ExecutionFailureBenchmark, TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR, None),
    "evaluation_failure": (EvaluationFailureBenchmark, TaskExecutionStatus.EVALUATION_FAILED, None),
}


def _one_task(timeout_seconds=None):
    if timeout_seconds is None:
        return TaskQueue.from_list([{"query": "q", "environment_data": {}}])
    return TaskQueue([Task(query="q", environment_data={}, protocol=TaskProtocol(timeout_seconds=timeout_seconds))])


def _many_tasks(n=4):
    return TaskQueue.from_list([{"query": f"q{i}", "environment_data": {}} for i in range(n)])


# --------------------------------------------------------------------------
# Schema-invariance tests
# --------------------------------------------------------------------------


@pytest.mark.core
@pytest.mark.parametrize("num_workers", [1, 2], ids=["sequential", "parallel"])
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_report_schema_consistent_across_lifecycle_outcomes(scenario, num_workers):
    """Reports carry the canonical schema regardless of where (or whether) a task failed."""
    benchmark_cls, expected_status, timeout = SCENARIOS[scenario]
    benchmark = benchmark_cls(num_workers=num_workers)

    reports = benchmark.run(_one_task(timeout), agent_data={"model": "test"})

    assert len(reports) == 1
    report = reports[0]
    assert report["status"] == expected_status.value
    assert_canonical_schema(report)


@pytest.mark.core
@pytest.mark.parametrize("num_workers", [1, 2], ids=["sequential", "parallel"])
def test_mixed_outcomes_in_one_run_all_share_schema(num_workers):
    """A run mixing successes and a (graceful) setup failure still yields uniform reports."""
    benchmark = SelectiveSetupFailureBenchmark(num_workers=num_workers)
    tasks = TaskQueue.from_list(
        [
            {"query": "ok1", "environment_data": {}},
            {"query": "fail-me", "environment_data": {}},
            {"query": "ok2", "environment_data": {}},
        ]
    )

    reports = benchmark.run(tasks, agent_data={"model": "test"})

    assert len(reports) == 3
    for report in reports:
        assert_canonical_schema(report)

    statuses = sorted(r["status"] for r in reports)
    assert statuses == sorted(
        [
            TaskExecutionStatus.SUCCESS.value,
            TaskExecutionStatus.SETUP_FAILED.value,
            TaskExecutionStatus.SUCCESS.value,
        ]
    )


@pytest.mark.core
def test_parallel_fallback_unexpected_worker_failure_produces_full_schema_report():
    """When a worker raises unexpectedly (no fail_on_* set), the parallel runner records a full-schema report and carries on."""
    benchmark = UnexpectedWorkerFailureBenchmark(num_workers=2)
    tasks = TaskQueue.from_list(
        [
            {"query": "ok1", "environment_data": {}},
            {"query": "boom", "environment_data": {}},
            {"query": "ok2", "environment_data": {}},
        ]
    )

    reports = benchmark.run(tasks, agent_data={"model": "test"})

    assert len(reports) == 3
    for report in reports:
        assert_canonical_schema(report)

    boom_reports = [r for r in reports if r["status"] == TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value]
    assert len(boom_reports) == 1
    assert "worker exploded" in boom_reports[0]["error"]["error_message"]
    assert boom_reports[0]["error"]["error_type"] == "RuntimeError"


@pytest.mark.core
@pytest.mark.parametrize("num_workers", [1, 2], ids=["sequential", "parallel"])
def test_error_always_populated_when_status_not_success(num_workers):
    """Whenever a report's status is not SUCCESS, ``report["error"]`` is populated."""
    for benchmark_cls, expected_status, timeout in SCENARIOS.values():
        benchmark = benchmark_cls(num_workers=num_workers)
        reports = benchmark.run(_one_task(timeout), agent_data={"model": "test"})
        report = reports[0]
        if report["status"] == TaskExecutionStatus.SUCCESS.value:
            assert report["error"] is None
        else:
            assert report["error"] is not None
            assert report["error"]["error_message"]


# --------------------------------------------------------------------------
# Parallel fail-fast tests (must match sequential semantics)
# --------------------------------------------------------------------------


@pytest.mark.core
def test_parallel_run_aborts_on_fail_on_task_error():
    """fail_on_task_error=True aborts a parallel run instead of swallowing the failure."""
    benchmark = ExecutionFailureBenchmark(num_workers=2, fail_on_task_error=True)
    with pytest.raises(RuntimeError, match="agent boom"):
        benchmark.run(_many_tasks(), agent_data={"model": "test"})


@pytest.mark.core
def test_parallel_run_aborts_on_fail_on_evaluation_error():
    """fail_on_evaluation_error=True aborts a parallel run instead of swallowing the failure."""
    benchmark = EvaluationFailureBenchmark(num_workers=2, fail_on_evaluation_error=True)
    with pytest.raises(ValueError, match="eval boom"):
        benchmark.run(_many_tasks(), agent_data={"model": "test"})


@pytest.mark.core
def test_parallel_run_aborts_on_fail_on_setup_error():
    """fail_on_setup_error=True aborts a parallel run instead of swallowing the failure."""
    benchmark = SetupFailureBenchmark(num_workers=2, fail_on_setup_error=True)
    with pytest.raises(RuntimeError, match="setup boom"):
        benchmark.run(_many_tasks(), agent_data={"model": "test"})


@pytest.mark.core
def test_parallel_unexpected_worker_failure_propagates_when_fail_fast():
    """An unexpected worker failure also aborts a parallel run when a fail_on_* flag is set."""
    benchmark = UnexpectedWorkerFailureBenchmark(num_workers=2, fail_on_task_error=True)
    tasks = TaskQueue.from_list(
        [
            {"query": "ok1", "environment_data": {}},
            {"query": "boom", "environment_data": {}},
        ]
    )
    with pytest.raises(RuntimeError, match="worker exploded"):
        benchmark.run(tasks, agent_data={"model": "test"})


@pytest.mark.core
def test_parallel_graceful_failure_keeps_running_with_full_schema():
    """With fail_on_* unset, a failed task yields a full-schema report and the run continues."""
    benchmark = ExecutionFailureBenchmark(num_workers=2, fail_on_task_error=False)
    reports = benchmark.run(_many_tasks(), agent_data={"model": "test"})

    assert len(reports) == 4
    for report in reports:
        assert_canonical_schema(report)
        assert report["status"] == TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value
