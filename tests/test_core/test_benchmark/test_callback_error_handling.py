"""Tests for callback error handling in benchmark execution.

These tests verify that callback exceptions are properly isolated and logged,
preventing one failing callback from disrupting the entire benchmark run.
"""

import pytest
import logging
from typing import List

from maseval import (
    BenchmarkCallback,
    Task,
    TaskQueue,
)
from conftest import DummyBenchmark


# ==================== Test Fixtures ====================


class FailingCallback(BenchmarkCallback):
    """Callback that raises an exception on specific methods."""

    def __init__(self, fail_on: str = "on_task_start"):
        self.fail_on = fail_on
        self.call_count = 0

    def on_task_start(self, benchmark, task):
        if self.fail_on == "on_task_start":
            raise RuntimeError("Intentional failure in on_task_start")

    def on_task_repeat_start(self, benchmark, task, repeat_idx):
        if self.fail_on == "on_task_repeat_start":
            raise ValueError("Intentional failure in on_task_repeat_start")

    def on_task_repeat_end(self, benchmark, report):
        self.call_count += 1
        if self.fail_on == "on_task_repeat_end":
            raise TypeError("Intentional failure in on_task_repeat_end")

    def on_task_end(self, benchmark, task, result):
        if self.fail_on == "on_task_end":
            raise KeyError("Intentional failure in on_task_end")


class TrackingCallback(BenchmarkCallback):
    """Callback that tracks which methods were called."""

    def __init__(self):
        self.calls: List[str] = []

    def on_run_start(self, benchmark):
        self.calls.append("run_start")

    def on_task_start(self, benchmark, task):
        self.calls.append(f"task_start:{task.query}")

    def on_task_repeat_start(self, benchmark, task, repeat_idx):
        self.calls.append(f"repeat_start:{task.query}:{repeat_idx}")

    def on_task_repeat_end(self, benchmark, report):
        self.calls.append(f"repeat_end:{report['task_id'][:8]}")

    def on_task_end(self, benchmark, task, result):
        self.calls.append(f"task_end:{task.query}")

    def on_run_end(self, benchmark, results):
        self.calls.append("run_end")


@pytest.fixture
def simple_tasks():
    """Create simple tasks for testing."""
    return TaskQueue.from_list(
        [
            {"query": "Task 1", "environment_data": {}},
            {"query": "Task 2", "environment_data": {}},
        ]
    )


# ==================== Error Suppression Tests ====================


@pytest.mark.core
class TestCallbackErrorSuppression:
    """Tests for callback error suppression in sequential execution."""

    def test_failing_callback_does_not_stop_execution(self, simple_tasks, caplog):
        """A failing callback should not prevent benchmark from completing."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_start")
        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        # Should complete despite callback failure
        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        assert len(reports) == 2
        assert all(r["status"] == "success" for r in reports)

        # Error should be logged
        assert "Callback" in caplog.text
        assert "on_task_start" in caplog.text
        assert "Intentional failure" in caplog.text

    def test_multiple_callbacks_one_fails_others_continue(self, simple_tasks, caplog):
        """Other callbacks should continue even if one fails."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_repeat_end")
        tracking_cb = TrackingCallback()

        benchmark = DummyBenchmark(
            callbacks=[failing_cb, tracking_cb],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        # Execution completes
        assert len(reports) == 2

        # Tracking callback still received all events
        assert "run_start" in tracking_cb.calls
        assert "task_start:Task 1" in tracking_cb.calls
        assert "task_start:Task 2" in tracking_cb.calls
        assert "run_end" in tracking_cb.calls

        # Error logged
        assert "on_task_repeat_end" in caplog.text

    def test_callback_fails_on_every_task(self, simple_tasks, caplog):
        """Execution continues even if callback fails on every task."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_repeat_end")
        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        # All tasks complete
        assert len(reports) == 2
        assert all(r["status"] == "success" for r in reports)

        # Callback was attempted for each task (even though it failed)
        assert failing_cb.call_count == 2

    def test_callback_error_in_on_run_start(self, simple_tasks, caplog):
        """Benchmark continues if callback fails in on_run_start."""
        caplog.set_level(logging.ERROR)

        class RunStartFailer(BenchmarkCallback):
            def on_run_start(self, benchmark):
                raise RuntimeError("Failed at run start")

        benchmark = DummyBenchmark(
            callbacks=[RunStartFailer()],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        assert len(reports) == 2
        assert "on_run_start" in caplog.text

    def test_callback_error_in_on_run_end(self, simple_tasks, caplog):
        """Benchmark completes and logs error if on_run_end fails."""
        caplog.set_level(logging.ERROR)

        class RunEndFailer(BenchmarkCallback):
            def on_run_end(self, benchmark, results):
                raise RuntimeError("Failed at run end")

        benchmark = DummyBenchmark(
            callbacks=[RunEndFailer()],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        # Reports are generated (run_end happens after report collection)
        assert len(reports) == 2
        assert "on_run_end" in caplog.text


# ==================== Parallel Execution Error Handling Tests ====================


@pytest.mark.core
class TestCallbackErrorHandlingParallel:
    """Tests for callback error handling in parallel execution."""

    def test_callback_error_in_parallel_execution(self, simple_tasks, caplog):
        """Callback errors in parallel execution should not crash workers."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_repeat_end")
        tracking_cb = TrackingCallback()

        benchmark = DummyBenchmark(
            callbacks=[failing_cb, tracking_cb],
        )

        # Run in parallel
        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        # All tasks complete
        assert len(reports) == 2
        assert all(r["status"] == "success" for r in reports)

        # Tracking callback still received events
        assert len(tracking_cb.calls) > 0

        # Errors logged
        assert "on_task_repeat_end" in caplog.text

    def test_multiple_parallel_tasks_with_failing_callback(self, caplog):
        """Callback failures should not interfere across parallel workers."""
        caplog.set_level(logging.ERROR)

        tasks = TaskQueue.from_list([{"query": f"Task {i}", "environment_data": {}} for i in range(5)])

        failing_cb = FailingCallback(fail_on="on_task_repeat_start")

        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # All 5 tasks complete despite callback failures
        assert len(reports) == 5
        assert all(r["status"] == "success" for r in reports)

        # Multiple errors logged (one per task)
        error_count = caplog.text.count("on_task_repeat_start")
        assert error_count >= 5


# ==================== Error Return Value Tests ====================


@pytest.mark.core
class TestCallbackErrorReturnValues:
    """Tests for _invoke_callbacks error return values."""

    def test_invoke_callbacks_returns_empty_list_on_success(self, simple_tasks):
        """_invoke_callbacks should return empty list when no errors occur."""
        tracking_cb = TrackingCallback()
        benchmark = DummyBenchmark(
            callbacks=[tracking_cb],
        )

        # Manually invoke callbacks to test return value
        errors = benchmark._invoke_callbacks("on_run_start", benchmark)

        assert errors == []
        assert "run_start" in tracking_cb.calls

    def test_invoke_callbacks_returns_error_list_on_failure(self, simple_tasks):
        """_invoke_callbacks should return list of exceptions when callbacks fail."""
        failing_cb1 = FailingCallback(fail_on="on_task_start")
        failing_cb2 = FailingCallback(fail_on="on_task_start")
        tracking_cb = TrackingCallback()

        benchmark = DummyBenchmark(
            callbacks=[failing_cb1, tracking_cb, failing_cb2],
        )

        task = Task(query="Test", environment_data={})
        errors = benchmark._invoke_callbacks("on_task_start", benchmark, task)

        # Two errors returned (from two failing callbacks)
        assert len(errors) == 2
        assert all(isinstance(e, RuntimeError) for e in errors)

        # Tracking callback still ran
        assert "task_start:Test" in tracking_cb.calls

    def test_invoke_callbacks_with_suppress_false_raises(self, simple_tasks):
        """With suppress_errors=False, first exception should be raised."""
        failing_cb = FailingCallback(fail_on="on_task_start")
        tracking_cb = TrackingCallback()

        benchmark = DummyBenchmark(
            callbacks=[failing_cb, tracking_cb],
        )

        task = Task(query="Test", environment_data={})

        with pytest.raises(RuntimeError, match="Intentional failure"):
            benchmark._invoke_callbacks(
                "on_task_start",
                benchmark,
                task,
                suppress_errors=False,
            )

        # Tracking callback was not called (execution stopped at first error)
        assert len(tracking_cb.calls) == 0


# ==================== Different Exception Types Tests ====================


@pytest.mark.core
class TestCallbackExceptionTypes:
    """Tests for handling different exception types from callbacks."""

    def test_value_error_in_callback(self, simple_tasks, caplog):
        """ValueError from callback should be caught and logged."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_repeat_start")
        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        assert len(reports) == 2
        assert "ValueError" in caplog.text

    def test_type_error_in_callback(self, simple_tasks, caplog):
        """TypeError from callback should be caught and logged."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_repeat_end")
        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        assert len(reports) == 2
        assert "TypeError" in caplog.text

    def test_key_error_in_callback(self, simple_tasks, caplog):
        """KeyError from callback should be caught and logged."""
        caplog.set_level(logging.ERROR)

        failing_cb = FailingCallback(fail_on="on_task_end")
        benchmark = DummyBenchmark(
            callbacks=[failing_cb],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        assert len(reports) == 2
        assert "KeyError" in caplog.text


# ==================== Integration Tests ====================


@pytest.mark.core
class TestCallbackErrorHandlingIntegration:
    """Integration tests for callback error handling with real scenarios."""

    def test_failing_callback_with_repeats(self, caplog):
        """Callback errors should be handled correctly with n_task_repeats > 1."""
        caplog.set_level(logging.ERROR)

        tasks = TaskQueue.from_list([{"query": "Task", "environment_data": {}}])

        failing_cb = FailingCallback(fail_on="on_task_repeat_end")

        benchmark = DummyBenchmark(
            n_task_repeats=3,
            callbacks=[failing_cb],
        )

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # 3 reports (one per repeat)
        assert len(reports) == 3

        # Callback attempted 3 times
        assert failing_cb.call_count == 3

        # 3 errors logged
        error_count = caplog.text.count("on_task_repeat_end")
        assert error_count >= 3

    def test_mixed_callbacks_some_fail_some_succeed(self, simple_tasks, caplog):
        """Mixed scenario: some callbacks fail, others succeed."""
        caplog.set_level(logging.ERROR)

        failing_cb1 = FailingCallback(fail_on="on_task_start")
        tracking_cb1 = TrackingCallback()
        failing_cb2 = FailingCallback(fail_on="on_task_repeat_end")
        tracking_cb2 = TrackingCallback()

        benchmark = DummyBenchmark(
            callbacks=[failing_cb1, tracking_cb1, failing_cb2, tracking_cb2],
        )

        reports = benchmark.run(simple_tasks, agent_data={"model": "test"})

        # Execution completes
        assert len(reports) == 2

        # Both tracking callbacks received all events
        assert len(tracking_cb1.calls) > 0
        assert len(tracking_cb2.calls) > 0
        assert tracking_cb1.calls == tracking_cb2.calls

        # Both types of errors logged
        assert "on_task_start" in caplog.text
        assert "on_task_repeat_end" in caplog.text
