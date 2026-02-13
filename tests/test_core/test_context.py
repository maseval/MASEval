"""Tests for TaskContext timeout handling.

These tests verify that TaskContext correctly tracks time, checks deadlines,
and raises TaskTimeoutError when appropriate.
"""

import pytest
import time

from maseval.core.context import TaskContext
from maseval.core.exceptions import TaskTimeoutError


@pytest.mark.core
class TestTaskContextBasics:
    """Tests for basic TaskContext functionality."""

    def test_context_no_timeout_never_expires(self):
        """Context without deadline should never expire."""
        context = TaskContext(deadline=None)

        assert context.deadline is None
        assert context.remaining is None
        assert context.is_expired is False

        # check_timeout should be no-op
        context.check_timeout()  # Should not raise

    def test_context_with_timeout_not_expired(self):
        """Context before deadline should show remaining time."""
        context = TaskContext(deadline=10.0)

        assert context.deadline == 10.0
        assert context.is_expired is False
        assert context.remaining is not None and context.remaining > 9.0
        assert context.elapsed < 1.0  # Just started

    def test_context_elapsed_increases(self):
        """Elapsed property should increase over time."""
        context = TaskContext(deadline=10.0)

        elapsed1 = context.elapsed
        time.sleep(0.05)
        elapsed2 = context.elapsed

        assert elapsed2 > elapsed1

    def test_context_remaining_decreases(self):
        """Remaining property should decrease over time."""
        context = TaskContext(deadline=10.0)

        remaining1 = context.remaining
        time.sleep(0.05)
        remaining2 = context.remaining

        assert remaining1 is not None and remaining2 is not None
        assert remaining2 < remaining1


@pytest.mark.core
class TestTaskContextTimeout:
    """Tests for TaskContext timeout behavior."""

    def test_context_is_expired_after_deadline(self):
        """Context after deadline should show is_expired=True."""
        context = TaskContext(deadline=0.01)  # Very short deadline

        time.sleep(0.02)  # Wait past deadline

        assert context.is_expired is True
        assert context.remaining == 0.0

    def test_check_timeout_raises_on_expiry(self):
        """check_timeout() should raise TaskTimeoutError when expired."""
        context = TaskContext(deadline=0.01)

        time.sleep(0.02)

        with pytest.raises(TaskTimeoutError) as exc_info:
            context.check_timeout()

        error: TaskTimeoutError = exc_info.value  # type: ignore[assignment]
        assert error.timeout == 0.01
        assert error.elapsed >= 0.01

    def test_check_timeout_includes_partial_traces(self):
        """TaskTimeoutError should include partial traces if set."""
        context = TaskContext(deadline=0.01)
        partial_traces = {"agents": {"agent1": {"steps": 3}}}
        context.set_collected_traces(partial_traces)

        time.sleep(0.02)

        with pytest.raises(TaskTimeoutError) as exc_info:
            context.check_timeout()

        error: TaskTimeoutError = exc_info.value  # type: ignore[assignment]
        assert error.partial_traces == partial_traces

    def test_check_timeout_no_traces_if_not_set(self):
        """TaskTimeoutError should have empty traces if not set."""
        context = TaskContext(deadline=0.01)

        time.sleep(0.02)

        with pytest.raises(TaskTimeoutError) as exc_info:
            context.check_timeout()

        error: TaskTimeoutError = exc_info.value  # type: ignore[assignment]
        assert error.partial_traces == {}

    def test_check_timeout_does_not_raise_before_deadline(self):
        """check_timeout() should not raise before deadline."""
        context = TaskContext(deadline=10.0)

        # Should not raise
        for _ in range(10):
            context.check_timeout()

    def test_set_collected_traces_updates_context(self):
        """set_collected_traces should store traces for later use."""
        context = TaskContext(deadline=10.0)

        traces = {"test": "data"}
        context.set_collected_traces(traces)

        assert context.collected_traces == traces

    def test_context_timing_accuracy(self):
        """Elapsed time should be reasonably accurate."""
        context = TaskContext(deadline=10.0)

        time.sleep(0.1)
        elapsed = context.elapsed

        # Allow 50ms tolerance
        assert 0.05 < elapsed < 0.2
