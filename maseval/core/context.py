"""Execution context for task timeout handling.

This module provides the TaskContext class for cooperative timeout checking
during task execution. The context tracks elapsed time and enables checkpoints
where tasks can gracefully exit if the deadline has passed.
"""

import time
from typing import Any, Dict, Optional

from .exceptions import TaskTimeoutError


class TaskContext:
    """Execution context for cooperative timeout checking.

    TaskContext provides a mechanism for tasks to voluntarily check for timeout
    conditions at defined checkpoints. This enables clean interruption without
    forcibly killing threads (which Python doesn't support well).

    The context is created with an optional deadline and tracks elapsed time.
    Tasks should call `check_timeout()` at natural checkpoints (e.g., between
    agent steps, after LLM calls) to allow graceful termination.

    Attributes:
        deadline: Maximum execution time in seconds, or None for no timeout.
        collected_traces: Traces collected before timeout (for partial results).

    Usage:
        context = TaskContext(deadline=60.0)

        for step in range(max_steps):
            context.check_timeout()  # Raises TaskTimeoutError if expired
            result = agent.run(query)
            # ... process result

        # Access timing info
        print(f"Elapsed: {context.elapsed}s")
        print(f"Remaining: {context.remaining}s")

    Thread Safety:
        TaskContext instances are not thread-safe. Each thread/task should
        have its own context instance.
    """

    def __init__(self, deadline: Optional[float] = None):
        """Initialize execution context.

        Args:
            deadline: Maximum execution time in seconds. If None, no timeout
                checking is performed and check_timeout() is a no-op.
        """
        self._deadline = deadline
        self._start_time = time.monotonic()
        self.collected_traces: Dict[str, Any] = {}

    @property
    def deadline(self) -> Optional[float]:
        """Maximum execution time in seconds, or None if no timeout."""
        return self._deadline

    @property
    def elapsed(self) -> float:
        """Time elapsed since context creation in seconds."""
        return time.monotonic() - self._start_time

    @property
    def remaining(self) -> Optional[float]:
        """Time remaining before deadline in seconds, or None if no deadline.

        Returns 0 if deadline has passed.
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self.elapsed)

    @property
    def is_expired(self) -> bool:
        """Whether the deadline has passed.

        Returns False if no deadline is set.
        """
        if self._deadline is None:
            return False
        return self.elapsed >= self._deadline

    def check_timeout(self) -> None:
        """Check if deadline has passed and raise TaskTimeoutError if so.

        This method should be called at natural checkpoints during task
        execution (e.g., between agent steps, after LLM calls). If the
        deadline has passed, it raises TaskTimeoutError with timing info.

        If no deadline is set, this is a no-op.

        Raises:
            TaskTimeoutError: If the deadline has passed.

        Usage:
            context = TaskContext(deadline=30.0)

            for step in range(max_steps):
                context.check_timeout()  # May raise
                result = agent.run(query)
        """
        if self.is_expired:
            raise TaskTimeoutError(
                f"Task exceeded {self._deadline}s deadline after {self.elapsed:.2f}s",
                component="timeout_check",
                elapsed=self.elapsed,
                timeout=self._deadline or 0.0,
                partial_traces=self.collected_traces,
            )

    def set_collected_traces(self, traces: Dict[str, Any]) -> None:
        """Store traces collected during execution for inclusion in timeout errors.

        Args:
            traces: Traces to store. These will be included in TaskTimeoutError
                if a timeout occurs.
        """
        self.collected_traces = traces
