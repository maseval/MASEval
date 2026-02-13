"""Tests for TaskProtocol and TimeoutAction.

These tests verify that TaskProtocol correctly configures task execution
parameters and that TimeoutAction enum values are correct.
"""

import pytest
from maseval import Task, TaskQueue
from maseval.core.task import TaskProtocol, TimeoutAction


@pytest.mark.core
class TestTimeoutAction:
    """Tests for TimeoutAction enum."""

    def test_enum_values(self):
        """TimeoutAction should have expected values."""
        assert TimeoutAction.SKIP.value == "skip"
        assert TimeoutAction.RETRY.value == "retry"
        assert TimeoutAction.EXTEND.value == "extend"

    def test_enum_members(self):
        """TimeoutAction should have expected members."""
        members = list(TimeoutAction)
        assert len(members) == 3
        assert TimeoutAction.SKIP in members
        assert TimeoutAction.RETRY in members
        assert TimeoutAction.EXTEND in members


@pytest.mark.core
class TestTaskProtocol:
    """Tests for TaskProtocol dataclass."""

    def test_default_values(self):
        """TaskProtocol should have sensible defaults."""
        protocol = TaskProtocol()

        assert protocol.timeout_seconds is None
        assert protocol.timeout_action == TimeoutAction.SKIP
        assert protocol.max_retries == 0
        assert protocol.priority == 0
        assert protocol.tags == {}

    def test_custom_values(self):
        """TaskProtocol should accept custom values."""
        protocol = TaskProtocol(
            timeout_seconds=60.0,
            timeout_action=TimeoutAction.RETRY,
            max_retries=3,
            priority=10,
            tags={"category": "hard", "group": "A"},
        )

        assert protocol.timeout_seconds == 60.0
        assert protocol.timeout_action == TimeoutAction.RETRY
        assert protocol.max_retries == 3
        assert protocol.priority == 10
        assert protocol.tags == {"category": "hard", "group": "A"}

    def test_tags_isolation(self):
        """Tags dict should be independent per instance."""
        p1 = TaskProtocol()
        p2 = TaskProtocol()

        p1.tags["key"] = "value"

        assert "key" not in p2.tags


@pytest.mark.core
class TestTaskWithProtocol:
    """Tests for Task with TaskProtocol integration."""

    def test_task_has_protocol_field(self):
        """Task dataclass should have protocol field."""
        task = Task(query="Test", environment_data={})

        assert hasattr(task, "protocol")
        assert isinstance(task.protocol, TaskProtocol)

    def test_task_default_protocol(self):
        """Task should have default protocol if not specified."""
        task = Task(query="Test")

        assert task.protocol.timeout_seconds is None
        assert task.protocol.priority == 0

    def test_task_custom_protocol(self):
        """Task should accept custom protocol."""
        protocol = TaskProtocol(
            timeout_seconds=30.0,
            priority=5,
        )
        task = Task(query="Test", protocol=protocol)

        assert task.protocol.timeout_seconds == 30.0
        assert task.protocol.priority == 5

    def test_task_queue_preserves_protocol(self):
        """TaskQueue should preserve protocol on tasks."""
        task1 = Task(query="Q1", protocol=TaskProtocol(priority=1))
        task2 = Task(query="Q2", protocol=TaskProtocol(priority=2))
        tasks = TaskQueue([task1, task2])

        first_task: Task = tasks[0]  # type: ignore[assignment]
        second_task: Task = tasks[1]  # type: ignore[assignment]

        assert first_task.protocol.priority == 1
        assert second_task.protocol.priority == 2
