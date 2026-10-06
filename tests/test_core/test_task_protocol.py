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
            priority=10,
            tags={"category": "hard", "group": "A"},
        )

        assert protocol.timeout_seconds == 60.0
        assert protocol.timeout_action == TimeoutAction.SKIP
        assert protocol.max_retries == 0
        assert protocol.priority == 10
        assert protocol.tags == {"category": "hard", "group": "A"}

    @pytest.mark.parametrize("timeout_action", [TimeoutAction.RETRY, TimeoutAction.EXTEND])
    def test_retry_timeout_actions_are_rejected(self, timeout_action):
        """Unsupported timeout retry actions should fail loudly."""
        with pytest.raises(ValueError, match="supports only TimeoutAction.SKIP"):
            TaskProtocol(timeout_action=timeout_action)

    @pytest.mark.parametrize("max_retries", [-1, 1, 5, 100])
    def test_max_retries_is_rejected(self, max_retries):
        """Any non-zero max_retries should fail loudly."""
        with pytest.raises(ValueError, match="max_retries is reserved"):
            TaskProtocol(max_retries=max_retries)

    def test_tags_isolation(self):
        """Tags dict should be independent per instance."""
        p1 = TaskProtocol()
        p2 = TaskProtocol()

        p1.tags["key"] = "value"

        assert "key" not in p2.tags

    def test_to_dict_defaults(self):
        """to_dict should return all fields with defaults."""
        protocol = TaskProtocol()
        result = protocol.to_dict()

        assert result == {
            "timeout_seconds": None,
            "timeout_action": "skip",
            "max_retries": 0,
            "priority": 0,
            "tags": {},
        }

    def test_to_dict_custom_values(self):
        """to_dict should serialize custom values and enums correctly."""
        protocol = TaskProtocol(
            timeout_seconds=60.0,
            priority=10,
            tags={"category": "hard"},
        )
        result = protocol.to_dict()

        assert result == {
            "timeout_seconds": 60.0,
            "timeout_action": "skip",
            "max_retries": 0,
            "priority": 10,
            "tags": {"category": "hard"},
        }

    def test_to_dict_returns_new_dict(self):
        """to_dict should return a new dict, not a reference to internal state."""
        protocol = TaskProtocol(tags={"key": "value"})
        result = protocol.to_dict()

        result["tags"]["key"] = "modified"
        assert protocol.tags["key"] == "value"


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
