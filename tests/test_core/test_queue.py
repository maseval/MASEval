"""Tests for TaskQueue implementations.

These tests verify that SequentialTaskQueue, PriorityTaskQueue, and AdaptiveTaskQueue
correctly order and iterate over tasks.
"""

import pytest
from typing import Any, Dict, List, Optional, Sequence

from maseval import Task
from maseval.core.task import (
    TaskProtocol,
    SequentialTaskQueue,
    PriorityTaskQueue,
    AdaptiveTaskQueue,
    TaskQueue,
    BaseTaskQueue,
)


# ==================== Fixtures ====================


@pytest.fixture
def tasks_with_priorities() -> List[Task]:
    """Create tasks with different priorities."""
    tasks = []
    for i, priority in enumerate([0, 5, 2, 8, 1]):
        task = Task(
            query=f"Query {i}",
            environment_data={"index": i},
            protocol=TaskProtocol(priority=priority),
        )
        tasks.append(task)
    return tasks


@pytest.fixture
def simple_tasks() -> List[Task]:
    """Simple task list for basic tests."""
    return [
        Task(query="Q1", environment_data={}),
        Task(query="Q2", environment_data={}),
        Task(query="Q3", environment_data={}),
    ]


# ==================== BaseTaskQueue Tests ====================


@pytest.mark.core
class TestBaseTaskQueue:
    """Tests for BaseTaskQueue common functionality."""

    def test_taskqueue_is_alias_for_sequential(self):
        """TaskQueue should be an alias for SequentialTaskQueue."""
        assert TaskQueue is SequentialTaskQueue

    def test_sequence_protocol(self, simple_tasks):
        """Queue should implement Sequence protocol."""
        queue = SequentialTaskQueue(simple_tasks)

        # __len__
        assert len(queue) == 3

        # __getitem__ with int
        assert queue[0].query == "Q1"
        assert queue[1].query == "Q2"
        assert queue[-1].query == "Q3"

        # __getitem__ with slice
        sliced = queue[1:]
        assert isinstance(sliced, BaseTaskQueue)
        assert len(sliced) == 2

    def test_append_and_extend(self, simple_tasks):
        """Queue should support append and extend."""
        queue = SequentialTaskQueue(simple_tasks[:2])
        assert len(queue) == 2

        queue.append(simple_tasks[2])
        assert len(queue) == 3

        queue.extend([Task(query="Q4"), Task(query="Q5")])
        assert len(queue) == 5

    def test_to_list(self, simple_tasks):
        """to_list() should return a copy of internal list."""
        queue = SequentialTaskQueue(simple_tasks)

        result = queue.to_list()

        assert result == simple_tasks
        assert result is not queue._tasks  # Should be a copy

    def test_from_list_with_tasks(self, simple_tasks):
        """from_list should accept Task objects."""
        queue = SequentialTaskQueue.from_list(simple_tasks)

        assert len(queue) == 3
        assert queue[0].query == "Q1"

    def test_from_list_with_dicts(self):
        """from_list should accept dicts and convert to Tasks."""
        data = [
            {"query": "Dict 1"},
            {"query": "Dict 2", "environment_data": {"key": "value"}},
        ]
        queue = SequentialTaskQueue.from_list(data)

        assert len(queue) == 2
        assert queue[0].query == "Dict 1"
        assert queue[1].environment_data == {"key": "value"}

    def test_from_list_type_error(self):
        """from_list should raise TypeError for invalid items."""
        with pytest.raises(TypeError, match="expects Task or dict"):
            SequentialTaskQueue.from_list(["not a task"])  # type: ignore[arg-type]  # intentional

    def test_from_json_file(self, tmp_path):
        """from_json_file should load tasks from JSON file."""
        import json

        data = {
            "data": [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
            ]
        }

        file_path = tmp_path / "tasks.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        queue = SequentialTaskQueue.from_json_file(file_path)

        assert len(queue) == 2
        assert queue[0].query == "Task 1"
        assert queue[1].query == "Task 2"

    def test_from_json_file_with_limit(self, tmp_path):
        """from_json_file should respect limit parameter."""
        import json

        data = {"data": [{"query": f"Task {i}"} for i in range(10)]}

        file_path = tmp_path / "tasks.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        queue = SequentialTaskQueue.from_json_file(file_path, limit=5)

        assert len(queue) == 5
        assert queue[4].query == "Task 4"

    def test_from_list_field_mapping(self):
        """from_list should map alternative field names."""
        # Test question -> query mapping and short_answer -> evaluation_data
        queue = SequentialTaskQueue.from_list([{"question": "What is 2+2?", "short_answer": "4"}])

        task = queue[0]
        assert task.query == "What is 2+2?"
        assert task.evaluation_data == {"short_answer": "4"}

    def test_repr(self, simple_tasks):
        """Queue should have informative repr."""
        queue = SequentialTaskQueue(simple_tasks)

        repr_str = repr(queue)
        # Should mention queue type and task count
        assert "SequentialTaskQueue" in repr_str or "TaskQueue" in repr_str or "3" in repr_str


# ==================== SequentialTaskQueue Tests ====================


@pytest.mark.core
class TestSequentialTaskQueue:
    """Tests for SequentialTaskQueue ordering."""

    def test_order_preserved(self, simple_tasks):
        """Tasks should be yielded in original order."""
        queue = SequentialTaskQueue(simple_tasks)

        queries = [task.query for task in queue]

        assert queries == ["Q1", "Q2", "Q3"]

    def test_all_tasks_yielded(self, simple_tasks):
        """All tasks should be yielded exactly once."""
        queue = SequentialTaskQueue(simple_tasks)

        count = sum(1 for _ in queue)

        assert count == 3

    def test_empty_collection(self):
        """Empty collection should yield nothing."""
        queue = SequentialTaskQueue([])

        items = list(queue)

        assert items == []

    def test_single_task(self):
        """Single task should be handled correctly."""
        queue = SequentialTaskQueue([Task(query="Only one")])

        items = list(queue)

        assert len(items) == 1
        assert items[0].query == "Only one"


# ==================== PriorityTaskQueue Tests ====================


@pytest.mark.core
class TestPriorityTaskQueue:
    """Tests for PriorityTaskQueue priority ordering."""

    def test_high_priority_first(self, tasks_with_priorities):
        """Higher priority tasks should come first (default)."""
        queue = PriorityTaskQueue(tasks_with_priorities)

        priorities = [task.protocol.priority for task in queue]

        assert priorities == [8, 5, 2, 1, 0]

    def test_low_priority_first_with_reverse_false(self, tasks_with_priorities):
        """Lower priority tasks should come first when reverse=False."""
        queue = PriorityTaskQueue(tasks_with_priorities, reverse=False)

        priorities = [task.protocol.priority for task in queue]

        assert priorities == [0, 1, 2, 5, 8]

    def test_stable_sort_for_equal_priorities(self):
        """Tasks with equal priority should maintain original order."""
        tasks = [
            Task(query="First", environment_data={}, protocol=TaskProtocol(priority=5)),
            Task(query="Second", environment_data={}, protocol=TaskProtocol(priority=5)),
            Task(query="Third", environment_data={}, protocol=TaskProtocol(priority=5)),
        ]
        queue = PriorityTaskQueue(tasks)

        queries = [task.query for task in queue]

        # Python's sort is stable, so original order should be preserved
        assert queries == ["First", "Second", "Third"]

    def test_default_priority_zero(self, simple_tasks):
        """Tasks without explicit priority should have priority 0."""
        queue = PriorityTaskQueue(simple_tasks)

        for task in queue:
            assert task.protocol.priority == 0

    def test_negative_priority(self):
        """Negative priorities should be handled correctly."""
        tasks = [
            Task(query="Low", environment_data={}, protocol=TaskProtocol(priority=-5)),
            Task(query="Normal", environment_data={}, protocol=TaskProtocol(priority=0)),
            Task(query="High", environment_data={}, protocol=TaskProtocol(priority=5)),
        ]
        queue = PriorityTaskQueue(tasks)

        queries = [task.query for task in queue]

        assert queries == ["High", "Normal", "Low"]


# ==================== AdaptiveTaskQueue Tests ====================


class ConcreteAdaptiveQueue(AdaptiveTaskQueue):
    """Concrete implementation of AdaptiveTaskQueue for testing."""

    def initial_state(self) -> Dict[str, Any]:
        """Return empty initial state."""
        return {}

    def select_next_task(self, remaining: Sequence[Task], state: Dict[str, Any]) -> Optional[Task]:
        """Select tasks in order (simple FIFO)."""
        if not remaining:
            return None
        return remaining[0]

    def update_state(self, task: Task, report: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state unchanged."""
        return state


@pytest.mark.core
class TestAdaptiveTaskQueue:
    """Tests for AdaptiveTaskQueue adaptive behavior."""

    def test_basic_iteration_with_completion(self, simple_tasks):
        """AdaptiveTaskQueue should yield all tasks when on_task_repeat_end is called."""
        queue = ConcreteAdaptiveQueue(simple_tasks)

        count = 0
        for task in queue:
            count += 1
            # Simulate callback from benchmark
            queue.on_task_repeat_end(None, {"task_id": str(task.id), "status": "success"})  # type: ignore[arg-type]

        assert count == 3

    def test_on_task_repeat_end_moves_to_completed(self, simple_tasks):
        """on_task_repeat_end should move task to completed list."""
        queue = ConcreteAdaptiveQueue(simple_tasks)
        task = next(iter(queue))

        assert len(queue._completed) == 0

        queue.on_task_repeat_end(None, {"task_id": str(task.id), "status": "success"})  # type: ignore[arg-type]

        assert len(queue._completed) == 1
        assert queue._completed[0][0].id == task.id

    def test_stop_terminates_iteration(self, simple_tasks):
        """Calling stop() should end iteration early."""
        queue = ConcreteAdaptiveQueue(simple_tasks)

        items = []
        for task in queue:
            items.append(task)
            queue.stop()  # Stop immediately after first yield

        assert len(items) == 1

    def test_stop_sets_flag(self, simple_tasks):
        """stop() should set the internal stop flag."""
        queue = ConcreteAdaptiveQueue(simple_tasks)

        assert queue._stop_flag is False

        queue.stop()

        assert queue._stop_flag is True

    def test_iterator_stops_when_empty(self):
        """Iterator should stop when no pending tasks."""
        queue = ConcreteAdaptiveQueue([])

        tasks_yielded = list(queue)
        assert len(tasks_yielded) == 0

    def test_remaining_decreases_after_completion(self, simple_tasks):
        """Remaining list should shrink as tasks complete."""
        queue = ConcreteAdaptiveQueue(simple_tasks)

        assert len(queue._remaining) == 3

        task = next(iter(queue))
        queue.on_task_repeat_end(None, {"task_id": str(task.id), "status": "success"})  # type: ignore[arg-type]

        assert len(queue._remaining) == 2
        assert len(queue._completed) == 1


# ==================== Queue Callback Tests ====================


@pytest.mark.core
class TestQueueCallbacks:
    """Tests for queue callback mechanisms."""

    def test_sequential_queue_iterates_all_tasks(self, simple_tasks):
        """SequentialTaskQueue should iterate through all tasks."""
        queue = SequentialTaskQueue(simple_tasks)

        tasks_yielded = list(queue)
        assert len(tasks_yielded) == len(simple_tasks)
