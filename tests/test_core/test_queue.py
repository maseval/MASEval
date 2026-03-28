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
    InformativeSubsetQueue,
    DISCOQueue,
)


class _FakeArray:
    """Pickle-serializable array-like for testing .tolist() conversion."""

    def tolist(self):
        return [1, 2, 3]

    def __iter__(self):
        return iter([1, 2, 3])


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


# ==================== InformativeSubsetQueue Tests ====================


@pytest.mark.core
class TestInformativeSubsetQueue:
    """Tests for InformativeSubsetQueue subset filtering."""

    def test_filters_to_indices(self, simple_tasks):
        """Only tasks at the given indices should be yielded."""
        queue = InformativeSubsetQueue(simple_tasks, indices=[0, 2])

        queries = [task.query for task in queue]

        assert queries == ["Q1", "Q3"]

    def test_preserves_index_order(self):
        """Tasks should be yielded in the order given by indices, not original order."""
        tasks = [Task(query=f"Q{i}") for i in range(5)]
        queue = InformativeSubsetQueue(tasks, indices=[4, 1, 3])

        queries = [task.query for task in queue]

        assert queries == ["Q4", "Q1", "Q3"]

    def test_none_indices_yields_all(self, simple_tasks):
        """indices=None should yield all tasks in original order."""
        queue = InformativeSubsetQueue(simple_tasks, indices=None)

        queries = [task.query for task in queue]

        assert queries == ["Q1", "Q2", "Q3"]

    def test_stores_all_tasks(self, simple_tasks):
        """_all_tasks should contain the full unfiltered list."""
        queue = InformativeSubsetQueue(simple_tasks, indices=[0])

        assert len(queue._all_tasks) == 3
        assert len(queue) == 1

    def test_out_of_range_indices_raises(self):
        """Out-of-range indices should raise IndexError."""
        tasks = [Task(query="Q0"), Task(query="Q1")]

        with pytest.raises(IndexError, match="out of range"):
            InformativeSubsetQueue(tasks, indices=[0, 5, 99])

    def test_empty_indices(self, simple_tasks):
        """Empty indices list should yield no tasks."""
        queue = InformativeSubsetQueue(simple_tasks, indices=[])

        assert list(queue) == []
        assert len(queue) == 0

    def test_is_subclass_of_sequential(self, simple_tasks):
        """InformativeSubsetQueue should be a SequentialTaskQueue."""
        queue = InformativeSubsetQueue(simple_tasks)
        assert isinstance(queue, SequentialTaskQueue)


# ==================== DISCOQueue Tests ====================


@pytest.mark.core
class TestDISCOQueue:
    """Tests for DISCOQueue diversity-based subset."""

    def test_filters_to_anchor_points(self):
        """Only tasks at anchor-point indices should be yielded."""
        tasks = [Task(query=f"Q{i}") for i in range(10)]
        queue = DISCOQueue(tasks, anchor_points=[2, 5, 8])

        queries = [task.query for task in queue]

        assert queries == ["Q2", "Q5", "Q8"]

    def test_none_anchor_points_yields_all(self, simple_tasks):
        """anchor_points=None should yield all tasks."""
        queue = DISCOQueue(simple_tasks, anchor_points=None)

        assert len(list(queue)) == 3

    def test_stores_anchor_points(self):
        """_anchor_points should be accessible."""
        tasks = [Task(query=f"Q{i}") for i in range(5)]
        anchor_pts = [0, 3, 4]
        queue = DISCOQueue(tasks, anchor_points=anchor_pts)

        assert queue._anchor_points == [0, 3, 4]

    def test_is_subclass_of_informative_subset(self, simple_tasks):
        """DISCOQueue should be an InformativeSubsetQueue."""
        queue = DISCOQueue(simple_tasks)
        assert isinstance(queue, InformativeSubsetQueue)

    def test_len_matches_anchor_count(self):
        """Queue length should match number of valid anchor points."""
        tasks = [Task(query=f"Q{i}") for i in range(10)]
        queue = DISCOQueue(tasks, anchor_points=[1, 3, 7])

        assert len(queue) == 3


@pytest.mark.core
class TestDISCOQueueLoadAnchorPoints:
    """Tests for DISCOQueue.load_anchor_points static method."""

    def test_load_from_json(self, tmp_path):
        """Should load anchor points from a JSON file."""
        import json

        path = tmp_path / "anchors.json"
        path.write_text(json.dumps([0, 5, 12, 99]))

        result = DISCOQueue.load_anchor_points(path)

        assert result == [0, 5, 12, 99]

    def test_load_from_pickle(self, tmp_path):
        """Should load anchor points from a pickle file."""
        import pickle

        path = tmp_path / "anchors.pkl"
        with open(path, "wb") as f:
            pickle.dump([2, 7, 15], f)

        result = DISCOQueue.load_anchor_points(path)

        assert result == [2, 7, 15]

    def test_load_converts_tolist(self, tmp_path):
        """Should call .tolist() on array-like objects (e.g. numpy arrays)."""
        import pickle

        path = tmp_path / "anchors.pkl"
        with open(path, "wb") as f:
            pickle.dump(_FakeArray(), f)

        result = DISCOQueue.load_anchor_points(path)

        assert result == [1, 2, 3]

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            DISCOQueue.load_anchor_points(tmp_path / "nonexistent.json")

    def test_accepts_string_path(self, tmp_path):
        """Should accept a string path, not just Path objects."""
        import json

        path = tmp_path / "anchors.json"
        path.write_text(json.dumps([10, 20]))

        result = DISCOQueue.load_anchor_points(str(path))

        assert result == [10, 20]

    def test_init_with_anchor_points_path(self, tmp_path):
        """DISCOQueue should load anchor points from file when anchor_points_path is given."""
        import json

        tasks = [Task(query=f"Q{i}") for i in range(10)]
        path = tmp_path / "anchors.json"
        path.write_text(json.dumps([2, 5, 8]))

        queue = DISCOQueue(tasks, anchor_points_path=path)

        assert len(queue) == 3
        assert queue._anchor_points == [2, 5, 8]

    def test_init_rejects_both_anchor_args(self, tmp_path):
        """DISCOQueue should raise ValueError when both anchor_points and anchor_points_path are given."""
        import json

        tasks = [Task(query=f"Q{i}") for i in range(5)]
        path = tmp_path / "anchors.json"
        path.write_text(json.dumps([0, 1]))

        with pytest.raises(ValueError, match="not both"):
            DISCOQueue(tasks, anchor_points=[0, 1], anchor_points_path=path)


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
