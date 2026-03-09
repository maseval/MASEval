"""Tests for Task.freeze() and Task.unfreeze().

These tests verify that freezing a task prevents accidental mutation of its
dictionary fields during a benchmark run, and that unfreezing restores
full mutability.
"""

import pytest
from maseval import Task, TaskFrozenError
from maseval.core.task import FrozenDict, TaskProtocol


@pytest.mark.core
class TestTaskFreeze:
    """Tests for Task.freeze()."""

    def test_freeze_returns_self(self):
        """freeze() should return the same task instance for chaining."""
        task = Task(query="test")
        result = task.freeze()
        assert result is task

    def test_is_frozen_after_freeze(self):
        """is_frozen should be True after freeze()."""
        task = Task(query="test")
        assert not task.is_frozen
        task.freeze()
        assert task.is_frozen

    def test_freeze_blocks_dict_mutation(self):
        """Frozen dict fields should reject item assignment with TaskFrozenError."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.environment_data["key"] = "new"

    def test_freeze_blocks_dict_item_deletion(self):
        """Frozen dict fields should reject item deletion with TaskFrozenError."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            del task.environment_data["key"]

    def test_freeze_blocks_all_dict_fields(self):
        """All four dict fields should be frozen."""
        task = Task(
            query="test",
            environment_data={"a": 1},
            user_data={"b": 2},
            evaluation_data={"c": 3},
            metadata={"d": 4},
        )
        task.freeze()

        for field_name in ("environment_data", "user_data", "evaluation_data", "metadata"):
            with pytest.raises(TaskFrozenError):
                getattr(task, field_name)["new_key"] = "value"

    def test_freeze_blocks_attribute_reassignment(self):
        """Frozen task should reject attribute reassignment."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.query = "changed"

        with pytest.raises(TaskFrozenError):
            task.environment_data = {"new": "dict"}

        with pytest.raises(TaskFrozenError):
            task.id = "new-id"

    def test_freeze_deep_nested_dicts(self):
        """Nested dicts should also be frozen recursively."""
        task = Task(
            query="test",
            environment_data={"outer": {"inner": {"deep": "value"}}},
        )
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.environment_data["outer"]["inner"]["deep"] = "changed"

        with pytest.raises(TaskFrozenError):
            task.environment_data["outer"]["new_key"] = "value"

    def test_freeze_preserves_data_access(self):
        """Frozen fields should still be readable."""
        data = {"key": "value", "nested": {"a": 1}}
        task = Task(query="test", environment_data=data)
        task.freeze()

        assert task.environment_data["key"] == "value"
        assert task.environment_data["nested"]["a"] == 1
        assert task.query == "test"

    def test_freeze_converts_to_frozen_dict(self):
        """Frozen dict fields should be FrozenDict instances."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        assert isinstance(task.environment_data, FrozenDict)

    def test_freeze_already_frozen_raises(self):
        """Calling freeze() on an already frozen task should raise."""
        task = Task(query="test")
        task.freeze()

        with pytest.raises(TaskFrozenError, match="already frozen"):
            task.freeze()

    def test_freeze_empty_dicts(self):
        """Freezing a task with empty dicts should work."""
        task = Task(query="test")
        task.freeze()
        assert task.is_frozen
        assert task.environment_data == {}

    def test_freeze_blocks_dict_clear(self):
        """Frozen dict fields should reject clear()."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.environment_data.clear()

    def test_freeze_blocks_dict_pop(self):
        """Frozen dict fields should reject pop()."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.environment_data.pop("key")

    def test_freeze_blocks_dict_update(self):
        """Frozen dict fields should reject update()."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()

        with pytest.raises(TaskFrozenError):
            task.environment_data.update({"new": "value"})


@pytest.mark.core
class TestTaskUnfreeze:
    """Tests for Task.unfreeze()."""

    def test_unfreeze_returns_self(self):
        """unfreeze() should return the same task instance for chaining."""
        task = Task(query="test")
        task.freeze()
        result = task.unfreeze()
        assert result is task

    def test_is_frozen_after_unfreeze(self):
        """is_frozen should be False after unfreeze()."""
        task = Task(query="test")
        task.freeze()
        task.unfreeze()
        assert not task.is_frozen

    def test_unfreeze_restores_dict_mutability(self):
        """Unfrozen dict fields should accept item assignment again."""
        task = Task(query="test", environment_data={"key": "value"})
        task.freeze()
        task.unfreeze()

        task.environment_data["key"] = "new"
        assert task.environment_data["key"] == "new"

    def test_unfreeze_restores_attribute_assignment(self):
        """Unfrozen task should accept attribute reassignment again."""
        task = Task(query="test")
        task.freeze()
        task.unfreeze()

        task.query = "changed"
        assert task.query == "changed"

    def test_unfreeze_restores_nested_dicts(self):
        """Nested dicts should be regular dicts after unfreeze."""
        task = Task(
            query="test",
            environment_data={"outer": {"inner": "value"}},
        )
        task.freeze()
        task.unfreeze()

        task.environment_data["outer"]["inner"] = "changed"
        assert task.environment_data["outer"]["inner"] == "changed"
        assert type(task.environment_data) is dict
        assert type(task.environment_data["outer"]) is dict

    def test_unfreeze_preserves_data(self):
        """Data should be identical after a freeze/unfreeze cycle."""
        original = {"key": "value", "nested": {"a": [1, 2, 3]}}
        task = Task(query="test", environment_data=original)
        task.freeze()
        task.unfreeze()

        assert task.environment_data == original

    def test_unfreeze_not_frozen_raises(self):
        """Calling unfreeze() on a non-frozen task should raise."""
        task = Task(query="test")

        with pytest.raises(TaskFrozenError, match="not frozen"):
            task.unfreeze()

    def test_freeze_unfreeze_cycle(self):
        """Multiple freeze/unfreeze cycles should work correctly."""
        task = Task(query="test", environment_data={"key": "value"})

        task.freeze()
        assert task.is_frozen

        task.unfreeze()
        assert not task.is_frozen
        task.environment_data["key"] = "v2"

        task.freeze()
        assert task.is_frozen
        assert task.environment_data["key"] == "v2"

        task.unfreeze()
        assert not task.is_frozen


@pytest.mark.core
class TestTaskFreezeWithProtocol:
    """Tests for freeze/unfreeze interaction with TaskProtocol."""

    def test_freeze_does_not_affect_protocol_scalars(self):
        """Protocol scalar fields should still be readable when frozen."""
        task = Task(query="test", protocol=TaskProtocol(priority=5, timeout_seconds=30.0))
        task.freeze()

        assert task.protocol.priority == 5
        assert task.protocol.timeout_seconds == 30.0

    def test_frozen_task_in_queue(self):
        """Frozen tasks should work in task queues (read-only iteration)."""
        from maseval import TaskQueue

        tasks = [Task(query=f"Q{i}") for i in range(3)]
        for t in tasks:
            t.freeze()

        queue = TaskQueue(tasks)
        for task in queue:
            assert task.is_frozen
            assert task.query.startswith("Q")


@pytest.mark.core
class TestTaskFreezeNonDictFields:
    """Tests that non-dict fields behave correctly with freeze."""

    def test_non_dict_values_in_dicts_unaffected(self):
        """Lists, strings, ints inside dicts should be preserved as-is."""
        task = Task(
            query="test",
            environment_data={"items": [1, 2, 3], "count": 42, "name": "test"},
        )
        task.freeze()

        assert task.environment_data["items"] == [1, 2, 3]
        assert task.environment_data["count"] == 42
        assert task.environment_data["name"] == "test"

    def test_lists_in_frozen_dicts_remain_mutable(self):
        """Lists inside frozen dicts are not converted (only dicts are frozen)."""
        task = Task(
            query="test",
            environment_data={"items": [1, 2, 3]},
        )
        task.freeze()

        # Lists inside frozen dicts remain mutable (freeze targets dicts only)
        task.environment_data["items"].append(4)
        assert task.environment_data["items"] == [1, 2, 3, 4]
