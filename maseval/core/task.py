from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, overload, TYPE_CHECKING
from uuid import uuid4
from collections.abc import Sequence
from typing import Iterable, List, Union, Iterator, Optional
import json
import pickle
from pathlib import Path
from enum import Enum

from .exceptions import TaskFrozenError

if TYPE_CHECKING:
    from .benchmark import Benchmark

# Import BenchmarkCallback at runtime to enable inheritance
from .callback import BenchmarkCallback


class TimeoutAction(Enum):
    """Action to take when a task timeout occurs."""

    SKIP = "skip"  # Mark as timed out, continue to next task
    RETRY = "retry"  # Retry once with same timeout
    EXTEND = "extend"  # Double timeout and retry


@dataclass
class TaskProtocol:
    """Configuration for how MASEval executes a task.

    This is a data container for execution parameters, separate from
    task content (query, environment_data, etc.). It controls the
    interface between the task and MASEval's execution engine.

    Note:
        Timeout checking is cooperative and currently only occurs at execution phase
        boundaries (after setup, before execution, before evaluation). Timeout detection
        during agent execution is not yet supported.

    Attributes:
        timeout_seconds: Maximum execution time for this task. None means no timeout.
        timeout_action: Action to take when timeout occurs.
        max_retries: Maximum retry attempts for transient failures (not timeouts).
        priority: Execution priority (higher = sooner). Used by adaptive task queues.
        tags: Arbitrary tags for filtering or grouping tasks.
    """

    timeout_seconds: Optional[float] = None
    timeout_action: TimeoutAction = TimeoutAction.SKIP
    max_retries: int = 0
    priority: int = 0
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary.

        Returns:
            Dictionary with all fields. Enum values are converted to strings.
        """
        return {
            "timeout_seconds": self.timeout_seconds,
            "timeout_action": self.timeout_action.value,
            "max_retries": self.max_retries,
            "priority": self.priority,
            "tags": dict(self.tags),
        }


class FrozenDict(dict):
    """A dict subclass that raises ``TaskFrozenError`` on any mutation attempt.

    Behaves like a regular dict for all read operations (iteration, lookup,
    ``len``, ``in``, etc.) but blocks writes so that frozen task data cannot
    be modified accidentally.
    """

    _FROZEN_MSG = "Cannot modify a frozen Task's data. Call task.unfreeze() first."

    def _raise(self, *_args: Any, **_kwargs: Any) -> Any:
        raise TaskFrozenError(self._FROZEN_MSG, component="Task")

    __setitem__ = _raise
    __delitem__ = _raise
    clear = _raise
    pop = _raise  # type: ignore[assignment]
    popitem = _raise
    setdefault = _raise  # type: ignore[assignment]
    update = _raise  # type: ignore[override]

    def __repr__(self) -> str:
        return f"FrozenDict({dict.__repr__(self)})"


def _freeze_value(value: Any) -> Any:
    """Recursively convert dicts to ``FrozenDict`` for read-only access."""
    if isinstance(value, dict) and not isinstance(value, FrozenDict):
        return FrozenDict({k: _freeze_value(v) for k, v in value.items()})
    return value


def _unfreeze_value(value: Any) -> Any:
    """Recursively convert ``FrozenDict`` back to regular dicts."""
    if isinstance(value, FrozenDict):
        return {k: _unfreeze_value(v) for k, v in value.items()}
    return value


# Dict field names on Task that freeze() / unfreeze() operate on.
_TASK_DICT_FIELDS = ("environment_data", "user_data", "evaluation_data", "metadata")


@dataclass
class Task:
    """A data container for a single benchmark task.

    Task data can optionally be frozen after loading to prevent accidental
    mutation during a benchmark run. Use ``freeze()`` to make all dictionary
    fields read-only, and ``unfreeze()`` to restore mutability.

    Note:
        It is strongly recommended to call ``freeze()`` once all task data has
        been assembled (e.g. after ``load_tasks()`` or inside ``setup_environment``).
        This guards against subtle bugs where benchmark components accidentally
        overwrite task data during execution.

    Attributes:
        query: The main input query or prompt for the task.
        id: A unique identifier for the task. Benchmarks can provide human-readable IDs
            (e.g., "task-000001", "retail_001"). Auto-generates a UUID string if not provided.
        environment_data: A dictionary of data needed to set up the environment for the task.
        evaluation_data: A dictionary of data needed to evaluate the agent's performance on the task.
        metadata: A dictionary for any additional metadata about the task.
        protocol: Execution protocol controlling timeout, retries, priority, and other runtime
            parameters. It provides fine-grained control over how MASEval runs the task. The
            protocol serves purely as a communication channel between the task instance and
            MASEval's execution engine; it does not impose any intrinsic semantics on the task
            content itself.
    """

    query: str
    id: str = field(default_factory=lambda: str(uuid4()))
    environment_data: Dict[str, Any] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)
    evaluation_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    protocol: TaskProtocol = field(default_factory=TaskProtocol)
    _frozen: bool = field(init=False, default=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Block attribute assignment when the task is frozen.

        The ``_frozen`` field itself can always be set (used internally by
        ``freeze`` / ``unfreeze``).  All other fields raise
        ``TaskFrozenError`` while the task is frozen.
        """
        if name != "_frozen" and getattr(self, "_frozen", False):
            raise TaskFrozenError(
                f"Cannot set '{name}' on a frozen Task. Call task.unfreeze() first.",
                component="Task",
            )
        object.__setattr__(self, name, value)

    @property
    def is_frozen(self) -> bool:
        """Whether this task's data is currently frozen (read-only).

        Returns:
            ``True`` if ``freeze()`` has been called and ``unfreeze()`` has not.
        """
        return self._frozen

    def freeze(self) -> "Task":
        """Make all dictionary fields read-only.

        Converts ``environment_data``, ``user_data``, ``evaluation_data``, and
        ``metadata`` (including nested dicts) to read-only wrappers and prevents
        attribute reassignment on the task. Subsequent attempts to mutate any of
        these fields raise ``TaskFrozenError``.

        Call ``unfreeze()`` to restore mutability.

        Returns:
            ``self``, for chaining (e.g. ``task.freeze().query``).

        Raises:
            TaskFrozenError: If the task is already frozen.

        Example:
            ```python
            task = Task(query="test", environment_data={"key": "value"})
            task.freeze()

            task.environment_data["key"] = "new"  # raises TaskFrozenError
            task.query = "changed"                 # raises TaskFrozenError
            ```
        """
        if self._frozen:
            raise TaskFrozenError("Task is already frozen.", component="Task")

        for field_name in _TASK_DICT_FIELDS:
            object.__setattr__(self, field_name, _freeze_value(getattr(self, field_name)))
        object.__setattr__(self, "_frozen", True)
        return self

    def unfreeze(self) -> "Task":
        """Restore mutability to all dictionary fields.

        Converts read-only wrappers back to regular dicts and re-enables
        attribute assignment on the task.

        Returns:
            ``self``, for chaining.

        Raises:
            TaskFrozenError: If the task is not currently frozen.

        Example:
            ```python
            task.freeze()
            # ... benchmark run ...
            task.unfreeze()
            task.environment_data["key"] = "updated"  # works again
            ```
        """
        if not self._frozen:
            raise TaskFrozenError("Task is not frozen.", component="Task")

        object.__setattr__(self, "_frozen", False)
        for field_name in _TASK_DICT_FIELDS:
            setattr(self, field_name, _unfreeze_value(getattr(self, field_name)))
        return self


# =============================================================================
# Task Queue Classes
# =============================================================================


class BaseTaskQueue(ABC, Sequence):
    """Abstract base class for task scheduling strategies.

    BaseTaskQueue provides a sequence-like interface for task execution.
    Concrete implementations can reorder tasks, skip tasks, or terminate
    early based on execution outcomes.

    Subclasses must implement ``__iter__`` to define the iteration order.
    For adaptive behavior based on task results, use ``AdaptiveTaskQueue``
    which integrates with the benchmark callback system.

    Attributes:
        _tasks: Internal list of tasks.

    Example:
        ```python
        queue = SequentialTaskQueue(tasks)

        for task in queue:
            report = execute_task(task)
            # Iterator handles termination automatically
        ```
    """

    def __init__(self, tasks: Iterable[Task]) -> None:
        """Initialize the task queue.

        Args:
            tasks: An iterable of Task objects to schedule.
        """
        self._tasks: List[Task] = list(tasks)

    def __len__(self) -> int:
        """Return the total number of tasks in the queue."""
        return len(self._tasks)

    @overload
    def __getitem__(self, idx: int) -> Task: ...

    @overload
    def __getitem__(self, idx: slice) -> "BaseTaskQueue": ...

    def __getitem__(  # type: ignore[invalid-method-override]
        self, idx: Union[int, slice]
    ) -> Union[Task, "BaseTaskQueue"]:
        """Get a task by index or a slice of tasks.

        Args:
            idx: Integer index or slice object.

        Returns:
            A single Task for integer index, or a new queue instance for slices.
        """
        if isinstance(idx, slice):
            # Return a new instance of the same type with sliced tasks
            return self.__class__(self._tasks[idx])
        return self._tasks[idx]

    @abstractmethod
    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in the scheduled execution order.

        Returns:
            Iterator yielding Task objects.
        """
        pass

    def append(self, task: Task) -> None:
        """Add a task to the end of the queue.

        Args:
            task: The task to append.
        """
        self._tasks.append(task)

    def extend(self, tasks: Iterable[Task]) -> None:
        """Add multiple tasks to the end of the queue.

        Args:
            tasks: An iterable of tasks to append.
        """
        self._tasks.extend(tasks)

    def to_list(self) -> List[Task]:
        """Return a copy of the internal task list.

        Returns:
            List of all tasks in the queue.
        """
        return list(self._tasks)

    @classmethod
    def from_list(cls, data: Iterable[Union[Task, dict]]) -> "BaseTaskQueue":
        """Create a queue from an iterable of Tasks or dicts.

        Args:
            data: An iterable of Task objects or dicts that can be converted to Tasks.

        Returns:
            A new queue instance containing the tasks.

        Raises:
            TypeError: If an item is neither a Task nor a dict.
        """
        tasks: List[Task] = []
        for item in data:
            if isinstance(item, Task):
                tasks.append(item)
            elif isinstance(item, dict):
                if "query" in item:
                    query = item["query"]
                    tasks.append(
                        Task(
                            query=query,
                            environment_data=item.get("environment_data", {}),
                            evaluation_data=item.get("evaluation_data", {}),
                            metadata=item.get("metadata", {}),
                        )
                    )
                else:
                    query_val = item.get("question") or item.get("prompt") or item.get("query") or ""
                    query = str(query_val) if query_val else ""
                    environment_data = (
                        item.get("environment_data") or {"text_content": item.get("text")}
                        if item.get("text")
                        else item.get("environment_data", {})
                    )
                    evaluation_data = (
                        item.get("evaluation_data") or {"short_answer": item.get("short_answer")}
                        if item.get("short_answer")
                        else item.get("evaluation_data", {})
                    )
                    tasks.append(
                        Task(
                            query=query,
                            environment_data=environment_data or {},
                            evaluation_data=evaluation_data or {},
                            metadata=item.get("metadata", {}),
                        )
                    )
            else:
                raise TypeError(f"{cls.__name__}.from_list expects Task or dict entries")
        return cls(tasks)

    @classmethod
    def from_json_file(cls, path: Union[str, Path], *, limit: Optional[int] = None) -> "BaseTaskQueue":
        """Load tasks from a JSON file.

        This helper understands the example file format used in ``examples/data.json``
        where the top-level object has a ``data`` list and optional ``metadata``.

        Args:
            path: Path to the JSON file.
            limit: Optional limit to the number of tasks to load.

        Returns:
            A new queue instance containing the loaded tasks.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        items = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

        if limit is not None:
            items = items[:limit]

        return cls.from_list(items)


class SequentialTaskQueue(BaseTaskQueue):
    """Execute tasks in their original order.

    This queue maintains the current sequential execution model, processing
    tasks in the order they appear in the input iterable. It's the default
    queue used when no explicit queue is provided.

    Example:
        ```python
        queue = SequentialTaskQueue(tasks)
        for task in queue:
            result = execute(task)
        ```
    """

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in original order."""
        return iter(self._tasks)


class InformativeSubsetQueue(SequentialTaskQueue):
    """Evaluates an informative subset of tasks in a specified order.

    Used for efficient evaluation where a carefully selected subset of tasks
    can predict performance on the full dataset. The subset is defined by
    ``indices`` — integer positions into the original task list. Only tasks
    at those positions are yielded, in the order given by ``indices``.

    The informativeness criterion (how the indices were chosen) is determined
    by the caller or by a subclass. This base class is criterion-agnostic.

    When ``indices`` is ``None``, all tasks are yielded in their original
    order (equivalent to ``SequentialTaskQueue``).

    Attributes:
        _all_tasks: The complete, unfiltered task list.
        _indices: The subset indices, or ``None``.

    Example:
        ```python
        # Evaluate only tasks at indices 0, 5, 12
        queue = InformativeSubsetQueue(tasks, indices=[0, 5, 12])

        for task in queue:
            result = execute(task)  # Only 3 tasks
        ```
    """

    def __init__(self, tasks: Iterable[Task], indices: Optional[List[int]] = None) -> None:
        """Initialize informative-subset task queue.

        Args:
            tasks: Full list of tasks (ordered by index).
            indices: Positions into ``tasks`` selecting which tasks to evaluate
                and in what order. If ``None``, evaluates all tasks in order.
        """
        all_tasks = list(tasks)
        self._all_tasks: List[Task] = all_tasks
        self._indices: Optional[List[int]] = indices

        if indices is not None:
            n_tasks = len(all_tasks)
            out_of_range = [idx for idx in indices if idx < 0 or idx >= n_tasks]
            if out_of_range:
                raise IndexError(
                    f"Indices {out_of_range} are out of range for task list of length {n_tasks}. "
                    "This likely means a mismatch between the task data and the index file."
                )
            filtered = [all_tasks[idx] for idx in indices]
            super().__init__(filtered)
        else:
            super().__init__(all_tasks)


class DISCOQueue(InformativeSubsetQueue):
    """Diversity-based informative subset using DISCO anchor points.

    Selects a diverse subset of tasks (anchor points) for evaluation. Full
    benchmark performance is then predicted from results on this subset using
    DISCO (Diversifying Sample Condensation for Efficient Model Evaluation).

    The informativeness criterion is **diversity**: anchor points are chosen
    to maximise disagreement across models, so that a small evaluation set
    captures the discriminative structure of the full benchmark.

    Reference: `DISCO: Diversifying Sample Condensation for Efficient Model
    Evaluation <https://arxiv.org/abs/2510.07959>`_

    Example:
        ```python
        queue = DISCOQueue(tasks, anchor_points=[0, 5, 12])
        # or load from file:
        queue = DISCOQueue(tasks, anchor_points_path="anchor_points.pkl")

        for task in queue:
            result = execute(task)  # Only anchor-point tasks
        ```
    """

    def __init__(
        self,
        tasks: Iterable[Task],
        anchor_points: Optional[List[int]] = None,
        anchor_points_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize DISCO task queue.

        Anchor points can be supplied directly via ``anchor_points`` or loaded
        from a file via ``anchor_points_path``.  Providing both is an error.

        Args:
            tasks: Full list of tasks (ordered by index).
            anchor_points: Diversity-selected indices into ``tasks``.
                Typically downloaded from a HuggingFace DISCO model repo.
                If ``None`` and ``anchor_points_path`` is also ``None``,
                evaluates all tasks in order.
            anchor_points_path: Path to a ``.json`` or ``.pkl`` file
                containing anchor-point indices.  Mutually exclusive with
                ``anchor_points``.
        """
        if anchor_points is not None and anchor_points_path is not None:
            raise ValueError("Provide either anchor_points or anchor_points_path, not both.")

        if anchor_points_path is not None:
            anchor_points = self.load_anchor_points(anchor_points_path)

        self._anchor_points: Optional[List[int]] = anchor_points
        super().__init__(tasks, indices=anchor_points)

    @staticmethod
    def load_anchor_points(path: Union[str, Path]) -> List[int]:
        """Load anchor points from a ``.json`` or ``.pkl`` file.

        Args:
            path: Path to anchor points file. JSON files should contain a
                list of integer indices. Pickle files may contain a list or
                a numpy array.

        Returns:
            List of integer anchor-point indices.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Anchor points file not found: {path}")

        if path.suffix.lower() == ".json":
            with open(path) as f:
                anchor_points = json.load(f)
        else:
            with open(path, "rb") as f:
                anchor_points = pickle.load(f)

        if hasattr(anchor_points, "tolist"):
            anchor_points = anchor_points.tolist()

        return list(anchor_points)


class PriorityTaskQueue(BaseTaskQueue):
    """Execute tasks ordered by priority.

    Tasks are sorted by ``task.protocol.priority`` at construction time.
    Higher priority values are executed first by default. Tasks with equal
    priority maintain their relative order from the original input (stable sort).

    This queue uses ``task.protocol.priority`` as the sole source of priority.
    Pre-compute priority values and assign them to tasks before creating the queue.

    Args:
        tasks: An iterable of Task objects to schedule.
        reverse: If True (default), higher priority values execute first.
            If False, lower priority values execute first.

    Example:
        ```python
        # Assign priorities based on your criteria
        for task in tasks:
            task.protocol.priority = compute_priority(task)

        # Create queue (higher priority first)
        queue = PriorityTaskQueue(tasks)

        # Or lower priority first
        queue = PriorityTaskQueue(tasks, reverse=False)
        ```
    """

    def __init__(self, tasks: Iterable[Task], reverse: bool = True) -> None:
        """Initialize priority queue with sorted tasks.

        Args:
            tasks: An iterable of Task objects to schedule.
            reverse: If True (default), higher priority values execute first.
        """
        task_list = list(tasks)
        # Stable sort by priority
        sorted_tasks = sorted(task_list, key=lambda t: t.protocol.priority, reverse=reverse)
        super().__init__(sorted_tasks)

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in priority order."""
        return iter(self._tasks)


class AdaptiveTaskQueue(BaseTaskQueue, BenchmarkCallback, ABC):
    """Abstract base class for adaptive task scheduling.

    AdaptiveTaskQueue enables dynamic task ordering based on execution results.
    It inherits from BenchmarkCallback to integrate with the benchmark's callback
    system, creating a clean bidirectional communication model:

    - **Benchmark → Queue**: Via iterator protocol (``for task in queue``)
    - **Queue → Benchmark**: Via callback (``on_task_repeat_end()``)

    The queue automatically moves completed tasks from ``_remaining`` to
    ``_completed`` and calls ``update_state()`` to let subclasses adapt their
    scheduling strategy based on task results.

    Subclasses must implement:
        - ``initial_state()``: Return initial state dict for adaptive algorithm
        - ``select_next_task(remaining, state)``: Choose the next task to execute
        - ``update_state(task, report, state)``: Update and return new state

    The state dict is managed by the base class: initialized via ``initial_state()``
    at iteration start, passed to both methods, and updated from ``update_state()``
    return value. This functional approach keeps state flow explicit while allowing
    subclasses to store any data they need.

    Internal state (managed by base class, do not modify directly):
        - ``_remaining``: Tasks not yet executed
        - ``_completed``: Completed tasks paired with their reports
        - ``_state``: Current adaptive state dict
        - ``_stop_flag``: Flag to signal early termination

    When used with ``Benchmark.run()``, the queue is automatically registered
    as a callback and receives ``on_task_repeat_end()`` notifications.

    Example:
        ```python
        class IRTTaskQueue(AdaptiveTaskQueue):
            '''Item Response Theory-based adaptive testing.'''

            def initial_state(self) -> Dict[str, Any]:
                return {"ability": 0.0}

            def select_next_task(
                self, remaining: Sequence[Task], state: Dict[str, Any]
            ) -> Optional[Task]:
                # Select task with difficulty closest to current ability estimate
                return min(
                    remaining,
                    key=lambda t: abs(t.metadata.get("difficulty", 0) - state["ability"])
                )

            def update_state(
                self, task: Task, report: Dict[str, Any], state: Dict[str, Any]
            ) -> Dict[str, Any]:
                # Update ability estimate based on task result
                correct = report.get("eval", [{}])[0].get("correct", False)
                return {"ability": state["ability"] + (0.5 if correct else -0.5)}

        queue = IRTTaskQueue(tasks)
        results = benchmark.run(queue)  # Auto-registered as callback
        ```
    """

    def __init__(self, tasks: Iterable[Task]) -> None:
        """Initialize adaptive queue.

        Args:
            tasks: An iterable of Task objects to schedule.
        """
        super().__init__(tasks)
        self._remaining: List[Task] = list(self._tasks)
        self._completed: List[Tuple[Task, Dict[str, Any]]] = []
        self._stop_flag: bool = False
        self._state: Dict[str, Any] = {}

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks selected by the adaptive algorithm.

        Initializes state via ``initial_state()`` at iteration start, then
        continues until ``select_next_task()`` returns None, ``_remaining``
        is empty, or ``stop()`` is called.

        Note: ``select_next_task()`` is only called when ``_remaining`` is
        non-empty, so implementers don't need to check for empty list.
        """
        self._state = self.initial_state()
        while not self._stop_flag and self._remaining:
            next_task = self.select_next_task(self._remaining, self._state)
            if next_task is not None:
                yield next_task
            else:
                break

    def on_task_repeat_end(self, benchmark: "Benchmark", report: Dict[str, Any]) -> None:
        """BenchmarkCallback hook called after each task repetition completes.

        This method extracts the task from the report, moves it from
        ``_remaining`` to ``_completed``, and calls ``update_state()``
        to let the subclass update its adaptive model.

        Args:
            benchmark: The benchmark instance (unused in this implementation).
            report: The execution report containing task_id and results.
        """
        # Extract task from report
        task_id_str = report.get("task_id")
        if task_id_str is None:
            return

        # Find the task in remaining list
        task = None
        for i, t in enumerate(self._remaining):
            if str(t.id) == task_id_str:
                task = self._remaining.pop(i)
                self._completed.append((task, report))
                break

        # If not found in remaining, check completed (for n_task_repeats > 1)
        if task is None:
            for t, _ in self._completed:
                if str(t.id) == task_id_str:
                    task = t
                    break

        # Update subclass state
        if task is not None:
            self._state = self.update_state(task, report, self._state)

    def stop(self) -> None:
        """Signal that no more tasks should be processed.

        Call this from ``update_state()`` to trigger early termination
        (e.g., when confidence threshold is reached).

        The ``_stop_flag`` is checked in ``__iter__``, which will stop yielding
        tasks and naturally terminate the benchmark's iteration loop via Python's
        iterator protocol.
        """
        self._stop_flag = True

    @abstractmethod
    def initial_state(self) -> Dict[str, Any]:
        """Return the initial state for adaptive selection.

        This state dict will be passed to ``select_next_task()`` and
        ``update_state()`` throughout the benchmark run. Store any data
        your adaptive algorithm needs (ability estimates, history, etc.).

        Returns:
            Initial state dict. Can contain any keys/values you need.
        """
        pass

    @abstractmethod
    def select_next_task(self, remaining: Sequence[Task], state: Dict[str, Any]) -> Optional[Task]:
        """Select the next task to execute.

        Implement this method to define your adaptive selection algorithm
        (e.g., IRT-based selection, uncertainty sampling, bandit algorithms).

        Args:
            remaining: Read-only sequence of tasks not yet executed.
                Do not modify this sequence; the queue manages task lifecycle.
            state: Current adaptive state from ``initial_state()`` or
                ``update_state()``.

        Returns:
            The next Task to execute from ``remaining``, or None to
            signal early termination.

        Note:
            This method is only called when ``remaining`` is non-empty,
            so you don't need to check for an empty sequence.
        """
        pass

    @abstractmethod
    def update_state(self, task: Task, report: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state after task completion.

        Implement this method to update ability estimates, difficulty models,
        or other adaptive state based on task results.

        Args:
            task: The task that just completed.
            report: The execution report containing status and eval results.
            state: Current state dict.

        Returns:
            Updated state dict (can be the same dict mutated, or a new dict).

        Note:
            Call ``self.stop()`` here to halt iteration before the next
            task selection.
        """
        pass


# Alias for the default queue type
TaskQueue = SequentialTaskQueue
