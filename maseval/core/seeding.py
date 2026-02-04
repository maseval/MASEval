"""Seed generation infrastructure for reproducible benchmark runs.

This module provides the `SeedGenerator` abstract base class and `DefaultSeedGenerator`
implementation for deriving deterministic seeds across benchmark components. Seeds are
derived from hierarchical paths (e.g., "agents/experimental") ensuring that adding or
removing components doesn't affect seeds for other components.

Example:
    ```python
    from maseval import Benchmark, DefaultSeedGenerator

    # Simple usage - just pass a seed
    benchmark = MyBenchmark(seed=42)

    # Custom generator
    generator = DefaultSeedGenerator(global_seed=42)
    benchmark = MyBenchmark(seed_generator=generator)

    # Disable seeding (derive_seed returns None)
    benchmark = MyBenchmark(seed=None)

    # In setup methods, derive seeds for components using hierarchical paths
    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        # Use child() for hierarchical namespacing - creates paths like "agents/orchestrator"
        agent_gen = seed_generator.child("agents")
        orchestrator_seed = agent_gen.derive_seed("orchestrator")  # Optional[int]

        # per_repetition controls variance across repetitions
        experimental_seed = agent_gen.derive_seed("experimental", per_repetition=True)
        baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)

        # Pass seed directly to your agent
        agent = MyAgent(seed=orchestrator_seed)
    ```
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from typing_extensions import Self
import hashlib

from .config import ConfigurableMixin


class SeedingError(Exception):
    """Raised when seeding is misconfigured or unsupported.

    Examples:
        - Calling `derive_seed()` before `for_task()` sets the task context
        - Calling `derive_seed(per_repetition=True)` before `for_repetition()` sets the rep context
        - Model adapter doesn't support seeding but seed was provided
    """

    pass


class SeedGenerator(ABC, ConfigurableMixin):
    """Abstract base class for seed generation.

    Subclass this to implement custom seeding strategies (e.g., database lookup,
    different hash algorithms, external seed services).

    The default implementation is `DefaultSeedGenerator`, which uses SHA-256 hashing
    and provides additional convenience methods like `child()` for hierarchical namespacing.

    Implementations must be thread-safe. The recommended pattern is to create isolated
    state in `for_task()` so each task repetition has its own generator instance.

    When `global_seed` is `None`, seeding is disabled and `derive_seed()` returns `None`.
    This allows seeds to flow through to model adapters which already accept `Optional[int]`.

    Subclasses must implement:

    - `global_seed` - Property returning the root seed (or `None` if disabled)
    - `derive_seed()` - Derive a seed for a named component (returns `None` if disabled)
    - `for_task()` - Create a generator scoped to a specific task
    - `for_repetition()` - Create a generator scoped to a specific repetition
    - `seed_log` - Property returning all derived seeds
    """

    @property
    @abstractmethod
    def global_seed(self) -> Optional[int]:
        """Root seed for the entire benchmark run, or `None` if seeding is disabled."""
        pass

    @abstractmethod
    def derive_seed(self, name: str, per_repetition: bool = True) -> Optional[int]:
        """Derive a seed for a named component.

        Args:
            name: Component identifier (e.g., "agent_x", "environment/tool_weather").
                Use "/" for hierarchical paths if desired.
            per_repetition: If True, seed varies per repetition. If False,
                seed is constant across repetitions of the same task.

        Returns:
            Deterministic seed derived from the generator's state and name,
            or `None` if `global_seed` is `None` (seeding disabled).

        Raises:
            SeedingError: If `global_seed` is set but required context (task_id, rep_index)
                is not set.
        """
        pass

    @abstractmethod
    def for_task(self, task_id: str) -> Self:
        """Create a generator scoped to a specific task.

        This creates a fresh generator with its own seed log. Call this once
        per task to isolate seed tracking between tasks.

        Args:
            task_id: The task identifier.

        Returns:
            New generator with task_id set and fresh log.
        """
        pass

    @abstractmethod
    def for_repetition(self, rep_index: int) -> Self:
        """Create a generator scoped to a specific repetition.

        Args:
            rep_index: The repetition index (0-based).

        Returns:
            New generator with rep_index set, preserving task scope and log.
        """
        pass

    @property
    @abstractmethod
    def seed_log(self) -> Dict[str, int]:
        """Return all seeds derived by this generator and its children.

        Returns:
            Dictionary mapping component paths to their derived seeds.
        """
        pass

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration for tracing integration.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `seeds` - Dictionary of all derived seeds

        Returns:
            Dictionary containing seed generator configuration.
        """
        return {
            **super().gather_config(),
            "seeds": self.seed_log,
        }


class DefaultSeedGenerator(SeedGenerator):
    """Default hash-based seed generator using SHA-256.

    Derives deterministic seeds from hierarchical paths. Thread-safe and immutable
    after construction (derives seeds via pure functions).

    When `global_seed` is `None`, seeding is disabled and `derive_seed()` returns `None`.
    This allows seeds to flow directly to model adapters which already accept `Optional[int]`.

    Provides `child()` method for hierarchical namespacing (not part of the ABC).
    Override `_compute_seed()` to customize the hash algorithm while keeping
    the rest of the infrastructure (logging, scoping, validation).

    All child generators share the same seed log for unified tracking.

    Example:
        ```python
        # Create root generator with seeding enabled
        gen = DefaultSeedGenerator(global_seed=42)

        # Or disable seeding (derive_seed returns None)
        gen = DefaultSeedGenerator(global_seed=None)

        # Scope to task and repetition
        task_gen = gen.for_task("task_001").for_repetition(0)

        # Use child() for hierarchical namespacing - creates paths like "agents/orchestrator"
        agent_gen = task_gen.child("agents")
        orchestrator_seed = agent_gen.derive_seed("orchestrator")  # Path: "agents/orchestrator"
        baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)  # Constant

        # Pass seed directly to your agent
        agent = MyAgent(seed=orchestrator_seed)  # Works with None
        ```

    Thread Safety:
        DefaultSeedGenerator is thread-safe by design through isolation:

        1. **Isolated logs per task**: `for_task()` creates a fresh seed log, so
           different tasks running in parallel threads don't share state.

        2. **Root generator is read-only**: The root generator only stores the
           global_seed and is never mutated after construction.

        3. **Children share parent's log**: Within a task, `child()` and
           `for_repetition()` share the same seed log. This is safe because
           a single task/repetition runs in a single thread.

        Parallel execution example::

            Thread 1 (task A, rep 0):
              task_gen_A = root.for_task("A").for_repetition(0)  # Fresh log
              child = task_gen_A.child("env")                     # Shares task_gen_A's log

            Thread 2 (task B, rep 0):
              task_gen_B = root.for_task("B").for_repetition(0)  # Different fresh log
              child = task_gen_B.child("env")                     # Shares task_gen_B's log

        If implementing a custom SeedGenerator subclass, ensure similar thread
        isolation by creating fresh state in `for_task()`.
    """

    def __init__(
        self,
        global_seed: Optional[int] = None,
        task_id: Optional[str] = None,
        rep_index: Optional[int] = None,
        path_prefix: str = "",
        _shared_log: Optional[Dict[str, int]] = None,
    ):
        """Initialize a DefaultSeedGenerator.

        Args:
            global_seed: Root seed for the entire benchmark run, or `None` to disable seeding.
                When `None`, `derive_seed()` returns `None` for all components.
            task_id: Current task identifier (set via `for_task()`).
            rep_index: Current repetition index (set via `for_repetition()`).
            path_prefix: Accumulated path from parent generators.
            _shared_log: Internal. Shared dict for logging all derived seeds.
                Do not pass this directly; it's managed by `child()` and `for_task()`.
        """
        super().__init__()
        self._global_seed = global_seed
        self._task_id = task_id
        self._rep_index = rep_index
        self._path_prefix = path_prefix
        self._shared_log = _shared_log if _shared_log is not None else {}

    @property
    def global_seed(self) -> Optional[int]:
        """Root seed for the entire benchmark run, or `None` if seeding is disabled."""
        return self._global_seed

    def derive_seed(self, name: str, per_repetition: bool = True) -> Optional[int]:
        """Derive a seed for a named component.

        Args:
            name: Component identifier (e.g., "agent_x", "tool_weather").
            per_repetition: If True, seed varies per repetition. If False,
                seed is constant across repetitions of the same task.

        Returns:
            Deterministic seed derived from (global_seed, task_id, [rep_index], path, name),
            or `None` if `global_seed` is `None` (seeding disabled).

        Raises:
            SeedingError: If `global_seed` is set but task_id is not set (call `for_task()` first),
                or if `per_repetition=True` and rep_index is not set.
        """
        # If seeding is disabled, return None without validation
        if self._global_seed is None:
            return None

        if self._task_id is None:
            raise SeedingError("task_id not set. Call for_task() first.")
        if per_repetition and self._rep_index is None:
            raise SeedingError("rep_index not set. Call for_repetition() first.")

        full_path = f"{self._path_prefix}/{name}" if self._path_prefix else name

        # Build components for seed computation
        if per_repetition:
            components = [self._global_seed, self._task_id, self._rep_index, full_path]
        else:
            components = [self._global_seed, self._task_id, full_path]

        seed = self._compute_seed(full_path, components)

        # Log with full path
        self._shared_log[full_path] = seed

        return seed

    def _compute_seed(self, full_path: str, components: List[Any]) -> int:
        """Compute seed from components using SHA-256.

        Override this method to use a different hash algorithm while keeping
        the rest of the DefaultSeedGenerator infrastructure.

        Args:
            full_path: The full hierarchical path (for reference, already in components).
            components: List of [global_seed, task_id, [rep_index], path] to hash.

        Returns:
            Integer seed in range [0, 2^31 - 1].
        """
        seed_string = ":".join(str(c) for c in components)
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF

    def child(self, name: str) -> Self:
        """Create a child generator scoped to a component.

        The child inherits context (global_seed, task_id, rep_index) and extends
        the path. All children share the same seed log.

        This method is specific to DefaultSeedGenerator and not part of the ABC.

        Args:
            name: Component name to add to the path.

        Returns:
            New DefaultSeedGenerator with extended path, sharing the same log.

        Example:
            ```python
            env_gen = seed_generator.child("environment")
            tools_gen = env_gen.child("tools")
            weather_seed = tools_gen.derive_seed("weather")  # Path: "environment/tools/weather"
            ```
        """
        new_path = f"{self._path_prefix}/{name}" if self._path_prefix else name
        return self.__class__(
            global_seed=self._global_seed,
            task_id=self._task_id,
            rep_index=self._rep_index,
            path_prefix=new_path,
            _shared_log=self._shared_log,
        )

    def for_task(self, task_id: str) -> Self:
        """Create a generator scoped to a specific task.

        Creates a fresh seed log for this task. Each task should have its own
        generator created via this method.

        Args:
            task_id: The task identifier.

        Returns:
            New DefaultSeedGenerator with task_id set and fresh log.
        """
        return self.__class__(
            global_seed=self._global_seed,
            task_id=task_id,
            rep_index=None,
            path_prefix="",
            _shared_log={},
        )

    def for_repetition(self, rep_index: int) -> Self:
        """Create a generator scoped to a specific repetition.

        Preserves the task scope and seed log. Seeds derived with
        `per_repetition=True` will include the rep_index in the hash.

        Args:
            rep_index: The repetition index (0-based).

        Returns:
            New DefaultSeedGenerator with rep_index set, preserving task scope and log.
        """
        return self.__class__(
            global_seed=self._global_seed,
            task_id=self._task_id,
            rep_index=rep_index,
            path_prefix=self._path_prefix,
            _shared_log=self._shared_log,
        )

    @property
    def seed_log(self) -> Dict[str, int]:
        """Return all seeds derived by this generator and its children.

        Returns:
            Copy of the seed log dictionary for safety.
        """
        return dict(self._shared_log)

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration for tracing integration.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `global_seed` - The root seed (or `None` if disabled)
        - `task_id` - Current task ID (if set)
        - `rep_index` - Current repetition index (if set)
        - `seeds` - Dictionary of all derived seeds (empty if seeding disabled)

        Returns:
            Dictionary containing seed generator configuration.
        """
        return {
            **super().gather_config(),
            "global_seed": self._global_seed,
            "task_id": self._task_id,
            "rep_index": self._rep_index,
        }
