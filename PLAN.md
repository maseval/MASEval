# Seeding System Implementation Plan

## Overview

Implement Option B from IDEA.md with child generator support from Option C. The system provides an abstract `SeedGenerator` base class that defines the interface, plus a `DefaultSeedGenerator` concrete implementation that derives seeds via SHA-256 hashing.

This follows the project's established pattern of "abstract base classes with optional default implementations" (see README.md).

## Design Goals

From IDEA.md Section 4:

1. **Opt-in** — Users who don't need seeding shouldn't be affected
2. **Derive by name, not index** — Seeds from identifiers, never positions
3. **Support repetitions** — Each repetition gets a different seed
4. **Selective variance** — Some components constant across reps, others vary
5. **Logging** — Seeds used must be recorded in results
6. **Fail explicitly** — Providers without seed support should error
7. **Extensible** — Future use cases (task queues, etc.) should work
8. **Easy to customize** — Users can implement custom seeding strategies by subclassing

## API Design

### Core Classes

```python
# maseval/core/seeding.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Self
import hashlib

from maseval.core.config import ConfigurableMixin


class SeedGenerator(ABC, ConfigurableMixin):
    """Abstract base class for seed generation.

    Subclass this to implement custom seeding strategies (e.g., database lookup,
    different hash algorithms, external seed services).

    The default implementation is `DefaultSeedGenerator`, which uses SHA-256 hashing
    and provides additional convenience methods like `child()` for hierarchical namespacing.
    """

    @property
    @abstractmethod
    def global_seed(self) -> int:
        """Root seed for the entire benchmark run."""
        pass

    @abstractmethod
    def derive_seed(self, name: str, per_repetition: bool = True) -> int:
        """Derive a seed for a named component.

        Args:
            name: Component identifier (e.g., "agent_x", "environment/tool_weather").
                Use "/" for hierarchical paths if desired.
            per_repetition: If True, seed varies per repetition. If False,
                seed is constant across repetitions of the same task.

        Returns:
            Deterministic seed derived from the generator's state and name.

        Raises:
            SeedingError: If required context (task_id, rep_index) is not set.
        """
        pass

    @abstractmethod
    def for_task(self, task_id: str) -> Self:
        """Create a generator scoped to a specific task.

        Returns:
            New generator with task_id set and fresh log.
        """
        pass

    @abstractmethod
    def for_repetition(self, rep_index: int) -> Self:
        """Create a generator scoped to a specific repetition.

        Returns:
            New generator with rep_index set, preserving task scope and log.
        """
        pass

    @property
    @abstractmethod
    def seed_log(self) -> Dict[str, int]:
        """Return all seeds derived by this generator and its children."""
        pass

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration for tracing integration."""
        return {
            **super().gather_config(),
            "seeds": self.seed_log,
        }


class DefaultSeedGenerator(SeedGenerator):
    """Default hash-based seed generator using SHA-256.

    Derives deterministic seeds from hierarchical paths. Thread-safe and immutable
    after construction (derives seeds via pure functions).

    Provides `child()` method for hierarchical namespacing (not part of the ABC).
    Override `_compute_seed()` to customize the hash algorithm while keeping
    the rest of the infrastructure (logging, scoping, validation).

    All child generators share the same seed log for unified tracking.
    """

    def __init__(
        self,
        global_seed: int,
        task_id: Optional[str] = None,
        rep_index: Optional[int] = None,
        path_prefix: str = "",
        _shared_log: Optional[Dict[str, int]] = None,  # Internal: shared across children
    ):
        """
        Args:
            global_seed: Root seed for the entire benchmark run.
            task_id: Current task identifier (set when entering task scope).
            rep_index: Current repetition index (set when entering rep scope).
            path_prefix: Accumulated path from parent generators.
            _shared_log: Internal. Shared dict for logging all derived seeds.
        """
        super().__init__()
        self._global_seed = global_seed
        self._task_id = task_id
        self._rep_index = rep_index
        self._path_prefix = path_prefix
        self._shared_log = _shared_log if _shared_log is not None else {}

    @property
    def global_seed(self) -> int:
        """Root seed for the entire benchmark run."""
        return self._global_seed

    def derive_seed(self, name: str, per_repetition: bool = True) -> int:
        """Derive a seed for a named component.

        Args:
            name: Component identifier (e.g., "agent_x", "tool_weather").
            per_repetition: If True, seed varies per repetition. If False,
                seed is constant across repetitions of the same task.

        Returns:
            Deterministic seed derived from (global_seed, task_id, [rep_index], path, name).

        Raises:
            SeedingError: If task_id is not set (call for_task() first).
        """
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

    def _compute_seed(self, full_path: str, components: list) -> int:
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

        Args:
            name: Component name to add to the path.

        Returns:
            New DefaultSeedGenerator with extended path, sharing the same log.
        """
        new_path = f"{self._path_prefix}/{name}" if self._path_prefix else name
        return self.__class__(
            global_seed=self._global_seed,
            task_id=self._task_id,
            rep_index=self._rep_index,
            path_prefix=new_path,
            _shared_log=self._shared_log,  # Share the log
        )

    def for_task(self, task_id: str) -> Self:
        """Create a generator scoped to a specific task.

        Returns:
            New DefaultSeedGenerator with task_id set and fresh log.
        """
        return self.__class__(
            global_seed=self._global_seed,
            task_id=task_id,
            rep_index=None,
            path_prefix="",
            _shared_log={},  # Fresh log for this task
        )

    def for_repetition(self, rep_index: int) -> Self:
        """Create a generator scoped to a specific repetition.

        Returns:
            New DefaultSeedGenerator with rep_index set, preserving task scope and log.
        """
        return self.__class__(
            global_seed=self._global_seed,
            task_id=self._task_id,
            rep_index=rep_index,
            path_prefix=self._path_prefix,
            _shared_log=self._shared_log,  # Preserve the log
        )

    @property
    def seed_log(self) -> Dict[str, int]:
        """Return all seeds derived by this generator and its children."""
        return dict(self._shared_log)  # Return copy for safety

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration for tracing integration."""
        return {
            **super().gather_config(),
            "global_seed": self._global_seed,
            "task_id": self._task_id,
            "rep_index": self._rep_index,
        }


class SeedingError(Exception):
    """Raised when seeding is misconfigured or unsupported."""
    pass
```

### Extension Patterns

The ABC + default implementation pattern makes it easy for users to customize:

| Use Case                   | What to Do                                                  |
| -------------------------- | ----------------------------------------------------------- |
| Different hash algorithm   | Subclass `DefaultSeedGenerator`, override `_compute_seed()` |
| Different logging strategy | Subclass `DefaultSeedGenerator`, override `derive_seed()`   |
| Need `child()` hierarchy   | Use or subclass `DefaultSeedGenerator`                      |
| Seed lookup from database  | Implement `SeedGenerator` ABC directly (5 methods)          |
| External seed service      | Implement `SeedGenerator` ABC directly (5 methods)          |

**Example: Custom hash algorithm**

```python
class MD5SeedGenerator(DefaultSeedGenerator):
    """Uses MD5 instead of SHA-256 (not recommended, just for illustration)."""

    def _compute_seed(self, full_path: str, components: list) -> int:
        seed_string = ":".join(str(c) for c in components)
        hash_bytes = hashlib.md5(seed_string.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF
```

**Example: Database-backed seeds**

```python
class DatabaseSeedGenerator(SeedGenerator):
    """Looks up seeds from a database for exact reproducibility."""

    def __init__(self, db_connection, run_id: str):
        super().__init__()
        self._db = db_connection
        self._run_id = run_id
        self._task_id = None
        self._rep_index = None
        self._log = {}

    @property
    def global_seed(self) -> int:
        return self._db.get_run_seed(self._run_id)

    def derive_seed(self, name: str, per_repetition: bool = True) -> int:
        key = (self._run_id, self._task_id, self._rep_index if per_repetition else None, name)
        seed = self._db.get_or_create_seed(key)
        self._log[name] = seed
        return seed

    def for_task(self, task_id: str) -> Self:
        new_gen = DatabaseSeedGenerator(self._db, self._run_id)
        new_gen._task_id = task_id
        return new_gen

    def for_repetition(self, rep_index: int) -> Self:
        new_gen = DatabaseSeedGenerator(self._db, self._run_id)
        new_gen._task_id = self._task_id
        new_gen._rep_index = rep_index
        new_gen._log = self._log  # Share log within task
        return new_gen

    @property
    def seed_log(self) -> Dict[str, int]:
        return dict(self._log)
```

### Integration with Benchmark

```python
class Benchmark(ABC):
    def __init__(
        self,
        ...,
        seed: Optional[int] = None,  # NEW: global seed for reproducibility
        seed_generator: Optional[SeedGenerator] = None,  # NEW: custom generator
    ):
        # Users can provide either a seed (uses DefaultSeedGenerator) or a custom generator
        if seed_generator is not None:
            self._seed_generator = seed_generator
        elif seed is not None:
            self._seed_generator = DefaultSeedGenerator(global_seed=seed)
        else:
            self._seed_generator = None

    @property
    def seed_generator(self) -> Optional[SeedGenerator]:
        """Current seed generator, or None if seeding disabled."""
        return self._seed_generator

    def _execute_task_repetition(self, task, agent_data, repeat_idx):
        # Create scoped generator for this task+rep
        if self._seed_generator is not None:
            task_gen = self._seed_generator.for_task(str(task.id)).for_repetition(repeat_idx)
            # Register for automatic config collection
            self.register("seeding", "seed_generator", task_gen)
        else:
            task_gen = None

        # Pass to setup methods (NEW parameter)
        environment = self.setup_environment(agent_data, task, seed_generator=task_gen)
        user = self.setup_user(agent_data, environment, task, seed_generator=task_gen)
        agents = self.setup_agents(agent_data, environment, task, user, seed_generator=task_gen)
        evaluators = self.setup_evaluators(environment, task, agents, user, seed_generator=task_gen)

        # ... rest of execution ...

        # Seeds automatically included via collect_all_configs()
        # Available at report["config"]["seeding"]["seed_generator"]["seeds"]
```

### Updated Setup Method Signatures

```python
# All setup methods gain an optional seed_generator parameter

@abstractmethod
def setup_environment(
    self,
    agent_data: Dict[str, Any],
    task: Task,
    seed_generator: Optional[SeedGenerator] = None,  # NEW
) -> Environment:
    pass

def setup_user(
    self,
    agent_data: Dict[str, Any],
    environment: Environment,
    task: Task,
    seed_generator: Optional[SeedGenerator] = None,  # NEW
) -> Optional[User]:
    pass

@abstractmethod
def setup_agents(
    self,
    agent_data: Dict[str, Any],
    environment: Environment,
    task: Task,
    user: Optional[User],
    seed_generator: Optional[SeedGenerator] = None,  # NEW
) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
    pass

@abstractmethod
def setup_evaluators(
    self,
    environment: Environment,
    task: Task,
    agents: Sequence[AgentAdapter],
    user: Optional[User],
    seed_generator: Optional[SeedGenerator] = None,  # NEW
) -> Sequence[Evaluator]:
    pass
```

### Updated get_model_adapter

```python
@abstractmethod
def get_model_adapter(
    self,
    model_id: str,
    seed: Optional[int] = None,  # NEW: seed for this model instance
    **kwargs,
) -> ModelAdapter:
    """
    Args:
        model_id: Model identifier.
        seed: Seed for deterministic generation. Passed to ModelAdapter.__init__.
            If the provider doesn't support seeding, adapter raises SeedingError.
        **kwargs: Registration info, etc.
    """
    pass

# Example implementation:
def get_model_adapter(self, model_id: str, seed: Optional[int] = None, **kwargs) -> ModelAdapter:
    adapter = GoogleGenAIModelAdapter(self.client, model_id=model_id, seed=seed)
    if "register_name" in kwargs:
        self.register("models", kwargs["register_name"], adapter)
    return adapter
```

## Usage Examples

### Basic Usage (Default Generator)

```python
class MyBenchmark(Benchmark):
    def setup_environment(self, agent_data, task, seed_generator=None):
        env = MyEnvironment(task.environment_data)

        if seed_generator is not None:
            # Option 1: Use child() for hierarchical namespacing (DefaultSeedGenerator only)
            env_gen = seed_generator.child("environment")
            for tool in env.tools:
                tool_seed = env_gen.derive_seed(tool.name)
                tool_model = self.get_model_adapter(model_id, seed=tool_seed)
                tool.set_simulator(tool_model)

            # Option 2: Use flat paths (works with any SeedGenerator)
            # for tool in env.tools:
            #     tool_seed = seed_generator.derive_seed(f"environment/{tool.name}")
            #     ...

        return env

    def setup_user(self, agent_data, environment, task, seed_generator=None):
        user_seed = None
        if seed_generator is not None:
            # Use child() to create logical namespace - results in "simulators/user"
            sim_gen = seed_generator.child("simulators")
            user_seed = sim_gen.derive_seed("user")

        user_model = self.get_model_adapter(model_id, seed=user_seed)
        return MyUser(model=user_model, ...)

    def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
        if seed_generator is not None:
            # Using child() for cleaner namespacing
            agent_gen = seed_generator.child("agents")
            # Vary experimental agent per rep, keep baseline constant
            experimental_seed = agent_gen.derive_seed("experimental", per_repetition=True)
            baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)
        else:
            experimental_seed = None
            baseline_seed = None

        # ... create agents with seeds ...

# Run with seeding (uses DefaultSeedGenerator internally)
benchmark = MyBenchmark(seed=42, n_task_repeats=3)
results = benchmark.run(tasks=tasks, agent_data=config)

# Results include seed log (via ConfigurableMixin integration)
for report in results:
    seed_config = report["config"]["seeding"]["seed_generator"]
    print(seed_config["global_seed"])  # 42
    print(seed_config["seeds"])
    # {"environment/tools/weather": 12345, "simulators/user": 67890, "agents/experimental": ...}
```

### Custom Generator

```python
# Use a custom generator for specialized seeding strategies
custom_gen = MyDatabaseSeedGenerator(db_connection, run_id="experiment_42")
benchmark = MyBenchmark(seed_generator=custom_gen, n_task_repeats=3)
results = benchmark.run(tasks=tasks, agent_data=config)
```

## Thread Safety

The `DefaultSeedGenerator` is thread-safe by design:

1. **Isolated logs per task repetition**: `for_task()` creates a fresh `_shared_log` dict. Each task repetition has its own log that isn't shared across threads.

2. **No cross-thread sharing**: The root `self._seed_generator` is read-only (only used to call `for_task()`). All mutable state (the log) lives in the task-scoped generator.

3. **Children share parent's log**: Within a single task repetition, `child()` generators (in `DefaultSeedGenerator`) share the same `_shared_log`. This is safe because a single task repetition runs in a single thread.

```
Thread 1 (task A, rep 0):
  task_gen_A = root.for_task("A").for_repetition(0)  # Fresh log
  child = task_gen_A.child("env")                     # Shares task_gen_A's log

Thread 2 (task B, rep 0):
  task_gen_B = root.for_task("B").for_repetition(0)  # Different fresh log
  child = task_gen_B.child("env")                     # Shares task_gen_B's log

# No cross-thread sharing of mutable state
```

**Note for custom implementations:** If you implement your own `SeedGenerator`, ensure similar thread isolation by creating fresh state in `for_task()`.

## Awkward Patterns / Open Questions

### 1. ~~Breaking Change to Setup Method Signatures~~ (Not a concern)

Per AGENTS.md: "The library is in early development, so breaking changes that are parsimonous are strongly preferred" and "We have zero obligation to maintain backwards compatibility."

**Decision:** Add `seed_generator: Optional[SeedGenerator] = None` parameter to all setup methods. This is the clean, explicit approach consistent with how other context (agent_data, task, environment, user) is already passed.

### 2. ModelAdapter Seed Support

Not all providers support seeding. The current `ModelAdapter` interface doesn't have seed support.

**Decision:** Add `seed` as a first-class parameter to `ModelAdapter.__init__`. This is explicit and avoids collision with user-provided `generation_params`.

```python
class ModelAdapter(ABC, TraceableMixin, ConfigurableMixin):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self._seed = seed
        self.logs: List[Dict[str, Any]] = []

    @property
    def seed(self) -> Optional[int]:
        """Seed for deterministic generation, or None if unseeded."""
        return self._seed
```

Each concrete adapter:

- Passes `self._seed` to the provider API in `_chat_impl`
- Raises `SeedingError` if seed is provided but provider doesn't support it
- Logs the seed in `gather_config()`

User overrides via `generation_params` always take precedence over the adapter's seed.

### 3. ~~Framework-Level Seeding~~ (Not a concern)

Users instantiate their own agent framework models. MASEval provides the `seed_generator` utility; users extract seeds and pass them to their framework's model constructors. This is the correct separation of concerns.

### 4. ~~Seed Logging Granularity~~ (Resolved)

**Decision:** Log all derivations. All children share the parent's `_shared_log`, so every `derive_seed()` call is recorded automatically. Enables debugging and reproducibility verification.

### 5. ~~Repetition Index vs. Repetition Seed~~ (Resolved)

**Decision:** Internal implementation detail. The `_compute_seed()` method handles both `per_repetition=True` (includes rep_index) and `per_repetition=False` (excludes rep_index) correctly:

```python
if per_repetition:
    components = [global_seed, task_id, rep_index, full_path]
else:
    components = [global_seed, task_id, full_path]  # No rep_index
```

This allows components to request either behavior within the same task repetition.

## Implementation Steps

### Phase 1: Core Infrastructure ✅ COMPLETE

1. ✅ Create `maseval/core/seeding.py` with `SeedGenerator` (ABC), `DefaultSeedGenerator`, and `SeedingError`
2. ✅ Add unit tests for seed derivation, thread safety, and extension patterns
3. ✅ Add `seed` and `seed_generator` parameters to `Benchmark.__init__`

### Phase 2: Benchmark Integration ✅ COMPLETE

4. ✅ Update `_execute_task_repetition` to create scoped generators
5. ✅ Add `seed_generator` parameter to all setup methods
6. ✅ Log seeds in `execution_configs`
7. ✅ Update existing benchmark implementations (MACS, Tau2) to use seeds

### Phase 3: Model Adapter Support ✅ COMPLETE

8. ✅ Add `seed` parameter to `ModelAdapter.__init__`
9. ✅ Update interface adapters to pass seed to provider APIs
10. ✅ Raise `SeedingError` for providers that don't support seeding

### Phase 4: Documentation & Examples ✅ COMPLETE

11. ✅ Update AGENTS.md with seeding guidance
12. ✅ Updated five_a_day example to use benchmark's seeding system (`Benchmark(seed=...)`)
    - Removed local `derive_seed()` utility from utils.py
    - Updated `setup_agents()` to build seeds dict and pass to builders (clean separation from agent specs)
    - All builder functions accept `seeds: Optional[Dict[str, int]]` as separate parameter
    - Updated notebook to use new seeding system with same clean pattern
13. Deferred: Add seeding section to documentation (docs/reference/seeding.md)
14. Deferred: Add a guide about seeding to the docs

**Implementation Summary:**
- All core phases (1-4) are complete
- 1366 tests pass (17 skipped for expected missing dependencies)
- Documentation tasks (13-14) deferred for separate PR

## Files to Create/Modify

### New Files

- `maseval/core/seeding.py` — `SeedGenerator` (ABC), `DefaultSeedGenerator`, `SeedingError`
- `tests/test_core/test_seeding.py` — Unit tests

### Modified Files

- `maseval/core/benchmark.py` — Add `seed` and `seed_generator` parameters, update setup calls
- `maseval/core/model.py` — Add seed parameter to ModelAdapter.**init**
- `maseval/core/__init__.py` — Export `SeedGenerator`, `DefaultSeedGenerator`, `SeedingError`
- `maseval/__init__.py` — Export `SeedGenerator`, `DefaultSeedGenerator`, `SeedingError`
- `maseval/interface/inference/*.py` — Pass seed to provider APIs, raise SeedingError if unsupported
- `maseval/benchmark/macs/macs.py` — Use seed_generator in setup methods
- `maseval/benchmark/tau2/tau2.py` — Use seed_generator in setup methods
- `examples/five_a_day_benchmark/five_a_day_benchmark.py` — Use new seeding system
- `AGENTS.md` — Document seeding and extension patterns
- `docs/reference/seeding.md` — User documentation with extension examples

## Testing Strategy

1. **Unit tests** for `DefaultSeedGenerator`:
   - Deterministic derivation (same inputs → same output)
   - Different paths → different seeds
   - `child()` generators extend paths correctly and share log
   - Thread safety under concurrent access
   - `per_repetition=False` produces constant seeds across reps
   - `global_seed` property returns correct value

2. **Unit tests** for `SeedGenerator` ABC:
   - Cannot instantiate directly (raises TypeError)
   - Subclasses must implement all 5 abstract methods (`global_seed`, `derive_seed`, `for_task`, `for_repetition`, `seed_log`)
   - Custom subclass with overridden `_compute_seed()` works correctly
   - Custom subclass without `child()` works (flat paths)

3. **Integration tests**:
   - Benchmark with `seed=42` produces reproducible results
   - Benchmark with `seed=None` works (seeding disabled)
   - Benchmark with custom `seed_generator` works
   - Seeds appear in result configs

## Test Analysis & Recommendations

### Current Test Coverage Status

| Component | Coverage | Status |
|-----------|----------|--------|
| `maseval/core/seeding.py` | 100% | ✅ Excellent unit tests |
| Model adapter seeding | ~85% | ⚠️ Basic acceptance tests only |
| Benchmark seeding integration | Not tested | ❌ Missing |

### Tests Currently Passing

The seeding unit tests in `tests/test_core/test_seeding.py` are well-structured:
- `TestDefaultSeedGenerator` - deterministic derivation, different seeds for different inputs
- `TestDefaultSeedGeneratorChild` - hierarchical namespacing with `child()`
- `TestDefaultSeedGeneratorSeedLog` - logging behavior
- `TestDefaultSeedGeneratorErrors` - error handling for missing context
- `TestSeedGeneratorABC` - abstract base class behavior
- `TestCustomHashAlgorithm` - extension via `_compute_seed()` override
- `TestSeedGeneratorGatherConfig` - config integration

### Missing Tests (Prioritized)

#### HIGH PRIORITY: Benchmark Seeding Integration

Create new test file: `tests/test_core/test_benchmark/test_seeding_integration.py`

```python
"""Tests for benchmark seeding integration.

These tests verify that the seeding system integrates correctly with
the benchmark execution lifecycle, including seed_generator propagation
to setup methods and seeding config in reports.
"""

import pytest
from maseval import TaskQueue
from maseval.core.seeding import DefaultSeedGenerator, SeedGenerator

pytestmark = pytest.mark.core


class TestBenchmarkSeedingInitialization:
    """Tests for Benchmark seed/seed_generator initialization."""

    def test_benchmark_seed_parameter_creates_generator(self):
        """seed parameter creates a DefaultSeedGenerator."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark(seed=42)

        assert benchmark.seed_generator is not None
        assert isinstance(benchmark.seed_generator, DefaultSeedGenerator)
        assert benchmark.seed_generator.global_seed == 42

    def test_benchmark_seed_generator_parameter(self):
        """seed_generator parameter is stored directly."""
        from conftest import DummyBenchmark

        custom_gen = DefaultSeedGenerator(global_seed=123)
        benchmark = DummyBenchmark(seed_generator=custom_gen)

        assert benchmark.seed_generator is custom_gen
        assert benchmark.seed_generator.global_seed == 123

    def test_benchmark_no_seed_no_generator(self):
        """No seed or seed_generator results in None."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark()

        assert benchmark.seed_generator is None

    def test_benchmark_seed_and_generator_raises_value_error(self):
        """Providing both seed and seed_generator raises ValueError."""
        from conftest import DummyBenchmark

        with pytest.raises(ValueError, match="Cannot provide both"):
            DummyBenchmark(seed=42, seed_generator=DefaultSeedGenerator(42))


class TestSeedGeneratorPropagation:
    """Tests verifying seed_generator is passed to all setup methods."""

    def test_seed_generator_passed_to_setup_environment(self):
        """setup_environment receives seed_generator."""
        from conftest import DummyBenchmark, DummyEnvironment

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_environment(self, agent_data, task, seed_generator=None):
                captured['seed_generator'] = seed_generator
                return DummyEnvironment(task.environment_data)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured['seed_generator'] is not None
        assert isinstance(captured['seed_generator'], SeedGenerator)

    def test_seed_generator_passed_to_setup_user(self):
        """setup_user receives seed_generator."""
        from conftest import DummyBenchmark

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_user(self, agent_data, environment, task, seed_generator=None):
                captured['seed_generator'] = seed_generator
                return None

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured['seed_generator'] is not None

    def test_seed_generator_passed_to_setup_agents(self):
        """setup_agents receives seed_generator."""
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                captured['seed_generator'] = seed_generator
                agent = DummyAgent()
                adapter = DummyAgentAdapter(agent, "test_agent")
                return [adapter], {"test_agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured['seed_generator'] is not None

    def test_seed_generator_passed_to_setup_evaluators(self):
        """setup_evaluators receives seed_generator."""
        from conftest import DummyBenchmark, DummyEvaluator

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator=None):
                captured['seed_generator'] = seed_generator
                return [DummyEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured['seed_generator'] is not None

    def test_seed_generator_none_when_no_seed(self):
        """seed_generator is None in setup methods when seeding disabled."""
        from conftest import DummyBenchmark

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                captured['seed_generator'] = seed_generator
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark()  # No seed
        benchmark.run(tasks, agent_data={})

        assert captured['seed_generator'] is None


class TestSeedingConfigInReports:
    """Tests verifying seeding config appears in benchmark reports."""

    def test_seeding_config_appears_in_report(self):
        """Seeding configuration appears in report config."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark(seed=42)

        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        config = reports[0]["config"]
        assert "seeding" in config
        assert "seed_generator" in config["seeding"]
        assert config["seeding"]["seed_generator"]["global_seed"] == 42

    def test_seeding_config_includes_seed_log(self):
        """Seeding config includes all derived seeds."""
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        class SeedUsingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    agent_gen = seed_generator.child("agents")
                    agent_gen.derive_seed("test_agent")
                agent = DummyAgent()
                adapter = DummyAgentAdapter(agent, "test_agent")
                return [adapter], {"test_agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = SeedUsingBenchmark(seed=42)

        reports = benchmark.run(tasks, agent_data={})

        config = reports[0]["config"]
        seeds = config["seeding"]["seed_generator"]["seeds"]
        assert "agents/test_agent" in seeds

    def test_no_seeding_config_when_disabled(self):
        """No seeding config when seeding is disabled."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()  # No seed

        reports = benchmark.run(tasks, agent_data={})

        config = reports[0]["config"]
        # seeding key may exist but be empty, or not exist at all
        if "seeding" in config:
            assert config["seeding"] == {} or config["seeding"].get("seed_generator") is None


class TestSeedingAcrossRepetitions:
    """Tests verifying seeding behavior across task repetitions."""

    def test_different_seeds_per_repetition(self):
        """Different repetitions get different seeds (per_repetition=True)."""
        from conftest import DummyBenchmark

        captured_seeds = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("agent", per_repetition=True)
                    captured_seeds.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42, n_task_repeats=3)
        benchmark.run(tasks, agent_data={})

        assert len(captured_seeds) == 3
        assert len(set(captured_seeds)) == 3  # All different

    def test_same_seed_across_repetitions_when_per_rep_false(self):
        """Same seed across repetitions when per_repetition=False."""
        from conftest import DummyBenchmark

        captured_seeds = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("baseline", per_repetition=False)
                    captured_seeds.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42, n_task_repeats=3)
        benchmark.run(tasks, agent_data={})

        assert len(captured_seeds) == 3
        assert len(set(captured_seeds)) == 1  # All same


class TestReproducibility:
    """Tests verifying reproducible benchmark runs with seeding."""

    def test_same_seed_produces_same_derived_seeds(self):
        """Same global seed produces identical derived seeds across runs."""
        from conftest import DummyBenchmark

        seeds_run1 = []
        seeds_run2 = []

        class CapturingBenchmark(DummyBenchmark):
            def __init__(self, capture_list, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._capture_list = capture_list

            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("agent")
                    self._capture_list.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])

        # Run 1
        benchmark1 = CapturingBenchmark(seeds_run1, seed=42)
        benchmark1.run(tasks, agent_data={})

        # Run 2
        benchmark2 = CapturingBenchmark(seeds_run2, seed=42)
        benchmark2.run(tasks, agent_data={})

        assert seeds_run1 == seeds_run2
```

#### MEDIUM PRIORITY: Model Adapter Seed Propagation

The implementations in `openai.py`, `litellm.py`, and `google_genai.py` all include this logic:
```python
# Add seed if set and not already in params (user params take precedence)
if self._seed is not None and "seed" not in params:
    params["seed"] = self._seed
```

**Current tests only verify seed acceptance/storage, NOT that seeds are passed to APIs.**

Add to `tests/test_core/test_model_adapter.py`:

```python
@pytest.mark.core
class TestModelAdapterSeedPropagation:
    """Tests verifying seeds are passed to underlying APIs.

    These tests verify the actual behavior: that seeds set on adapters
    are passed through to the underlying provider API calls.
    """

    def test_openai_adapter_passes_seed_to_api(self):
        """OpenAI adapter includes seed in API call."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test", tool_calls=None))]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4", seed=42)
        adapter.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("seed") == 42

    def test_openai_adapter_no_seed_when_not_set(self):
        """OpenAI adapter doesn't include seed when not set."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test", tool_calls=None))]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4")  # No seed
        adapter.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "seed" not in call_kwargs

    def test_openai_user_seed_overrides_adapter_seed(self):
        """User-provided seed in generation_params takes precedence."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test", tool_calls=None))]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4", seed=42)
        adapter.chat(
            [{"role": "user", "content": "test"}],
            generation_params={"seed": 999}  # Should override adapter seed
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("seed") == 999  # User's seed, not adapter's

    def test_litellm_adapter_passes_seed_to_api(self):
        """LiteLLM adapter includes seed in API call."""
        from unittest.mock import patch, MagicMock
        from maseval.interface.inference import LiteLLMModelAdapter

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test", tool_calls=None, role="assistant"))]
        mock_response.usage = None
        mock_response.model = "gpt-4"

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            adapter = LiteLLMModelAdapter(model_id="gpt-4", seed=42)
            adapter.chat([{"role": "user", "content": "test"}])

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get("seed") == 42

    def test_google_adapter_passes_seed_to_config(self):
        """Google GenAI adapter includes seed in generation config."""
        from unittest.mock import MagicMock, patch
        from maseval.interface.inference import GoogleGenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.candidates = []
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.types.GenerateContentConfig") as mock_config:
            adapter = GoogleGenAIModelAdapter(client=mock_client, model_id="gemini-pro", seed=42)
            adapter.chat([{"role": "user", "content": "test"}])

            # Verify seed was passed to GenerateContentConfig
            config_kwargs = mock_config.call_args[1]
            assert config_kwargs.get("seed") == 42
```

### Implementation Checklist

#### High Priority: Benchmark Seeding Integration ✅ COMPLETE
- [x] Create `tests/test_core/test_benchmark/test_seeding_integration.py`
- [x] Add `TestBenchmarkSeedingInitialization` class (4 tests)
  - [x] `test_benchmark_seed_parameter_creates_generator`
  - [x] `test_benchmark_seed_generator_parameter`
  - [x] `test_benchmark_no_seed_no_generator`
  - [x] `test_benchmark_seed_and_generator_raises_value_error`
- [x] Add `TestSeedGeneratorPropagation` class (5 tests)
  - [x] `test_seed_generator_passed_to_setup_environment`
  - [x] `test_seed_generator_passed_to_setup_user`
  - [x] `test_seed_generator_passed_to_setup_agents`
  - [x] `test_seed_generator_passed_to_setup_evaluators`
  - [x] `test_seed_generator_none_when_no_seed`
- [x] Add `TestSeedingConfigInReports` class (3 tests)
  - [x] `test_seeding_config_appears_in_report`
  - [x] `test_seeding_config_includes_seed_log`
  - [x] `test_no_seeding_config_when_disabled`
- [x] Add `TestSeedingAcrossRepetitions` class (2 tests)
  - [x] `test_different_seeds_per_repetition`
  - [x] `test_same_seed_across_repetitions_when_per_rep_false`
- [x] Add `TestReproducibility` class (2 tests)
  - [x] `test_same_seed_produces_same_derived_seeds`
  - [x] `test_different_seeds_produce_different_derived_seeds`
- [x] Add `TestSeedGeneratorScoping` class (3 tests) - bonus tests
  - [x] `test_seed_generator_scoped_to_task`
  - [x] `test_seed_generator_scoped_to_repetition`
  - [x] `test_child_generators_share_seed_log`

#### Medium Priority: Model Adapter Seed Propagation ✅ COMPLETE
- [x] Add `TestModelAdapterSeedPropagation` to `test_model_adapter.py` (8 tests)
  - [x] `test_openai_adapter_passes_seed_to_api`
  - [x] `test_openai_adapter_no_seed_when_not_set`
  - [x] `test_openai_user_seed_overrides_adapter_seed`
  - [x] `test_litellm_adapter_passes_seed_to_api`
  - [x] `test_litellm_adapter_no_seed_when_not_set`
  - [x] `test_litellm_user_seed_overrides_adapter_seed`
  - [x] `test_google_adapter_passes_seed_to_config`
  - [x] `test_google_adapter_no_seed_when_not_set`

#### Verification ✅ COMPLETE
- [x] Run full test suite to verify all new tests pass (1459 passed, 2 skipped)
- [x] Run linter to confirm code quality (All checks passed!)
- [x] Verify no regressions in existing tests
