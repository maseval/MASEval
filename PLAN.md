# Seeding System Implementation Plan

## Overview

Implement Option B from IDEA.md with child generator support from Option C. The system provides a centralized `SeedGenerator` that derives seeds by path and can spawn scoped child generators.

## Design Goals

From IDEA.md Section 4:
1. **Opt-in** — Users who don't need seeding shouldn't be affected
2. **Derive by name, not index** — Seeds from identifiers, never positions
3. **Support repetitions** — Each repetition gets a different seed
4. **Selective variance** — Some components constant across reps, others vary
5. **Logging** — Seeds used must be recorded in results
6. **Fail explicitly** — Providers without seed support should error
7. **Extensible** — Future use cases (task queues, etc.) should work

## API Design

### Core Classes

```python
# maseval/core/seeding.py

from maseval.core.config import ConfigurableMixin

class SeedGenerator(ConfigurableMixin):
    """Generates deterministic seeds from hierarchical paths.

    Thread-safe. Immutable after construction (derives seeds via pure functions).
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
        self._global_seed = global_seed
        self._task_id = task_id
        self._rep_index = rep_index
        self._path_prefix = path_prefix
        self._shared_log = _shared_log if _shared_log is not None else {}

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

        seed = self._compute_seed(name, per_repetition)

        # Log with full path
        full_path = f"{self._path_prefix}/{name}" if self._path_prefix else name
        self._shared_log[full_path] = seed

        return seed

    def _compute_seed(self, name: str, per_repetition: bool) -> int:
        """Internal: compute deterministic seed from components."""
        full_path = f"{self._path_prefix}/{name}" if self._path_prefix else name

        # Include rep_index only if per_repetition=True
        if per_repetition:
            components = [self._global_seed, self._task_id, self._rep_index, full_path]
        else:
            components = [self._global_seed, self._task_id, full_path]

        seed_string = ":".join(str(c) for c in components)
        hash_bytes = hashlib.sha256(seed_string.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF

    def child(self, name: str) -> "SeedGenerator":
        """Create a child generator scoped to a component.

        The child inherits context (global_seed, task_id, rep_index) and extends
        the path. All children share the same seed log.

        Args:
            name: Component name to add to the path.

        Returns:
            New SeedGenerator with extended path, sharing the same log.
        """
        new_path = f"{self._path_prefix}/{name}" if self._path_prefix else name
        return SeedGenerator(
            global_seed=self._global_seed,
            task_id=self._task_id,
            rep_index=self._rep_index,
            path_prefix=new_path,
            _shared_log=self._shared_log,  # Share the log
        )

    def for_task(self, task_id: str) -> "SeedGenerator":
        """Create a generator scoped to a specific task.

        Returns:
            New SeedGenerator with task_id set and fresh log.
        """
        return SeedGenerator(
            global_seed=self._global_seed,
            task_id=task_id,
            rep_index=None,
            path_prefix="",
            _shared_log={},  # Fresh log for this task
        )

    def for_repetition(self, rep_index: int) -> "SeedGenerator":
        """Create a generator scoped to a specific repetition.

        Returns:
            New SeedGenerator with rep_index set, preserving task scope and log.
        """
        return SeedGenerator(
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
            "global_seed": self._global_seed,
            "task_id": self._task_id,
            "rep_index": self._rep_index,
            "seeds": self.seed_log,
        }


class SeedingError(Exception):
    """Raised when seeding is misconfigured or unsupported."""
    pass
```

### Integration with Benchmark

```python
class Benchmark(ABC):
    def __init__(
        self,
        ...,
        seed: Optional[int] = None,  # NEW: global seed for reproducibility
    ):
        self._seed_generator: Optional[SeedGenerator] = None
        if seed is not None:
            self._seed_generator = SeedGenerator(global_seed=seed)

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

## Usage Example

```python
class MyBenchmark(Benchmark):
    def setup_environment(self, agent_data, task, seed_generator=None):
        env = MyEnvironment(task.environment_data)

        if seed_generator is not None:
            env_gen = seed_generator.child("environment")
            for tool in env.tools:
                tool_seed = env_gen.derive_seed(tool.name)
                tool_model = self.get_model_adapter(model_id, seed=tool_seed)
                tool.set_simulator(tool_model)

        return env

    def setup_user(self, agent_data, environment, task, seed_generator=None):
        user_seed = None
        if seed_generator is not None:
            user_seed = seed_generator.derive_seed("user_simulator")

        user_model = self.get_model_adapter(model_id, seed=user_seed)
        return MyUser(model=user_model, ...)

    def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
        if seed_generator is not None:
            agent_gen = seed_generator.child("agents")
            # Vary experimental agent per rep, keep baseline constant
            experimental_seed = agent_gen.derive_seed("experimental", per_repetition=True)
            baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)
        else:
            experimental_seed = None
            baseline_seed = None

        # ... create agents with seeds ...

# Run with seeding
benchmark = MyBenchmark(seed=42, n_task_repeats=3)
results = benchmark.run(tasks=tasks, agent_data=config)

# Results include seed log (via ConfigurableMixin integration)
for report in results:
    seed_config = report["config"]["seeding"]["seed_generator"]
    print(seed_config["global_seed"])  # 42
    print(seed_config["seeds"])
    # {"environment/tool_weather": 12345, "user_simulator": 67890, "agents/experimental": ...}
```

## Thread Safety

The `SeedGenerator` is thread-safe by design:

1. **Isolated logs per task repetition**: `for_task()` creates a fresh `_shared_log` dict. Each task repetition has its own log that isn't shared across threads.

2. **No cross-thread sharing**: The root `self._seed_generator` is read-only (only used to call `for_task()`). All mutable state (the log) lives in the task-scoped generator.

3. **Children share parent's log**: Within a single task repetition, `child()` generators share the same `_shared_log`. This is safe because a single task repetition runs in a single thread.

```
Thread 1 (task A, rep 0):
  task_gen_A = root.for_task("A").for_repetition(0)  # Fresh log
  child = task_gen_A.child("env")                     # Shares task_gen_A's log

Thread 2 (task B, rep 0):
  task_gen_B = root.for_task("B").for_repetition(0)  # Different fresh log
  child = task_gen_B.child("env")                     # Shares task_gen_B's log

# No cross-thread sharing of mutable state
```

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

### Phase 1: Core Infrastructure
1. Create `maseval/core/seeding.py` with `SeedGenerator` and `SeedingError`
2. Add unit tests for seed derivation and thread safety
3. Add `seed` parameter to `Benchmark.__init__`

### Phase 2: Benchmark Integration
4. Update `_execute_task_repetition` to create scoped generators
5. Add `seed_generator` parameter to all setup methods
6. Log seeds in `execution_configs`
7. Update existing benchmark implementations (MACS, Tau2) to use seeds

### Phase 3: Model Adapter Support
8. Add `seed` parameter to `ModelAdapter.__init__`
9. Update interface adapters to pass seed to provider APIs
10. Raise `SeedingError` for providers that don't support seeding

### Phase 4: Documentation & Examples
11. Update AGENTS.md with seeding guidance
12. Update five_a_day example to use new seeding system
13. Add seeding section to documentation

## Files to Create/Modify

### New Files
- `maseval/core/seeding.py` — SeedGenerator, SeedingError
- `tests/test_core/test_seeding.py` — Unit tests

### Modified Files
- `maseval/core/benchmark.py` — Add seed parameter, update setup calls
- `maseval/core/model.py` — Add seed parameter to ModelAdapter.__init__
- `maseval/core/__init__.py` — Export SeedGenerator, SeedingError
- `maseval/__init__.py` — Export SeedGenerator, SeedingError
- `maseval/interface/inference/*.py` — Pass seed to provider APIs, raise SeedingError if unsupported
- `maseval/benchmark/macs/macs.py` — Use seed_generator in setup methods
- `maseval/benchmark/tau2/tau2.py` — Use seed_generator in setup methods
- `examples/five_a_day_benchmark/five_a_day_benchmark.py` — Use new seeding system
- `AGENTS.md` — Document seeding
- `docs/reference/seeding.md` — User documentation

## Testing Strategy

1. **Unit tests** for `SeedGenerator`:
   - Deterministic derivation (same inputs → same output)
   - Different paths → different seeds
   - Child generators extend paths correctly
   - Thread safety under concurrent access
   - `per_repetition=False` produces constant seeds across reps

2. **Integration tests**:
   - Benchmark with `seed=42` produces reproducible results
   - Benchmark with `seed=None` works (seeding disabled)
   - Seeds appear in result configs
