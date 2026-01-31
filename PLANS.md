# PLANS.md - Robust Seeding System for MASEval

## Overview

MASEval currently lacks a systematic seeding mechanism for reproducibility. This plan proposes a comprehensive seeding system that enables deterministic benchmark runs while maintaining flexibility for variance measurement.

## Current State Analysis

### Components with Randomness Potential

| Component         | Location                                 | Current State            | Priority |
| ----------------- | ---------------------------------------- | ------------------------ | -------- |
| LLM Model Calls   | `core/model.py`, `interface/inference/*` | No seed parameter        | CRITICAL |
| Tool Simulators   | `core/simulator.py`                      | Uses model (unseeded)    | CRITICAL |
| User Simulators   | `core/user.py`, `core/simulator.py`      | Uses model (unseeded)    | CRITICAL |
| Task Ordering     | `core/task.py`                           | Sequential/Priority only | HIGH     |
| Task Sampling     | `benchmark/*/data_loader.py`             | Sequential only          | HIGH     |
| Task Repetitions  | `core/benchmark.py`                      | Deterministic loops      | HIGH     |
| Environment Setup | `core/environment.py`                    | Implementation-dependent | MEDIUM   |
| Evaluators        | `core/evaluator.py`                      | Implementation-dependent | MEDIUM   |

### Reproducibility Gaps

1. **No global seed setting** - Components manage randomness independently
2. **No seed propagation** - Seed doesn't cascade from benchmark to sub-components
3. **No seed in model parameters** - ModelAdapter and LLMSimulator don't support seed
4. **No deterministic task ordering** - No reproducible shuffling capability
5. **No sampling control** - Data loaders don't support seeded sampling
6. **No multi-run reproducibility** - Running same benchmark twice yields different results

---

## Proposed Architecture

### Design Principles

1. **Hierarchical Seeding** - Global seed propagates deterministically to all components
2. **Explicit Over Implicit** - Seeds are explicitly passed, never hidden in global state
3. **Opt-in Complexity** - Simple use cases remain simple; advanced control is available
4. **Configuration Tracking** - All seeds used are recorded in results for reproducibility

### Seed Hierarchy

```
Benchmark (global_seed)
├── TaskQueue (derived: global_seed + "task_queue")
│   └── Shuffle/Sample operations
├── Per-Task Seeds (derived: global_seed + task_id + repetition)
│   ├── Environment (derived: task_seed + "environment")
│   ├── User Simulator (derived: task_seed + "user")
│   ├── Agent Execution (derived: task_seed + "agent" + agent_name)
│   │   └── Model Calls (derived: agent_seed + call_index)
│   └── Evaluator (derived: task_seed + "evaluator")
```

### Seed Derivation Function

```python
def derive_seed(parent_seed: int, component: str) -> int:
    """Deterministically derive a child seed from parent seed and component identifier."""
    import hashlib
    combined = f"{parent_seed}:{component}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

#### 1.1 Create `maseval/core/seed.py`

New module containing seeding utilities:

```python
"""Seeding infrastructure for reproducible benchmark execution."""

import hashlib
import random
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SeedState:
    """Tracks seed state for a component."""
    seed: int
    component_path: str
    call_count: int = 0

    def next_seed(self) -> int:
        """Get next deterministic seed for sequential operations."""
        self.call_count += 1
        return derive_seed(self.seed, str(self.call_count))

def derive_seed(parent_seed: int, component: str) -> int:
    """Deterministically derive a child seed from parent seed and component identifier."""
    combined = f"{parent_seed}:{component}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big')

def create_rng(seed: int) -> random.Random:
    """Create an isolated Random instance with the given seed."""
    return random.Random(seed)

class SeedManager:
    """Manages seed hierarchy and distribution for a benchmark run."""

    def __init__(self, global_seed: Optional[int] = None):
        self._global_seed = global_seed if global_seed is not None else random.randint(0, 2**32 - 1)
        self._component_states: Dict[str, SeedState] = {}

    @property
    def global_seed(self) -> int:
        return self._global_seed

    def get_seed(self, component_path: str) -> int:
        """Get or create a seed for a component path."""
        if component_path not in self._component_states:
            seed = derive_seed(self._global_seed, component_path)
            self._component_states[component_path] = SeedState(seed, component_path)
        return self._component_states[component_path].seed

    def get_rng(self, component_path: str) -> random.Random:
        """Get an isolated RNG for a component."""
        seed = self.get_seed(component_path)
        return create_rng(seed)

    def get_config(self) -> Dict[str, Any]:
        """Return seed configuration for logging/reproducibility."""
        return {
            "global_seed": self._global_seed,
            "component_seeds": {
                path: state.seed for path, state in self._component_states.items()
            }
        }
```

#### 1.2 Update `maseval/core/__init__.py`

Export seeding utilities:

```python
from maseval.core.seed import SeedManager, derive_seed, create_rng
```

### Phase 2: Benchmark Integration (Core Flow)

#### 2.1 Update `Benchmark.__init__()` in `maseval/core/benchmark.py`

Add seed parameter:

```python
def __init__(
    self,
    ...,
    seed: Optional[int] = None,  # NEW: Global seed for reproducibility
    ...
):
    ...
    self._seed_manager = SeedManager(seed)
```

#### 2.2 Update `Benchmark.run()`

Propagate seed manager to execution:

```python
def run(self, tasks, agent_data, ...):
    # Log seed configuration at run start
    self._callbacks.on_run_start(
        ...,
        seed_config=self._seed_manager.get_config()
    )
    ...
```

#### 2.3 Update Task Execution Loop

Pass task-specific seeds:

```python
def _execute_task(self, task, repetition_index, ...):
    task_seed_path = f"task/{task.protocol.task_id}/rep/{repetition_index}"
    task_seed = self._seed_manager.get_seed(task_seed_path)

    # Pass to environment setup
    environment = self.setup_environment(
        task,
        seed=self._seed_manager.get_seed(f"{task_seed_path}/environment")
    )

    # Pass to user setup
    user = self.setup_user(
        task,
        seed=self._seed_manager.get_seed(f"{task_seed_path}/user")
    )
    ...
```

### Phase 3: Model Adapter Seeding (LLM Reproducibility)

#### 3.1 Update `ModelAdapter` Protocol in `maseval/core/model.py`

Add seed support to generation params:

```python
def chat(
    self,
    messages: Union[List[Dict[str, Any]], MessageHistory],
    generation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,  # NEW: Explicit seed parameter
    ...
) -> ChatResponse:
    """
    Args:
        seed: Random seed for reproducible generation. Support varies by provider:
            - OpenAI: Supported via 'seed' parameter
            - Anthropic: Not directly supported (use temperature=0)
            - Google: Supported via 'seed' parameter
            - HuggingFace: Supported via generator seed
    """
    if seed is not None:
        generation_params = generation_params or {}
        generation_params["seed"] = seed
    ...
```

#### 3.2 Update Interface Implementations

**OpenAI Adapter (`interface/inference/openai.py`):**

```python
def _chat_impl(self, messages, generation_params=None, ...):
    params = generation_params or {}
    if "seed" in params:
        # OpenAI supports seed parameter directly
        api_params["seed"] = params.pop("seed")
    ...
```

**Anthropic Adapter (`interface/inference/anthropic.py`):**

```python
def _chat_impl(self, messages, generation_params=None, ...):
    params = generation_params or {}
    if "seed" in params:
        # Anthropic doesn't support seed - log warning, use temperature=0 for determinism
        seed = params.pop("seed")
        logger.warning(f"Anthropic does not support seed parameter (requested: {seed}). "
                      "For reproducibility, consider temperature=0.")
    ...
```

**Google GenAI Adapter (`interface/inference/google_genai.py`):**

```python
def _chat_impl(self, messages, generation_params=None, ...):
    params = generation_params or {}
    if "seed" in params:
        # Google supports seed in generation config
        api_params["generation_config"]["seed"] = params.pop("seed")
    ...
```

### Phase 4: Simulator Seeding (Tool & User)

#### 4.1 Update `LLMSimulator` in `maseval/core/simulator.py`

Add seed state management:

```python
class LLMSimulator:
    def __init__(
        self,
        model: ModelAdapter,
        ...,
        seed: Optional[int] = None,  # NEW
    ):
        self._seed_state = SeedState(seed, "simulator") if seed else None

    def __call__(self, generation_params=None, **inputs):
        params = generation_params or {}
        if self._seed_state:
            params["seed"] = self._seed_state.next_seed()
        return self._generate(params, **inputs)
```

#### 4.2 Update `ToolLLMSimulator`

Inherit seed support from base:

```python
class ToolLLMSimulator(LLMSimulator):
    def __init__(self, ..., seed: Optional[int] = None):
        super().__init__(..., seed=seed)
```

#### 4.3 Update `UserLLMSimulator` and `AgenticUserLLMSimulator`

Same pattern - accept and propagate seed.

### Phase 5: Task Queue Seeding (Ordering)

#### 5.1 Add `ShuffledTaskQueue` in `maseval/core/task.py`

New task queue with reproducible shuffling:

```python
class ShuffledTaskQueue(TaskQueue):
    """Task queue that shuffles tasks with a seed for reproducibility."""

    def __init__(self, tasks: Sequence[Task], seed: Optional[int] = None):
        self._original_tasks = list(tasks)
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._shuffled_tasks = self._original_tasks.copy()
        self._rng.shuffle(self._shuffled_tasks)
        super().__init__(self._shuffled_tasks)

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ShuffledTaskQueue",
            "seed": self._seed,
            "task_count": len(self._original_tasks),
        }
```

#### 5.2 Add `SampledTaskQueue`

For random sampling of tasks:

```python
class SampledTaskQueue(TaskQueue):
    """Task queue that samples a subset of tasks with a seed."""

    def __init__(
        self,
        tasks: Sequence[Task],
        sample_size: int,
        seed: Optional[int] = None,
        replacement: bool = False,
    ):
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

        if replacement:
            sampled = self._rng.choices(tasks, k=sample_size)
        else:
            sampled = self._rng.sample(list(tasks), min(sample_size, len(tasks)))

        super().__init__(sampled)
```

### Phase 6: Data Loader Seeding (Benchmark-Specific)

#### 6.1 Update `load_tasks()` in Data Loaders

**MACS (`benchmark/macs/data_loader.py`):**

```python
def load_tasks(
    domain: str,
    limit: Optional[int] = None,
    shuffle: bool = False,  # NEW
    seed: Optional[int] = None,  # NEW
    sample: Optional[int] = None,  # NEW: Random sample size
) -> List[MACSTask]:
    tasks = _load_all_tasks(domain)

    if sample is not None:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, min(sample, len(tasks)))
    elif shuffle:
        rng = random.Random(seed)
        tasks = tasks.copy()
        rng.shuffle(tasks)

    if limit is not None:
        tasks = tasks[:limit]

    return tasks
```

**Tau2 (`benchmark/tau2/data_loader.py`):**
Same pattern with split-aware sampling.

### Phase 7: Environment Seeding (Optional)

#### 7.1 Update `Environment` Protocol

Add optional seed parameter:

```python
class Environment(Protocol):
    def setup_state(
        self,
        task: Task,
        seed: Optional[int] = None,  # NEW: For reproducible initialization
    ) -> None:
        """Initialize environment state for task execution."""
        ...
```

Implementations can use this for:

- Random initial states
- Stochastic environment dynamics
- Simulated external systems

### Phase 8: Configuration & Logging

#### 8.1 Update `gather_config()` Methods

Include seed information in configuration:

```python
class Benchmark(ConfigurableMixin):
    def gather_config(self) -> Dict[str, Any]:
        config = super().gather_config()
        config["seed_config"] = self._seed_manager.get_config()
        return config
```

#### 8.2 Update Callbacks

Add seed information to callback events:

```python
def on_run_start(self, ..., seed_config: Optional[Dict[str, Any]] = None):
    """Called when benchmark run starts."""
    ...

def on_task_start(self, ..., task_seed: Optional[int] = None):
    """Called when task execution starts."""
    ...
```

---

## API Examples

### Basic Usage (Simple)

```python
from maseval import Benchmark

# Reproducible run with explicit seed
benchmark = MyBenchmark(seed=42)
results = benchmark.run(tasks, agent_data=config)

# Run again with same seed - identical results
benchmark2 = MyBenchmark(seed=42)
results2 = benchmark2.run(tasks, agent_data=config)
assert results == results2
```

### Advanced Usage (Task Control)

```python
from maseval.core.task import ShuffledTaskQueue, SampledTaskQueue

# Shuffled task order with seed
tasks = load_tasks("travel", shuffle=True, seed=42)
queue = ShuffledTaskQueue(tasks, seed=42)
benchmark.run(queue, ...)

# Random sample of tasks
tasks = load_tasks("travel", sample=10, seed=42)
```

### Model-Level Seeding

```python
# Pass seed to model calls for reproducibility
response = model.chat(
    messages=[{"role": "user", "content": "..."}],
    generation_params={"temperature": 0.7},
    seed=12345
)
```

### Variance Measurement

```python
# Run with different seeds to measure variance
results = []
for seed in range(10):
    benchmark = MyBenchmark(seed=seed)
    result = benchmark.run(tasks, agent_data=config)
    results.append(result)

# Analyze variance across seeds
variance = compute_variance(results)
```

---

## Migration Guide

### Breaking Changes

None - all seed parameters are optional with backward-compatible defaults.

### Opt-in Reproducibility

Users can enable reproducibility by:

1. Setting `seed` parameter in `Benchmark.__init__()`
2. Using seeded data loaders
3. Using `ShuffledTaskQueue` or `SampledTaskQueue`

### Provider-Specific Notes

| Provider     | Seed Support | Workaround                     |
| ------------ | ------------ | ------------------------------ |
| OpenAI       | Full         | Use `seed` parameter           |
| Anthropic    | None         | Use `temperature=0`            |
| Google GenAI | Full         | Use `seed` parameter           |
| HuggingFace  | Full         | Set generator seed             |
| LiteLLM      | Varies       | Depends on underlying provider |

---

## Testing Strategy

### Unit Tests

```python
def test_derive_seed_deterministic():
    """Same inputs produce same outputs."""
    assert derive_seed(42, "component") == derive_seed(42, "component")

def test_derive_seed_different_components():
    """Different components get different seeds."""
    assert derive_seed(42, "a") != derive_seed(42, "b")

def test_seed_manager_hierarchy():
    """Child seeds are deterministic from parent."""
    mgr = SeedManager(42)
    seed1 = mgr.get_seed("task/1/environment")
    seed2 = mgr.get_seed("task/1/user")
    assert seed1 != seed2  # Different components

    mgr2 = SeedManager(42)
    assert mgr2.get_seed("task/1/environment") == seed1  # Reproducible
```

### Integration Tests

```python
@pytest.mark.core
def test_benchmark_reproducibility():
    """Same seed produces identical results."""
    tasks = load_tasks("domain", limit=5)

    results1 = Benchmark(seed=42).run(tasks, agent_data=config)
    results2 = Benchmark(seed=42).run(tasks, agent_data=config)

    assert results1 == results2

@pytest.mark.core
def test_different_seeds_different_results():
    """Different seeds can produce different results."""
    tasks = load_tasks("domain", limit=5, shuffle=True, seed=1)
    tasks2 = load_tasks("domain", limit=5, shuffle=True, seed=2)

    # Task order should differ
    assert [t.protocol.task_id for t in tasks] != [t.protocol.task_id for t in tasks2]
```

---

## Implementation Order

1. **Phase 1** - Core Infrastructure (`seed.py`) - Foundation for all other phases
2. **Phase 2** - Benchmark Integration - Enable top-level seed control
3. **Phase 5** - Task Queue Seeding - Reproducible task ordering
4. **Phase 6** - Data Loader Seeding - Reproducible data sampling
5. **Phase 3** - Model Adapter Seeding - LLM call reproducibility
6. **Phase 4** - Simulator Seeding - Tool/User simulation reproducibility
7. **Phase 7** - Environment Seeding - Optional, implementation-dependent
8. **Phase 8** - Configuration & Logging - Full observability

---

## Success Criteria

1. **Reproducibility** - Same seed produces byte-identical results (where provider supports it)
2. **Isolation** - Component seeds don't interfere with each other
3. **Observability** - All seeds used are logged and recoverable
4. **Simplicity** - Basic usage remains simple (single seed parameter)
5. **Flexibility** - Advanced users can control seeds at any level
6. **Documentation** - Clear guidance on provider-specific limitations

---

## User Workflow Analysis

### Current User Workflows (from Examples)

Analysis of the three example benchmarks reveals different approaches to seeding:

#### 1. MACS Benchmark (`examples/macs_benchmark/`)

**Current CLI:**

```bash
uv run python examples/macs_benchmark.py \
    --framework smolagents \
    --domain travel \
    --limit 5 \
    --repeats 3
```

**Workflow:** No seeding at all. Users cannot reproduce runs.

#### 2. Tau2 Benchmark (`examples/tau2_benchmark/`)

**Current CLI:**

```bash
uv run python examples/tau2_benchmark/tau2_benchmark.py \
    --framework default \
    --domain retail \
    --temperature 0.0  # Only reproducibility control
```

**Workflow:** Uses `temperature=0` as a workaround for determinism. No explicit seed.

#### 3. Five-A-Day Benchmark (`examples/five_a_day_benchmark/`)

**Current CLI:**

```bash
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py \
    --framework smolagents \
    --seed 42  # Has seeding!
```

**Workflow:** Manually implements seeding:

- Accepts `--seed` CLI argument
- Uses custom `derive_seed(base_seed, task_id, agent_id)` in `utils.py`
- Passes seeds to model factory per-agent
- Seeds stored in agent specs during data loading

**Code Pattern (five_a_day):**

```python
# In data loading (line 892-894):
for agent_spec in config["agents"]:
    agent_spec["seed"] = derive_seed(seed, task_id, agent_spec["agent_id"])

# In model creation (line 77-97):
def get_model(model_id, framework, temperature, seed=None):
    return LiteLLMModel(..., seed=seed)

# In agent building (line 255-256):
seed = primary_spec.get("seed")
model = get_model(model_id, "smolagents", temperature, seed)
```

### Key Insight

The five_a_day example shows that **users need seeding** but are implementing it ad-hoc. This pattern should be standardized in the core library.

---

## Alternative Workflow Designs

### Design A: Benchmark-Level Seed (Recommended)

**Concept:** Single seed parameter on Benchmark that propagates automatically.

```python
# Simple case - just add seed
benchmark = MACSBenchmark(seed=42, callbacks=[logger])
results = benchmark.run(tasks, agent_data=config)

# CLI integration
parser.add_argument("--seed", type=int, default=None)
benchmark = MACSBenchmark(seed=args.seed, ...)
```

**Pros:**

- Minimal API change (one new parameter)
- Automatic propagation - users don't manage child seeds
- Backward compatible (seed=None means non-deterministic)

**Cons:**

- Less control over individual components
- Benchmark subclasses must call `super().__init__()` correctly

**Example Workflow:**

```bash
# Reproducible run
uv run python examples/macs_benchmark.py --domain travel --seed 42

# Variance study (run 10 times with different seeds)
for seed in {1..10}; do
    uv run python examples/macs_benchmark.py --domain travel --seed $seed
done
```

### Design B: Run-Level Seed

**Concept:** Seed specified at run time, allowing multiple runs with different seeds.

```python
benchmark = MACSBenchmark(callbacks=[logger])

# Different seeds per run without reinstantiation
results1 = benchmark.run(tasks, agent_data=config, seed=42)
results2 = benchmark.run(tasks, agent_data=config, seed=43)
```

**Pros:**

- More flexible for variance studies
- Single benchmark instance, multiple seeded runs
- Natural for hyperparameter sweeps

**Cons:**

- Seed not visible in benchmark config
- Could be confusing: is benchmark stateful?
- Harder to reason about reproducibility

### Design C: Seed Context Manager

**Concept:** Context-based seeding that affects all operations within scope.

```python
from maseval import seed_context

with seed_context(42):
    tasks = load_tasks("travel", shuffle=True)  # Uses context seed
    benchmark = MACSBenchmark(callbacks=[logger])
    results = benchmark.run(tasks, agent_data=config)  # Uses context seed
```

**Pros:**

- Affects all operations uniformly
- No API changes to existing functions
- Familiar pattern from numpy/torch

**Cons:**

- Implicit behavior - harder to debug
- Global state can cause issues in concurrent code
- Doesn't match MASEval's explicit design philosophy

### Design D: Layered Seed Override

**Concept:** Hierarchical seeds with explicit override capability.

```python
# Global seed with local overrides
benchmark = MACSBenchmark(
    seed=42,
    seed_overrides={
        "task_shuffle": 123,  # Override shuffle seed
        "user_simulator": 456,  # Override user seed
    }
)

# Or via data loader
tasks = load_tasks("travel", shuffle=True, seed=123)  # Independent seed
benchmark = MACSBenchmark(seed=42)  # Execution seed
results = benchmark.run(tasks, agent_data=config)
```

**Pros:**

- Maximum flexibility
- Clear separation of concerns
- Supports advanced reproducibility studies

**Cons:**

- More complex API
- Users must understand seed hierarchy
- More ways to make mistakes

### Design E: Configuration Object

**Concept:** Dedicated configuration object for all seed-related settings.

```python
from maseval import SeedConfig, MACSBenchmark

seed_config = SeedConfig(
    global_seed=42,
    shuffle_seed=None,  # Derive from global
    per_agent_seeds={"supervisor": 100},  # Override specific agent
    log_seeds=True,  # Include in results
)

benchmark = MACSBenchmark(seed_config=seed_config, callbacks=[logger])
results = benchmark.run(tasks, agent_data=config)
```

**Pros:**

- All seed settings in one place
- Self-documenting
- Easy to serialize/deserialize for reproducibility

**Cons:**

- New class to learn
- Overkill for simple cases
- Breaking change if made required

---

## Recommended Approach: Hybrid of A + D

Combine simplicity of Design A with flexibility of Design D:

### Simple Case (Most Users)

```python
# Just works - single seed controls everything
benchmark = MACSBenchmark(seed=42)
results = benchmark.run(tasks, agent_data=config)
```

### Advanced Case (Researchers)

```python
# Data loading with its own seed (for reproducible splits)
tasks = load_tasks("travel", shuffle=True, seed=100)

# Benchmark with different seed (for execution randomness)
benchmark = MACSBenchmark(seed=42)
results = benchmark.run(tasks, agent_data=config)

# Access derived seeds for debugging
print(benchmark.seed_manager.get_config())
# {"global_seed": 42, "component_seeds": {"task/1/user": 8273645, ...}}
```

### CLI Pattern for All Examples

```python
# Standard CLI arguments for all benchmarks
parser.add_argument("--seed", type=int, default=None,
    help="Random seed for reproducibility (default: None for non-deterministic)")
parser.add_argument("--shuffle", action="store_true",
    help="Shuffle task order (uses --seed if provided)")

# Usage
benchmark = BenchmarkClass(seed=args.seed, ...)
tasks = load_tasks(domain, shuffle=args.shuffle, seed=args.seed)
```

### Resulting User Workflow

```bash
# Non-reproducible (current default behavior)
uv run python examples/macs_benchmark.py --domain travel

# Reproducible with shuffled tasks
uv run python examples/macs_benchmark.py --domain travel --seed 42 --shuffle

# Variance study
for i in {1..10}; do
    uv run python examples/macs_benchmark.py --domain travel --seed $i --shuffle
done

# Debug specific run (recover exact execution)
uv run python examples/macs_benchmark.py --domain travel --seed 42 --task-id task_001
```

---

## Workflow Comparison Table

| Design             | API Complexity | User Effort    | Flexibility | Debuggability | Recommended For |
| ------------------ | -------------- | -------------- | ----------- | ------------- | --------------- |
| A: Benchmark-level | Low            | Minimal        | Medium      | Good          | Most users      |
| B: Run-level       | Low            | Minimal        | High        | Medium        | Sweep studies   |
| C: Context         | Medium         | Low            | Medium      | Poor          | NumPy users     |
| D: Layered         | High           | High           | Very High   | Excellent     | Researchers     |
| E: Config object   | Medium         | Medium         | Very High   | Excellent     | Large projects  |
| **Hybrid A+D**     | Low-Medium     | Minimal-Medium | High        | Excellent     | **All users**   |

---

## Open Questions

1. **Should seed be in `run()` or `__init__()`?**
   - Proposal: `__init__()` for consistency with other config
   - Alternative: `run()` allows different seeds per run without reinstantiation

2. **How to handle providers without seed support?**
   - Proposal: Log warning, recommend temperature=0
   - Alternative: Raise exception to force user acknowledgment

3. **Should we support numpy/torch seeding?**
   - Currently only stdlib `random` is used
   - Could add `numpy_seed`, `torch_seed` if needed by environments

4. **Per-repetition seed strategy?**
   - Proposal: Derive from `task_seed + repetition_index`
   - Alternative: All repetitions use same seed (measure LLM variance only)

5. **Should `load_tasks()` use benchmark seed automatically?**
   - Proposal: Keep data loading independent (explicit seed)
   - Alternative: Benchmark could control data loading seed too

6. **How to handle existing five_a_day seeding pattern?**
   - Proposal: Migrate to new system, deprecate custom `derive_seed`
   - Alternative: Support both patterns during transition
