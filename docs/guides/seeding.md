# Seeding for Reproducibility

## Overview

MASEval provides a seeding system for reproducible benchmark runs. Seeds cascade from a global seed through all components, enabling deterministic behavior when model providers support seeding.

This guide covers:

- **Basic usage**: Enabling seeding in benchmarks
- **Selective variance**: Controlling which components vary per repetition
- **Custom generators**: Implementing alternative seeding strategies
- **Provider support**: Which model providers support seeding

!!! info "When to use seeding"

    Seeding is most useful when:

    - Comparing agent architectures under controlled conditions
    - Running ablation studies where you need reproducibility
    - Debugging issues that only appear intermittently
    - Creating reproducible baselines for publication

    Note that not all model providers support seeding, and even those that do may only offer "best-effort" determinism.

## Basic Usage

### Enabling Seeding

Pass a `seed` parameter when creating your benchmark:

```python
from maseval import Benchmark

# Simple: pass a seed integer
benchmark = MyBenchmark(seed=42)
results = benchmark.run(tasks, agent_data=config)
```

This creates a `DefaultSeedGenerator` internally and passes it to all setup methods.

### Using Seeds in Setup Methods

All setup methods receive an optional `seed_generator` parameter. Use it to derive seeds for your components:

```python
from maseval import Benchmark, SeedGenerator
from typing import Optional

class MyBenchmark(Benchmark):
    def setup_agents(
        self,
        agent_data,
        environment,
        task,
        user,
        seed_generator: Optional[SeedGenerator] = None,
    ):
        # Derive a seed for your agent using hierarchical paths
        agent_seed = None
        if seed_generator is not None:
            # Use child() to create logical namespaces - results in "agents/orchestrator"
            agent_gen = seed_generator.child("agents")
            agent_seed = agent_gen.derive_seed("orchestrator")

        # Pass seed to model adapter
        model = self.get_model_adapter(model_id, seed=agent_seed)
        agent = MyAgent(model=model)
        # ... rest of setup
```

Seeds are derived from hierarchical paths, so `derive_seed("orchestrator")` within a `child("agents")` context produces `"agents/orchestrator"`, which is different from `"agents/baseline"`.

## Selective Variance with `per_repetition`

When running multiple repetitions of the same task, you may want some components to vary while others remain constant. The `per_repetition` flag controls this:

```python
def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
    if seed_generator is not None:
        # Use child() to group agent seeds under "agents/" namespace
        agent_gen = seed_generator.child("agents")

        # Varies per repetition - different seed for rep 0, 1, 2, ...
        # Results in path: "agents/experimental"
        experimental_seed = agent_gen.derive_seed("experimental", per_repetition=True)

        # Constant across repetitions - same seed for rep 0, 1, 2, ...
        # Results in path: "agents/baseline"
        baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)
```

**Use cases:**

| `per_repetition` | Behavior                         | Use case                              |
| ---------------- | -------------------------------- | ------------------------------------- |
| `True` (default) | Seed varies per repetition       | Experimental agents, ablation studies |
| `False`          | Seed constant across repetitions | Baseline agents, control conditions   |

## Hierarchical Namespacing

For complex systems with many components, use `child()` to create hierarchical namespaces:

```python
def setup_environment(self, agent_data, task, seed_generator=None):
    if seed_generator is not None:
        # Create a child generator for environment components
        env_gen = seed_generator.child("environment")

        # Further nest tools under "environment/tools/"
        tools_gen = env_gen.child("tools")
        weather_seed = tools_gen.derive_seed("weather")  # "environment/tools/weather"
        search_seed = tools_gen.derive_seed("search")    # "environment/tools/search"

def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
    if seed_generator is not None:
        # Create a child generator for agents
        agent_gen = seed_generator.child("agents")

        orchestrator_seed = agent_gen.derive_seed("orchestrator")  # "agents/orchestrator"

        # Nest workers under "agents/workers/"
        worker_gen = agent_gen.child("workers")
        analyst_seed = worker_gen.derive_seed("analyst")           # "agents/workers/analyst"
```

Child generators share the same seed log, so all derived seeds are recorded together.

!!! note "Flat paths work too"

    You can use flat paths directly without `child()`:

    ```python
    seed_generator.derive_seed("environment/tools/weather")
    seed_generator.derive_seed("agents/orchestrator")
    ```

    Both approaches produce identical seeds. Use `child()` when it makes your code cleaner.

## How Seed Derivation Works

This section demonstrates the core mechanics of seed derivation with concrete examples.

### Basic Example

```python
from maseval import DefaultSeedGenerator

# Create a generator with a global seed
gen = DefaultSeedGenerator(global_seed=0)

# Scope to a task and repetition (required before deriving seeds)
task_gen = gen.for_task("task_1").for_repetition(0)

# Derive seeds for components
agent_seed = task_gen.derive_seed("agent")
print(agent_seed)  # 778051139

# Different paths produce different seeds
env_seed = task_gen.derive_seed("environment")
print(env_seed)  # 1348051591

# Child generators extend the path
tools_gen = task_gen.child("tools")
weather_seed = tools_gen.derive_seed("weather")  # Path: "tools/weather"
print(weather_seed)  # 1528663065
```

### Determinism: Same Path = Same Seed

The key property of the seed generator is **determinism**: the same path always produces the same derived seed, even when called multiple times.

```python
gen = DefaultSeedGenerator(global_seed=0).for_task("task_1").for_repetition(0)

# Call the same path twice on the same generator
seed1 = gen.derive_seed("agent")
seed2 = gen.derive_seed("agent")

print(seed1)  # 778051139
print(seed2)  # 778051139
assert seed1 == seed2  # Always true
```

This also works across separate generator instances with the same configuration:

```python
# Two separate generators with identical configuration
gen1 = DefaultSeedGenerator(global_seed=0).for_task("task_1").for_repetition(0)
gen2 = DefaultSeedGenerator(global_seed=0).for_task("task_1").for_repetition(0)

seed1 = gen1.derive_seed("agent")
seed2 = gen2.derive_seed("agent")

assert seed1 == seed2  # Always true
```

This is what enables reproducibility - if you record the global seed used in an experiment, you can recreate the exact same derived seeds later.

### Different Global Seeds = Different Results

Changing the global seed changes all derived seeds:

```python
# With seed=0
gen_0 = DefaultSeedGenerator(global_seed=0).for_task("task_1").for_repetition(0)
print(gen_0.derive_seed("agent"))        # 778051139
print(gen_0.derive_seed("environment"))  # 1348051591

# With seed=1 - same paths, different seeds
gen_1 = DefaultSeedGenerator(global_seed=1).for_task("task_1").for_repetition(0)
print(gen_1.derive_seed("agent"))        # 1297896250
print(gen_1.derive_seed("environment"))  # 886012105
```

This allows you to run multiple independent experiments by simply changing the global seed.

### Seed Log

The generator tracks all derived seeds, which is useful for debugging and reproducibility:

```python
gen = DefaultSeedGenerator(global_seed=42).for_task("task_1").for_repetition(0)

# Derive several seeds
gen.derive_seed("agent")
tools_gen = gen.child("tools")
tools_gen.derive_seed("weather")
tools_gen.derive_seed("search")

# Inspect what was derived
print(gen.seed_log)
# {'agent': 1608637542, 'tools/weather': 353148029, 'tools/search': 906566780}
```

The seed log is included in benchmark reports automatically, so you always have a record of which seeds were used.

## Model Provider Support

Not all providers support seeding. Here's the current status:

| Provider     | Support       | Notes                                                   |
| ------------ | ------------- | ------------------------------------------------------- |
| OpenAI       | Best-effort   | Seed parameter accepted, but determinism not guaranteed |
| Google GenAI | Supported     | Seed parameter passed to generation config              |
| LiteLLM      | Pass-through  | Passes seed to underlying provider                      |
| HuggingFace  | Supported     | Uses `transformers.set_seed()`                          |
| Anthropic    | Not supported | Raises `SeedingError` if seed provided                  |

If you pass a seed to an adapter that doesn't support seeding, it raises `SeedingError` at creation time:

```python
from maseval import SeedingError

try:
    adapter = AnthropicModelAdapter(client, model_id="claude-3", seed=42)
except SeedingError as e:
    print(f"Provider doesn't support seeding: {e}")
```

## Seed Logging

All derived seeds are automatically logged and included in results:

```python
results = benchmark.run(tasks, agent_data=config)

for report in results:
    seed_config = report["config"]["seeding"]["seed_generator"]
    print(f"Global seed: {seed_config['global_seed']}")
    print(f"Task: {seed_config['task_id']}")
    print(f"Repetition: {seed_config['rep_index']}")
    print(f"Seeds used: {seed_config['seeds']}")
    # Output: {"agents/orchestrator": 12345, "agents/workers/analyst": 67890, ...}
```

This enables exact reproduction of benchmark runs and debugging of seed-related issues.

## Custom Seed Generators

### Using a Different Hash Algorithm

Subclass `DefaultSeedGenerator` and override `_compute_seed()`:

```python
import hashlib
from maseval import DefaultSeedGenerator

class MD5SeedGenerator(DefaultSeedGenerator):
    """Uses MD5 instead of SHA-256."""

    def _compute_seed(self, full_path: str, components: list) -> int:
        seed_string = ":".join(str(c) for c in components)
        hash_bytes = hashlib.md5(seed_string.encode()).digest()
        return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF

# Use it
benchmark = MyBenchmark(seed_generator=MD5SeedGenerator(global_seed=42))
```

### Implementing a Custom Generator

For completely custom seeding strategies (e.g., database-backed seeds), implement the `SeedGenerator` ABC:

```python
from maseval import SeedGenerator
from typing import Dict, Any
from typing_extensions import Self

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

## Tips

**For reproducibility**: Always log and report the global seed used. Include it in publications and experiment tracking.

**For debugging**: Use `seed_generator.seed_log` to inspect which seeds were derived during a run.

**For baselines**: Use `per_repetition=False` for baseline agents that should remain constant while you vary experimental agents.

**For provider compatibility**: Check provider support before relying on seeding. OpenAI's seeding is "best-effort" and may not be perfectly deterministic.
