# Seeding System for MASEval

## 1. Motivation

Unlike single-model benchmarks, multi-agent evaluation involves many stochastic components: agents, tool simulators, user simulators, evaluators, and environments themselves. Some make LLM calls; others use randomness directly (e.g., an environment that samples "how much do N apples cost?" with random N to prevent overfitting to leaked training data). Each component needs a seed.

Naive seeding breaks easily. Consider 10 tools seeded by index:

```python
for i, tool in enumerate(tools):
    tool.seed = derive_seed(parent, i)
```

Remove tool 7 for a debugging run → tools 8, 9, 10 now receive different seeds. Your "controlled" experiment just changed three unrelated components.

The same problem applies to agents, evaluators, any ordered collection. A robust seeding system must derive seeds from stable identifiers (names), not positions.

## 2. User Workflows

Reproducibility is essential for evaluation. Researchers need to:

1. **Reproduce a full benchmark run** — Run the same evaluation twice and get identical results.

2. **Reproduce a single task** — Debug or analyze one specific task with the same conditions.

3. **Measure variance** — Run the same task N times with different seeds to quantify outcome variance.

4. **Attribute variance to components** — Keep some components constant (e.g., tools, user simulator) while varying others (e.g., one agent) to isolate which component contributes to outcome variance.

## 3. Current State

MASEval has no systematic seeding. Components that require seeding:

| Component       | Why                                                                   |
| --------------- | --------------------------------------------------------------------- |
| Evaluators      | LLM-as-judge evaluators are non-deterministic                         |
| Environments    | May randomize task parameters, initial state, or dynamics             |
| Tool Simulators | LLM-simulated responses vary; real tools may have stochastic behavior |
| User Simulators | LLM-simulated user responses vary                                     |
| Agents          | Agent LLM calls are non-deterministic                                 |

Currently, users cannot reproduce runs. The only workaround is `temperature=0`, which doesn't guarantee determinism and isn't supported by all providers.

## 4. Design Requirements

- **Opt-in** — Users who don't need seeding or bring their own shouldn't be affected
- **Derive by name, not index** — Seeds must be derived from identifiers (task.id, tool name, agent name), never from indices. This ensures adding/removing a component doesn't shift seeds for other components.
- **Support repetitions** — Each repetition of a task should get a different seed
- **Selective variance** — Users should be able to keep some components constant across repetitions while varying others
- **Logging** — Seeds used must be recorded in results for reproducibility
- **Fail explicitly** — Providers without seed support should raise an error, not silently proceed
- **Extensible to task queues** — Future stochastic queue strategies (e.g., Thompson sampling) should be able to use the same seeding infrastructure

## 5. Proposed Options

### Option A: Decentralized (Two Seeds as Parameters)

Benchmark computes two seeds per task and passes both to setup methods:

```python
task_seed = derive_seed(global_seed, task.id)                    # constant across reps
rep_seed = derive_seed(global_seed, f"{task.id}/{rep_index}")    # varies per rep
```

Setup methods receive both:

```python
def setup_environment(self, agent_data, task, task_seed=None, rep_seed=None):
    seed = rep_seed  # common case: vary per repetition
    tool_seed = derive_seed(seed, "tool_x")  # local derivation
```

Selective variance by choosing which seed to derive from:

```python
def setup_agents(self, ..., task_seed=None, rep_seed=None):
    experimental_seed = derive_seed(rep_seed, "agent_x")   # varies
    baseline_seed = derive_seed(task_seed, "agent_y")      # constant
```

Utility function for local derivation:

```python
def derive_seed(parent: int, child: str) -> int:
    return int(hashlib.sha256(f"{parent}:{child}".encode()).digest()[:4].hex(), 16)
```

| Pros                                                   | Cons                                        |
| ------------------------------------------------------ | ------------------------------------------- |
| No new classes, just parameters and a utility function | Two parameters on every setup method        |
| Components are independent and testable                | Logging must be handled separately          |
| Explicit — code shows which seed is used               | Developers must understand two-seed pattern |

---

### Option B: Centralized (Single Generator, Flat Paths)

One `SeedGenerator` object. Components request seeds by path:

```python
class SeedGenerator:
    def derive_seed(self, path: str, task_id: str, rep_index: int,
                    per_repetition: bool = True) -> int:
        """Derive seed for a component path."""
```

Usage:

```python
def setup_environment(self, agent_data, task):
    seed = self.seed_generator.derive_seed(
        "environment", task.id, rep_index, per_repetition=True
    )
    tool_seed = self.seed_generator.derive_seed(
        "environment/tool_x", task.id, rep_index, per_repetition=True
    )
```

Selective variance via `per_repetition` flag:

```python
experimental_seed = generator.derive_seed("agent_x", task.id, rep, per_repetition=True)
baseline_seed = generator.derive_seed("agent_y", task.id, rep, per_repetition=False)
```

| Pros                                           | Cons                                           |
| ---------------------------------------------- | ---------------------------------------------- |
| Single interface with clear flag               | Context (task_id, rep_index) passed every call |
| Automatic logging — generator tracks all seeds | Components coupled to generator                |
| Extensible — subclass for custom strategies    | Flat paths don't reflect ownership hierarchy   |

---

### Option C: Hierarchical (Child Generators)

Generator spawns child generators. Each component receives a scoped generator:

```python
class SeedGenerator:
    def derive_seed(self, name: str, per_repetition: bool = True) -> int:
        """Derive a seed for a child component."""

    def child(self, name: str, per_repetition: bool = True) -> 'SeedGenerator':
        """Create a child generator scoped to a component."""
```

Benchmark creates child generators per task:

```python
task_generator = root_generator.child(task.id).child(str(rep_index))
env_generator = task_generator.child("environment")
self.setup_environment(agent_data, task, seed_generator=env_generator)
```

Components receive their own generator:

```python
def setup_environment(self, agent_data, task, seed_generator=None):
    for tool in tools:
        tool_seed = seed_generator.derive_seed(tool)
        # OR pass child generator to tool
        tool_generator = seed_generator.child(tool)
```

Selective variance by choosing `per_repetition` when creating children:

```python
experimental_gen = task_generator.child("agent_x", per_repetition=True)
baseline_gen = task_generator.child("agent_y", per_repetition=False)
```

| Pros                                           | Cons                                    |
| ---------------------------------------------- | --------------------------------------- |
| Maintains ownership hierarchy                  | More complex — generators passed around |
| Each component testable with its own generator | More objects to manage                  |
| Automatic logging via parent chain             | Developers must understand nesting      |
| Extensible at each level                       |                                         |

---

## 6. Comparison

| Criterion          | A: Decentralized             | B: Centralized      | C: Hierarchical         |
| ------------------ | ---------------------------- | ------------------- | ----------------------- |
| API complexity     | Low                          | Medium              | Medium-High             |
| New classes        | None (utility function)      | SeedGenerator       | SeedGenerator           |
| Logging            | Manual                       | Automatic           | Automatic               |
| Testability        | High                         | Medium              | High                    |
| Ownership model    | Local derivation             | Flat paths          | Nested generators       |
| Extensibility      | Low                          | High                | High                    |
| Selective variance | Choose task_seed vs rep_seed | per_repetition flag | per_repetition on child |
