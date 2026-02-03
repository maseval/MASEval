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

### Technical Workflows

Fundamental capabilities any seeding system must enable:

1. **Full Benchmark Reproduction** — Run a benchmark with global seed 42, get results A. Run again with seed 42, get identical results A. Run with seed 43, get different but internally consistent results B. The global seed cascades: global → task → component, so changing the global seed changes everything downstream.

2. **Single Task Reproduction** — Re-run task T in isolation with the same seed it received during a full benchmark run. Debug or analyze a specific failure without re-running the entire benchmark.

3. **Checkpoint Resumption** — Benchmark crashes at task 50. Resume from task 51 with correct seeds—tasks 51+ receive the same seeds they would have in a complete run. Requires seeds to derive from task.id, not execution order.

4. **Variance Quantification** — Run task T with K different repetition seeds to measure outcome distribution. Answers: "How reliable is my system on this task?"

### Research Questions

Research questions and methodologies that proper seeding infrastructure enables:

1. **Component Variance Attribution** — Vary one component's seed across repetitions while holding others constant. Isolates which component (agent, tool simulator, evaluator) drives outcome variance.

2. **Regression Detection** — Same tasks, same seeds, before and after a code change. Outcome differences indicate behavioral changes, not random noise. Useful for CI pipelines.

3. **Failure Mode Analysis** — Run many seeds, cluster the failures. Systematic bugs fail across all seeds; edge cases fail only on specific seeds. Separates "broken" from "flaky."

4. **Determinism Verification** — Run seed X twice, assert identical traces. Validates that seeding actually works end-to-end. Useful for CI and for catching components that ignore seeds.

5. **Model Version Drift Detection** — Same code, same seeds, different results months later. Indicates the provider silently updated the model. Seeding enables detection (you can prove something changed externally) but cannot prevent it.

## 3. Current State

MASEval has no systematic seeding. To understand what infrastructure is needed, here's what a user must do today to implement seeding manually.

### Current Manual Workflow

**Step 1: Define a seed derivation utility**

```python
# utils.py
def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive unique seed from base seed and component identifiers."""
    seed_string = f"{base_seed}:" + ":".join(str(c) for c in components)
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "big") & 0x7FFFFFFF
```

**Step 2: Derive seeds during data loading**

```python
def load_benchmark_data(seed: Optional[int] = None, ...):
    for idx, (task_dict, config) in enumerate(zip(tasks_raw, configs_raw)):
        task_id = task_dict["metadata"]["task_id"]

        # Manually derive seeds for each agent
        if seed is not None:
            for agent_spec in config["agents"]:
                agent_spec["seed"] = derive_seed(seed, task_id, agent_spec["agent_id"])

        configs_data.append(config)

    return TaskQueue(tasks_data), configs_data
```

**Step 3: Pass seeds through agent setup**

```python
def build_smolagents_single_agent(all_tool_adapters, primary_spec, ...):
    # Extract seed from agent spec
    seed = primary_spec.get("seed")

    # Pass to model
    model = LiteLLMModel(
        model_id="gemini/gemini-2.5-flash",
        seed=seed,
    )

    agent = ToolCallingAgent(model=model, tools=tools, ...)
    return SmolAgentAdapter(agent, primary_spec["agent_id"])
```

### What This Doesn't Cover

The manual approach above only seeds **agents**. A complete solution would also need to seed:

| Component       | Current Status                                                        |
| --------------- | --------------------------------------------------------------------- |
| Agents          | Manual seeding possible (shown above)                                 |
| User Simulators | No seeding — `setup_user()` doesn't receive seeds                     |
| Tool Simulators | No seeding — `setup_environment()` doesn't receive seeds              |
| Evaluators      | No seeding — `setup_evaluators()` doesn't receive seeds               |
| Environments    | No seeding — randomized parameters can't be controlled                |

### Problems with Manual Seeding

1. **Boilerplate** — Every benchmark must implement seed derivation and plumbing
2. **Incomplete** — Only covers components where users write custom setup code
3. **No logging** — Seeds used aren't automatically recorded in results
4. **Error-prone** — Easy to forget a component or derive seeds inconsistently

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
