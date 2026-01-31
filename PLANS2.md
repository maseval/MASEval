# PLANS2.md - Robust Seeding System for MASEval (Revised)

## Corrected Priorities

| Component               | Priority     | Rationale                                                                    |
| ----------------------- | ------------ | ---------------------------------------------------------------------------- |
| Evaluators              | CRITICAL     | Determines scores. LLM-as-judge evaluators are non-deterministic!            |
| Environment             | CRITICAL     | State that agents interact with must be reproducible                         |
| Tool Simulators         | CRITICAL     | LLM-simulated tool responses affect agent behavior                           |
| User Simulators         | CRITICAL     | Same - simulated user drives multi-turn                                      |
| Agents                  | CRITICAL     | Agent decisions must be reproducible                                         |
| Models in General       | CRITICAL     | Model calls are everywhere and need to be capable of deterministic behavior. |
| Task Ordering           | Nice-to-have | Pattern needed for future stochastic queues (Thompson sampling)              |
| Stochastic Data Loading | Not needed   | See below                                                                    |

### On Stochastic Data Loading

Not relevant for evaluation:

- You want to evaluate on a fixed, deliberate set of tasks
- Subsets should be intentional, not random
- If you need bootstrap sampling for confidence intervals, that's analysis-time, not loading-time

The only edge case might be "random sample for quick iteration," but that's a development convenience, not an eval concern. Dropped from scope.

---

## Workflow Analysis

### Workflow 1: Seed per task from data

```python
# In tasks.json
{"task_id": "task_001", "seed": 12345, ...}
```

| Pros                                    | Cons                                                 |
| --------------------------------------- | ---------------------------------------------------- |
| Explicit, tasks are self-contained      | More work for data creators                          |
| Easy to reproduce a single task         | Hard to run "same tasks with different global seeds" |
| Works well for fixed benchmark datasets | Inflexible for variance studies                      |

**Verdict:** Good for published benchmarks with fixed seeds. Not sufficient alone.

### Workflow 2: Benchmark-level seed with derivation

```python
benchmark = Benchmark(seed=42)
# Internally derives: task_seed = derive(42, task.id)
```

**Key question: What to derive from?**

| Option                 | Behavior                                | Pros                            | Cons                               |
| ---------------------- | --------------------------------------- | ------------------------------- | ---------------------------------- |
| `task.id`              | Same task always gets same derived seed | Reproducible, order-independent | Requires unique task IDs           |
| `task index`           | Seed depends on position in queue       | Simple                          | Changing queue changes seeds - BAD |
| `task.id + repetition` | Different seed per repetition           | Enables variance measurement    | More complex derivation            |

**Verdict:** Derive from `task.id` (not index). For repetitions, derive from `task.id + repetition_index`.

### Workflow 3: Selective seeding across repetitions

Question: "Should we support keeping some components seeded during repetition and others not?"

**Use case:** "Measure LLM variance while keeping tool responses deterministic"

```python
# Hypothetical
benchmark = Benchmark(
    seed=42,
    vary_per_repetition=["agent"],  # Re-seed agent each rep
    constant_per_repetition=["tools", "user"],  # Keep same seed
)
```

| Option              | Behavior                                                       |
| ------------------- | -------------------------------------------------------------- |
| **A: All vary**     | `seed = derive(global, task_id, rep_index, component)`         |
| **B: All constant** | `seed = derive(global, task_id, component)` - same across reps |
| **C: Configurable** | User specifies which components vary                           |

**Analysis:**

- Option A is the natural default - each repetition is independent
- Option B is odd - why repeat if everything is identical?
- Option C is powerful but complex

**Recommendation:** Default to **A** (all vary per repetition). For advanced use case C, users can manually set seeds in task data. Not worth the API complexity for a niche use case.

### Workflow 4: Who manages seeds?

**Option A: Benchmark handles internally**

```python
# User never sees seeds
benchmark = Benchmark(seed=42)
results = benchmark.run(tasks, agent_data)
# Internally, benchmark passes seeds to setup_environment(), setup_user(), etc.
```

**Option B: Components request from benchmark**

```python
class MyEnvironment(Environment):
    def setup_state(self, task, benchmark):
        seed = benchmark.get_seed(f"env/{task.id}")
        # use seed
```

**Option C: Hierarchical - parent passes to children**

```python
# Benchmark gives environment a seed
# Environment derives seeds for its tools
env_seed = benchmark.get_seed(f"env/{task.id}")
tool_seed = derive(env_seed, tool_name)
```

| Option          | Pros                                           | Cons                                 |
| --------------- | ---------------------------------------------- | ------------------------------------ |
| A: Internal     | Simple for users, no API changes to components | Less flexibility, harder to debug    |
| B: Request      | Explicit, traceable                            | Components must know about benchmark |
| C: Hierarchical | Natural ownership, tools belong to environment | More complex derivation chain        |

**Recommendation:** Hybrid of A and C.

- Benchmark derives a seed per task+repetition
- Benchmark passes this seed to `setup_environment(task, seed=X)`
- Environment owns derivation for its children (tools)
- Same pattern for user simulator

This matches ownership: environment creates tools, so environment seeds them.

### Workflow 5: LLM seeding per call

**Question:** Do we use the same seed for all calls, or derive per call?

**How LLM APIs work:**

- OpenAI: `seed` parameter - same seed + same input = same output
- The seed makes the _sampling_ deterministic, not the output invariant to input

**Options:**

| Strategy              | Behavior                                      | Use case                   |
| --------------------- | --------------------------------------------- | -------------------------- |
| Same seed all calls   | Each unique prompt gets reproducible response | Simple, usually sufficient |
| Increment per call    | Call 1 uses seed, call 2 uses seed+1, etc.    | Paranoid reproducibility   |
| Derive per call index | `derive(component_seed, call_index)`          | Full traceability          |

**Analysis:**
For evaluation, **same seed for all calls** is correct:

- Different prompts naturally produce different outputs
- The seed ensures "if I run this exact prompt again, I get the same answer"
- Incrementing adds complexity without clear benefit

**Exception:** If the same prompt is called multiple times (rare in eval), you might want different responses. But for eval, you probably want the _same_ response for reproducibility.

**Recommendation:** Same seed for all calls within a component. Simple and sufficient.

### Workflow 6: Seed registry vs utility function

**Option A: Just a utility function**

```python
from maseval.core.seed import derive_seed

seed = derive_seed(parent_seed, "component_name")
```

**Option B: Central registry**

```python
benchmark.seed_registry.get("env/task_001/tool_x")  # Returns seed
benchmark.seed_registry.dump()  # Returns all seeds for logging
```

| Option      | Pros                                                 | Cons                                   |
| ----------- | ---------------------------------------------------- | -------------------------------------- |
| A: Utility  | Simple, stateless, no magic                          | No automatic logging, users must track |
| B: Registry | Automatic tracking, easy to dump for reproducibility | Global state, more complex             |

**Recommendation:** Both.

- Provide the utility function for flexibility
- Benchmark maintains a registry internally
- Registry is dumped to results automatically

```python
# Utility for advanced users
from maseval.core.seed import derive_seed

# Internal registry exposed via results
results = benchmark.run(...)
print(results.seed_log)  # All seeds used
```

---

## Derivation Characteristics

**Question:** Names or paths?

| Approach   | Example                                   | Pros                       | Cons                      |
| ---------- | ----------------------------------------- | -------------------------- | ------------------------- |
| **Names**  | `derive(42, "user_simulator")`            | Simple, short              | Must be globally unique   |
| **Paths**  | `derive(42, "task/001/rep/0/user")`       | Hierarchical, no conflicts | Verbose, rigid structure  |
| **Hybrid** | User says "user", internally becomes path | Best of both               | Implementation complexity |

**Recommendation:** Use **paths** internally, but make them implicit:

```python
# User just creates components normally
environment = self.setup_environment(task)  # Gets seed automatically

# Internally, benchmark tracks:
# "task/{task.id}/rep/{rep}/environment" -> seed 8273645
# "task/{task.id}/rep/{rep}/environment/tool_x" -> seed 1829374
```

Users never see paths unless debugging. The structure is:

```
task/{task_id}/rep/{rep_index}/
├── environment/
│   ├── tool_1
│   ├── tool_2
│   └── ...
├── user_simulator
├── agent/{agent_name}
└── evaluator/{evaluator_name}
```

---

## Where to Record Seeds

**In results structure:**

```python
results = {
    "run_metadata": {
        "global_seed": 42,
        "seed_derivation": "sha256",  # Method used
    },
    "task_results": [
        {
            "task_id": "task_001",
            "repetition": 0,
            "task_seed": 8273645,  # Derived seed for this task+rep
            "component_seeds": {  # Optional, for debugging
                "environment": 1234,
                "user_simulator": 5678,
                "evaluator/accuracy": 9012,
            },
            "score": 0.85,
            ...
        }
    ]
}
```

This enables:

1. **Reproduce entire run:** Use `global_seed`
2. **Reproduce single task:** Use `task_seed`
3. **Debug component:** Use `component_seeds`

---

## On Providers Without Seeding

**Recommendation:** Fail explicitly.

```python
def chat(self, messages, seed=None, ...):
    if seed is not None and not self._supports_seeding:
        raise SeedingNotSupportedError(
            f"{self.__class__.__name__} does not support seeding. "
            "Set temperature=0 for best-effort determinism, or use seed=None."
        )
```

Users must consciously opt out:

```python
# Option 1: Accept non-determinism
benchmark = Benchmark(seed=42, allow_unseeded_providers=True)

# Option 2: Use temperature=0 workaround
model = AnthropicAdapter(..., generation_params={"temperature": 0})
```

---

## Summary: Recommended Design

1. **Benchmark-level seed** that derives per task using `task.id` (not index)
2. **Per-repetition derivation** using `task.id + rep_index` - all components vary
3. **Hierarchical ownership**: Benchmark → Environment → Tools
4. **Same seed for all LLM calls** within a component
5. **Utility function + internal registry** - simple API, automatic logging
6. **Path-based derivation** internally, invisible to users
7. **Seeds recorded** in run metadata and per-task results
8. **Explicit failure** for unsupported providers
