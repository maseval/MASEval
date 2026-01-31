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

**Use case:** "Keep tools, user, and evaluator constant across 5 repetitions. Only vary agent X to measure its specific contribution to outcome variance."

```python
# Hypothetical config-based approach (rejected - too complex)
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

**The problem with single-seed approach:**

Once `rep_index` is baked into a seed, all children vary together. You can't selectively "undo" it for specific components.

**Solution: Two seeds**

Benchmark computes and passes both:
- `task_seed` = derived from `task.id` only → **constant across repetitions**
- `rep_seed` = derived from `task.id + rep_index` → **varies per repetition**

```python
def setup_agents(self, ..., task_seed=None, rep_seed=None):
    for agent in agents:
        if agent.name == "varying_agent":
            seed = derive_seed(rep_seed, agent.name)   # varies per rep
        else:
            seed = derive_seed(task_seed, agent.name)  # constant across reps
```

**Recommendation:** Pass both `task_seed` and `rep_seed` to all setup methods. The common case uses `rep_seed` (all vary), but developers can selectively use `task_seed` for components that should remain constant across repetitions.

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

- Benchmark derives two seeds per task: `task_seed` (constant) and `rep_seed` (varies per rep)
- Benchmark passes both to `setup_environment(task, task_seed=X, rep_seed=Y)`
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
2. **Two seeds per task**: `task_seed` (constant across reps) and `rep_seed` (varies per rep)
3. **Selective variance**: Developers choose which seed to use per component for variance attribution studies
4. **Hierarchical ownership**: Benchmark → Environment → Tools
5. **Same seed for all LLM calls** within a component
6. **Utility function + internal registry** - simple API, automatic logging
7. **Path-based derivation** internally, invisible to users
8. **Seeds recorded** in run metadata and per-task results
9. **Explicit failure** for unsupported providers

---

## Final Recommendation: Seed Injection with Hierarchical Derivation

### Core Principle

**Seeds flow down through method parameters, not through global state or registries.**

```python
# User API - single entry point
benchmark = Benchmark(seed=42)
results = benchmark.run(tasks, agent_data)
```

Internally, the benchmark:

1. Derives two seeds per task:
   - `task_seed = derive_seed(global_seed, task.id)` → constant across repetitions
   - `rep_seed = derive_seed(global_seed, f"{task.id}/{rep_index}")` → varies per repetition
2. Passes both seeds to all setup methods via parameters
3. Each component chooses which seed to use (or derives children from either)

### The Interface

```python
# Benchmark passes both seeds to all setup methods
def setup_environment(self, agent_data, task, task_seed=None, rep_seed=None) -> Environment:
    ...

def setup_user(self, agent_data, environment, task, task_seed=None, rep_seed=None) -> User:
    ...

def setup_evaluators(self, environment, task, agents, user, task_seed=None, rep_seed=None) -> List[Evaluator]:
    ...
```

### Component Responsibility

Each component that receives seeds owns derivation for its children. The developer chooses which seed to use based on whether the component should vary per repetition:

```python
def setup_environment(self, agent_data, task, task_seed=None, rep_seed=None):
    env = MyEnvironment(task)

    # Common case: use rep_seed so tools vary per repetition
    seed = rep_seed

    if seed is not None:
        for tool_name in task.environment_data["tools"]:
            tool_seed = derive_seed(seed, tool_name)
            simulator = ToolLLMSimulator(model, seed=tool_seed)
            env.add_tool(tool_name, simulator)

    return env

def setup_agents(self, agent_data, environment, task, user, task_seed=None, rep_seed=None):
    # Advanced: keep some agents constant, vary others
    for agent in agents:
        if agent.name == "experimental_agent":
            # This agent varies per repetition - measure its variance contribution
            seed = derive_seed(rep_seed, agent.name)
        else:
            # These agents stay constant across repetitions
            seed = derive_seed(task_seed, agent.name)
        ...
```

### The Utility Function

Single stateless function, usable anywhere:

```python
from maseval.core.seed import derive_seed

def derive_seed(parent_seed: int, child_name: str) -> int:
    """Deterministically derive a child seed from parent seed and name."""
    import hashlib
    combined = f"{parent_seed}:{child_name}"
    return int(hashlib.sha256(combined.encode()).digest()[:4].hex(), 16)
```

### Simulator/Model Interface

Simulators and models receive their seed at construction:

```python
class ToolLLMSimulator:
    def __init__(self, model, prompt_template, seed=None):
        self._seed = seed

    def __call__(self, **inputs):
        return self.model.generate(
            prompt,
            generation_params={"seed": self._seed}
        )
```

### Why This Pattern is Clean

| Property | How It's Achieved |
|----------|-------------------|
| **Single entry point** | Just `Benchmark(seed=42)` |
| **Explicit propagation** | Seeds flow through method signatures |
| **No global state** | Stateless utility function |
| **Follows ownership** | Environment seeds its tools, User seeds its simulator |
| **Testable** | Any component can be tested with a specific seed |
| **Opt-in everywhere** | `seed=None` is the default at every level |
| **Composable** | Components don't know about Benchmark, just their own seed |
| **Variance attribution** | Two seeds (`task_seed`, `rep_seed`) enable isolating variance sources |

### Logging

The benchmark internally tracks what seeds it computed:

```python
class Benchmark:
    def __init__(self, seed=None):
        self._global_seed = seed
        self._seed_log = {}  # Populated during run

    def _get_seeds(self, task, rep_index):
        if self._global_seed is None:
            return None, None

        task_seed = derive_seed(self._global_seed, task.id)
        rep_seed = derive_seed(self._global_seed, f"{task.id}/{rep_index}")

        # Log both
        self._seed_log[f"{task.id}/task_seed"] = task_seed
        self._seed_log[f"{task.id}/rep/{rep_index}"] = rep_seed

        return task_seed, rep_seed
```

Results include the seed log:

```python
results = benchmark.run(tasks, agent_data)
# results includes:
{
    "global_seed": 42,
    "seeds": {
        "task_001/task_seed": 7382910,      # constant across reps
        "task_001/rep/0": 8273645,          # rep 0
        "task_001/rep/1": 1928374,          # rep 1
        ...
    }
}
```

### What Components Don't Need to Do

- No calling `benchmark.get_seed()` - they receive seeds via parameters
- No managing registries - benchmark handles logging
- No knowing about paths - they just use names for derivation
- No complex configuration - just choose `task_seed` or `rep_seed` per component

### Edge Cases

**Provider doesn't support seeding:**

```python
# In ModelAdapter
def generate(self, prompt, generation_params=None):
    seed = generation_params.get("seed") if generation_params else None
    if seed is not None and not self._supports_seeding:
        raise SeedingNotSupportedError(...)
```

**User wants to override a specific component's seed:**

```python
# In task data
{"task_id": "task_001", "environment_seed": 99999, ...}

# In setup_environment
def setup_environment(self, agent_data, task, task_seed=None, rep_seed=None):
    # Task-level override takes precedence
    override = task.environment_data.get("environment_seed")
    seed = override if override is not None else rep_seed
    ...
```

---

## Pattern Summary

The pattern is: **"Receive both seeds, choose based on variance needs, derive for children."**

```
Benchmark(seed=42)
    │
    │  Computes per task+rep:
    │    task_seed = derive(42, task.id)           # constant across reps
    │    rep_seed = derive(42, task.id/rep_index)  # varies per rep
    │
    ├─► setup_environment(task_seed=T, rep_seed=R)
    │       │
    │       │  Developer chooses: use T (constant) or R (varying)
    │       │
    │       └─► derive_seed(chosen, "tool_name") → ToolSimulator(seed=Y)
    │
    ├─► setup_user(task_seed=T, rep_seed=R)
    │       └─► derive_seed(chosen, "simulator") → UserSimulator(seed=Y)
    │
    ├─► setup_agents(task_seed=T, rep_seed=R)
    │       ├─► derive_seed(R, "experimental_agent")  # varies - measure its variance
    │       └─► derive_seed(T, "baseline_agent")      # constant - control
    │
    └─► setup_evaluators(task_seed=T, rep_seed=R)
            └─► derive_seed(chosen, "llm_judge") → LLMJudge(seed=Y)
```

**Key properties:**
- No magic, no global state, just explicit parameter passing
- Two seeds enable selective variance attribution studies
- Common case: just use `rep_seed` everywhere (all vary)
- Advanced case: mix `task_seed` and `rep_seed` to isolate variance sources
- Works with existing architecture, minimal API changes
- Fully opt-in at every level
