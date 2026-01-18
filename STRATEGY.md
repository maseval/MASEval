# ARE/Gaia2 Integration Strategy for MASEval

This document proposes integration approaches for adding Meta's ARE (Agent Research Environments) with the Gaia2 benchmark to MASEval.

## Background

### What is ARE?

Meta's ARE is a research platform for evaluating AI agents in dynamic, multi-step environments. Key characteristics:

- **Simulation-driven execution**: Time-based event loop with real-time state evolution
- **11 interactive applications**: Calendar, Email, Contacts, Messaging, Shopping, Cab, City, FileSystem, etc.
- **Event DAG system**: Scenarios define events with dependencies that trigger state changes
- **Dynamic validation**: Graph-based judge evaluating event timing, order, and correctness
- **Asynchronous execution**: Surfaces failure modes invisible in static benchmarks

### What is Gaia2?

Gaia2 is a benchmark built on ARE evaluating general agent capabilities across 7 dimensions:
- **Execution**: Multi-step instruction following (near-solved)
- **Search**: Cross-source information gathering (near-solved)
- **Ambiguity**: Clarification of conflicting requests (challenging)
- **Adaptability**: Response to environmental changes (challenging)
- **Time/Temporal**: Time-sensitive actions (hardest)
- **Agent2Agent**: Multi-agent collaboration
- **Noise**: Robustness to API failures

**Scale**: 800 scenarios across 10 pre-populated "universe" personas, with ~1000+ human-created scenarios.

### MASEval Architecture Overview

MASEval uses a task-driven orchestration pattern:

```
Benchmark.run(tasks)
  └→ For each Task:
       └→ setup_environment()  → Environment with tools
       └→ setup_user()         → Optional user simulator
       └→ setup_agents()       → Agent system
       └→ run_agents()         → Execute task
       └→ setup_evaluators()   → Create evaluators
       └→ evaluate()           → Compute metrics (with full traces)
```

**MASEval's trace system is comprehensive**:
- `TraceableMixin.gather_traces()` collects from all components
- `ToolInvocationHistory` records every call with inputs, outputs, timestamps
- `MessageHistory` captures full conversation flow
- `ComponentRegistry.collect_traces()` aggregates across agents, tools, models, environment, user
- Evaluators receive the **complete trace dict** via `filter_traces()` + `__call__()`

Current benchmarks (MACS, Tau2) demonstrate this:
- **MACSEvaluator**: Evaluates system-side assertions by examining tool invocation sequences
- **Tau2Evaluator**: Validates database state changes and can replay tool calls from traces
- Full event reconstruction is possible from MASEval traces

### Key Architecture Difference: Time Simulation

The core difference is NOT about trace availability, but about **how time and events work**:

| Aspect | MASEval | ARE |
|--------|---------|-----|
| Execution model | Task-driven (sequential) | Simulation-driven (event loop) |
| Time | Implicit (turn count) | Explicit simulation clock (seconds) |
| Environment events | Agent-initiated only | Concurrent (env can inject events at t=X) |
| Temporal validation | Not built-in | Native (action X within 10s of event Y) |
| State evolution | Tools modify on-demand | Clock-driven + event-triggered |

**What ARE adds that MASEval lacks**:
1. **Simulation clock**: Time advances in real seconds, events scheduled at specific times
2. **Environment-initiated events**: "User sends message at t=30s" independent of agent
3. **Temporal constraint validation**: Judge checks timing relationships in event DAG

### Critical Insight: ARE's Time Control is Tool-Based

A key architectural discovery: ARE exposes time simulation **through tools**, not through a custom execution loop. The `SystemApp` provides:

```python
# are/simulation/apps/system.py
class SystemApp(App):
    def get_current_time(self) -> dict:
        """Returns simulation timestamp, datetime, weekday."""

    def wait_for_notification(self, timeout: int) -> None:
        """Wait until notification received OR timeout elapses.
        Advances simulation time, processes scheduled events."""
```

**How temporal scenarios work in ARE:**

Example: "Send message. Wait 3 minutes. If no response, order a cab."

1. Agent calls `send_message_to_user("Are you coming?")`
2. Agent calls `wait_for_notification(180)` ← simulation advances 180s
3. During that window, scheduled events fire (e.g., user message at t=30s)
4. Tool returns early if notification arrives, or after timeout
5. Agent checks results and decides next action

**Why this matters for MASEval integration:**

Since ARE's time control is tool-based:
- **No execution loop changes needed** in MASEval core
- Everything fits within `Environment.create_tools()`
- The ARE simulation runs internally; agent interacts purely via tool calls
- MASEval's existing trace system captures all tool invocations naturally

This makes Approach C (Hybrid Wrapper) significantly cleaner than initially anticipated. The wrapper simply:
1. Starts ARE's internal event loop
2. Exposes all app tools (including `SystemApp.wait_for_notification`)
3. Agent tool calls interact with the live simulation
4. MASEval traces everything normally

---

## Integration Approaches

### Approach A: ARE as External Dependency (Delegation)

**Concept**: Install ARE as a pip dependency and delegate scenario execution to ARE's `run_dataset()` function, then translate results to MASEval's format.

```python
# maseval/benchmark/gaia2/gaia2.py
from are.simulation.benchmark import run_dataset
from are.simulation.scenarios.config import MultiScenarioRunnerConfig

class Gaia2Benchmark(Benchmark):
    """Thin wrapper that delegates to ARE's execution engine."""

    def run(self, tasks, agent_data):
        # Convert MASEval tasks to ARE scenarios iterator
        scenarios = self._tasks_to_scenarios(tasks)

        # Configure ARE runner
        config = MultiScenarioRunnerConfig(
            model=agent_data["model"],
            model_provider=agent_data["provider"],
            scenario_timeout=agent_data.get("timeout", 300),
        )

        # Delegate to ARE
        are_result = run_dataset(scenarios, config)

        # Translate to MASEval reports
        return self._translate_results(are_result, tasks)
```

**Directory Structure**:
```
maseval/benchmark/gaia2/
├── __init__.py
├── gaia2.py           # Delegation wrapper
├── data_loader.py     # Load from HuggingFace dataset
└── result_adapter.py  # Translate ARE results to MASEval format
```

**Tradeoffs**:

| Pros | Cons |
|------|------|
| Minimal code to maintain | Less control over execution flow |
| Automatic updates from ARE upstream | Dependency on external package versioning |
| Uses ARE's validated judge system | Cannot integrate MASEval callbacks during execution |
| Guarantees result replication | Model adapter incompatibility (ARE uses litellm) |
| Simple implementation (~500 lines) | Limited customization of agent behavior |

**Key Considerations**:
- ARE uses `litellm` for model inference; MASEval users with custom `ModelAdapter` would need adaptation layer
- ARE's agent implementation (ReAct loop) may differ from user's custom agents
- Trace format translation required for MASEval compatibility
- pyproject.toml would add `meta-agents-research-environments` as optional dependency

---

### Approach B: Full Port (Native Implementation)

**Concept**: Port ARE's core components to MASEval, creating native implementations of the simulation environment, apps, event system, and judge.

```python
# maseval/benchmark/gaia2/environment.py
class Gaia2Environment(Environment):
    """Native MASEval environment with ARE-style simulation."""

    def __init__(self, task_data):
        self.time_manager = TimeManager()
        self.apps = self._create_apps(task_data)
        self.event_queue = EventQueue()

    def setup_state(self, task_data):
        # Initialize apps with scenario data
        for app in self.apps:
            app.load_state(task_data.get(app.name, {}))
        return {"apps": self.apps, "time": 0}

    def create_tools(self):
        # Aggregate tools from all apps
        tools = {}
        for app in self.apps:
            for tool in app.get_tools():
                tools[tool.name] = self._wrap_tool(tool)
        return tools

    def advance_time(self, seconds):
        # Process pending events in time window
        self.time_manager.advance(seconds)
        self._process_ready_events()
```

**Directory Structure**:
```
maseval/benchmark/gaia2/
├── __init__.py
├── gaia2.py               # Gaia2Benchmark implementation
├── environment.py         # Simulation environment
├── event_system.py        # Event queue, DAG, time manager
├── evaluator.py           # Graph-based judge
├── data_loader.py         # HuggingFace data loading
├── apps/
│   ├── __init__.py
│   ├── base.py            # App base class
│   ├── calendar.py        # Ported calendar app
│   ├── email.py           # Ported email app
│   ├── messaging.py       # Ported messaging app
│   ├── contacts.py        # Ported contacts app
│   ├── shopping.py        # Ported shopping app
│   └── ...                # 6+ more apps
├── judge/
│   ├── __init__.py
│   ├── graph_judge.py     # Event graph validation
│   ├── tool_checkers.py   # Per-tool argument validators
│   └── llm_judge.py       # LLM-based judgment
└── prompt_templates/
    ├── judge.txt
    └── agent_system.txt
```

**Estimated Scope**:
- ~15,000+ lines of ported code
- 11 apps to port and maintain
- Complex event system with time management
- Graph-based judge with LLM integration

**Tradeoffs**:

| Pros | Cons |
|------|------|
| Full MASEval integration | Large maintenance burden (~15k+ lines) |
| Native callbacks and tracing | Risk of divergence from ARE upstream |
| Custom agent flexibility | Time-intensive implementation (weeks) |
| No external dependency | Must replicate judge fidelity |
| Framework-agnostic tools | App bugs must be debugged independently |

**Key Considerations**:
- ARE apps have complex interdependencies (protocols, shared state)
- Event timing validation requires careful port of TimeManager
- Judge accuracy is critical for benchmark validity
- Would need extensive testing against ARE's validation suite

---

### Approach C: Hybrid Wrapper (Recommended)

**Concept**: Use ARE as a dependency for the simulation engine and judge, but create a MASEval-native wrapper that:
1. Integrates with MASEval's `Benchmark` lifecycle
2. Provides native `Environment` access to ARE's apps/tools
3. Translates traces bidirectionally for full observability
4. Allows custom agent implementations via MASEval's `AgentAdapter`

**Key insight**: Because ARE's time control is tool-based (see above), this wrapper requires **no changes to MASEval's core execution loop**. The simulation complexity is fully encapsulated within `Environment`.

```python
# maseval/benchmark/gaia2/gaia2.py
from are.simulation.environment import Environment as AREEnvironment, EnvironmentConfig
from are.simulation.scenarios import Scenario
from are.simulation.validation import JudgeFactory

class Gaia2Environment(Environment):
    """MASEval Environment wrapping ARE's simulation.

    The ARE simulation runs its own internal event loop. Agent interaction
    happens purely through tool calls - including time control via
    SystemApp.wait_for_notification(). No special execution loop needed.
    """

    def __init__(self, scenario: Scenario):
        self._are_env = AREEnvironment(EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
        ))
        self._scenario = scenario

    def setup_state(self, task_data):
        # Initialize ARE scenario and start simulation
        self._scenario.initialize()
        for app in self._scenario.apps:
            self._are_env.register_app(app)

        # Start ARE's internal event loop (runs in background)
        self._are_env.run(self._scenario)

        return {"are_env": self._are_env, "scenario": self._scenario}

    def create_tools(self):
        # Wrap ALL app tools including SystemApp (time control)
        # Agent uses these to interact with live simulation:
        # - Calendar, Email, Messaging, etc. for actions
        # - SystemApp.get_current_time() for time queries
        # - SystemApp.wait_for_notification(timeout) for temporal scenarios
        tools = {}
        for app in self._scenario.apps:
            for tool in app.get_tools():
                tools[tool.name] = AREToolWrapper(tool, self)
        return tools

    def cleanup(self):
        """Stop ARE simulation when task completes."""
        if self._are_env:
            self._are_env.stop()


class Gaia2Benchmark(Benchmark):
    """Hybrid benchmark using ARE simulation with MASEval orchestration.

    Note: No custom execution loop needed. Agent interacts with the
    running ARE simulation purely through tool calls.
    """

    def setup_environment(self, agent_data, task):
        scenario = self._load_scenario(task)
        return Gaia2Environment(scenario)

    def setup_evaluators(self, environment, task, agents, user):
        # Use ARE's validated judge system
        judge_config = GraphPerEventJudgeConfig(
            engine=self._get_judge_engine(task)
        )
        judge = JudgeFactory.create(judge_config)
        return [Gaia2Evaluator(judge, environment._scenario, environment._are_env)]

    def run_agents(self, agents, task, environment, query):
        # Standard MASEval agent execution - no special handling needed
        # Agent tool calls interact with the live ARE simulation
        try:
            result = agents[0].run(query)
        finally:
            environment.cleanup()
        return result


class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    """Wraps ARE AppTool for MASEval compatibility and tracing."""

    def __init__(self, are_tool, environment):
        self.are_tool = are_tool
        self.environment = environment
        self.name = are_tool.name
        self.description = are_tool.description
        self.inputs = self._extract_schema(are_tool)
        self.history = ToolInvocationHistory()

    def __call__(self, **kwargs):
        # Execute in ARE's live simulation
        result = self.are_tool(**kwargs)

        # Record for MASEval tracing (captures time, inputs, outputs)
        self.history.add_invocation(
            inputs=kwargs,
            outputs=result,
            meta={"simulation_time": self.environment._are_env.time_manager.time()}
        )
        return result
```

**Directory Structure**:
```
maseval/benchmark/gaia2/
├── __init__.py
├── gaia2.py           # Gaia2Benchmark, Gaia2User
├── environment.py     # Gaia2Environment wrapping ARE
├── evaluator.py       # Gaia2Evaluator using ARE judge
├── tool_wrapper.py    # AREToolWrapper for tracing
├── data_loader.py     # load_tasks(), configure_model_ids()
└── prompt_templates/
    └── user_simulator.txt
```

**Tradeoffs**:

| Pros | Cons |
|------|------|
| Best of both worlds | Moderate complexity |
| Full MASEval integration | Still depends on ARE package |
| Uses validated ARE judge | Must handle ARE's litellm vs MASEval adapters |
| Native callbacks and tracing | Need wrapper code for tools (~1000 lines) |
| Custom agents supported | Some version coupling risk |
| Moderate code (~2000 lines) | |

---

## Detailed Comparison

| Criterion | A: Delegation | B: Full Port | C: Hybrid |
|-----------|--------------|--------------|-----------|
| Implementation effort | Low (~500 lines) | High (~15k+ lines) | Medium (~2k lines) |
| MASEval integration depth | Shallow | Deep | Deep |
| Maintenance burden | Low | High | Medium |
| Result replication fidelity | High | Medium (risk) | High |
| Custom agent support | Limited | Full | Full |
| Callback/tracing support | Post-hoc only | Full | Full |
| Upstream dependency | Yes | No | Yes |
| Time to implement | Days | Weeks | 1-2 weeks |

---

## Recommendation: Approach C (Hybrid Wrapper)

**Rationale**:

1. **Clean architectural fit**: ARE's tool-based time control means the entire simulation can be encapsulated within MASEval's `Environment` class. No changes to MASEval core required - the complexity stays in the benchmark module where it belongs.

2. **Replication fidelity**: Using ARE's judge system ensures results match the leaderboard methodology exactly. Porting the judge risks subtle divergences that could invalidate comparisons.

3. **MASEval integration**: Unlike pure delegation (Approach A), the hybrid approach provides:
   - Full callback support (on_task_start, on_task_end, etc.)
   - Native trace collection compatible with MASEval's registry
   - Support for custom `AgentAdapter` implementations
   - Framework-agnostic tool wrappers
   - Simulation time captured in tool invocation metadata

4. **Practical maintenance**: ARE is actively developed (v1.2.0). Porting 15k+ lines creates a maintenance burden and divergence risk. The hybrid approach isolates MASEval-specific code (~2k lines) while leveraging ARE's validated components.

5. **Agent flexibility**: The wrapper allows users to bring their own agent frameworks (smolagents, LangGraph, LlamaIndex) while still using ARE's simulation engine. The agent just needs to use the provided tools - including `wait_for_notification()` for temporal scenarios.

### Implementation Plan

1. **Phase 1: Core Integration**
   - Add `meta-agents-research-environments` as optional dependency
   - Implement `Gaia2Environment` wrapping ARE's Environment
   - Implement `AREToolWrapper` for tool tracing
   - Implement `Gaia2Evaluator` delegating to ARE's JudgeFactory

2. **Phase 2: Data Loading**
   - Implement `load_tasks()` from HuggingFace dataset
   - Implement `configure_model_ids()` for model configuration
   - Add scenario-to-Task conversion

3. **Phase 3: User Simulation**
   - Implement `Gaia2User` for multi-turn scenarios
   - Handle stop conditions and temporal reasoning

4. **Phase 4: Testing & Validation**
   - Validate against ARE's reference results
   - Test with multiple agent frameworks
   - Add example notebooks

### Dependency Addition

```toml
# pyproject.toml
[project.optional-dependencies]
gaia2 = ["meta-agents-research-environments>=1.2.0"]
```

### Model Adapter Bridge

ARE uses `litellm` internally. To support MASEval's `ModelAdapter`:

```python
class MASEvalModelBridge:
    """Bridge MASEval ModelAdapter to ARE's litellm interface."""

    def __init__(self, model_adapter: ModelAdapter):
        self.adapter = model_adapter

    def completion(self, messages, **kwargs):
        response = self.adapter.chat(messages, **kwargs)
        return self._to_litellm_format(response)
```

This allows users to use their existing model adapters with Gaia2.

---

## Future Consideration: Native Time Simulation in MASEval

The hybrid approach (Approach C) is complete and requires no MASEval core changes. However, if future benchmarks emerge that need ARE-style time simulation **without** depending on the ARE package, MASEval could add native support:

**Potential `maseval.core.simulation` module**:
- `TimeManager`: Simulation clock with configurable resolution
- `EventQueue`: Priority queue for scheduled environment events
- `SimulatedEnvironment(Environment)`: Base class with `advance_time()` and event scheduling
- Time-aware `ToolInvocationHistory` (already partially supported via `meta` field)

**When this would make sense**:
- Multiple benchmarks need time simulation (not just Gaia2)
- Users want to avoid the ARE dependency (~50MB+ with all apps)
- Custom temporal scenarios beyond what ARE provides

**When the hybrid approach is better**:
- Single benchmark (Gaia2) - no need to generalize yet
- ARE's judge system is critical for leaderboard comparability
- ARE is actively maintained and battle-tested

**Recommendation**: Proceed with hybrid approach. Revisit native simulation only if:
1. Gaia2 integration reveals friction with ARE dependency
2. Other time-simulation benchmarks emerge
3. Users request lighter-weight temporal environments

---

## References

- ARE Paper: https://arxiv.org/abs/2509.17158
- ARE Repository: https://github.com/facebookresearch/meta-agents-research-environments
- Gaia2 Dataset: https://huggingface.co/datasets/meta-agents-research-environments/gaia2
- Gaia2 Leaderboard: https://huggingface.co/spaces/meta-agents-research-environments/leaderboard
- ARE Documentation: https://facebookresearch.github.io/meta-agents-research-environments/
- License: MIT (confirmed in repository LICENSE file)
