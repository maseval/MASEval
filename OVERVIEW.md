# AgentAdapter Message Accumulation & Tracing: Investigation Overview

## 1. Status Quo: How Each Framework Handles Messages Across Invocations

### smolagents

**Source of truth**: `agent.memory.steps` (via `write_memory_to_messages()`)

**Key finding**: By default, smolagents **resets memory on every `.run()` call**.

```python
# smolagents/agents.py - MultiStepAgent.run()
def run(self, task, stream=False, reset=True, ...):  # reset=True is the DEFAULT
    if reset:
        self.memory.reset()   # Clears all steps
        self.monitor.reset()
```

The MASEval adapter calls `self.agent.run(query)` without specifying `reset=False`, meaning:
- Each invocation starts with a **clean memory**
- `get_messages()` (which calls `write_memory_to_messages()`) returns **only the current run's messages**
- Messages from previous invocations are **lost** from the agent's perspective

However, smolagents *can* accumulate if `reset=False` is passed. This is used in multi-turn scenarios like GradioUI.

**What smolagents does well for tracing**: Each `ActionStep` in memory has a `model_input_messages` field that captures **exactly what the LLM saw** when generating that step's response. This is the gold standard for answering "what context did the agent have?"

**Documentation sources**: [smolagents agents.py source](https://github.com/huggingface/smolagents), `AgentMemory.reset()` clears `self.steps = []`.

---

### LangGraph

**Source of truth depends on mode**:
- Stateless (no checkpointer): `_last_result` cache in adapter
- Stateful (with checkpointer + thread_id): `graph.get_state(config).values["messages"]`

**Key finding**: LangGraph has **two completely different behaviors** depending on configuration.

#### Stateless Mode (current default in adapter)

```python
# maseval langgraph adapter _run_agent():
initial_state = {"messages": [HumanMessage(content=query)]}
result = self.agent.invoke(initial_state)
self._last_result = result  # Overwritten each call
```

- Each `invoke()` creates a **fresh state** with only the new user message
- The result is cached in `_last_result`, which is **overwritten** on the next call
- `get_messages()` returns **only the last invocation's messages**
- Previous invocations' messages are **lost**

#### Stateful Mode (with checkpointer + thread_id)

When a checkpointer and `thread_id` are configured, LangGraph **accumulates messages across invocations** via its checkpoint system. The `add_messages` reducer in `MessagesState` appends new messages to the checkpointed state.

```python
# From LANGCHAIN.md documentation:
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

# First call
graph.invoke({"messages": [{"role": "user", "content": "hi! I'm bob"}]}, config)
# Second call - messages from first call are still in checkpoint state
graph.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, config)
# Agent remembers "bob" because all messages are accumulated in checkpoint
```

In this mode, `get_state(config).values["messages"]` returns **ALL messages from all invocations**, with no boundary markers between invocations. The `get_state_history(config)` API provides step-by-step checkpoint snapshots, which is LangGraph's mechanism for seeing intermediate states.

**Important caveat**: Accumulation depends on using `MessagesState` (which has the `add_messages` reducer annotation). A plain `TypedDict` with `messages: list` won't automatically accumulate; behavior depends on node implementation.

**Documentation source**: `LANGCHAIN.md` (local), LangGraph memory documentation.

---

### LlamaIndex

**Source of truth**: Per-run `Context` store (no persistent state on the workflow object)

**Key finding**: LlamaIndex AgentWorkflow is **stateless between runs by default**. There is **no `.memory` attribute** on `AgentWorkflow` itself.

```python
# LlamaIndex AgentWorkflow.run():
def run(self, user_msg=None, chat_history=None, memory=None, ctx=None, ...):
    if ctx is not None and ctx.is_running:
        return super().run(ctx=ctx, **kwargs)
    else:
        # Creates fresh Context internally when ctx=None
        start_event = start_event or AgentWorkflowStartEvent(...)
        return super().run(start_event=start_event, ctx=ctx)
```

The MASEval adapter does NOT pass `ctx` between calls:
```python
# maseval llamaindex adapter _run_agent_sync():
async def run_async():
    handler = self.agent.run(user_msg=query)  # No ctx passed
    result = await handler
    return result
```

- Each `.run()` creates a **fresh Context** with a **fresh `ChatMemoryBuffer`**
- `handler.ctx` (which contains the memory) is **discarded** after each call
- The adapter's `get_messages()` tries `self.agent.memory` (which doesn't exist on AgentWorkflow), then falls back to `_message_cache`
- `_message_cache` is **overwritten** each run via `_extract_messages_from_result()`
- Previous invocations' messages are **lost**

**To persist state**, you must explicitly pass `ctx=handler.ctx` or create a shared `Context` object.

**Documentation sources**: [LlamaIndex Agent State docs](https://docs.llamaindex.ai/en/stable/understanding/agent/state/), [LlamaIndex Agent Memory docs](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/), [AgentWorkflow source](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/agent/workflow/multi_agent_workflow.py).

---

## 2. Summary Table: Current Behavior

| Framework   | Default Accumulation | Can Accumulate? | MASEval Adapter Behavior | `get_messages()` Returns |
|-------------|---------------------|-----------------|--------------------------|--------------------------|
| smolagents  | No (`reset=True`)   | Yes (`reset=False`) | Per-run only (resets each time) | Current run's messages only |
| LangGraph (stateless) | No | No (unless checkpointer added) | Per-run only (`_last_result` overwritten) | Last run's messages only |
| LangGraph (stateful)  | Yes (via checkpoint) | Yes | Accumulated (via `get_state`) | All messages, no invocation boundaries |
| LlamaIndex  | No (fresh Context)  | Yes (pass `ctx`) | Per-run only (cache overwritten) | Last run's messages only |

**Conclusion**: Your intuition was partially correct. smolagents *can* keep track of messages internally (via `reset=False`), but the MASEval adapter doesn't use this. Currently, **all three adapters effectively reset messages per invocation** in the MASEval integration (unless LangGraph is configured with a checkpointer).

---

## 3. The Tracing Problem

### Your Scenario

Single agent, two invocations, 4 message pairs each:

```
Invocation 1:          Invocation 2:
  msg_in_1               msg_in_5
  msg_out_1              msg_out_5
  msg_in_2               msg_in_6
  msg_out_2              msg_out_6
  msg_in_3               msg_in_7
  msg_out_3              msg_out_7
  msg_in_4               msg_in_8
  msg_out_4              msg_out_8
```

### Problem 1: "How do we tell whether `msg_out_7` had only `msg_in_7` or all previous messages available?"

**Current state**: We can't. The traces don't capture what the LLM actually *saw* when generating each response.

- **smolagents** actually *does* have this data in `ActionStep.model_input_messages` (what the LLM saw for each step). This is captured in `adapter.logs` already. But it's a smolagents-specific feature.
- **LangGraph stateless**: The LLM only saw messages from the current invocation (the graph node receives the state, which only contains what was passed in).
- **LangGraph stateful**: The LLM saw ALL accumulated messages (from checkpoint + new input). But we don't capture a snapshot of "what the LLM saw at step N."
- **LlamaIndex**: The LLM only saw messages from the current invocation (fresh ChatMemoryBuffer each time). No record of what was visible.

**What we need**: A `model_input_messages` field (like smolagents has) for every adapter, capturing the actual LLM input at each step.

### Problem 2: "How do we see when invocations happened? Post-hoc instead of just one long list of runs?"

**Current state**: No invocation boundaries are recorded.

- `get_messages()` returns either a single run's messages (smolagents, LlamaIndex, stateless LangGraph) or a flat accumulated list (stateful LangGraph) with no boundary markers.
- `adapter.logs` (for LangGraph/LlamaIndex) has one entry per `run()` call with timestamps, but this is separate from the message history.
- smolagents `logs` property generates entries from `memory.steps`, which reset each run.

**The gap**: There's no unified structure that links "these messages belong to invocation N" with "the invocation started/ended at time T."

### Problem 3: "Some agents persist state across runs or not. Traces should be collected permanently."

**Current state**: This is the core tension.

- **Agent execution**: Should respect the framework's memory behavior (reset vs. accumulate). This affects agent functioning and scientific integrity.
- **Tracing/evaluation**: Should capture everything permanently across all invocations, regardless of whether the agent "forgot" previous messages.

Currently, `gather_traces()` is called (presumably) at the end, and returns whatever `get_messages()` returns at that moment. For frameworks that reset, this means **only the last run's messages are captured in traces**. Everything before the last run is lost.

---

## 4. Analysis: What Needs to Change

### Core Issue

The adapter conflates two concerns:
1. **Agent memory management** (what the agent sees during execution)
2. **Trace collection** (what the evaluator needs to see post-hoc)

These should be decoupled. The adapter should:
- Let the agent manage its own memory (respecting `reset=True/False`, checkpointers, context, etc.)
- Independently accumulate a complete trace of ALL messages across ALL invocations

### What a Solution Needs

1. **Invocation-scoped trace capture**: After each `run()`, capture a snapshot of `get_messages()` tagged with an invocation ID/index and timestamp. This builds an ordered list of `(invocation_id, timestamp, messages)` tuples.

2. **Cumulative trace storage**: A list-of-invocations structure rather than a flat list of messages:
   ```python
   traces = [
       {"invocation": 0, "timestamp": "...", "messages": [msg_in_1, msg_out_1, ...]},
       {"invocation": 1, "timestamp": "...", "messages": [msg_in_5, msg_out_5, ...]},
   ]
   ```

3. **Context visibility metadata**: Record whether the agent had access to prior invocations' messages:
   ```python
   {
       "invocation": 1,
       "had_prior_context": True,  # agent saw msgs from invocation 0
       "messages": [...],          # messages in this invocation
       "agent_visible_messages": [...],  # ALL messages the agent could see (including prior)
   }
   ```

4. **Framework-agnostic implementation**: This should live in the base `AgentAdapter` class (in `run()`) so all adapters automatically get it, without requiring framework-specific code.

### Proposed Approach (sketch)

The base `AgentAdapter.run()` method already wraps `_run_agent()`. It could be extended to:

```python
# In AgentAdapter base class
def run(self, query: str) -> Any:
    for cb in self.callbacks:
        cb.on_run_start(self)

    invocation_start = datetime.now().isoformat()
    result = self._run_agent(query)

    # Capture trace snapshot AFTER run (agent's memory is populated)
    messages_snapshot = self.get_messages().to_list()
    self._invocation_traces.append({
        "invocation": len(self._invocation_traces),
        "started_at": invocation_start,
        "completed_at": datetime.now().isoformat(),
        "query": query,
        "messages": messages_snapshot,
    })

    for cb in self.callbacks:
        cb.on_run_end(self, result)

    return result
```

Then `gather_traces()` would include `self._invocation_traces` as a structured field.

### Design Questions to Resolve

1. **Should `get_messages()` return per-invocation or cumulative messages?**
   - Per-invocation: matches current behavior for most frameworks, but loses accumulated state for stateful LangGraph
   - Cumulative: would need adapter-level accumulation for stateless frameworks, conflating adapter and agent state
   - Recommendation: keep `get_messages()` reflecting the agent's actual state (per-invocation for stateless, cumulative for stateful). Add a separate `get_all_traces()` or `get_invocation_history()` for the full trace.

2. **Should the adapter control the agent's memory behavior?**
   - For smolagents: Should the adapter pass `reset=False`? This changes agent behavior.
   - For LlamaIndex: Should the adapter store and pass `ctx` between runs? This changes agent behavior.
   - Recommendation: **No.** The adapter should not override the user's configuration. If a benchmark needs multi-turn persistence, the user should configure the agent accordingly (pass `reset=False`, add a checkpointer, create a shared Context). The adapter should trace whatever happens, not control it.

3. **Where should trace accumulation live?**
   - Option A: In base `AgentAdapter.run()` (automatic for all adapters)
   - Option B: In a separate `TracingMiddleware` or callback
   - Recommendation: Option A is simplest and ensures no adapter can forget to do it.

4. **What about the `logs` property?**
   - LangGraph and LlamaIndex append to `self.logs` (a list) across runs, so logs DO accumulate
   - smolagents overrides `logs` as a property reading from `agent.memory.steps`, which resets each run
   - This inconsistency means smolagents loses per-step logs from prior invocations
   - This should be fixed: smolagents `logs` should also accumulate

---

## 5. Summary of Findings

| Question | Answer |
|----------|--------|
| Do smolagents messages accumulate? | **Not by default** (`reset=True`). Can accumulate with `reset=False`. MASEval adapter uses default (resets). |
| Do LangGraph messages accumulate? | **Only with checkpointer + thread_id**. Without checkpointer (current default), no. |
| Do LlamaIndex messages accumulate? | **Not by default**. Each `.run()` gets fresh Context. Must pass `ctx` to accumulate. |
| Can we tell what context an agent had? | **Only for smolagents** (via `model_input_messages`). Not for LangGraph or LlamaIndex. |
| Can we see invocation boundaries? | **No.** No framework or adapter marks where one invocation ends and another begins. |
| Are traces collected permanently? | **No.** smolagents loses steps on reset. LangGraph/LlamaIndex overwrite `_last_result`/`_message_cache`. Only `self.logs` accumulates (and smolagents `logs` property doesn't). |

### The Fundamental Issue

The current design assumes `get_messages()` is the primary trace mechanism. But `get_messages()` reflects the **agent's current state** (which may have been reset), not the **evaluation's trace needs** (which require a permanent, structured record of everything).

Tracing and agent memory are different concerns that need different data structures.

---

## 6. The Internal Agent-to-Agent Call Problem

### The Scenario

Consider a multi-agent benchmark with 4 agents, each wrapped in an `AgentAdapter`:

```
Benchmark registers:
  AgentAdapter(agent_1, "agent_1")
  AgentAdapter(agent_2, "agent_2")
  AgentAdapter(agent_3, "agent_3")
  AgentAdapter(agent_4, "agent_4")

But at runtime, the FRAMEWORK orchestrates agents directly:
  benchmark calls adapter_1.run("task")
    → agent_1 internally calls agent_2.run(...)    ← bypasses AgentAdapter
      → agent_2 resets memory, executes, produces messages
    → agent_1 internally calls agent_3.run(...)    ← bypasses AgentAdapter
      → agent_3 executes
    → agent_1 internally calls agent_2.run(...) AGAIN  ← bypasses AgentAdapter
      → agent_2 resets memory AGAIN — first run's messages are GONE
    → ...

When benchmark finally calls gather_traces():
  adapter_2.get_messages()  → only sees agent_2's LAST run
  adapter_2.logs            → smolagents: only last run; LangGraph/LlamaIndex: never populated
                               (because adapter.run() was never called)
```

### Why This Is Worse Than the Multi-Invocation Problem

The multi-invocation problem (Section 3) assumed the benchmark calls `adapter.run()` each time. At least then, the adapter's `run()` method executes and *could* capture traces.

Here, the adapter's `run()` is **never called** for agents 2-4. The adapter is a dead wrapper — the framework bypasses it entirely. This means:
- No callbacks fire (`on_run_start`/`on_run_end`)
- No logs are created (LangGraph/LlamaIndex append to `self.logs` in `_run_agent()`)
- `get_messages()` shows whatever the agent's memory happens to contain at the moment `gather_traces()` is called
- If the agent reset memory N times during execution, N-1 runs are lost forever

### What Each Framework's Internal Agent-to-Agent Calling Looks Like

**smolagents managed agents**: The parent agent calls sub-agents via `execute_tool_call()`. Sub-agents run their full loop internally.

**LangGraph multi-agent**: Agents are subgraphs or nodes that invoke each other as graph steps. Execution flows through the graph runtime.

**LlamaIndex AgentWorkflow**: Multiple agents hand off to each other within the workflow. Execution flows through the workflow runtime.

---

## 7. Available Hook Mechanisms (Verified from Documentation)

### smolagents: `step_callbacks`

**Source**: [smolagents/agents.py v1.24.0](https://github.com/huggingface/smolagents/blob/v1.24.0/src/smolagents/agents.py)

```python
# Constructor parameter:
step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None

# List format: callbacks registered for ActionStep only (backward compat)
# Dict format: callbacks mapped to specific step types
```

**When it fires**: In `_finalize_step()` after each ActionStep, PlanningStep, or FinalAnswerStep completes.

**What it receives**: `(memory_step, agent=self)` — the completed step object containing:
- `step_number`, `model_input_messages`, `model_output_message`
- `tool_calls`, `observations`, `action_output`
- `timing`, `token_usage`, `error`

**Critical limitation**: Step callbacks do **NOT** fire for managed agents' internal steps. Only the parent agent's callbacks fire. The managed agent's results are returned as tool call output to the parent, but the managed agent's internal steps are invisible.

**Implication for MASEval**: To capture managed agent steps, we'd need to install `step_callbacks` on EACH managed agent individually, not just the top-level agent.

### LangGraph/LangChain: `BaseCallbackHandler`

**Source**: LangChain Reference — Callbacks (local file `Callbacks | LangChain Reference.html`)

```python
class BaseCallbackHandler:
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, tags, metadata, **kwargs): ...
    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs): ...
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id, **kwargs): ...
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id, **kwargs): ...
    def on_tool_end(self, output, *, run_id, parent_run_id, **kwargs): ...
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, **kwargs): ...
    def on_chain_end(self, outputs, *, run_id, parent_run_id, **kwargs): ...
    def on_agent_action(self, action, *, run_id, parent_run_id, **kwargs): ...
    def on_agent_finish(self, finish, *, run_id, parent_run_id, **kwargs): ...
```

**Key feature**: Every callback receives `run_id` and `parent_run_id`, enabling reconstruction of the full call tree.

**How to attach**: Pass via `config={"callbacks": [handler]}` at invoke time. Callbacks propagate through chains and subgraphs.

**Propagation**: Callbacks DO propagate to subgraphs and nested chains. When a parent graph invokes a subgraph, the parent's callbacks fire for the subgraph's events too.

**Implication for MASEval**: A single `BaseCallbackHandler` installed at the top-level graph would capture ALL events across all sub-agents. The `parent_run_id` chain gives us the execution tree for free.

### LlamaIndex: Instrumentation Module + Legacy CallbackManager

**Source**: `LLAMAINDEXINSTRUMENT.md`, `LLAMAINDEXCALLBACK.md` (local files)

**New system (v0.10.20+)**: `instrumentation` module with span handlers and event handlers.
```python
from llama_index.core.instrumentation import get_dispatcher

root_dispatcher = get_dispatcher()
root_dispatcher.add_span_handler(my_handler)
root_dispatcher.add_event_handler(my_handler)
```

**Legacy system**: `CallbackManager` with event types:
- `LLM`, `EMBEDDING`, `QUERY`, `RETRIEVE`, `SYNTHESIZE`, `TOOL`, etc.

**OpenTelemetry native**: `llama-index-observability-otel` package provides direct OTel export.
```python
from llama_index.observability.otel import LlamaIndexOpenTelemetry
instrumentor = LlamaIndexOpenTelemetry()
instrumentor.start_registering()
```

**Global handler**: `set_global_handler("simple")` or `set_global_handler("arize_phoenix")` etc. — catches ALL LlamaIndex operations globally, including within AgentWorkflow sub-agents.

**Implication for MASEval**: The global handler / dispatcher approach captures everything across all agents in the workflow. This is the most "automatic" of the three frameworks.

---

## 8. The OpenTelemetry Question

All three frameworks are converging on OpenTelemetry:
- **LlamaIndex**: Native OTel support via `llama-index-observability-otel`
- **LangChain**: OTel integrations via OpenLLMetry, Langfuse, etc.
- **smolagents**: Has structured logging that could be mapped to OTel spans

### Pros of an OTel-based approach for MASEval

1. **Unified standard**: One tracing format across all frameworks
2. **Built-in structure**: Spans with parent-child relationships, trace IDs, timestamps
3. **Invocation boundaries**: Each `.run()` could be a parent span; each LLM call a child span
4. **Call tree reconstruction**: `parent_run_id` / parent span gives us the agent-to-agent call graph
5. **Rich ecosystem**: Can export to Jaeger, Phoenix, Grafana, etc. for visualization
6. **No framework-specific code**: Instrumentation libraries already exist

### Cons of an OTel-based approach

1. **Heavy dependency**: `opentelemetry-sdk` + framework-specific instrumentation packages
2. **Abstraction mismatch**: OTel is designed for general observability, not eval-specific message tracing. MASEval needs "what messages did the LLM see?" — OTel gives "how long did the span take?"
3. **Data extraction**: Getting message content OUT of OTel spans back into MASEval's `MessageHistory` format requires parsing span attributes
4. **Setup complexity**: Requires configuring exporters, processors, etc.
5. **Not all frameworks equal**: smolagents' OTel story is weaker than LlamaIndex's

### Recommendation: Hybrid Approach

Use **framework-native hooks** (not OTel) as the primary mechanism, with optional OTel export:

1. **Primary**: Install framework-specific callbacks/hooks that write to the adapter's trace buffer in MASEval's native format. This gives us exactly the data we need (messages, context, invocation boundaries) without extra dependencies.

2. **Optional**: For users who want OTel observability, document how to also attach OTel instrumentation alongside MASEval's hooks. They're not mutually exclusive.

---

## 9. Recommended Architecture

### The Hook Pattern

Each adapter installs a framework-specific hook on the agent at initialization time. The hook writes to a persistent trace buffer on the adapter that survives memory resets.

```
AgentAdapter
  ├── agent (the wrapped framework agent)
  ├── _trace_buffer: List[TraceEvent]     ← permanent, never reset
  └── _hook (framework-specific)
        └── on_step/on_llm/etc → appends to _trace_buffer

When agent runs (even internally, bypassing adapter):
  hook fires → TraceEvent written to _trace_buffer

gather_traces():
  returns _trace_buffer contents (complete, structured, with boundaries)
```

### Framework-Specific Hook Installation

**smolagents**:
```python
class SmolAgentAdapter(AgentAdapter):
    def __init__(self, agent_instance, name, ...):
        super().__init__(agent_instance, name, ...)
        # Install step_callback on the agent itself
        self._install_trace_hook()
        # Also install on all managed agents
        for managed in getattr(agent_instance, 'managed_agents', {}).values():
            self._install_trace_hook_on(managed)

    def _install_trace_hook(self):
        # Register callback that writes to self._trace_buffer
        def trace_callback(step, agent):
            self._trace_buffer.append({
                "agent_name": agent.name,
                "step": step.dict(),
                "model_input_messages": step.model_input_messages,
                "timestamp": step.timing.end_time if step.timing else None,
            })
        self.agent.step_callbacks.register(ActionStep, trace_callback)
        self.agent.step_callbacks.register(PlanningStep, trace_callback)
```

**LangGraph**:
```python
class LangGraphAgentAdapter(AgentAdapter):
    def __init__(self, agent_instance, name, ..., config=None):
        super().__init__(agent_instance, name, ...)
        # Create callback handler that writes to trace buffer
        self._trace_handler = MASEvalLangChainHandler(self._trace_buffer)
        # Inject into config so it propagates to all subgraphs
        if config:
            config.setdefault("callbacks", []).append(self._trace_handler)
```

**LlamaIndex**:
```python
class LlamaIndexAgentAdapter(AgentAdapter):
    def __init__(self, agent_instance, name, ...):
        super().__init__(agent_instance, name, ...)
        # Use LlamaIndex's global dispatcher or instrumentation
        # This captures ALL events across all agents in the workflow
        dispatcher = get_dispatcher()
        self._trace_handler = MASEvalLlamaIndexHandler(self._trace_buffer)
        dispatcher.add_span_handler(self._trace_handler)
        dispatcher.add_event_handler(self._trace_handler)
```

### What The Trace Buffer Would Contain

```python
_trace_buffer = [
    # Invocation boundary
    {"type": "invocation_start", "agent": "agent_1", "query": "task", "timestamp": "..."},

    # LLM call with full context (what the LLM saw)
    {"type": "llm_call", "agent": "agent_1", "input_messages": [...], "output": "...",
     "tokens": {"input": 50, "output": 30}, "timestamp": "..."},

    # Sub-agent call
    {"type": "invocation_start", "agent": "agent_2", "query": "subtask", "parent_agent": "agent_1", "timestamp": "..."},
    {"type": "llm_call", "agent": "agent_2", "input_messages": [...], "output": "...", "timestamp": "..."},
    {"type": "invocation_end", "agent": "agent_2", "result": "...", "timestamp": "..."},

    # More of agent_1
    {"type": "llm_call", "agent": "agent_1", "input_messages": [...], "output": "...", "timestamp": "..."},
    {"type": "invocation_end", "agent": "agent_1", "result": "...", "timestamp": "..."},
]
```

This structure answers ALL three original questions:
1. **What context did the agent have?** → `input_messages` on each `llm_call` event
2. **Where do invocations start/end?** → `invocation_start` / `invocation_end` events
3. **Permanent trace collection?** → `_trace_buffer` is never reset, captures everything

### Key Design Principle

**The adapter should NOT change the agent's behavior.** The hooks are read-only observers. Whether the agent resets memory, uses checkpointers, or persists context is the user's choice. The hooks just record what happens.
