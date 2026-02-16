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
