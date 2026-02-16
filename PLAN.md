# Plan: Fix AgentAdapter Tracing & Message Accumulation

## Context

The OVERVIEW.md investigation identified four problems with MASEval's AgentAdapter tracing:

1. **No trace accumulation across invocations** — All adapters lose messages from prior `run()` calls
2. **No invocation boundaries** — No way to tell post-hoc which messages belong to which invocation
3. **smolagents `logs` doesn't accumulate** — The `logs` property reads from `agent.memory.steps` which resets each run (LangGraph, LlamaIndex, and CAMEL all accumulate properly)
4. **Internal agent-to-agent calls bypass adapters** — Framework orchestrators call sub-agents directly; adapter.run() is never called

This plan addresses all four problems:
- **Phase 1** (sections 1.1–1.6): Base adapter invocation tracing + smolagents logs fix
- **Phase 2** (sections 2.0–2.4): Framework-specific hooks to capture internal agent-to-agent calls

---

## Phase 1: Base Adapter Invocation Tracing

### 1.1 Changes to `maseval/core/agent.py`

**Add `_invocation_traces` to `__init__`:**

```python
def __init__(self, agent_instance, name, callbacks=None):
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self.logs = []
    self._invocation_traces = []  # NEW: permanent trace buffer
```

**Modify `run()` to capture invocation snapshots:**

After `_run_agent()` completes (and before `on_run_end` callbacks), snapshot the current messages and metadata into `_invocation_traces`. This happens whether or not the agent reset its memory — we capture whatever `get_messages()` returns at that moment.

```python
def run(self, query: str) -> Any:
    for cb in self.callbacks:
        cb.on_run_start(self)

    invocation_start = datetime.now().isoformat()
    result = self._run_agent(query)

    # Capture invocation trace snapshot
    try:
        messages_snapshot = self.get_messages().to_list()
    except Exception:
        messages_snapshot = []

    self._invocation_traces.append({
        "invocation": len(self._invocation_traces),
        "started_at": invocation_start,
        "completed_at": datetime.now().isoformat(),
        "query": query,
        "messages": messages_snapshot,
        "status": "success",
    })

    for cb in self.callbacks:
        cb.on_run_end(self, result)

    return result
```

**Handle errors** — if `_run_agent()` raises, still record the invocation (with status "error") before re-raising:

```python
def run(self, query: str) -> Any:
    for cb in self.callbacks:
        cb.on_run_start(self)

    invocation_start = datetime.now().isoformat()
    try:
        result = self._run_agent(query)
    except Exception as e:
        # Record failed invocation
        self._invocation_traces.append({
            "invocation": len(self._invocation_traces),
            "started_at": invocation_start,
            "completed_at": datetime.now().isoformat(),
            "query": query,
            "messages": [],
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        })
        raise

    # Capture successful invocation
    try:
        messages_snapshot = self.get_messages().to_list()
    except Exception:
        messages_snapshot = []

    self._invocation_traces.append({
        "invocation": len(self._invocation_traces),
        "started_at": invocation_start,
        "completed_at": datetime.now().isoformat(),
        "query": query,
        "messages": messages_snapshot,
        "status": "success",
    })

    for cb in self.callbacks:
        cb.on_run_end(self, result)

    return result
```

**Update `gather_traces()` to include invocation traces:**

```python
def gather_traces(self):
    history = self.get_messages()
    return {
        **super().gather_traces(),
        "name": self.name,
        "agent_type": type(self.agent).__name__,
        "message_count": len(history),
        "messages": history.to_list() if history else [],
        "callbacks": [type(cb).__name__ for cb in self.callbacks],
        "logs": self.logs,
        "invocation_traces": self._invocation_traces,  # NEW
    }
```

**Import needed:** Add `from datetime import datetime` to the top of `agent.py`.

### 1.2 Changes to `maseval/interface/agents/smolagents.py`

**Problem:** The `logs` property reads dynamically from `self.agent.memory.steps`, which resets on each `run()` call. Steps from previous invocations are lost.

**Fix:** Add `_accumulated_logs` list. After each `_run_agent()` call, snapshot the current memory steps into `_accumulated_logs`. Change the `logs` property to return `_accumulated_logs`.

```python
def __init__(self, agent_instance, name, callbacks=None):
    # Still skip super().__init__() to avoid self.logs = [] conflicting with property
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self._accumulated_logs = []
    self._invocation_traces = []  # Must initialize since we skip super().__init__()
```

Refactor the current `logs` property body into a helper method `_extract_current_logs()`:

```python
def _extract_current_logs(self) -> List[Dict[str, Any]]:
    """Extract logs from the agent's current memory state."""
    # (move existing logs property body here, unchanged)
    ...

@property
def logs(self) -> List[Dict[str, Any]]:
    """Return accumulated logs from all invocations."""
    return self._accumulated_logs
```

In `_run_agent()`, after calling `self.agent.run(query)`, snapshot:

```python
def _run_agent(self, query: str) -> str:
    _check_smolagents_installed()
    final_answer = self.agent.run(query)

    # Snapshot current memory steps into accumulated logs
    current_logs = self._extract_current_logs()
    self._accumulated_logs.extend(current_logs)

    return final_answer
```

### 1.3 Changes to `maseval/interface/agents/camel.py`

**Problem:** CAMEL also skips `super().__init__()`, so `_invocation_traces` won't be initialized.

**Fix:** Add `self._invocation_traces = []` in CamelAgentAdapter's `__init__`:

```python
def __init__(self, agent_instance, name, callbacks=None):
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self._responses = []
    self._errors = []
    self._invocation_traces = []  # NEW: must initialize since we skip super().__init__()
```

### 1.4 Changes to `tests/conftest.py` (DummyAgentAdapter)

No changes needed — DummyAgentAdapter calls `super().__init__()` indirectly (through `AgentAdapter.__init__`), so `_invocation_traces` will be initialized automatically.

---

## Phase 1: Testing Strategy

### 1.5 Contract test changes (`tests/test_contract/test_agent_adapter_contract.py`)

**Tighten `test_adapter_logs_accumulate_across_runs`:**
Currently says "we accept both behaviors as long as logs are populated." Change to **require accumulation** — logs from run 1 must still be present after run 2.

```python
def test_adapter_logs_accumulate_across_runs(self, framework):
    """Test that logs accumulate across multiple runs."""
    mock_llm = MockLLM(responses=["First response", "Second response"])
    agent = create_agent_for_framework(framework, mock_llm)
    adapter = create_adapter_for_framework(framework, agent)

    adapter.run("First query")
    logs_after_first = len(adapter.logs)
    assert logs_after_first > 0

    adapter.run("Second query")
    logs_after_second = len(adapter.logs)

    # Logs MUST accumulate (not reset)
    assert logs_after_second > logs_after_first
```

**Add new contract tests for invocation traces:**

```python
def test_adapter_invocation_traces_populated(self, framework):
    """Test that _invocation_traces is populated after run()."""
    ...
    adapter.run("Test query")
    assert len(adapter._invocation_traces) == 1
    trace = adapter._invocation_traces[0]
    assert trace["invocation"] == 0
    assert trace["query"] == "Test query"
    assert trace["status"] == "success"
    assert "started_at" in trace
    assert "completed_at" in trace
    assert isinstance(trace["messages"], list)

def test_adapter_invocation_traces_accumulate(self, framework):
    """Test that invocation traces accumulate across multiple runs."""
    ...
    adapter.run("First query")
    adapter.run("Second query")
    assert len(adapter._invocation_traces) == 2
    assert adapter._invocation_traces[0]["query"] == "First query"
    assert adapter._invocation_traces[1]["query"] == "Second query"
    assert adapter._invocation_traces[0]["invocation"] == 0
    assert adapter._invocation_traces[1]["invocation"] == 1

def test_adapter_invocation_traces_in_gather_traces(self, framework):
    """Test that gather_traces() includes invocation_traces."""
    ...
    adapter.run("Test query")
    traces = adapter.gather_traces()
    assert "invocation_traces" in traces
    assert len(traces["invocation_traces"]) == 1

def test_adapter_invocation_traces_on_error(self, framework):
    """Test that invocation traces are recorded even when run fails."""
    # Use a framework-specific setup that causes an error
    # (may need to be a separate non-parametrized test or handle frameworks individually)
```

### 1.6 Framework-specific integration test changes

**smolagents** (`test_smolagents_integration.py`):

Existing tests that directly manipulate `agent.memory.steps` and then check `adapter.logs` will need updating because `logs` now returns `_accumulated_logs` (populated via `_run_agent()`, not dynamically from memory).

- `test_smolagents_adapter_logs_property` — Change to call `adapter._extract_current_logs()` instead of `adapter.logs` (tests the conversion logic itself)
- `test_smolagents_adapter_logs_with_errors` — Same: use `_extract_current_logs()`
- `test_smolagents_adapter_logs_empty_when_no_steps` — Same: use `_extract_current_logs()`
- **Add new test:** `test_smolagents_logs_accumulate_across_runs` — Run agent twice with `FakeSmolagentsModel`, verify `adapter.logs` has entries from both runs
- **Add new test:** `test_smolagents_invocation_traces` — Run agent, verify `adapter._invocation_traces` is populated

**LangGraph** (`test_langgraph_integration.py`):
- **Add:** Test that `_invocation_traces` captures per-invocation messages correctly
- **Add:** Test that `_invocation_traces[0]["messages"]` has messages from only run 1

**LlamaIndex** (`test_llamaindex_integration.py`):
- **Add:** Same as LangGraph — verify invocation traces per-invocation

---

## Phase 2: Framework-Specific Hooks

**Goal:** Capture execution events even when internal agent-to-agent calls bypass `adapter.run()`. Each framework provides a hook mechanism; we install read-only observers that write into a `_trace_buffer` on the adapter.

### 2.0 Changes to `maseval/core/agent.py`

**Add `_trace_buffer` to `__init__`:**

```python
def __init__(self, agent_instance, name, callbacks=None):
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self.logs = []
    self._invocation_traces = []    # Phase 1
    self._trace_buffer = []         # Phase 2: framework hook events
```

**Update `gather_traces()` to include trace buffer:**

```python
def gather_traces(self):
    history = self.get_messages()
    return {
        **super().gather_traces(),
        "name": self.name,
        "agent_type": type(self.agent).__name__,
        "message_count": len(history),
        "messages": history.to_list() if history else [],
        "callbacks": [type(cb).__name__ for cb in self.callbacks],
        "logs": self.logs,
        "invocation_traces": self._invocation_traces,
        "trace_buffer": self._trace_buffer,
    }
```

### 2.1 smolagents: `step_callbacks` via `CallbackRegistry`

**API:** `agent.step_callbacks.register(step_cls, callback)` — the `CallbackRegistry` (in `smolagents/memory.py`) fires callbacks in `_finalize_step()` with signature `callback(memory_step, agent=self)`.

**Key detail:** Callbacks do NOT propagate to managed agents. We must register on `agent.managed_agents[name]` for each managed agent.

**Changes to `maseval/interface/agents/smolagents.py`:**

Add `_install_hooks()` called from `__init__`:

```python
def __init__(self, agent_instance, name, callbacks=None):
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self._accumulated_logs = []
    self._invocation_traces = []
    self._trace_buffer = []
    self._install_hooks()

def _install_hooks(self):
    """Register step_callbacks on agent and all managed agents."""
    from smolagents.memory import ActionStep, PlanningStep
    if hasattr(self.agent, 'step_callbacks'):
        self.agent.step_callbacks.register(ActionStep, self._on_step)
        self.agent.step_callbacks.register(PlanningStep, self._on_step)
    # Also register on managed agents
    if hasattr(self.agent, 'managed_agents') and self.agent.managed_agents:
        for managed_agent in self.agent.managed_agents.values():
            if hasattr(managed_agent, 'step_callbacks'):
                managed_agent.step_callbacks.register(ActionStep, self._on_step)
                managed_agent.step_callbacks.register(PlanningStep, self._on_step)

def _on_step(self, memory_step, agent=None):
    """Callback fired by smolagents after each step finalization."""
    from smolagents.memory import ActionStep, PlanningStep
    entry = {
        "source": "smolagents_step_callback",
        "step_type": type(memory_step).__name__,
        "agent_name": getattr(agent, 'name', None),
    }
    if isinstance(memory_step, ActionStep):
        entry["step_number"] = memory_step.step_number
        entry["has_error"] = memory_step.error is not None
        if memory_step.tool_calls:
            entry["tool_calls"] = [tc.name for tc in memory_step.tool_calls]
    elif isinstance(memory_step, PlanningStep):
        entry["plan_length"] = len(memory_step.plan) if memory_step.plan else 0
    self._trace_buffer.append(entry)
```

### 2.2 LangGraph: `BaseCallbackHandler` in config

**API:** `langchain_core.callbacks.base.BaseCallbackHandler` — mix of `LLMManagerMixin`, `ChainManagerMixin`, `ToolManagerMixin`, `CallbackManagerMixin`, `RunManagerMixin`. Passed via `config={"callbacks": [handler]}` to `graph.invoke()`. LangGraph propagates callbacks to all subgraphs automatically. Each callback receives `run_id` and `parent_run_id`.

**Changes to `maseval/interface/agents/langgraph.py`:**

Add a private handler class and install it:

```python
class _MASEvalLangChainHandler:
    """Read-only callback handler that captures execution events."""

    def __init__(self, trace_buffer: list):
        self._trace_buffer = trace_buffer
        # BaseCallbackHandler attributes
        self.raise_error = False
        self.run_inline = True
        self.ignore_llm = False
        self.ignore_chain = False
        self.ignore_agent = False
        self.ignore_retriever = True
        self.ignore_retry = True

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self._trace_buffer.append({
            "source": "langgraph_callback",
            "event": "chain_start",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "chain_type": serialized.get("id", [])[-1] if serialized.get("id") else None,
        })

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._trace_buffer.append({
            "source": "langgraph_callback",
            "event": "chain_end",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        })

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._trace_buffer.append({
            "source": "langgraph_callback",
            "event": "llm_end",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        })

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, tags=None, metadata=None, inputs=None, **kwargs):
        self._trace_buffer.append({
            "source": "langgraph_callback",
            "event": "tool_start",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tool_name": serialized.get("name"),
        })

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._trace_buffer.append({
            "source": "langgraph_callback",
            "event": "tool_end",
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        })

    # No-op stubs for remaining required methods
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs): pass
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs): pass
    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs): pass
    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs): pass
    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs): pass
    def on_llm_new_token(self, token, *, run_id, parent_run_id=None, **kwargs): pass
```

**Note:** We use duck-typing instead of inheriting from `BaseCallbackHandler` to avoid importing `langchain_core` at module level. LangChain's callback manager checks for method presence, not class hierarchy.

In `LangGraphAgentAdapter.__init__()`:

```python
def __init__(self, agent_instance, name, callbacks=None, config=None):
    super().__init__(agent_instance, name, callbacks)
    self._langgraph_config = config
    self._last_result = None
    self._hook_handler = _MASEvalLangChainHandler(self._trace_buffer)
```

In `_run_agent()`, inject the handler into config:

```python
# Build config with our callback handler injected
invoke_config = dict(self._langgraph_config) if self._langgraph_config else {}
existing_callbacks = invoke_config.get("callbacks", []) or []
invoke_config["callbacks"] = existing_callbacks + [self._hook_handler]

# Use invoke_config instead of self._langgraph_config for the invoke() call
```

### 2.3 LlamaIndex: Instrumentation Dispatcher

**API:** `llama_index_instrumentation.get_dispatcher()` returns/creates dispatchers. `Dispatcher.add_span_handler(handler)` adds a `BaseSpanHandler`. The `SimpleSpanHandler` tracks `completed_spans` and `dropped_spans` with timing, parent hierarchy, and tags. Events propagate up the dispatcher tree.

**Key detail:** The dispatcher is global — all LlamaIndex operations go through it. We use an `_active` flag on the handler to only record spans during our adapter's `run()`.

**Changes to `maseval/interface/agents/llamaindex.py`:**

Add a private span handler class:

```python
class _MASEvalSpanHandler(BaseSpanHandler):
    """Read-only span handler that records LlamaIndex execution spans."""

    _trace_buffer: list = PrivateAttr(default_factory=list)
    _active: bool = PrivateAttr(default=False)

    def class_name(cls) -> str:
        return "_MASEvalSpanHandler"

    def new_span(self, id_, bound_args, instance=None, parent_span_id=None, tags=None, **kwargs):
        if not self._active:
            return None
        from llama_index_instrumentation.span.simple import SimpleSpan
        return SimpleSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})

    def prepare_to_exit_span(self, id_, bound_args, instance=None, result=None, **kwargs):
        span = self.open_spans.get(id_)
        if span is None:
            return None
        from datetime import datetime
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds()
        self._trace_buffer.append({
            "source": "llamaindex_span",
            "event": "span_exit",
            "span_id": id_,
            "parent_id": span.parent_id,
            "duration": span.duration,
        })
        return span

    def prepare_to_drop_span(self, id_, bound_args, instance=None, err=None, **kwargs):
        span = self.open_spans.get(id_)
        if span is None:
            return None
        self._trace_buffer.append({
            "source": "llamaindex_span",
            "event": "span_drop",
            "span_id": id_,
            "parent_id": span.parent_id,
            "error": str(err) if err else None,
        })
        return span
```

In `LlamaIndexAgentAdapter.__init__()`:

```python
def __init__(self, agent_instance, name, callbacks=None):
    super().__init__(agent_instance, name, callbacks)
    self._last_result = None
    self._message_cache = []
    self._span_handler = _MASEvalSpanHandler()
    self._span_handler._trace_buffer = self._trace_buffer  # share buffer with base
    self._install_hooks()

def _install_hooks(self):
    """Add span handler to LlamaIndex's global dispatcher."""
    from llama_index_instrumentation import get_dispatcher
    dispatcher = get_dispatcher()
    dispatcher.add_span_handler(self._span_handler)
```

In `_run_agent()`, bracket execution with `_active` flag:

```python
def _run_agent(self, query):
    ...
    self._span_handler._active = True
    try:
        result = self._run_agent_sync(query)
        ...
    except Exception as e:
        ...
        raise
    finally:
        self._span_handler._active = False
```

### 2.4 CAMEL-AI: Explicitly Skipped

**CAMEL-AI has no agent-level hook mechanism.** Unlike the other three frameworks:

- smolagents has `step_callbacks` (`CallbackRegistry`) fired after each step
- LangGraph has `BaseCallbackHandler` propagated through all subgraphs
- LlamaIndex has a global instrumentation `Dispatcher` with `BaseSpanHandler`

CAMEL's `ChatAgent.step()` provides **no callback, hook, or instrumentation API**. The only callback system in CAMEL is `WorkforceCallback` which operates at the **task orchestration level** (task created/assigned/completed), not at the individual agent step level.

**What this means:** `CamelAgentAdapter._trace_buffer` will always be empty. The adapter captures what it can via `self._responses` (stored in `_run_agent()`), but there is no way to observe internal agent execution (tool calls, reasoning steps, sub-agent delegation) without CAMEL adding agent-level hooks.

**Future options if CAMEL adds hooks:**
1. If CAMEL adds step-level callbacks to `ChatAgent` (like smolagents' `CallbackRegistry`), register a callback in `_install_hooks()`
2. If CAMEL exposes an instrumentation/tracing API, integrate similarly to LlamaIndex's dispatcher
3. The `WorkforceCallback` system could be used to improve `CamelWorkforceTracer` (replace private attribute access with event-driven capture), but that's a separate enhancement

**Changes to `maseval/interface/agents/camel.py`:**

1. Add `self._invocation_traces = []` and `self._trace_buffer = []` to `__init__` (required since CAMEL skips `super().__init__()`)
2. Add a prominent warning to the class docstring about unreliable tracing

```python
def __init__(self, agent_instance, name, callbacks=None):
    self.agent = agent_instance
    self.name = name
    self.callbacks = callbacks or []
    self.messages = None
    self._responses = []
    self._errors = []
    self._invocation_traces = []
    self._trace_buffer = []  # CAMEL has no agent-level hook API — buffer stays empty
```

**Docstring warning to add to `CamelAgentAdapter`:**

```python
class CamelAgentAdapter(AgentAdapter):
    """An AgentAdapter for CAMEL-AI ChatAgent.

    .. warning::
        **Unreliable tracing.** CAMEL-AI's ChatAgent does not expose any
        callback, hook, or instrumentation API for individual agent steps.
        Unlike smolagents (step_callbacks), LangGraph (BaseCallbackHandler),
        and LlamaIndex (instrumentation Dispatcher), there is no way to
        observe internal execution events (tool calls, reasoning steps,
        sub-agent delegation) from outside the agent.

        Consequences:
        - ``_trace_buffer`` is always empty (no framework hooks to tap into)
        - ``logs`` only captures data from ``ChatAgentResponse.info`` returned
          by ``step()`` — if the agent performs internal tool calls or
          multi-step reasoning, those details may be lost
        - ``get_messages()`` relies on CAMEL's memory system, which may not
          reflect the full execution history
        - For Workforce orchestration, use ``CamelWorkforceTracer`` which
          can tap into ``WorkforceCallback`` events at the task level

        This will be improved when CAMEL-AI adds agent-level instrumentation.
        Track: https://github.com/camel-ai/camel

    ...
    """
```

---

## Phase 2: Testing Strategy

### 2.5 Contract test additions (`tests/test_contract/test_agent_adapter_contract.py`)

```python
def test_adapter_trace_buffer_exists(self, framework):
    """Test that _trace_buffer is initialized."""
    adapter = create_adapter_for_framework(framework, ...)
    assert hasattr(adapter, '_trace_buffer')
    assert isinstance(adapter._trace_buffer, list)

def test_adapter_trace_buffer_in_gather_traces(self, framework):
    """Test that gather_traces() includes trace_buffer."""
    adapter = create_adapter_for_framework(framework, ...)
    adapter.run("Test query")
    traces = adapter.gather_traces()
    assert "trace_buffer" in traces
```

### 2.6 Framework-specific hook tests

**smolagents** (`test_smolagents_integration.py`):
- `test_smolagents_hook_captures_steps` — Run agent, verify `adapter._trace_buffer` has entries with `source == "smolagents_step_callback"`
- `test_smolagents_hook_captures_managed_agent_steps` — Create agent with managed agent, run, verify buffer captures steps from both agents

**LangGraph** (`test_langgraph_integration.py`):
- `test_langgraph_hook_captures_chain_events` — Run graph, verify `adapter._trace_buffer` has `chain_start`/`chain_end` events
- `test_langgraph_hook_has_run_ids` — Verify each event in buffer has `run_id` and `parent_run_id`

**LlamaIndex** (`test_llamaindex_integration.py`):
- `test_llamaindex_hook_captures_spans` — Run agent, verify `adapter._trace_buffer` has `span_exit` entries
- `test_llamaindex_hook_only_active_during_run` — Verify buffer is empty before `run()` and only populated during execution

---

## Updated Files to Modify

| File | Change |
|------|--------|
| `maseval/core/agent.py` | Add `_invocation_traces`, `_trace_buffer`, modify `run()`, update `gather_traces()`, add `datetime` import |
| `maseval/interface/agents/smolagents.py` | Fix logs accumulation, add `_install_hooks()`, `_on_step()`, init `_invocation_traces` + `_trace_buffer` |
| `maseval/interface/agents/langgraph.py` | Add `_MASEvalLangChainHandler` class, create handler in `__init__`, inject in `_run_agent()` |
| `maseval/interface/agents/llamaindex.py` | Add `_MASEvalSpanHandler` class, install in `__init__`, bracket `_run_agent()` with `_active` flag |
| `maseval/interface/agents/camel.py` | Add `_invocation_traces` + `_trace_buffer` to `__init__`, add unreliable-tracing warning to docstring |
| `tests/test_contract/test_agent_adapter_contract.py` | Tighten logs test, add invocation trace tests, add trace buffer tests |
| `tests/test_interface/test_agent_integration/test_langgraph_integration.py` | Add invocation trace + hook tests |
| `tests/test_interface/test_agent_integration/test_llamaindex_integration.py` | Add invocation trace + hook tests |
| `tests/test_interface/test_agent_integration/test_smolagents_integration.py` | Add logs accumulation + hook tests |

## Files NOT Modified (no changes needed)

| File | Why |
|------|-----|
| `tests/conftest.py` | DummyAgentAdapter calls `super().__init__()` — gets both buffers automatically |

---

## Verification

1. `uv run ruff format . && uv run ruff check . --fix` — formatting and linting
2. `uv run pytest -m contract -v` — contract tests (most critical)
3. `uv run pytest -m interface -v` — all framework integration tests
4. `uv run pytest -v` — full default test suite
5. Verify no existing tests break (especially the tightened logs accumulation test)
