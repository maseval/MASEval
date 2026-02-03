# CAMEL-AI Integration Plan

## Overview

This document outlines the implementation plan for integrating CAMEL-AI as an agent framework adapter in MASEval.

## CAMEL-AI Ecosystem Overview

**CAMEL-AI** is an open-source, modular framework for building intelligent multi-agent systems. The ecosystem includes:

| Project | Description | Type |
|---------|-------------|------|
| **CAMEL** | Core multi-agent framework | Agent Framework |
| **OWL** | Optimized Workforce Learning - extends CAMEL's RolePlaying | Agent Framework (extends CAMEL) |
| **CRAB** | Cross-environment Agent Benchmark for cross-device tasks | Benchmark/Environment |

### Sources
- CAMEL Documentation: https://docs.camel-ai.org/
- CAMEL GitHub: https://github.com/camel-ai/camel
- OWL GitHub: https://github.com/camel-ai/owl
- OWL EnhancedRolePlaying: https://github.com/camel-ai/owl/blob/main/owl/utils/enhanced_role_playing.py
- CRAB Website: https://crab.camel-ai.org/

---

## Terminology Clarification

**Important**: The term "Task" is overloaded between MASEval and CAMEL-AI:

| Term | Definition | Scope |
|------|------------|-------|
| **maseval.Task** | A data point in a benchmark dataset. One `maseval.Task` = one evaluation problem (e.g., one GAIA question). | Evaluation level |
| **camelai.Task** | An internal work unit that agents decompose and solve. Many `camelai.Tasks` can exist within solving ONE `maseval.Task`. | Agent collaboration level |

These operate at **different levels** and do not conflict.

---

## Architectural Analysis: CAMEL vs MASEval

### Key Insight: Complementary Orchestration Layers

MASEval and CAMEL orchestrate at **different levels** - they complement rather than compete:

| Component | Orchestrates | Level | Scope |
|-----------|--------------|-------|-------|
| **MASEval Benchmark** | Evaluation lifecycle | Evaluation | Running many `maseval.Tasks`, collecting traces, metrics, repetitions |
| **MASEval `execution_loop`** | Agent-user turns | Interaction | Multi-turn conversation within ONE `maseval.Task` |
| **CAMEL RolePlaying** | Agent-agent turns | Interaction | Alternates between `assistant_agent` and `user_agent` |
| **CAMEL Workforce** | Agent collaboration | Collaboration | Task decomposition, assignment, retries within ONE problem |

**Key realization**: Workforce handles how agents collaborate to solve ONE `maseval.Task` by decomposing it into many `camelai.Tasks`. MASEval handles running MANY `maseval.Tasks` and collecting evaluation metrics. **They work together, not against each other.**

### Agent-to-Agent Evaluation

MASEval already supports agent-to-agent evaluation in multiple ways:

1. **Multi-agent frameworks without User**: LangGraph, AutoGen, and other frameworks have multiple agents communicating internally. MASEval wraps and evaluates the entire system. The `maseval.User` is optional - `setup_user()` can return `None`.

2. **LLM-simulated user**: The `maseval.User` with `UserLLMSimulator` IS an LLM-based agent simulating user behavior.

3. **Potential: Framework-specific agent as user**: With a `User` abstraction, any agent implementation could serve as the "user" side (see Option B).

### Component Deep Dive

#### RolePlaying
- Contains two `ChatAgent` instances as **public attributes**: `assistant_agent`, `user_agent`
- `step()` method alternates between agents, returns `Tuple[ChatAgentResponse, ChatAgentResponse]`
- Handles termination via `terminated` flags
- Each agent maintains its own memory via `ChatAgent.memory`
- **State worth tracing**: Configuration (task_prompt, roles), step count, termination reason
- **Does NOT maintain**: Aggregate metrics or conversation-level logging beyond individual agents

#### OWL (OwlRolePlaying)
```python
class OwlRolePlaying(RolePlaying):  # Inherits from CAMEL's RolePlaying
    self.assistant_agent: ChatAgent  # Accessible
    self.user_agent: ChatAgent       # Accessible
```
- Adds enhanced prompts and GAIA-specific formatting (`<analysis>`, `<final_answer>` tags)
- Same underlying architecture - agents are extractable
- No additional runtime state beyond RolePlaying

#### Workforce
Complex task orchestrator with rich state worth tracing:

| Attribute | Description | Tracing Value |
|-----------|-------------|---------------|
| `_children` | List of worker nodes (agents) | Worker roster |
| `_pending_tasks` / `_completed_tasks` | Task queues | Task lifecycle |
| `_task_dependencies` | Dependency graph | How problem was decomposed |
| `_assignees` | Worker assignments | Who did what |
| `_snapshots` | Checkpoints | Execution history |
| `WorkforceMetrics` | Lifecycle events | Task creation, assignment, completion, failure |
| Recovery strategies | RETRY, REPLAN, REASSIGN, DECOMPOSE, CREATE_WORKER | Debugging agent behavior |

#### Memory
- `agent.memory.get_context()` returns `(List[OpenAIMessage], int)` - messages + token count
- Already traced in our `CamelAgentAdapter.get_messages()` implementation

---

## Phase 1: ChatAgent Adapter

### Status: IMPLEMENTED (Testing Incomplete)

Basic CAMEL-AI integration supporting the core `ChatAgent` class. Implementation complete, but not yet added to contract tests.

### Completed Items

| Task | Status | Notes |
|------|--------|-------|
| Create `maseval/interface/agents/camel.py` | Done | `CamelAgentAdapter` and `CamelLLMUser` |
| Update `maseval/interface/agents/__init__.py` | Done | Conditional import added |
| Add `camel` optional dependency to `pyproject.toml` | Done | `camel-ai>=0.2.0` |
| Add `camel` pytest marker | Done | In `pyproject.toml` |
| Create integration tests | Done | 17 tests, all passing |
| Create documentation | Done | `docs/interface/agents/camel.md` |
| Update mkdocs.yml navigation | Done | Added to Interface > Agents |
| Update CHANGELOG.md | Done | Under Unreleased |
| Linting (`ruff format`, `ruff check`) | Done | Passes |
| Type checking (`ty check`) | Done | Passes |
| Integration tests (`pytest -m camel`) | Done | 17/17 passing |
| **Add to contract tests** | TODO | See "Outstanding (Phase 1 - Testing)" below |
| **Improve test coverage** | TODO | `_run_agent()`, error paths, multi-turn |

### Implementation Details

**CamelAgentAdapter**:
- Wraps CAMEL `ChatAgent` instances
- Uses `agent.step(user_msg)` for execution
- Fetches messages from `agent.memory.get_context()` (source of truth pattern)
- Captures response metadata (`terminated`, `info`) in traces
- Extracts tool and model configuration in `gather_config()`

**CamelLLMUser**:
- Extends `LLMUser` class
- `get_tool()` returns CAMEL `FunctionTool` wrapping `respond()`

### Design Decisions Made

1. **Memory as Source of Truth**: Using `agent.memory.get_context()` following AGENTS.md Pattern 1
2. **Minimal Message Conversion**: CAMEL already uses OpenAI-compatible format
3. **Step-based Execution**: Using `step()` method (synchronous, single-turn)
4. **No `camel_optional.py`**: Not needed since tool creation is simple (inline import)

---

## Phase 2: Multi-Agent Support

Phase 2 consists of incremental improvements that build on each other. Core library changes come first, followed by interface-specific implementations.

### 2.1 User Abstraction (Core Library)

**Status: IMPLEMENTED**

**Motivation**: The current `maseval.User` forces LLM simulation via `UserLLMSimulator`. This is too restrictive for:
- Using a smolagents agent as user
- Using CAMEL's `user_agent` from RolePlaying
- Connecting user to an MCP server
- Scripted/deterministic users for testing
- Human-in-the-loop scenarios

**Design Decisions**:

1. **Method naming: `respond()` instead of `simulate_response()`**

   The current `simulate_response()` name implies the response is being "simulated", which is only accurate for LLM-based users. Other user types don't "simulate":
   - A human-in-the-loop actually responds
   - A CAMEL agent acting as user processes and responds
   - A scripted test user returns pre-defined values

   Using `respond(message: str) -> str` is generic and accurate for all implementations.

2. **`get_tool()` is optional, not abstract**

   Some frameworks (smolagents, CAMEL) use a tool-based pattern where agents invoke an `AskUser` tool to interact with the user. Other frameworks handle user interaction through message passing, callbacks, or other mechanisms.

   Making `get_tool()` optional with a default returning `None` allows:
   - Framework-specific subclasses to override and provide tools
   - Frameworks that don't need tools to ignore it
   - The core `respond()` method remains the universal interface

3. **Class naming: `LLMUser` instead of `SimulatedUser`**

   `LLMUser` is clearer - it explicitly states the user is backed by a language model. "Simulated" is vague and could be confused with other simulation types.

4. **`AgenticUser` → `AgenticLLMUser`**

   The current `AgenticUser` (user with tool access) should be renamed to `AgenticLLMUser` to match the new naming convention.

5. **Class naming: `User` for the abstract interface**

   The abstract base class is named `User` (not `BaseUser`) to match MASEval's conventions - other abstract classes like `AgentAdapter`, `Environment`, `Evaluator` don't use a "Base" prefix. This keeps type hints clean (`user: User`) and follows Python's ABC convention (`Mapping`, `Sequence`, etc.).

6. **Clean break, no backwards compatibility aliases**

   Per AGENTS.md: *"This project is early-release. Clean, maintainable code is the priority - not backwards compatibility."*

**Core changes** (in `maseval/core/user.py`):
```python
class User(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract interface for user interaction during evaluation.

    A user represents the entity that interacts with agents. This could be
    an LLM simulating a human, a scripted response sequence, a real human,
    or another agent system.
    """

    @abstractmethod
    def get_initial_query(self) -> str:
        """Return the initial query to start the conversation."""
        ...

    @abstractmethod
    def respond(self, message: str) -> str:
        """Respond to a message from the agent.

        Args:
            message: The agent's message or question.

        Returns:
            The user's response.
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the user interaction should terminate."""
        ...

    def get_tool(self):
        """Return a framework-compatible tool for agent interaction.

        Some frameworks (smolagents, CAMEL) use a tool-based pattern where
        agents invoke an AskUser tool. Override this in subclasses for
        frameworks that need it. Returns None by default.

        Returns:
            Framework-specific tool, or None if not applicable.
        """
        return None


class LLMUser(User):
    """User simulated by a language model.

    Uses an LLM to generate realistic user responses based on a
    user profile and scenario description.
    """
    def __init__(
        self,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        name: str = "user",
        initial_query: Optional[str] = None,
        template: Optional[str] = None,
        max_try: int = 3,
        max_turns: int = 1,
        stop_tokens: Optional[List[str]] = None,
        early_stopping_condition: Optional[str] = None,
    ):
        ...
    # Current User implementation body (without abstract get_tool)


class AgenticLLMUser(LLMUser):
    """LLM-simulated user with access to tools.

    Extends LLMUser with the ability to use tools (e.g., check order status,
    lookup information) during the conversation.
    """
    # Current AgenticUser implementation, renamed
```

**Benefits**:
- Follows MASEval's "Abstract Base Classes" principle
- Enables any agent implementation as user (CAMEL, smolagents, LangGraph, MCP, etc.)
- Library-wide improvement, not CAMEL-specific
- Cleaner separation: core interface vs. framework-specific tool integration
- Generic method names work for all user types

**Effort**: Medium (core refactor, but library is early-release)

**Files to change**:

*Core:*
- `maseval/core/user.py` - Refactor `User` → `User` + `LLMUser`, rename `AgenticUser` → `AgenticLLMUser`, rename `simulate_response` → `respond`
- `maseval/core/__init__.py` - Export `User`, `LLMUser`, `AgenticLLMUser`
- `maseval/core/benchmark.py` - Update type hints to use `User`
- `tests/test_core/test_user.py` - Update tests for new structure

*Interface (framework-specific users now extend `LLMUser` and override `get_tool()`):*
- `maseval/interface/agents/smolagents.py` - `SmolAgentLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/langgraph.py` - `LangGraphLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/llamaindex.py` - `LlamaIndexLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/camel.py` - `CamelLLMUser(LLMUser)` overrides `get_tool()`

---

### 2.2 CAMEL Interface: CamelAgentUser

**Status: IMPLEMENTED**

**Depends on**: 2.1 (User abstraction)

Once `User` exists, add CAMEL-specific implementation for using a CAMEL ChatAgent as the user:

```python
# In maseval/interface/agents/camel.py

class CamelAgentUser(User):
    """User backed by a CAMEL ChatAgent.

    Wraps a CAMEL ChatAgent to act as the user in MASEval's evaluation loop.
    Useful for using RolePlaying's `user_agent` with MASEval.
    """

    def __init__(self, user_agent: ChatAgent, initial_query: str, max_turns: int = 10):
        self.user_agent = user_agent
        self._initial_query = initial_query
        self._turn_count = 0
        self._max_turns = max_turns

    def get_initial_query(self) -> str:
        return self._initial_query

    def respond(self, message: str) -> str:
        """Forward the message to the CAMEL agent and return its response."""
        self._turn_count += 1
        response = self.user_agent.step(message)
        return response.msgs[0].content

    def is_done(self) -> bool:
        return self._turn_count >= self._max_turns

    def get_tool(self):
        """Return a CAMEL FunctionTool for agent-to-user interaction."""
        from camel.toolkits import FunctionTool
        return FunctionTool(self.respond)
```

**Use case**: Using RolePlaying's `user_agent` as the user in MASEval's execution loop, enabling agent-to-agent evaluation where one CAMEL agent acts as the user.

---

### 2.3 Documentation: Using Workforce/RolePlaying with MASEval

**Location**: TBD (possibly `docs/interface/agents/camel.md` or a dedicated guide)

Document how developers can use CAMEL's multi-agent orchestrators with MASEval by extracting individual agents for tracing.

**Pattern for Workforce**:
```python
from camel.societies.workforce import Workforce
from maseval.interface.agents.camel import CamelAgentAdapter

class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        # Create Workforce
        workforce = Workforce(...)
        self._workforce = workforce  # Store for run_agents

        # Extract and wrap individual workers for tracing
        worker_adapters = {}
        for worker in workforce._children:
            adapter = CamelAgentAdapter(worker.agent, name=worker.name)
            worker_adapters[worker.name] = adapter

        # Return empty agents_to_run (we'll run Workforce directly)
        return [], worker_adapters

    def run_agents(self, agents, task, environment, query):
        # Run Workforce (handles internal orchestration)
        result = self._workforce.process(query)
        return result
```

**Pattern for RolePlaying**:
```python
from camel.societies import RolePlaying
from maseval.interface.agents.camel import CamelAgentAdapter

class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        # Create RolePlaying
        role_playing = RolePlaying(
            assistant_role_name="Assistant",
            user_role_name="User",
            task_prompt=task.query,
        )

        # Extract and wrap the assistant agent
        assistant_adapter = CamelAgentAdapter(
            role_playing.assistant_agent,
            name="camel_assistant"
        )

        return [assistant_adapter], {"assistant": assistant_adapter}
```

**Key points to document**:
- MASEval evaluates, CAMEL orchestrates agent collaboration - they're complementary
- Individual agent messages are traced via `CamelAgentAdapter`
- Orchestration state (task decomposition, assignments) requires additional tracers (see 2.5)

---

### 2.4 CAMEL Interface: Prefixed Execution Loop

**Status: IMPLEMENTED**

**Context**: The `execution_loop` method is **designed to be overridden**. From the docstring:

> *"Override this method in your benchmark subclass to implement custom interaction patterns (e.g., agent-initiated conversations, different termination conditions, or specialized query routing)."*

Provide a reusable execution loop function for RolePlaying semantics:

```python
# In maseval/interface/agents/camel.py

def camel_role_playing_execution_loop(
    benchmark: Benchmark,
    role_playing: RolePlaying,
    agents: Sequence[AgentAdapter],
    task: Task,
    environment: Environment,
    user: Optional[User],
    max_steps: int = 10,
) -> Any:
    """Execution loop using CAMEL RolePlaying's step() semantics.

    Use this in your benchmark's execution_loop override:

        def execution_loop(self, agents, task, environment, user):
            return camel_role_playing_execution_loop(
                self, self._role_playing, agents, task, environment, user
            )

    Args:
        benchmark: The benchmark instance (for max_invocations, callbacks, etc.)
        role_playing: The CAMEL RolePlaying instance
        agents: Agent adapters (for tracing, not execution)
        task: Current task
        environment: Current environment
        user: Optional user (ignored - RolePlaying uses its own user_agent)
        max_steps: Maximum RolePlaying steps

    Returns:
        Final answer from the assistant agent
    """
    role_playing.init_chat()

    final_answer = None
    for _ in range(max_steps):
        assistant_response, user_response = role_playing.step()

        if assistant_response.msgs:
            final_answer = assistant_response.msgs[-1].content

        if assistant_response.terminated or user_response.terminated:
            break

    return final_answer
```

**Usage**:
```python
class CamelRolePlayingBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        self._role_playing = RolePlaying(
            assistant_role_name="Assistant",
            user_role_name="User",
            task_prompt=task.query,
        )

        # Wrap both agents for tracing
        assistant = CamelAgentAdapter(self._role_playing.assistant_agent, "assistant")
        user_agent = CamelAgentAdapter(self._role_playing.user_agent, "user_agent")

        return [assistant], {"assistant": assistant, "user_agent": user_agent}

    def execution_loop(self, agents, task, environment, user):
        return camel_role_playing_execution_loop(
            self, self._role_playing, agents, task, environment, user
        )
```

**Benefits**:
- Reusable - developers don't need to write the loop themselves
- Consistent behavior across CAMEL-based benchmarks
- Both agents traced via adapters

---

### 2.5 CAMEL Interface: Orchestration Tracers

**Status: IMPLEMENTED**

**Motivation**: RolePlaying and Workforce maintain orchestration state that individual agent traces don't capture. Provide lightweight `TraceableMixin` wrappers for this state.

**RolePlayingTracer**:
```python
class CamelRolePlayingTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL RolePlaying."""

    def __init__(self, role_playing: RolePlaying, name: str = "role_playing"):
        self.role_playing = role_playing
        self.name = name
        self._step_count = 0
        self._termination_reason: Optional[str] = None

    def record_step(self, assistant_response, user_response):
        """Call this after each step() to track progress."""
        self._step_count += 1
        if assistant_response.terminated:
            self._termination_reason = "assistant_terminated"
        elif user_response.terminated:
            self._termination_reason = "user_terminated"

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "role_playing_orchestration",
            "step_count": self._step_count,
            "termination_reason": self._termination_reason,
        }

    def gather_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_prompt": self.role_playing.task_prompt,
            "assistant_role": getattr(self.role_playing, 'assistant_role_name', None),
            "user_role": getattr(self.role_playing, 'user_role_name', None),
        }
```

**WorkforceTracer**:
```python
class CamelWorkforceTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL Workforce."""

    def __init__(self, workforce: Workforce, name: str = "workforce"):
        self.workforce = workforce
        self.name = name

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "workforce_orchestration",
            "task_decomposition": self._extract_task_graph(),
            "worker_assignments": dict(getattr(self.workforce, '_assignees', {})),
            "completed_tasks": self._extract_completed_tasks(),
            "pending_tasks": len(getattr(self.workforce, '_pending_tasks', [])),
        }

    def _extract_task_graph(self) -> Dict[str, Any]:
        deps = getattr(self.workforce, '_task_dependencies', {})
        return {task_id: list(dep_ids) for task_id, dep_ids in deps.items()}

    def _extract_completed_tasks(self) -> List[Dict[str, Any]]:
        completed = getattr(self.workforce, '_completed_tasks', [])
        return [{"id": t.id, "content": t.content} for t in completed]

    def gather_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mode": str(getattr(self.workforce, 'mode', 'unknown')),
            "workers": [w.name for w in getattr(self.workforce, '_children', [])],
        }
```

**Usage**:
```python
class MyBenchmark(Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        workforce = Workforce(...)
        self._workforce = workforce

        # Create tracer and register it
        workforce_tracer = CamelWorkforceTracer(workforce)
        self.register(workforce_tracer)  # Traces included in gather_traces()

        # Wrap individual workers for message tracing
        worker_adapters = {}
        for worker in workforce._children:
            adapter = CamelAgentAdapter(worker.agent, name=worker.name)
            worker_adapters[worker.name] = adapter

        return [], worker_adapters
```

**What gets traced**:
- Individual agents: Full message history via `CamelAgentAdapter`
- Orchestration: Task decomposition, assignments, lifecycle via tracer

**Note**: Workforce tracer accesses private attributes (`_children`, `_assignees`, etc.) which may change with CAMEL updates. Document this caveat.

---

## Integration Summary

| CAMEL Component | Integration Approach | Phase |
|-----------------|---------------------|-------|
| **ChatAgent** | `CamelAgentAdapter` | 1 (Done) |
| **RolePlaying** | Extract agents + `camel_role_playing_execution_loop` + optional tracer | 2.3, 2.4, 2.5 |
| **OWL** | Same as RolePlaying | 2.3, 2.4, 2.5 |
| **Workforce** | Extract workers + optional tracer | 2.3, 2.5 |
| **user_agent (from RolePlaying)** | `CamelAgentUser` | 2.1, 2.2 |
| **Memory** | Already traced via `get_context()` | 1 (Done) |

---

## Recommendations

### Implementation Order

1. **Phase 2.1: User Abstraction** (Core)
   - Library-wide improvement, benefits all frameworks
   - Unblocks flexible user implementations

2. **Phase 2.2: CamelAgentUser** (Interface)
   - Depends on 2.1
   - Enables RolePlaying's user_agent in MASEval

3. **Phase 2.3: Documentation**
   - Can be done in parallel with 2.1/2.2
   - Document Workforce/RolePlaying usage patterns

4. **Phase 2.4: camel_role_playing_execution_loop** (Interface)
   - Reusable execution loop for RolePlaying
   - Can be done after 2.3

5. **Phase 2.5: Orchestration Tracers** (Interface)
   - Optional add-on for debugging/analysis
   - Can be done independently

### Decision Framework

| If you need... | Use |
|----------------|-----|
| Evaluate CAMEL ChatAgents | `CamelAgentAdapter` (Phase 1) |
| Use Workforce with MASEval | Extract workers, optionally add `CamelWorkforceTracer` (2.3, 2.5) |
| Use RolePlaying with MASEval's User | Extract `assistant_agent` only (2.3) |
| Use RolePlaying's `user_agent` | `CamelAgentUser` after User refactor (2.1, 2.2) |
| Use RolePlaying's exact interaction semantics | `camel_role_playing_execution_loop` (2.4) |
| Trace orchestration state | `CamelRolePlayingTracer` / `CamelWorkforceTracer` (2.5) |

---

## Open Questions

1. **Documentation location**: Should Workforce/RolePlaying usage patterns go in `docs/interface/agents/camel.md` or a dedicated guide (e.g., `docs/guides/camel-multi-agent.md`)?

2. **Workforce private attributes**: The tracer accesses `_children`, `_assignees`, `_task_dependencies`, etc. Should we document this as "may break with CAMEL updates" or request stable APIs from CAMEL-AI?

3. **Trace category**: Should orchestration traces go to a dedicated category (e.g., `orchestration`) or use the existing `other` category in `collect_all_traces()`?

---

## Files Reference

### Completed (Phase 1 + Phase 2)
- `maseval/interface/agents/camel.py` - All CAMEL components:
  - `CamelAgentAdapter` - Wraps CAMEL ChatAgent
  - `CamelLLMUser` - LLM-simulated user with CAMEL tool
  - `CamelAgentUser` - User backed by CAMEL ChatAgent (Phase 2.2)
  - `camel_role_playing_execution_loop` - Execution loop for RolePlaying (Phase 2.4)
  - `CamelRolePlayingTracer` - RolePlaying orchestration tracer (Phase 2.5)
  - `CamelWorkforceTracer` - Workforce orchestration tracer (Phase 2.5)
- `maseval/interface/agents/__init__.py` - Conditional import (all 6 exports)
- `pyproject.toml` - Dependency and marker
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Integration tests (44 tests)
- `docs/interface/agents/camel.md` - Documentation
- `mkdocs.yml` - Navigation
- `CHANGELOG.md` - Entry

### Outstanding (Phase 1 - Testing)

**Contract Tests Gap**: CAMEL is NOT included in contract tests.

The contract tests (`tests/test_contract/test_agent_adapter_contract.py`) validate that all `AgentAdapter` implementations honor the same interface. Currently tested frameworks:
- `dummy`
- `smolagents`
- `langgraph`
- `llamaindex`
- `camel` **MISSING**

**Required changes to add CAMEL to contract tests:**

1. Add `"camel"` to the `@pytest.mark.parametrize("framework", [...])` decorator
2. Add CAMEL case to `create_agent_for_framework()`:
   ```python
   elif framework == "camel":
       pytest.importorskip("camel")
       from camel.agents import ChatAgent
       from camel.messages import BaseMessage

       # Create a mock ChatAgent for testing
       class MockCamelAgent:
           """Mock CAMEL ChatAgent for contract testing."""
           def __init__(self, response: str = "Test response"):
               self.response = response
               self.memory = MockMemory()

           def step(self, user_msg):
               # Return mock ChatAgentResponse
               ...

       return MockCamelAgent(response=mock_llm.responses[0])
   ```

3. Add CAMEL case to `create_adapter_for_framework()`:
   ```python
   elif framework == "camel":
       pytest.importorskip("camel")
       from maseval.interface.agents.camel import CamelAgentAdapter
       return CamelAgentAdapter(agent, "test_agent", callbacks=callbacks)
   ```

---

### Phase 2.1: User Abstraction (Core) ✅ IMPLEMENTED

**Core changes:**
- `maseval/core/user.py` - Refactor `User` → `User` + `LLMUser`, rename `AgenticUser` → `AgenticLLMUser`, rename `simulate_response()` → `respond()`
- `maseval/core/__init__.py` - Export `User`, `LLMUser`, `AgenticLLMUser`
- `maseval/core/benchmark.py` - Update type hints to use `User`
- `tests/test_core/test_user.py` - Update tests for new structure and method names

**Interface updates** (existing users extend `LLMUser` and override `get_tool()`):
- `maseval/interface/agents/smolagents.py` - `SmolAgentLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/langgraph.py` - `LangGraphLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/llamaindex.py` - `LlamaIndexLLMUser(LLMUser)` overrides `get_tool()`
- `maseval/interface/agents/camel.py` - `CamelLLMUser(LLMUser)` overrides `get_tool()`

---

### Phase 2.2: CamelAgentUser (Interface) ✅ IMPLEMENTED

- `maseval/interface/agents/camel.py` - Add `CamelAgentUser(User)`
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Add CamelAgentUser tests (12 tests)

---

### Phase 2.3: Documentation

- `docs/interface/agents/camel.md` - Add Workforce/RolePlaying usage patterns
- Or: `docs/guides/camel-multi-agent.md` - Dedicated guide

---

### Phase 2.4: Prefixed Execution Loop (Interface) ✅ IMPLEMENTED

- `maseval/interface/agents/camel.py` - Add `camel_role_playing_execution_loop()`
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Add execution loop tests (4 tests)

---

### Phase 2.5: Orchestration Tracers (Interface) ✅ IMPLEMENTED

- `maseval/interface/agents/camel.py` - Add `CamelRolePlayingTracer`, `CamelWorkforceTracer`
- `tests/test_interface/test_agent_integration/test_camel_integration.py` - Add tracer tests (11 tests)
