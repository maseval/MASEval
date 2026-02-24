# Tau 2 Benchmark - Provenance Documentation

This document tracks the source mapping between MASEval's Tau 2 implementation and the original tau2-bench codebase.

## Base Version

**tau2-bench v0.2.0** (commit `f8de30c`, 2025-10-06)

Repository: https://github.com/sierra-research/tau2-bench
License: MIT License
Copyright: (c) 2025 Sierra Research

## Why v0.2.0?

| Factor           | v0.2.0                              | HEAD (main)                    |
| ---------------- | ----------------------------------- | ------------------------------ |
| **Domain tools** | Complete                            | Identical (no changes)         |
| **Core tasks**   | 50/114/114 (airline/retail/telecom) | Same "base" split              |
| **Dependencies** | Simpler (no gymnasium)              | +gymnasium for RL training     |
| **Stability**    | Formal release                      | Includes experimental features |

## Component Mapping

### Core Infrastructure

| MASEval Component | tau2-bench Source      | Notes                           |
| ----------------- | ---------------------- | ------------------------------- |
| `data_loader.py`  | `data/tau2/domains/*/` | Downloads from v0.2.0 tag       |
| `utils.py`        | `tau2/utils/`          | Hashing, file loading utilities |

### Retail Domain

| MASEval Component          | tau2-bench Source                                | Notes                        |
| -------------------------- | ------------------------------------------------ | ---------------------------- |
| `domains/retail/models.py` | `src/tau2/domains/retail/data_model.py`          | Pydantic models for entities |
| `domains/retail/db.py`     | `src/tau2/domains/retail/data_model.py:RetailDB` | Database class               |
| `domains/retail/tools.py`  | `src/tau2/domains/retail/tools.py`               | All retail tools ported      |

### Airline Domain

| MASEval Component           | tau2-bench Source                                  | Notes                        |
| --------------------------- | -------------------------------------------------- | ---------------------------- |
| `domains/airline/models.py` | `src/tau2/domains/airline/data_model.py`           | Pydantic models for entities |
| `domains/airline/db.py`     | `src/tau2/domains/airline/data_model.py:AirlineDB` | Database class               |
| `domains/airline/tools.py`  | `src/tau2/domains/airline/tools.py`                | All airline tools ported     |

### Telecom Domain

| MASEval Component           | tau2-bench Source                                  | Notes                        |
| --------------------------- | -------------------------------------------------- | ---------------------------- |
| `domains/telecom/models.py` | `src/tau2/domains/telecom/data_model.py`           | Pydantic models for entities |
| `domains/telecom/db.py`     | `src/tau2/domains/telecom/data_model.py:TelecomDB` | Database class               |
| `domains/telecom/tools.py`  | `src/tau2/domains/telecom/tools.py`                | All telecom tools ported     |

### Benchmark Components

| MASEval Component                       | tau2-bench Source                             | Notes                         |
| --------------------------------------- | --------------------------------------------- | ----------------------------- |
| `Tau2Environment`                       | `src/tau2/environment/environment.py`         | Uses MASEval Environment base |
| `Tau2User`                              | `src/tau2/user/user_simulator.py`             | Uses MASEval User base        |
| `Tau2Evaluator._evaluate_environment`   | `src/tau2/evaluator/evaluator_env.py`         | DB state comparison           |
| `Tau2Evaluator._evaluate_actions`       | `src/tau2/evaluator/evaluator_action.py`      | Tool call verification        |
| `Tau2Evaluator._evaluate_communication` | `src/tau2/evaluator/evaluator_communicate.py` | Communication checks          |

### Improvements Adopted from HEAD

| Feature                     | Description                              | Rationale                                |
| --------------------------- | ---------------------------------------- | ---------------------------------------- |
| Evaluator termination logic | Explicit `AGENT_STOP`, `USER_STOP` check | More defensive than v0.2.0's reject-list |

## Data Files

Domain data is downloaded from the v0.2.0 tag:

```
https://github.com/sierra-research/tau2-bench/tree/v0.2.0/data/tau2/domains/
```

| Domain  | Files                               | Task Count (base split) |
| ------- | ----------------------------------- | ----------------------- |
| airline | db.json, tasks.json, policy.md      | 50                      |
| retail  | db.json, tasks.json, policy.md      | 114                     |
| telecom | db.toml, tasks.json, main_policy.md | 114                     |

## MASEval-Specific Additions

These components are new implementations that don't have direct tau2-bench equivalents:

| Component                     | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `Tau2Benchmark`               | Abstract benchmark base following MASEval patterns |
| `Tau2ToolBase`                | Base class for tools with MASEval tracing          |
| `compute_pass_at_k()`         | Pass@k metric computation                          |
| TraceableMixin integration    | Execution tracing for all components               |
| ConfigurableMixin integration | Configuration gathering                            |

## Intentional Implementation Differences from Original tau2-bench

MASEval's tau2 implementation does not exactly replicate the original tau2 benchmark implementation.

The following intentional differences exist between MASEval and the original tau2-bench:

### User Simulator Prompt

**Location:** `prompt_templates/user_simulator.txt`

| Difference                                                                | Rationale                                                   |
| ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Explicit stop guidance ("When the agent confirms...END the conversation") | Prevents unnecessary conversation continuation              |
| "Do NOT ask unnecessary follow-up questions"                              | Reduces conversation length without affecting task validity |
| JSON output format required                                               | Structured parsing for MASEval framework                    |

These changes result in shorter conversations.

### Order ID Normalization (Retail Domain)

**Location:** `domains/retail/tools.py:66-68`

```python
if not order_id.startswith("#"):
    order_id = f"#{order_id}"
```

MASEval normalizes order IDs by adding the `#` prefix if missing. The original tau2-bench does not normalize, causing "Order not found" errors when LLMs omit the prefix. This is a minor leniency that improves usability without fundamentally changing task difficulty.

## Known Architectural Divergences

These structural differences arise from MASEval's framework architecture. They are
documented here for transparency.

### Step Counting

**Original:** `max_steps=100` (default) counts ALL message exchanges (agent→user,
user→agent, agent→env, env→agent). One orchestrator-level counter for the entire
conversation. Source: tau2-bench `orchestrator.py` default `max_steps=100`.

**MASEval:** Two separate mechanisms:

- `max_invocations=50` (benchmark-level): agent-user interaction rounds
- `max_tool_calls=50` (agent-internal): tool calls per agent turn

With ~4 message exchanges per invocation, `max_steps=100 ≈ 25 invocations`. The
value 50 provides headroom.

### Agent Architecture

**Original:** Orchestrator calls `agent.generate_next_message()` which returns ONE
message (text or tool calls). Tool routing happens at the orchestrator level.

**MASEval:** `DefaultTau2Agent.run()` handles tool routing internally via a ReAct
loop, returning only the final text response.

### Max Tool Calls Error Message

**Original:** When `max_steps` is reached, the orchestrator stops. No error message
is generated.

**MASEval:** When `max_tool_calls` is reached, `DefaultTau2Agent._generate_with_tools()`
returns: `"I apologize, but I've encountered an issue processing your request. Please try again."`
This message does not exist in the original and could affect evaluation of
communication-check assertions.

### max_errors

**Original:** Has `max_errors=10` parameter but `num_errors` is never incremented
(dead code). Source: tau2-bench `orchestrator.py`.

**MASEval:** Does not implement error tracking.

### User Initial Query

**Original:** User simulator generates initial query via LLM in response to the
agent's greeting.

**MASEval:** Uses pre-set `task.query` as the initial query. The greeting is injected
into the user's message history (so subsequent `respond()` calls see it), but the
initial query itself is not LLM-generated.

### Separate Message Histories

**Original:** The orchestrator maintains ONE trajectory (list of messages). Both
agent and user see filtered views of this shared trajectory.

**MASEval:** Agent and user maintain separate `MessageHistory` instances.

### User Tool Execution Routing

**Original:** User tool calls are routed through the orchestrator to the environment,
each consuming a step.  The environment's `make_tool_call()` dispatches based on the
`requestor` field, `sync_tools()` runs after every step, and tool call/result messages
appear in the global trajectory.

**MASEval:** User tools are executed inline within `Tau2User._generate_response()`.
The user LLM generates a tool call, the wrapped callable is invoked directly, the
result is appended to the user's internal messages, and the loop continues until a
text response is produced.  Tool calls still trigger `sync_tools()` via the wrapper,
and internal step counting (`_last_respond_steps`) tracks the correct number of
messages produced.

**Impact on evaluation:** None for current task data.  The evaluator reconstructs DB
state by replaying tool calls from the trajectory (including user tool calls with
their `requestor` field).  Communication evaluation only examines assistant messages.
Action evaluation extracts both assistant and user tool calls, but no task pairs user
golden actions with `ACTION` in its `reward_basis`.

### Content-as-List Handling

**Original:** Uses typed Pydantic `AssistantMessage` where `content: Optional[str]`
is always a string.  Content-as-list never occurs.

**MASEval:** Uses plain dictionaries for messages to support multiple LLM providers.
Some providers (e.g., Anthropic) return `content` as a list of blocks
(`[{"type": "text", "text": "..."}]`) instead of a plain string.  The communication
evaluator (`_evaluate_communication`) joins list blocks into a single string before
performing substring matching.  Without this, evaluation would crash with
`AttributeError` for list-typed content.

### TelecomDB Structure

**Original:** `TelecomDB` and `TelecomUserDB` are separate objects.  `TelecomTools`
receives `TelecomDB`, `TelecomUserTools` receives `TelecomUserDB`.  Hashes are
computed independently.

**MASEval:** `TelecomUserDB` is embedded as a `user_db` field inside `TelecomDB`.
Both toolkits share the same `TelecomDB` instance.  `TelecomUserTools` accesses user
state via `self.db.user_db`.  The `get_db_hash()` method on `Tau2Environment`
explicitly excludes `user_db` from the agent-side hash to match the original's
independent hashing.

## Validation Strategy

1. **Deterministic evaluators** (env, action): Exact DB state hash match with upstream v0.2.0
2. **LLM-based evaluators** (communicate, NL): Within ±3% of upstream v0.2.0
3. **Contract tests**: Tool sequences produce identical state changes

## Key Characteristics

| Aspect          | Description                                       |
| --------------- | ------------------------------------------------- |
| Tools           | Real Python implementations with business logic   |
| State           | Modifies actual database state during execution   |
| Evaluation      | Deterministic database state verification         |
| Reproducibility | Exact state matching for consistent results       |

## License

The tau2-bench code is used under the MIT License. See the original repository for full license text.
