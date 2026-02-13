# GAIA2 Bug Report

Line-by-line comparison of `maseval/benchmark/gaia2/` against ARE source (`~/Repositories/AREFork/`).

---

## BUG 1: Wrong notification system verbosity (HIGH)

**File:** `maseval/benchmark/gaia2/environment.py:149`

**Problem:** MASEval creates the ARE environment without specifying a notification system:

```python
self._are_env = AREEnvironment(config)
```

ARE's `Environment.__init__` defaults to `NotificationSystem` which is
`VerboseNotificationSystem(verbosity_level=VerbosityLevel.LOW)`
(ARE `notification_system.py:378`). But ARE's scenario runner uses
`VerboseNotificationSystem()` which defaults to `VerbosityLevel.MEDIUM`
(ARE `scenario_runner.py:282`).

- **LOW** (what MASEval gets): Only user messages and system notifications
  (due reminders, wait-for-notification timeouts).
- **MEDIUM** (what ARE uses): Also emails (`EmailClientApp`), messaging
  (`MessagingApp`), shopping orders (`ShoppingApp`), cab rides (`CabApp`),
  calendar events (`CalendarApp`).

**Impact:** In multi-turn scenarios involving environment changes (new
emails, messages, shopping updates, cab status changes, calendar events),
the agent won't receive these notifications. This silently breaks
scenarios that depend on the agent reacting to environment-driven
events — the agent simply never learns about them.

**ARE reference:**
- `scenario_runner.py:282`: `notification_system=VerboseNotificationSystem()`
- `notification_system.py:363-369`: `VerboseNotificationSystem` defaults to `MEDIUM`
- `notification_system.py:117-169`: `get_notification_tools()` defines per-verbosity tool lists

**Fix:** Pass `VerboseNotificationSystem()` when creating the ARE environment.

---

## BUG 2: Agent loop architecture mismatch (HIGH)

**File:** `maseval/benchmark/gaia2/gaia2.py` — `DefaultGaia2Agent._react_loop()`

**Problem:** ARE uses a **two-level loop** architecture:

- **Outer loop** (`are_simulation_main.py:agent_loop()`, line 270): loops
  over turns (`max_turns = scenario.nb_turns`).
- **Inner loop** (`base_agent.py:execute_agent_loop()`, line 775): loops
  over step iterations (`max_iterations = 80`).

The GAIA2-specific termination step (`get_gaia2_termination_step()` in
`termination_methods/are_simulation.py:135-139`) terminates the **inner
loop** on BOTH `send_message_to_user` AND `wait_for_notification`.
The `termination_state_update()` function (line 21-37) sets:

- `TERMINATED` for `send_message_to_user` → outer loop increments turn count.
- `PAUSED` for `wait_for_notification` → outer loop continues without
  incrementing turns.

MASEval has a **single flat loop** where `wait_for_notification` is
treated as a regular tool call (just another observation).

**Behavioral differences:**

1. **User message formatting between turns:** In ARE, user messages from
   the notification queue are delivered via `react_agent.run(task=task)`
   which formats them as `[TASK]: \n{content}\n` (ARE `base_agent.py:96`).
   In MASEval, they're delivered as
   `User messages updates:\n***\n{content}\n***\n`.

2. **Env notification re-queuing:** ARE's outer loop puts environment
   notifications BACK into the queue (`are_simulation_main.py:352-353`)
   so they're processed by the inner loop's pre-step handler. MASEval
   directly injects them into messages.

3. **Agent state boundary:** ARE calls `react_agent.run(task=..., reset=False)`
   between turns, which preserves state but appends a new TaskLog. MASEval
   never creates this turn boundary.

**Impact:** The LLM sees different prompt formatting for multi-turn
interactions, which can change agent behavior. For single-turn scenarios
(majority of tasks), this difference is less impactful.

**ARE reference:**
- `are_simulation_main.py:230-326`: `agent_loop()` — outer turn loop
- `base_agent.py:775-854`: `execute_agent_loop()` — inner step loop
- `termination_methods/are_simulation.py:76-139`: termination conditions
- `termination_methods/are_simulation.py:21-37`: `termination_state_update()`
- `base_agent.py:93-113`: `DEFAULT_STEP_2_MESSAGE` — message format templates

**Fix:** Restructure `DefaultGaia2Agent` to use two nested loops matching
ARE's architecture. `wait_for_notification` must terminate the inner loop
with PAUSED state; outer loop re-polls notifications and continues.

---

## BUG 3: Top-level ARE import in tool_wrapper.py (MEDIUM)

**File:** `maseval/benchmark/gaia2/tool_wrapper.py:12`

**Problem:**

```python
from are.simulation.tool_utils import AppToolAdapter
```

This is a top-level import (not guarded by `try/except`). The import
chain is:

```
__init__.py → gaia2.py → environment.py → tool_wrapper.py
                                            ↓
                              from are.simulation.tool_utils import AppToolAdapter  ← FAILS
```

**Importing `maseval.benchmark.gaia2` will fail with `ImportError` if
ARE is not installed.** All other ARE imports in the codebase use lazy
imports with `try/except` guards.

**Impact:** Users who install maseval without the gaia2 extra get an
unhelpful `ImportError` on import, rather than a descriptive error when
they actually try to use GAIA2 functionality.

**Fix:** Move `AppToolAdapter` import inside `Gaia2GenericTool.__init__()`,
matching the lazy-import pattern used elsewhere in the codebase.

---

## BUG 4: Judge turn advancement may conflict with trigger conditions (MEDIUM)

**File:** `maseval/benchmark/gaia2/evaluator.py:180-188`

**Problem:** `preprocess_scenario()` sets up trigger conditions that call
`judge.__call__()` during the ARE simulation's event loop (ARE
`utils.py:131`). MASEval's evaluator ALSO calls `judge()` after the
simulation completes to advance intermediate turns:

```python
while judge.state.turn_idx < last_intermediate_turn:
    judgment = judge(are_env)
```

If trigger conditions fired during the simulation, the judge's `turn_idx`
was already advanced. Each `judge.__call__()` calls `update_state()`
which both increments `turn_idx` AND appends to `turn_to_agent_events`
(ARE `judge.py:364-367`). When called post-hoc with the full event log,
`extract_agent_events()` may produce different results than incremental
extraction during the simulation.

The code comments claim idempotency via the `while turn_idx < ...` guard,
but `update_state()` has side effects beyond just incrementing the index.

**Impact:** Potentially incorrect evaluation for multi-turn scenarios.
Severity depends on whether trigger conditions actually fire during
MASEval's agent run (they may not due to BUG 2).

**ARE reference:**
- `utils.py:121-153`: trigger condition and validation function setup
- `validation/judge.py:359-367`: `update_state()` — increments `turn_idx` and extracts events
- `validation/base.py:91-124`: `validate()` — checks `is_last_turn`

**Fix:** Verify whether trigger conditions fire during MASEval runs.
If they do, the evaluator must detect already-advanced state and skip.
Current `while` guard may be sufficient but needs validation.

---

## BUG 5: Missing `environment_hints` passthrough (LOW)

**File:** `maseval/benchmark/gaia2/gaia2.py:569`

**Problem:**

```python
environment_formatted = environment_template.format(
    tool_descriptions=tool_descriptions, environment_hints=""
)
```

The `environment_hints` is always empty string. In ARE,
`environment_hints` is also typically empty for the default JSON agent,
but ARE's `system_prompt.py` provides a mechanism for scenario-specific
hints via `{environment_hints}` placeholder.

**Impact:** No impact for standard scenarios. Could matter for custom
scenarios that provide specific hints.

**Fix:** Low priority. Read `environment_hints` from the scenario if
available.