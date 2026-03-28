# AREEnvironment Issues

Issues identified by comparing `AREEnvironment` (generic ARE wrapper in `maseval/interface/environments/`) with `Gaia2Environment` (dataset-specific implementation in `maseval/benchmark/gaia2/`).

Reviewed through scientific coding principles: silent failures that could produce wrong-but-plausible benchmark results are treated as high priority.

**Design principle:** maseval's `Benchmark` class provides `fail_on_task_error`, `fail_on_setup_error`, `fail_on_evaluation_error`, etc. flags that give users explicit control over error tolerance. Environment and tool code must propagate errors so the benchmark runner can classify them (agent fault vs. infrastructure fault) and respect the user's `fail_on_*` settings. Silent swallowing inside environment code bypasses this mechanism entirely — the user asked for strict mode but gets silent degradation.

## High Priority

### 1. Oracle mode silently degrades to empty data

`maseval/interface/environments/are.py:200-203` uses `hasattr` checks that silently fall back to empty dicts/lists:

```python
oracle_traces = {
    "apps_state": oracle_env.get_apps_state() if hasattr(oracle_env, "get_apps_state") else {},
    "world_logs": oracle_env.get_world_logs() if hasattr(oracle_env, "get_world_logs") else [],
}
```

If the ARE API changes or the methods are missing, oracle traces will be `{}` and `[]` — structurally valid but scientifically empty. Downstream evaluation code will run without error and produce meaningless scores. This is a textbook "wrong result that looks right" scenario.

`Gaia2Environment` avoids this entirely by delegating to ARE's canonical `preprocess_scenario()`.

No tests cover oracle mode, so this cannot be caught by CI.

**Fix:** Remove the `hasattr` fallbacks. Call the methods directly — if they don't exist, the crash immediately tells you the ARE API changed. Alternatively, delegate to `preprocess_scenario()`. Add oracle mode tests.

### 2. Missing simulation time tracking in AREToolWrapper

`Gaia2GenericTool` records simulation time before/after each tool call (`maseval/benchmark/gaia2/tool_wrapper.py:123-150`). `AREToolWrapper` only records wall-clock time (`maseval/interface/environments/are_tool_wrapper.py:122-128`).

Simulation time is the temporal coordinate of the experiment. Without it, any analysis of agent behavior in time-sensitive ARE scenarios is done against wall-clock time, which conflates LLM latency with simulated environment dynamics. Results would be unreproducible across different hardware or API providers.

**Fix:** Add `_get_simulation_time()` helper and record `simulation_time_before`, `simulation_time_after`, `simulation_time_elapsed` in the invocation `meta` dict, matching `Gaia2GenericTool`.

### 3. Schema extraction silently fabricates defaults

`AREToolWrapper._extract_schema()` (`maseval/interface/environments/are_tool_wrapper.py:90-95`):

```python
"type": getattr(arg, "arg_type", "string"),   # fabricates "string" if missing
if not getattr(arg, "has_default", True):      # assumes optional if missing
```

If an ARE tool arg is missing `arg_type`, it silently becomes `"string"`. If `has_default` is missing, it silently becomes optional. The schema will look valid but will be wrong — agents will receive incorrect type information and parameter requirements, producing tool calls that either fail in hard-to-trace ways or succeed with wrong arguments.

`Gaia2GenericTool` accesses these directly (`arg.arg_type`, `arg.has_default`) — it will crash immediately if the structure is unexpected, which is the correct behavior.

**Fix:** Remove the silent defaults. Access `arg.arg_type` and `arg.has_default` directly. If ARE tools don't have these attributes, that's a real problem that should surface immediately. Additionally, use ARE's `AppToolAdapter` as the canonical source of tool metadata, as `Gaia2GenericTool` does.

### 4. `poll_notifications()` silently swallows all exceptions

Both `AREEnvironment` (`are.py:330-331`) and `Gaia2Environment` (`environment.py:328-329`):

```python
except Exception:
    return [], [], False
```

If the notification system has a bug, returns corrupt data, or the ARE API changes, this catch-all returns "no notifications, simulation not stopped" — a plausible-looking empty result. The agent loop continues as if nothing happened, missing user messages and environment events. The benchmark produces scores based on an agent that never received its inputs.

This is distinct from lifecycle cleanup (where swallowing errors is acceptable). Notification polling is part of the data path — it directly affects what the agent sees and does.

Additionally, this bypasses maseval's `fail_on_*_error` mechanism: even if the user configured strict mode via `fail_on_task_error` or `fail_on_setup_error` to catch infrastructure failures, these errors are swallowed before the benchmark runner ever sees them.

**Fix:** Remove the bare `except Exception`. Catch only the specific exceptions that represent expected transient conditions (if any). Let unexpected errors propagate — the benchmark runner classifies them (`ENVIRONMENT_ERROR` during execution, `SETUP_FAILED` during setup) and the user's `fail_on_*` settings decide whether to abort or continue.

## Medium Priority

### 5. AUI tool filtering missing from AREEnvironment

`Gaia2Environment` filters out `AgentUserInterface` message-retrieval tools and sets `wait_for_user_response = False` (`maseval/benchmark/gaia2/environment.py:178-213`), matching ARE's default agent behavior. `AREEnvironment` doesn't.

Anyone using `AREEnvironment` with ARE's notification-based message delivery will get tools that block waiting for a response or duplicate user messages. This is a correctness issue for interactive scenarios, not just a convenience gap.

**Fix:** Add an opt-in parameter (e.g. `filter_aui_tools=True`) to `AREEnvironment.__init__`, or at minimum document the behavior prominently so users of the generic wrapper know they must handle this themselves.

### 6. Inconsistent error handling in AREEnvironment lifecycle methods

`cleanup()` wraps in try/except (`are.py:360-364`), but `pause()` and `resume_with_offset()` propagate exceptions (`are.py:268-280`). `Gaia2Environment` wraps all lifecycle methods consistently.

For lifecycle methods, the question is: "if this fails, can the experiment still produce correct results?"

- `cleanup()`: Swallowing is acceptable — the task is already done, we're tearing down.
- `pause()` / `resume_with_offset()`: These control simulation time during agent execution. A failure here means time is advancing when it shouldn't be (or vice versa), directly affecting the experimental conditions. The current inconsistency means `pause()` will crash the benchmark run while `cleanup()` won't — but neither behavior is fully correct.

Crucially, swallowing these errors removes the user's ability to control error tolerance via `fail_on_task_error` / `fail_on_setup_error`. The user opted into strict mode precisely to catch conditions like "simulation time wasn't paused during LLM generation." Silent swallowing overrides that choice.

**Fix:** Let `pause()` and `resume_with_offset()` propagate exceptions. The benchmark runner classifies them and the user's `fail_on_*` settings decide the outcome. `cleanup()` is the only lifecycle method where swallowing is acceptable (teardown after task completion).

### 7. No `get_turn_notifications()` on AREEnvironment

`Gaia2Environment` has `get_turn_notifications()` (`maseval/benchmark/gaia2/environment.py:331-380`) which re-queues environment notifications for the inner agent loop — essential for multi-turn ARE scenarios. Without it, environment notifications are consumed by the outer turn loop and never reach the agent's step loop.

Anyone building a multi-turn agent on `AREEnvironment` will silently lose environment notifications between turns.

**Fix:** Add `get_turn_notifications()` to `AREEnvironment`, or document that `poll_notifications()` alone is insufficient for multi-turn agent loops and that users must implement re-queuing themselves.

## Low Priority

### 8. Metadata extraction should use AppToolAdapter

`Gaia2GenericTool` delegates metadata extraction to ARE's `AppToolAdapter` (`maseval/benchmark/gaia2/tool_wrapper.py:68-75`) — the canonical source of truth for tool name, description, inputs, and output_type. `AREToolWrapper` reads these attributes directly from the ARE tool object.

This works today but couples `AREToolWrapper` to ARE's internal tool structure rather than its public adapter API.

**Fix:** Use `AppToolAdapter` in `AREToolWrapper` for metadata extraction.
