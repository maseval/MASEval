# GAIA2 Bug Analysis

Line-by-line comparison of MASEval's GAIA2 implementation against the original ARE (Agent Research Environments) source at `~/Repositories/AREFork` (v1.2.0).

---

## Bug 1: poll_notifications uses wall-clock time instead of simulation time

**Severity:** Critical — notifications may be missed or retrieved prematurely.

**MASEval:** `environment.py:268`
```python
timestamp = datetime.now(tz=timezone.utc)
unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)
```

**ARE:** `agents/default_agent/steps/are_simulation.py:30-32`
```python
unhandled_notifications = agent.notification_system.message_queue.get_by_timestamp(
    timestamp=datetime.fromtimestamp(agent.make_timestamp(), tz=timezone.utc),
)
```

**Explanation:** ARE's notification system timestamps messages using **simulation time** (via `time_manager.time()`). The `get_by_timestamp()` method filters `event.timestamp <= timestamp`. Using wall-clock time (`datetime.now()`) instead of simulation time means:
- Wall-clock time is typically far ahead of simulation time, so messages that shouldn't be visible yet (because simulation hasn't advanced that far) would be returned prematurely.
- Conversely, if wall-clock happens to be behind for any reason, messages could be missed.

**Fix:** Use the ARE environment's current simulation time:
```python
sim_time = self._are_env.current_time
timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
```

---

## Bug 2: Tool inputs format mismatch — JSON schema vs ARE's flat dict

**Severity:** Critical — tool descriptions in the system prompt differ from ARE, affecting agent behavior.

**MASEval:** `tool_wrapper.py:129-142` — `_extract_inputs()` returns JSON schema format:
```python
{"properties": {"arg_name": {"type": "string", "description": "..."}}, "required": ["arg_name"]}
```

**ARE:** `tool_utils.py:572-578` — `AppToolAdapter` creates a flat dict:
```python
{"arg_name": {"description": "...", "type": "string", "default": "..."}}
```

**Impact:** The system prompt tool descriptions use `tool.inputs` directly (via `_build_tool_descriptions()` at `gaia2.py:435`). The LLM sees a completely different inputs format than ARE provides:
- ARE: `Takes inputs: {'timeout': {'description': 'Timeout in seconds', 'type': 'integer'}}`
- MASEval: `Takes inputs: {'properties': {'timeout': {'type': 'int', 'description': 'Timeout in seconds'}}, 'required': ['timeout']}`

Additionally, ARE converts Python types to HuggingFace types (`int` → `"integer"`, `float` → `"number"`, `bool` → `"boolean"` per `python_to_hf_type` at `tool_utils.py:533-541`). MASEval uses raw `arg.arg_type` which is the Python type string.

**Fix:** Match ARE's `AppToolAdapter` format exactly:
```python
def _extract_inputs(are_tool):
    inputs = {}
    for arg in are_tool.args:
        inputs[arg.name] = {
            "description": arg.description,
            "type": python_to_hf_type.get(str(arg.arg_type), "any"),
        }
        if arg.has_default:
            inputs[arg.name]["default"] = arg.default
    return inputs
```

---

## Bug 3: Missing AUI tool filtering — 4 extra tools exposed to agent

**Severity:** High — agent sees tools it shouldn't have, potentially altering behavior.

**ARE:** `are_simulation_main.py:206-228` — `remove_aui_irrelevant_tools()`:
1. Sets `aui.wait_for_user_response = False` on the AgentUserInterface app
2. Removes 4 tools:
   - `AgentUserInterface__get_last_message_from_user`
   - `AgentUserInterface__get_last_message_from_agent`
   - `AgentUserInterface__get_last_unread_messages`
   - `AgentUserInterface__get_all_messages`

**MASEval:** `environment.py:183-188` — `create_tools()` wraps ALL tools from ALL apps without filtering.

**Explanation:** ARE removes these tools because user messages are delivered via the notification system, not via explicit tool calls. Having them available in MASEval means:
- Extra tools pollute the system prompt (4 extra tool descriptions)
- The agent might call these tools instead of relying on the notification system
- `wait_for_user_response` not being set to `False` could cause blocking behavior

**Fix:** Filter out AUI tools and set `wait_for_user_response = False` in `create_tools()`.

---

## Bug 4: Missing tool description prefix and suffix

**Severity:** High — tool descriptions differ from ARE, changing prompt semantics.

**MASEval:** `tool_wrapper.py:70-93` — `_extract_description()` reads raw `_public_description`:
```python
desc = getattr(are_tool, "_public_description", None)
```

**ARE:** `tool_utils.py:547-582` — `AppToolAdapter.__init__()` adds prefix and suffix:
```python
self.description = f"Acts on app {app_tool.app_name}: {self.description}"
# ... later ...
if self.actual_return_type:
    self.description += f" Returns: {self.actual_return_type}"
```

**Impact:** Every tool description in ARE starts with `"Acts on app Calendar: "` (or Email, Messaging, etc.) and ends with `" Returns: str"` (or the actual return type). MASEval omits both, meaning:
- The agent doesn't know which app a tool belongs to
- The agent doesn't know what type each tool returns
- System prompt content differs significantly from ARE's reference

**Fix:** Reconstruct the full description:
```python
app_name = getattr(are_tool, "app_name", "")
desc = f"Acts on app {app_name}: {raw_desc}"
if return_type:
    desc += f" Returns: {return_type}"
```

---

## Bug 5: output_type hardcoded to "string" instead of derived from tool

**Severity:** Medium — tool description claims all tools return strings.

**MASEval:** `tool_wrapper.py:65`
```python
self.output_type: str = "string"
```

**ARE:** `tool_utils.py:554-570` — derives from `app_tool.return_type`:
```python
if return_type_str in python_to_hf_type:
    self.output_type = python_to_hf_type[return_type_str]
else:
    self.output_type = "any"
```

**Impact:** The system prompt line `Returns an output of type: {tool.output_type}` always says `"string"` in MASEval, regardless of the actual return type.

---

## Bug 6: None start_time produces empty string instead of epoch date

**Severity:** High — agent lacks temporal context entirely when start_time is None.

**MASEval:** `gaia2.py:514-516` — `_get_current_time_description()`:
```python
start_time = environment.get_start_time()
if start_time is None:
    return ""
```

**ARE:** `are_simulation_main.py:156-158`:
```python
date_str = datetime.fromtimestamp(
    scenario.start_time or 0, tz=timezone.utc
).strftime("%Y-%m-%d %H")
```

**Explanation:** ARE uses `scenario.start_time or 0`, defaulting to Unix epoch (1970-01-01 00). MASEval returns an empty string when `start_time` is None, so the `<<curent_time_description>>` placeholder becomes empty and the agent gets no time context in the system prompt.

**Fix:** Default to 0 (epoch) when start_time is None, matching ARE.

---

## Bug 7: Missing Boolean replacement (True/False → true/false)

**Severity:** Medium — JSON parsing can fail when LLM outputs Python-style booleans.

**ARE:** `agents/llm/litellm/litellm_engine.py:125`
```python
res = res.replace("False", "false").replace("True", "true")
```

**MASEval:** `gaia2.py:574-621` — `_parse_json_blob()` has no Boolean replacement.

**Explanation:** LLMs frequently output Python-style `True`/`False` in JSON blobs. ARE pre-processes the raw LLM output to replace these with JSON-valid `true`/`false` before any parsing. MASEval skips this, causing `json.loads()` to fail on `{"action": "tool", "action_input": {"flag": True}}`.

Note: ARE does this in the LLM engine layer (before stop-token truncation), while MASEval would need to do it either in `_parse_json_blob()` or in `_apply_stop_truncation()`.

**Fix:** Add Boolean replacement before JSON parsing:
```python
json_str = json_str.replace("False", "false").replace("True", "true")
```

---

## Bug 8: Missing additional_system_prompt from scenario

**Severity:** High — scenario-specific instructions are not included in the prompt.

**ARE:** `are_simulation_main.py:138-145` — `init_system_prompt()`:
```python
additional_system_prompt = scenario.additional_system_prompt
if additional_system_prompt is not None:
    self.react_agent.init_system_prompts["system_prompt"] += (
        "\n\n" + additional_system_prompt
    )
```

**MASEval:** `gaia2.py:524-571` — `_build_system_prompt()` never reads or appends `scenario.additional_system_prompt`.

**Explanation:** Some scenarios have additional instructions (e.g., persona constraints, domain-specific rules). ARE appends these to the system prompt. MASEval ignores them entirely, so the agent may miss critical scenario-specific context.

**Fix:** After assembling the prompt, append the scenario's additional system prompt if present.

---

## Bug 9: Error handling swallows errors as string observations instead of raising

**Severity:** High — changes error propagation behavior, affecting message format and retry logic.

**MASEval:** `gaia2.py:1115-1126` — `_execute_tool()`:
```python
if tool_name not in self.tools:
    return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
try:
    result = self.tools[tool_name](**tool_args)
    return str(result)
except Exception as e:
    return f"Error executing tool '{tool_name}': {e}"
```

**ARE:** `tools/json_action_executor.py:197-227` — `execute_tool_call()`:
```python
if tool_name not in self.tools:
    raise UnavailableToolAgentError(f"Error: unknown tool {tool_name}, ...")
try:
    observation = self.tools[tool_name](**arguments)
    return observation
except Exception as e:
    raise JsonExecutionAgentError(
        f"Error in tool call execution: {e}\n"
        f"You should only use this tool with a correct input.\n"
        f"As a reminder, this tool's description is the following:\n"
        f"{get_tool_description_with_args(self.tools[tool_name])}"
    )
```

**Differences:**
1. ARE **raises** exceptions; they propagate to `base_agent.py:839` where `log_error(e)` creates an ErrorLog. MASEval **catches** and returns error strings as observations.
2. ARE's error message for execution failures includes the full tool description as a reminder. MASEval's error message is generic.
3. ARE's error message for unknown tools says `"should be instead one of"`. MASEval says `"Available tools:"`.
4. Because MASEval catches errors, they appear as `Observation:` messages instead of `ERROR:` messages in the agent's context — the agent perceives errors as successful observations.

**Fix:** Raise errors from `_execute_tool()` and catch them in `_react_loop()` to format as `ERROR:` messages with the tool description reminder.

---

## Bug 10: Step counter not incremented on errors — duplicate step numbers

**Severity:** Medium — error messages use stale step numbers, confusing the agent.

**MASEval:** `gaia2.py:977` — `self._step_count += 1` only after successful action parsing, before tool execution. On errors in the except block at line 1008, the step count hasn't been incremented.

**ARE:** `base_agent.py:450-451`:
```python
if role in ["observation", "error"]:
    id_output_step += 1
```

ARE increments `id_output_step` for **both** observations and errors during history building. This means every step (success or failure) gets a unique step number.

**Impact:** In MASEval, if step 3 fails, the error message says `[OUTPUT OF STEP 3]`. If the retry also fails, it again says `[OUTPUT OF STEP 3]`. In ARE, the retry would say `[OUTPUT OF STEP 4]`.

**Fix:** Increment `_step_count` before the `[OUTPUT OF STEP ...]` message in the error path too.

---

## Bug 11: Missing `{environment_hints}` placeholder in prompt template

**Severity:** Low (currently always empty in default configs, but non-empty for APP_AGENT preset).

**ARE:** `prompts/system_prompt.py:157` — the `ARE_SIMULATION_ENVIRONMENT_INSTRUCTIONS` template includes:
```
{environment_hints}
```

This is filled with `APP_AGENT_HINTS` for the `ARE_SIMULATION_APP_AGENT_SYSTEM_PROMPT` preset and empty string `""` for others.

**MASEval:** `prompt_templates/environment_instructions.txt` does not contain `{environment_hints}`.

**Impact:** For the default JSON agent config (which ARE uses for GAIA2), `environment_hints` is always `""`, so this has no practical effect currently. However, it means MASEval can't support the APP_AGENT prompt variant without template changes.

---

## Bug 12: `__repr__` method broken due to inputs format mismatch

**Severity:** Low — affects debugging/logging output, not runtime behavior.

**MASEval:** `tool_wrapper.py:229`
```python
args = ", ".join(f"{k}: {v['type']}" for k, v in self.inputs.items())
```

Since `self.inputs` returns `{"properties": {...}, "required": [...]}` (Bug 2), iterating `.items()` yields `("properties", {...})` and `("required", [...])`, not the actual arguments. This crashes because `v['type']` on a list raises `TypeError`.

---

## Summary

| # | Bug | File | Severity |
|---|-----|------|----------|
| 1 | Wall-clock time for notifications | environment.py:268 | Critical |
| 2 | Wrong tool inputs format | tool_wrapper.py:129-142 | Critical |
| 3 | Missing AUI tool filtering | environment.py:183-188 | High |
| 4 | Missing description prefix/suffix | tool_wrapper.py:70-93 | High |
| 5 | Hardcoded output_type "string" | tool_wrapper.py:65 | Medium |
| 6 | Empty string for None start_time | gaia2.py:514-516 | High |
| 7 | Missing Boolean replacement | gaia2.py:574-621 | Medium |
| 8 | Missing additional_system_prompt | gaia2.py:524-571 | High |
| 9 | Errors swallowed as observations | gaia2.py:1115-1126 | High |
| 10 | Step counter not incremented on errors | gaia2.py:977, 1008 | Medium |
| 11 | Missing {environment_hints} placeholder | environment_instructions.txt | Low |
| 12 | __repr__ broken by inputs format | tool_wrapper.py:229 | Low |
