## GAIA2 Faithfulness Report: maseval vs ARE (AREFork)

Here are all instances where the maseval implementation diverges from the original ARE implementation:

---

### 1. **Scenario Duration Not Set (Critical)**

**ARE:** `preprocess_scenario()` sets `scenario.duration` to `max_scenario_duration` (default `1800` seconds / 30 minutes) before the simulation runs. For "time" capability, it uses `MAX_TIME_SCENARIO_DURATION = 420` (7 minutes). (`config.py:18-20`, `utils.py:69-76`)

**maseval:** Never calls `preprocess_scenario()`. The `scenario.duration` stays at whatever was loaded from JSON (possibly `None`). The fallback `86400` in `data_loader.py:203` is stored in `environment_data["duration"]` but never used — `setup_state()` reads `scenario.duration` directly. If `scenario.duration` is `None`, ARE's `EnvironmentConfig` defaults to 60 seconds.

**Impact:** Simulations may run with wrong duration — either 60s (too short) or whatever the JSON contains, rather than the correct 1800s/420s.

---

### 2. **Task Timeout: 600s vs 1860s**

**ARE:** `DEFAULT_SCENARIO_TIMEOUT = 1860` (31 minutes). (`config.py:20`)

**maseval:** `DEFAULT_TIMEOUT_SECONDS = 600.0` (10 minutes). (`data_loader.py:40`)

**Impact:** Tasks may time out prematurely in maseval, especially complex scenarios that need up to 30 minutes.

---

### 3. **Iteration Counting Logic (Behavioral)**

**ARE:** `iterations` counter is incremented in the `finally` block on EVERY iteration, including errors and invalid formats. Termination checks `agent.iterations >= agent.max_iterations`. (`base_agent.py:849`)

**maseval:** `_iteration_count` only incremented on successful action parse. Format retries tracked separately in `_format_retry_count`. (`gaia2.py:601`)

**Impact:** ARE's agent terminates after 80 total loop iterations (errors included). maseval's agent can do 80 successful iterations PLUS up to 10 format retries per iteration = potentially far more LLM calls before terminating.

---

### 4. **Max Iterations Termination Behavior**

**ARE:** When `max_iterations` reached, the agent calls `send_message_to_user` with "Max iterations (80) reached. Stopping." through the actual tool, recording it in the event log. Then logs `MaxIterationsAgentError`. (`are_simulation.py:109-116`)

**maseval:** Returns the string `"Max iterations (80) reached."` as a Python return value. Does NOT call the `send_message_to_user` tool. (`gaia2.py:632`)

**Impact:** The judge evaluates completed events in the simulation. ARE's max-iteration message is recorded as an event; maseval's is not. This may affect evaluation results.

---

### 5. **System Prompt: Missing Current Time**

**ARE:** Injects `"Today's date in 'YYYY-MM-DD HH' format is {date_str}"` from `scenario.start_time` into the system prompt. (`are_simulation_main.py:156-164`)

**maseval:** Does not include any current time information in the system prompt. (`environment_instructions.txt`)

**Impact:** The agent doesn't know the starting simulation time, which is critical for time-sensitive tasks.

---

### 6. **System Prompt: Agent Hints Included vs Excluded**

**ARE:** Default agent uses `DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT` with `json_agent_hints=""` (empty). (`system_prompt.py:182-190`)

**maseval:** `agent_instructions.txt` includes the `JSON_AGENT_HINTS` block ("EXECUTION GUIDELINES: Take one action at a time..."). (`agent_instructions.txt:53-56`)

**Impact:** Different agent behavior due to extra instructions in the prompt.

---

### 7. **System Prompt: Notification System Description**

**ARE:** Dynamically generates notification policy from `get_notification_system_prompt()` based on the actual notification system config and scenario apps. (`are_simulation_main.py:147-154`)

**maseval:** Hardcoded generic notification description. (`environment_instructions.txt:20-22`)

**Impact:** Agent receives different (less specific) notification policy information.

---

### 8. **Tool Description Format**

**ARE:** Uses Jinja2 template: `- {{ tool.name }}: {{ tool.description }}\n    Takes inputs: {{tool.inputs}}\n    Returns an output of type: {{tool.output_type}}`. Tool inputs are rendered as raw dict. (`tool_box.py:16-20`)

**maseval:** Custom format: `Tool: {name}\nDescription: {desc}\nParameters:\n    - {param}: {type} (required/optional) - {desc}`. (`gaia2.py:331-351`)

**Impact:** LLM sees different tool description formatting, which can affect how it constructs tool calls.

---

### 9. **Message History Format**

**ARE:** Uses log-based message construction with specific templates: `[TASK]: \n{content}\n`, `[OUTPUT OF STEP {i}] Observation:\n***\n{content}\n***\n`, error messages with "Now let's retry" suffix. (`base_agent.py:93-113`)

**maseval:** Simple `{"role": "user/assistant", "content": "..."}` format. Observations formatted as `"Observation: {result}"`. (`gaia2.py:559-629`)

**Impact:** Significant difference in how conversation history is presented to the LLM.

---

### 10. **Pre-step Notification Polling Missing**

**ARE:** Has `get_are_simulation_update_pre_step()` as a conditional pre-step that polls for environment notifications before each agent step. (`agent_factory.py:37`)

**maseval:** No pre-step functions. Notifications are only received when the agent explicitly calls `wait_for_notification`. (`gaia2.py:562-633`)

**Impact:** Agent may miss asynchronous notifications (e.g., incoming messages) that arrive between iterations.

---

### 11. **Environment Stop Message Not Checked**

**ARE:** Termination condition checks `agent.notification_system.message_queue.has_environment_stop_message()`. (`are_simulation.py:105-107`)

**maseval:** No environment stop message checking. (`gaia2.py:568`)

**Impact:** Agent may continue running after the environment signals it should stop.

---

### 12. **JSON Parsing: Different Error Handling**

**ARE:** On JSONDecodeError, raises `JsonParsingAgentError` with detailed error. No trailing comma fix. (`json_action_executor.py:33-57`)

**maseval:** On JSONDecodeError, tries to fix trailing commas and retry. Returns `None` instead of raising. (`gaia2.py:414-424`)

**Impact:** maseval is more lenient, accepting malformed JSON that ARE would reject. This changes which agent outputs count as valid actions vs errors.

---

### 13. **`action_input` Default Value**

**ARE:** Missing `action_input` defaults to empty string `""`. (`json_action_executor.py:64-70`)

**maseval:** Missing `action_input` defaults to empty dict `{}`. (`gaia2.py:470`)

**Impact:** Tools receiving `""` vs `{}` may behave differently.

---

### 14. **Evaluation: Exceptions Scored as 0.0 vs Excluded**

**ARE:** Exceptions and "no_validation" get `score=None` and are EXCLUDED from success rate calculations. (`hf_upload_utils.py:33-52`, `report_stats.py`)

**maseval:** Evaluation exceptions result in `gsr=0.0, passed=False`. They are counted as failures in metrics. (`evaluator.py:153-163`)

**Impact:** maseval inflates failure rates by counting infrastructure errors as agent failures.

---

### 15. **Partial GSR Always Equals GSR**

**ARE:** `GraphPerEventJudge` can produce partial success rates based on fraction of matched oracle events.

**maseval:** Sets `partial_gsr = gsr` unconditionally (always 0.0 or 1.0). (`evaluator.py:146`)

**Impact:** Partial success information is lost.

---

### 16. **LLM Judge Not Implemented**

**maseval:** Stores `use_llm_judge` and `model` but never references them in `__call__()`. Always creates `GraphPerEventJudgeConfig()` regardless. (`evaluator.py:55-56`, `evaluator.py:131-132`)

**Impact:** LLM-based judging is advertised but non-functional.

---

### 17. **Turn Initialization Skipped**

**ARE:** Calls `scenario.initialize_turns()` with trigger conditions for online validation during simulation. (`utils.py:145-150`)

**maseval:** Does not call `initialize_turns()`. Only calls `build_event_id_to_turn_idx()`. (`environment.py:125`)

**Impact:** Online validation during simulation is skipped. May cause issues with judge trigger conditions.

---

### 18. **Duration Fallback in data_loader is Invented**

**maseval:** `"duration": getattr(scenario, "duration", 86400)` — the `86400` (24 hours) fallback is invented and doesn't exist anywhere in ARE. (`data_loader.py:203`)

**ARE:** Duration defaults to `None` in `Scenario` class, then gets set to `1800` or `420` during preprocessing.

**Impact:** Violates AGENTS.md scientific integrity guidelines: "Only copy defaults that exist in the source."

---

### 19. **`judge_type` Stored But Ignored**

**maseval:** Stores `judge_type` from scenario metadata but always uses `GraphPerEventJudgeConfig()`. (`evaluator.py:61`, `evaluator.py:131`)

**Impact:** Scenarios requiring different judge types (e.g., `InContextJudge`) would use the wrong judge.

---

### 20. **Simulated Generation Time Not Implemented**

**ARE:** Pauses/resumes the environment during LLM generation to simulate realistic generation times. Configurable via `SimulatedGenerationTimeConfig`. (`base_agent.py:623-689`)

**maseval:** No simulated generation time support.

**Impact:** Simulation time advances differently, which could affect time-sensitive scenarios.

---

### Summary of Severity

| # | Issue | Severity | Done |
|---|-------|----------|------|
| 1 | Scenario duration not set (preprocess_scenario skipped) | **Critical** | |
| 2 | Task timeout 600s vs 1860s | **High** | |
| 3 | Iteration counting (errors excluded vs included) | **High** | |
| 4 | Max iterations doesn't call send_message_to_user tool | **High** | |
| 5 | Missing current time in system prompt | **High** | |
| 6 | Agent hints included vs excluded in prompt | **Medium** | |
| 7 | Notification system description differs | **Medium** | |
| 8 | Tool description format differs | **Medium** | |
| 9 | Message history format differs | **Medium** | |
| 10 | Pre-step notification polling missing | **Medium** | |
| 11 | Environment stop message not checked | **Medium** | |
| 12 | JSON parsing more lenient | **Low** | |
| 13 | action_input default "" vs {} | **Low** | |
| 14 | Exceptions scored as 0.0 vs excluded | **High** | |
| 15 | Partial GSR always equals GSR | **Medium** | |
| 16 | LLM judge not implemented | **Medium** | |
| 17 | Turn initialization skipped | **Medium** | |
| 18 | Invented 86400 duration fallback | **Medium** | |
| 19 | judge_type stored but ignored | **Low** | |
| 20 | Simulated generation time not implemented | **Low** | |
