# MultiAgentBench: MARBLE Engine Coordination Loop Bypassed

## Bug

`run_agents()` ([multiagentbench.py:274-321](maseval/benchmark/multiagentbench/multiagentbench.py#L274-L321)) calls each agent's `run(query)` **once**. `max_iterations` and `coordinate_mode` are stored but never used for loop control or dispatch.

## Expected Behavior

MARBLE's `Engine.start()` dispatches to one of four coordination modes, each running `while current_iteration < max_iterations` with an LLM-based `EnginePlanner` that assigns tasks, summarizes results, and decides when to stop:

- **Star**: Planner assigns tasks to agents each iteration (database domain)
- **Graph**: All agents act, then self-plan via `plan_task()` in subsequent rounds (research, bargaining)
- **Chain**: One agent acts, picks next agent via `plan_next_agent()`, chain limit = `max_iterations * len(agents)`
- **Tree**: Root delegates recursively via `plan_tasks_for_children()` (coding domain)

## What MASEval Does vs. Should Do

| Step                            | MARBLE Engine                                    | MASEval                                                     |
| ------------------------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| Agents, Environment, AgentGraph | Created and wired up                             | **Same** (via `create_marble_agents`, `_setup_agent_graph`) |
| SharedMemory                    | Created                                          | **Missing**                                                 |
| EnginePlanner                   | Created; assigns tasks, summarizes, decides stop | **Missing**                                                 |
| `max_iterations`                | Controls loop bound                              | Stored, **unused**                                          |
| `coordinate_mode`               | Selects coordination method                      | Stored, **unused**                                          |
| Coordination loop               | 4 mode-specific multi-iteration loops            | **Each agent acts once**                                    |

MASEval creates MARBLE agents, wraps them in `MarbleAgentAdapter` for tracing, and sets up `AgentGraph` with relationships — but never instantiates `Engine`, `EnginePlanner`, or `SharedMemory`. The `raw_marble_config` needed to build a MARBLE `Config` is already stored in `environment_data["raw_marble_config"]`.

## Proposed Fixes

### Option A: Subclass Engine, Inject Pre-Created Components (~150 lines)

Create `MASEvalEngine(Engine)` that skips `__init__`'s factory methods and uses the agents/environment MASEval already created. Call `engine.start()` to run MARBLE's native coordination.

- **Pro**: Uses MARBLE's exact logic for all 4 modes; low drift risk
- **Con**: Fragile coupling to Engine internals (attribute names, init order). Engine calls `agent.act()` directly, bypassing `MarbleAgentAdapter._run_agent()` — traces must be extracted post-hoc. Must suppress Engine's `_write_to_jsonl()` side effect and internal `Evaluator`.

### Option B: Reimplement Coordination Loops (~500-800 lines)

Port all 4 coordination modes into MASEval, calling agents through adapters. Use MARBLE's `EnginePlanner` and `SharedMemory` directly.

- **Pro**: Full MASEval tracing on every `act()` call; no Engine internals dependency; no side effects
- **Con**: Largest effort. Must faithfully port ~800 lines including edge cases (Minecraft `block_hit_rate`, tree recursion, chain agent selection). Drift risk if MARBLE updates coordination logic.

### Option C: Let Engine Run Natively (~80-120 lines)

Build a MARBLE `Config` from `raw_marble_config`, instantiate `Engine(config)`, call `engine.start()`. Extract results afterward.

- **Pro**: Simplest; zero drift risk; guaranteed correctness for all modes
- **Con**: Engine creates its own agents/environment, so `setup_agents()`/`setup_environment()` become stubs (strains base class contract). MASEval tracing fully bypassed — traces only from post-hoc extraction. Same `_write_to_jsonl()` and internal `Evaluator` side effects as Option A.

### Comparison

| Criteria            | A (Subclass)  | B (Reimplement)  | C (Native)    |
| ------------------- | ------------- | ---------------- | ------------- |
| Correctness         | High          | High if faithful | Highest       |
| MASEval tracing     | Post-hoc      | Full             | Post-hoc      |
| Effort              | Medium        | High             | Low           |
| Drift risk          | Medium        | High             | Low           |
| Side effects        | Must suppress | None             | Must suppress |
| Base class contract | OK            | OK               | Strained      |

## Further point to validate:

data_loader.py:266 falls back to 10 when the JSONL lacks environment.max_iterations, which is incorrect for research (3), bargaining (3), and coding/database (5)
