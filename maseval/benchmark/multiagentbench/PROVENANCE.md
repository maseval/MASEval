# MARBLE Integration Provenance

## Source Information

- **Original Repository**: https://github.com/ulab-uiuc/MARBLE (where the original work was done)
- **Fork Used**: https://github.com/cemde/MARBLE (contains bug fixes)
- **Version**: Currently unpinned (tracking latest from fork while bug fixes are being added)
- **License**: MIT (Copyright 2024 Haofei Yu)
- **Vendoring**: Permitted by MIT license with attribution

**Note**: Once the fork is stable, we will pin to a specific commit hash for reproducibility.

### Why We Use a Fork

We vendor from https://github.com/cemde/MARBLE rather than the original repository because:

- The fork contains critical bug fixes needed for integration with MASEval
- All credit for the original work goes to the MARBLE team (Zhu et al., 2025)
- The fork maintains the same MIT license and contains no API changes, only bug fixes

## Reference

**Paper**: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents"

- arXiv: https://arxiv.org/abs/2503.01935
- Authors: Zhu et al., 2025
- Publication Date: 2025

## License Text (MIT)

```
MIT License

Copyright (c) 2024 Haofei Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Known Issues in MARBLE

1. **Missing method**: `AgentGraph.get_agent_profiles_linked()` does not exist but is
   called in `engine.py:702`. This breaks chain coordination mode.

2. **SharedMemory naming**: Despite the name, `SharedMemory` is instantiated per-agent
   in `BaseAgent.__init__()` and is NOT shared between agents. Use `msg_box` for
   inter-agent communication.

3. **Environment constructor signature**: Some environments expect different constructor
   arguments. Check each environment's `__init__` signature before use.

4. **`evaluate_planning` `.format()` crash** (`evaluator.py:108`,
   `evaluator_prompts.json` line 9): The Planning prompt contains single-brace JSON
   examples (`{"rating": X}`). When `evaluate_planning` calls `.format(summary=...)`,
   Python parses `{"rating": X}` as a format field → `KeyError`. Graph mode is
   unaffected (planning evaluation is always `-1`). Star/chain/tree modes crash.
   MASEval patches the same bug in `evaluate_communication` via
   `_create_patched_marble_evaluator()` but deliberately leaves `evaluate_planning`
   unchanged to match MARBLE's crash behavior.

5. **Chain communication assertion bug** (`engine.py:720-725`): In `chain_coordinate`,
   engine.py:720 stores raw `communication` (None or dict) into
   `iteration_data["communications"]`, then asserts it's a list at L725 → crashes if
   the agent actually communicated. MASEval's `_chain_coordinate` uses
   `adapter._communication_log` (always a list), avoiding the crash.

## Architectural Differences from MARBLE

### Result summarization before evaluation

In MARBLE, agent results are summarized in the engine's coordination loop
(`Engine._summarize_results()` + `EnginePlanner.summarize_output()`) before
reaching the evaluator. MASEval does not use MARBLE's engine loop, so this
summarization logic has been moved into `MultiAgentBenchEvaluator` (see
`_summarize_results()` and `_summarize_output()` in `evaluator.py`). The
behaviour is identical: each agent result is truncated to 1000 characters, then
an LLM call condenses the truncated output into a compact summary. The
truncation length is configurable via `result_truncation_length`.

## Faithfulness Audit (2026-02-18)

A line-by-line audit was performed comparing MASEval's reproduction mode
(`MarbleMultiAgentBenchBenchmark`) against MARBLE's `engine.py`, `evaluator.py`,
and domain configs. The goal: ensure the main experiment from the paper
(graph-mesh, 6 domains, 5 models) can be faithfully reproduced.

### Fixes Applied

| ID | Severity | File | Fix |
|---|---|---|---|
| D01 | CRITICAL | `multiagentbench.py` | Fixed import `marble.utils.utils` → `marble.llms.model_prompting` |
| D02 | CRITICAL | `marble_adapter.py` | Added Minecraft agent registration (`env.register_agent`) matching `engine.py:169-173` |
| D03 | CRITICAL | `data_loader.py` | Infer missing `scenario`/`task_id` fields for minecraft JSONL entries |
| D04 | CRITICAL | `data_loader.py` | Per-domain `max_iterations` defaults from MARBLE YAML configs (research=3, coding=5, database=5, bargaining=3, minecraft=20) |
| D05 | CRITICAL | `data_loader.py` | Per-domain `coordinate_mode` defaults (bargaining=chain, others=graph) |
| D06 | CRITICAL | `data_loader.py` | Per-domain `environment.type` defaults (Research, Coding, DB, WorldSimulation, Minecraft) |
| D07 | CRITICAL | `multiagentbench.py` | Resolve `score.json` path via `_MARBLE_ROOT` instead of hardcoded `../data/score.json` |
| D08 | CRITICAL | `multiagentbench.py` | Resolve `workspace/solution.py` via `_MARBLE_ROOT` instead of hardcoded relative path |
| D11 | CRITICAL | `multiagentbench.py` | Unified `coordinate_mode` defaults to `"graph"` across all 3 locations |
| D12 | CRITICAL | `data_loader.py` | Per-domain `memory.type` default (`SharedMemory`) for blank JSONL fields |
| D13 | HIGH | `multiagentbench.py` | Evaluator model default `gpt-4o-mini` → `gpt-3.5-turbo` (matching MARBLE) |
| D14 | HIGH | `multiagentbench.py` | Agent model default `gpt-4o-mini` → `""` (matching MARBLE Config) |
| D16 | HIGH | `marble_adapter.py` | Replaced auto-generated agent IDs with `ValueError` (matching MARBLE's `assert`) |
| D20 | MODERATE | `data_loader.py` | Replaced `or 10` falsy trap with domain-specific defaults |

### Known MARBLE Bugs Faithfully Preserved

These bugs exist in MARBLE and are intentionally kept in MASEval to match
MARBLE's behavior exactly:

- **D09**: Agent errors silently swallowed in all coordination loops (`engine.py:254,373,541`)
- **D10**: Bargaining evaluation returns -1 on JSON parse failure (`evaluator.py:230-282`)
- **D17**: Chain coordination silently loops same agent on error (`engine.py:709-716`)
- **D18**: `_format_agent_tasks` bare except changes formatting (`engine.py:1052-1058`)

### Remaining Divergences (Non-Reproduction Mode Only)

These affect `MultiAgentBenchEvaluator` (the abstract base class for
framework-agnostic evaluation) but NOT `MarbleMultiAgentBenchBenchmark`
(reproduction mode):

- **D19**: Coding prompt template (`coding.txt`) missing strict scoring directives
- **D22-D28**: Various defensive fallbacks in evaluator helper methods
- **D27**: Planning + KPI evaluation not implemented in abstract evaluator

## Local Patches Applied

None currently. All bug fixes are maintained in the fork.

## Update Process

To update MARBLE to a newer version from the fork:

1. `cd maseval/benchmark/multiagentbench/marble`
2. `git remote set-url origin https://github.com/cemde/MARBLE.git` (if needed)
3. `git fetch origin`
4. `git log --oneline origin/main` (review changes)
5. `git checkout <new-commit-hash>`
6. Run integration tests
7. Update this file with new version info
