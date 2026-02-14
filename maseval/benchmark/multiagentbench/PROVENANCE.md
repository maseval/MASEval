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
