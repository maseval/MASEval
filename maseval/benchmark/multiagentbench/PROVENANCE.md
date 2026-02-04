# MARBLE Integration Provenance

## Source Information

- **Source Repository**: https://github.com/ulab-uiuc/MARBLE
- **Version**: Not yet pinned (clone latest and test)
- **License**: MIT (Copyright 2024 Haofei Yu)
- **Vendoring**: Permitted by MIT license with attribution

## Reference

**Paper**: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents"

- arXiv: https://arxiv.org/abs/2503.01935
- Authors: Haofei Yu et al.
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

## Local Patches Applied

None currently. Document any patches here if applied.

## Update Process

To update MARBLE to a newer version:

1. `cd maseval/benchmark/multiagentbench/marble`
2. `git fetch origin`
3. `git log --oneline origin/main` (review changes)
4. `git checkout <new-commit-hash>`
5. Run integration tests
6. Update this file with new version info
