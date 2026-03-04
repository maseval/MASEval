"""Agent adapters for MultiAgentBench.

Original Repository: https://github.com/ulab-uiuc/MARBLE
Fork Used: https://github.com/cemde/MARBLE (contains bug fixes for MASEval integration)
Code License: MIT

Citation:
    Zhu, et al. (2025). MultiAgentBench: Evaluating the Collaboration and Competition
    of LLM agents. arXiv:2503.01935.
"""

from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
    MarbleAgentAdapter,
)

__all__ = ["MarbleAgentAdapter"]
