"""
ColBench benchmark integration for MASEval.

Evaluates LLM agents on collaborative backend-programming tasks from
Facebook's Collaborative Agent Bench (ColBench / sweet_rl).
"""

from .colbench import ColBenchBenchmark
from .user import ColBenchUser, DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT
from .environment import ColBenchEnvironment
from .evaluator import ColBenchCodeEvaluator, check_correctness
from .agent import ColBenchAgentInner, DEFAULT_AGENT_CODE_PROMPT
from .openai_model_adapter import OpenAIModelAdapter

__all__ = [
    "ColBenchBenchmark",
    "ColBenchUser",
    "ColBenchEnvironment",
    "ColBenchCodeEvaluator",
    "ColBenchAgentInner",
    "OpenAIModelAdapter",
    "DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT",
    "DEFAULT_AGENT_CODE_PROMPT",
    "check_correctness",
]