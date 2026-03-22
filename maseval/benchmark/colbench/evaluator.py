"""
ColBench Code Evaluator — MASEval Evaluator adapter.

Matches the ``Evaluator`` ABC interface:
    - ``__init__(task, environment, user)`` — receives task/env for ground truth
    - ``filter_traces(traces)`` — extracts user traces (where the answer lives)
    - ``__call__(traces, final_answer)`` — runs unit tests, returns metrics

Safety:
    Generated code is executed with 1-second timeouts and blocked-pattern guards.
"""

from __future__ import annotations

import logging
import signal
from typing import Any, Dict, Optional

from maseval.core.evaluator import Evaluator
from maseval.core.environment import Environment
from maseval.core.task import Task
from maseval.core.user import User

logger = logging.getLogger(__name__)

# ── Blocked patterns (from sweet_rl/utils/code_utils.py) ─────────────────
_BLOCKED_PATTERNS = [
    "import os", "from os", "import sys", "from sys",
    "open(", "print(", "write", "sudo", "transformers",
    "exit(", "quit(", "argparse",
]


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def _get_function_output(function_definition: str, test_case: str) -> Any:
    try:
        exec(function_definition)
        return eval(test_case)
    except Exception:
        return None


def _queue_get_function_output(function_definition, test_case, queue):
    queue.put(_get_function_output(function_definition, test_case))


def _strip_markdown_fences(code: str) -> str:
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    return code.strip()


def check_correctness(
    ground_truth_function: str,
    test_function: str,
    test_cases: Dict[str, str],
) -> float:
    """Score a generated function against ground-truth using unit tests.

    Faithfully replicates ``code_utils.check_correctness()``.
    """
    if not test_cases:
        return 0.0

    for pat in _BLOCKED_PATTERNS:
        if pat in test_function:
            return 0.0

    test_function = _strip_markdown_fences(test_function)

    num_correct = 0
    for _name, test_expr in test_cases.items():
        gt_output = _get_function_output(ground_truth_function, test_expr)

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(1)
        try:
            test_output = _get_function_output(test_function, test_expr)
        except TimeoutError:
            test_output = None
        finally:
            signal.alarm(0)

        try:
            if gt_output == test_output and gt_output is not None:
                num_correct += 1
        except ValueError:
            pass

    return num_correct / len(test_cases)


def _extract_agent_code(
    final_answer: Any,
    traces: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Extract agent's code from available sources.

    Priority:
        1. traces (user traces with extracted answer)
        2. final_answer with "I WANT TO ANSWER:" stripped
        3. final_answer as-is
    """
    traces = traces or {}

    # Source 1: already-parsed answer from traces
    answer = traces.get("answer")
    if answer and isinstance(answer, str) and answer.strip():
        return answer.strip()

    # Source 2-3: final_answer
    if final_answer and isinstance(final_answer, str) and final_answer.strip():
        text = final_answer
        if "I WANT TO ANSWER:" in text:
            text = text.split("I WANT TO ANSWER:", 1)[1]
        return text.strip()

    return None


# ── MASEval Evaluator ────────────────────────────────────────────────────

class ColBenchCodeEvaluator(Evaluator):
    """Evaluates ColBench backend-programming trajectories.

    Uses ``task.evaluation_data`` for ground truth and test cases.
    Extracts the agent's code from user traces (where ``ColBenchUser``
    stores the parsed answer) via ``filter_traces()``.
    """

    def __init__(
        self,
        task: Task,
        environment: Environment,
        user: Optional[User] = None,
    ):
        super().__init__(task, environment, user)
        # Read ground truth from ColBenchEnvironment (set in setup_state)
        # This is more reliable than task.evaluation_data which may not be stored
        self.ground_truth = getattr(environment, 'ground_truth', '')
        self.test_cases = getattr(environment, 'test_cases', {})

        # Fallback to task.evaluation_data if environment doesn't have it
        if not self.ground_truth:
            eval_data = getattr(task, 'evaluation_data', None) or {}
            self.ground_truth = eval_data.get("ground_truth", "")
            self.test_cases = eval_data.get("test_cases", {})

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user traces where the answer is stored."""
        return traces.get("user", {}) or {}

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run unit tests and return correctness metrics.

        Args:
            traces: Filtered traces (user traces from filter_traces).
            final_answer: The last return value from run_agents().
        """
        agent_code = _extract_agent_code(final_answer, traces)

        if not agent_code:
            return {
                "correctness": 0.0,
                "success": False,
                "num_tests": len(self.test_cases),
                "num_passed": 0,
                "error": "no_answer_extracted",
            }

        correctness = check_correctness(
            self.ground_truth, agent_code, self.test_cases
        )
        num_passed = int(round(correctness * len(self.test_cases)))

        return {
            "correctness": correctness,
            "success": correctness == 1.0,
            "num_tests": len(self.test_cases),
            "num_passed": num_passed,
        }