"""
ColBench Environment — MASEval Environment adapter.

In MASEval, Environment.__init__(task_data) automatically calls:
    1. setup_state(task_data) → stored as self.state
    2. create_tools()         → stored as self.tools

So all initialization happens via task_data, not constructor args.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from maseval.core.environment import Environment


class ColBenchEnvironment(Environment):
    """MASEval Environment for a single ColBench task instance.

    Holds the ground-truth artifacts needed for post-hoc evaluation.
    No tools — the agent interacts only via dialogue with the User.

    Attributes (set in setup_state):
        problem_description: Task description shown to the agent.
        ground_truth:        Reference code — hidden from agent, used by evaluator.
        test_cases:          Unit-test expressions for backend tasks.
        task_type:           "code" or "html".
    """

    def __init__(self, task_data: Dict[str, Any], **kwargs):
        # Don't set attributes here — super().__init__() calls setup_state()
        # before we'd get a chance, and create_tools() runs too.
        super().__init__(task_data=task_data, **kwargs)

    def setup_state(self, task_data: dict) -> Any:
        """Extract task artifacts from task_data."""
        self.problem_description = str(task_data.get("problem_description", ""))
        self.ground_truth = str(task_data.get("ground_truth", ""))
        self.test_cases = task_data.get("test_cases", {}) or {}
        self.task_type = task_data.get("task_type", "code")
        return task_data  # stored as self.state

    def create_tools(self) -> Dict[str, Any]:
        """ColBench has no tools — the agent interacts only via dialogue."""
        return {}

    # ── Tracing / Config ─────────────────────────────────────────────────

    def gather_traces(self) -> Dict[str, Any]:
        return {
            **super().gather_traces(),
            "task_type": self.task_type,
            "problem_description_length": len(self.problem_description),
            "ground_truth_length": len(self.ground_truth),
            "num_test_cases": len(self.test_cases),
        }

    def gather_config(self) -> Dict[str, Any]:
        return {
            **super().gather_config(),
            "task_type": self.task_type,
            "num_test_cases": len(self.test_cases),
        }