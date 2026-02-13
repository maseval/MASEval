from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .environment import Environment
from .task import Task
from .user import User


class Evaluator(ABC):
    """Abstract base class for evaluating agent task performance.

    Evaluator provides a structured pattern for assessing how well agents perform on tasks.
    It separates trace filtering from evaluation logic, enabling the same evaluation criteria
    to be applied across different tools, agents, or execution contexts by simply changing
    which traces are analyzed.

    The two-stage pattern (filter, then evaluate) promotes code reuse: you can test whether
    agents correctly use different tools to accomplish the same goal without duplicating
    evaluation logic. For example, to verify financial accuracy regardless of which banking
    API is called, implement the evaluation logic once in `__call__` and create different
    `filter_traces` implementations for each API.

    How to use:
        1. **Subclass Evaluator** and implement both abstract methods
        2. **Implement filter_traces** to extract relevant traces from the full execution history
        3. **Implement __call__** to compute metrics from the filtered traces and final answer
        4. **Optionally use task, environment, and user** data from `__init__` for ground truth comparison
        5. **Return a dictionary** of metrics from `__call__` for aggregation and reporting

        Example workflow:
            ```python
            class ToolUsageEvaluator(Evaluator):
                def __init__(self, task, environment, user=None):
                    super().__init__(task, environment, user)
                    self.expected_tool = "calculator"
                    self.expected_result = task.evaluation_data["correct_answer"]

                def filter_traces(self, traces):
                    # Extract only calculator tool invocations
                    return traces.get("tools", {}).get(self.expected_tool, {})

                def __call__(self, traces, final_answer=None):
                    # Evaluate using filtered calculator traces
                    invocations = traces.get("invocations", [])

                    if not invocations:
                        return {"tool_used": False, "correct_result": False}

                    tool_output = invocations[0].get("outputs", [])
                    correct = tool_output == self.expected_result

                    return {
                        "tool_used": True,
                        "correct_result": correct,
                        "accuracy": 1.0 if correct else 0.0
                    }

            # Reuse the same evaluation logic for a different tool
            class AlternativeToolEvaluator(ToolUsageEvaluator):
                def filter_traces(self, traces):
                    # Only change: filter for alternative API instead
                    return traces.get("tools", {}).get("math_api", {})
            ```

        The Benchmark framework orchestrates the evaluation lifecycle by collecting traces
        during task execution, calling `filter_traces` to extract relevant data, passing
        filtered traces and final answer to `__call__`, and aggregating results across
        evaluators and task repetitions.

    Error-handling guideline:
        Let unexpected evaluator failures raise exceptions instead of swallowing them.
        The benchmark runner handles these failures based on `fail_on_evaluation_error`:
        if enabled, the exception is re-raised; if disabled, the task report is marked
        as `evaluation_failed` and execution continues.

    Return format:
        The `__call__` method must return a dictionary of metrics. Keys are metric names,
        values are numeric scores, booleans, or serializable data. These results are:
        - Aggregated across task repetitions for statistical robustness
        - Reported in benchmark results for analysis
        - Used by downstream analysis tools

        Example return formats:
            ```python
            # Binary success metrics
            {"task_completed": True, "error": None}

            # Continuous accuracy scores
            {"precision": 0.85, "recall": 0.92, "f1_score": 0.88}

            # Detailed breakdowns
            {
                "total_steps": 12,
                "successful_steps": 10,
                "efficiency": 0.83,
                "step_details": ["step1", "step2", ...]
            }
            ```

    Args:
        task: Task instance containing query, metadata, and evaluation ground truth
        environment: Environment instance providing state and tool availability
        user: Optional User instance with user-specific data for personalized evaluation


    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        pass

    @abstractmethod
    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Compute evaluation metrics from filtered traces and final answer.

        This method contains the core evaluation logic. It receives traces pre-filtered
        by `filter_traces()` and the agent's final answer, then computes performance metrics.

        Args:
            traces: Filtered execution traces from filter_traces() containing only relevant data
            final_answer: The final answer or output from the agent system (may be None)

        Returns:
            Dictionary of evaluation metrics. Keys are metric names, values are scores or data.
            Must be JSON-serializable for reporting and aggregation.
        """
        pass

    @abstractmethod
    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for this evaluator.

        This method filters the complete execution history to only the data needed for
        evaluation. By separating filtering from evaluation, the same evaluation logic
        in `__call__` can be reused with different trace sources (e.g., different tools,
        agents, or APIs) by implementing different filters.

        Args:
            traces: Complete execution traces dictionary containing agent messages, tool calls,
                model invocations, environment state, and other execution data

        Returns:
            Filtered subset of traces relevant to this evaluator, or an empty dict if
            no filtering is needed

        Example:
            ```python
            # Filter to specific tool traces
            return traces.get("tools", {}).get("email_send", {})

            # Filter to specific agent messages
            return traces.get("agents", {}).get("researcher", {})

            # No filtering needed (use all traces)
            return traces
            ```
        """
        pass
