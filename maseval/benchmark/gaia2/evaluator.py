"""Gaia2 Benchmark - Evaluator.

Evaluates Gaia2 scenarios using ARE's judge system with MASEval trace integration.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

from typing import Any, Dict, List, Optional

from maseval import Evaluator, Task, TaskExecutionStatus

from maseval.benchmark.gaia2.environment import Gaia2Environment


# Statuses where agent is accountable (included in scoring)
SCOREABLE_STATUSES = frozenset(
    {
        TaskExecutionStatus.SUCCESS.value,
        TaskExecutionStatus.AGENT_ERROR.value,
        TaskExecutionStatus.TASK_TIMEOUT.value,
    }
)


class Gaia2Evaluator(Evaluator):
    """Evaluates Gaia2 scenarios using ARE's judge system.

    Uses ARE's GraphPerEventJudge for deterministic evaluation based on
    the event DAG. Supports optional LLM-based judge for complex assertions.

    The evaluator compares completed events in the simulation against
    oracle (expected) events to compute Goal Success Rate (GSR).
    """

    def __init__(
        self,
        task: Task,
        environment: Gaia2Environment,
        user: Optional[Any] = None,
        use_llm_judge: bool = False,
        model: Optional[Any] = None,
    ):
        """Initialize the evaluator.

        Args:
            task: Task being evaluated
            environment: Gaia2Environment instance
            user: Optional user simulator (not used in Gaia2)
            use_llm_judge: Whether to use LLM-based judge
            model: Optional ModelAdapter for LLM-based evaluation
        """
        self.task = task
        self.environment = environment
        self.user = user
        self.use_llm_judge = use_llm_judge
        self.model = model

        # Extract evaluation data from task
        eval_data = task.evaluation_data
        self.oracle_events = eval_data.get("oracle_events", [])
        self.judge_type = eval_data.get("judge_type", "graph_per_event")

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool invocations and environment state for evaluation.

        Args:
            traces: Full execution traces

        Returns:
            Dict with:
                - tool_invocations: List of all tool calls with timing
                - simulation_time: Final simulation time
                - scenario_id: For correlation
        """
        tool_traces = traces.get("environment", {}).get("tools", {})

        # Flatten all tool invocations
        invocations = []
        for tool_name, tool_data in tool_traces.items():
            for inv in tool_data.get("invocations", []):
                invocations.append(
                    {
                        "tool": tool_name,
                        "inputs": inv.get("inputs", {}),
                        "outputs": inv.get("outputs"),
                        "simulation_time": inv.get("meta", {}).get("simulation_time_after"),
                        "wall_time": inv.get("meta", {}).get("wall_time"),
                    }
                )

        return {
            "tool_invocations": invocations,
            "simulation_time": traces.get("environment", {}).get("final_simulation_time", 0),
            "scenario_id": self.task.metadata.get("scenario_id"),
        }

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate using ARE's judge system.

        Args:
            traces: Filtered execution traces
            final_answer: Final answer from agent (not used in Gaia2)

        Returns:
            Dict with:
                - gsr: Goal Success Rate (0.0 or 1.0)
                - partial_gsr: Partial success rate
                - passed: Boolean indicating full success
                - event_results: Per-event evaluation results
                - capability: Task capability type
        """
        # Import ARE judge (required dependency for Gaia2)
        from are.simulation.validation import JudgeFactory  # type: ignore[import-not-found]
        from are.simulation.validation.config import GraphPerEventJudgeConfig  # type: ignore[import-not-found]

        # Create ARE judge
        judge_config = GraphPerEventJudgeConfig()
        judge = JudgeFactory.create(judge_config)

        # Get ARE environment and completed events
        are_env = self.environment.get_are_environment()
        if are_env is None:
            return {
                "gsr": 0.0,
                "partial_gsr": 0.0,
                "passed": False,
                "error": "ARE environment not available",
                "capability": self.task.metadata.get("capability"),
            }

        try:
            completed_events = are_env.get_completed_events()
        except AttributeError:
            completed_events = []

        # Run ARE's judge
        try:
            result = judge.evaluate(
                oracle_events=self.oracle_events,
                completed_events=completed_events,
                scenario=self.environment.get_scenario(),
            )

            # Convert ARE result to MASEval format
            gsr = 1.0 if result.passed else 0.0
            partial_gsr = getattr(result, "partial_score", gsr)

            return {
                "gsr": gsr,
                "partial_gsr": partial_gsr,
                "passed": result.passed,
                "event_results": getattr(result, "event_results", []),
                "capability": self.task.metadata.get("capability"),
                "tool_call_count": len(traces.get("tool_invocations", [])),
                "final_simulation_time": traces.get("simulation_time", 0),
            }

        except Exception as e:
            # Return failure on evaluation error
            return {
                "gsr": 0.0,
                "partial_gsr": 0.0,
                "passed": False,
                "error": str(e),
                "capability": self.task.metadata.get("capability"),
                "tool_call_count": len(traces.get("tool_invocations", [])),
                "final_simulation_time": traces.get("simulation_time", 0),
            }


def compute_gaia2_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all Gaia2 benchmark results.

    Infrastructure errors are excluded from scoring metrics.
    Uses SCOREABLE_STATUSES to determine which results count toward agent score.

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with:
            - total_tasks: Total number of tasks
            - scored_tasks: Tasks included in scoring
            - gsr: Overall Goal Success Rate
            - partial_gsr: Average partial GSR
            - by_capability: Metrics broken down by capability type
            - status_counts: Count by status
    """
    if not results:
        return {
            "total_tasks": 0,
            "scored_tasks": 0,
            "gsr": 0.0,
            "partial_gsr": 0.0,
            "by_capability": {},
            "status_counts": {},
        }

    total_tasks = len(results)
    scored_tasks = 0
    total_gsr = 0.0
    total_partial_gsr = 0.0
    status_counts: Dict[str, int] = {}
    by_capability: Dict[str, Dict[str, Any]] = {}

    for res in results:
        status = res.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        if status not in SCOREABLE_STATUSES:
            continue  # Skip infrastructure errors

        scored_tasks += 1
        evals = res.get("eval") or []

        for entry in evals:
            gsr = entry.get("gsr", 0.0)
            partial_gsr = entry.get("partial_gsr", 0.0)
            capability = entry.get("capability", "unknown")

            total_gsr += gsr
            total_partial_gsr += partial_gsr

            # Track by capability
            if capability not in by_capability:
                by_capability[capability] = {
                    "count": 0,
                    "gsr_sum": 0.0,
                    "partial_gsr_sum": 0.0,
                    "passed": 0,
                }

            by_capability[capability]["count"] += 1
            by_capability[capability]["gsr_sum"] += gsr
            by_capability[capability]["partial_gsr_sum"] += partial_gsr
            if entry.get("passed", False):
                by_capability[capability]["passed"] += 1

    # Compute averages
    overall_gsr = total_gsr / scored_tasks if scored_tasks > 0 else 0.0
    overall_partial_gsr = total_partial_gsr / scored_tasks if scored_tasks > 0 else 0.0

    # Compute per-capability averages
    for cap, data in by_capability.items():
        count = data["count"]
        data["gsr"] = data["gsr_sum"] / count if count > 0 else 0.0
        data["partial_gsr"] = data["partial_gsr_sum"] / count if count > 0 else 0.0
        data["pass_rate"] = data["passed"] / count if count > 0 else 0.0
        del data["gsr_sum"]
        del data["partial_gsr_sum"]

    return {
        "total_tasks": total_tasks,
        "scored_tasks": scored_tasks,
        "gsr": overall_gsr,
        "partial_gsr": overall_partial_gsr,
        "by_capability": by_capability,
        "status_counts": status_counts,
    }
