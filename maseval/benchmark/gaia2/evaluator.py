"""Gaia2 Benchmark - Evaluator.

Evaluates Gaia2 scenarios using ARE's judge system with MASEval trace integration.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
"""

import logging
from typing import Any, Dict, List, Optional

from maseval import Evaluator, Task, TaskExecutionStatus

from maseval.benchmark.gaia2.environment import Gaia2Environment

logger = logging.getLogger(__name__)


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

    Uses ARE's ``GraphPerEventJudge`` which combines deterministic hard checks
    (exact value matching) with LLM-based soft checks (semantic comparison of
    content like email bodies and calendar descriptions).

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

        Uses the judge created during ``preprocess_scenario()`` (attached to the
        scenario object) rather than creating a new one. This ensures turn
        initialization and judge state are consistent.

        Exceptions return ``gsr=None`` (excluded from scoring), matching ARE's
        behavior where exceptions/no_validation get ``score=None``.
        ARE benchmark/hf_upload_utils.py:33-52, benchmark/report_stats.py

        Args:
            traces: Filtered execution traces
            final_answer: Final answer from agent (not used in Gaia2)

        Returns:
            Dict with evaluation results. ``gsr`` is None for evaluation errors
            (excluded from scoring) or a float for valid results.
        """
        # Get ARE environment
        are_env = self.environment.get_are_environment()
        if are_env is None:
            # Infrastructure error: return None score (excluded from scoring)
            # ARE benchmark/hf_upload_utils.py:47-48
            return {
                "gsr": None,
                "partial_gsr": None,
                "passed": False,
                "status": "no_validation",
                "error": "ARE environment not available",
                "capability": self.task.metadata.get("capability"),
            }

        try:
            # Use the scenario's judge (created during preprocess_scenario)
            # ARE scenarios/scenario_imported_from_json/utils.py:112
            scenario = self.environment.get_scenario()
            judge = getattr(scenario, "judge", None)

            if judge is None:
                # Fallback: create judge if not available on scenario
                from are.simulation.validation import GraphPerEventJudgeConfig, JudgeFactory  # type: ignore[import-not-found]

                judge_engine_config = self.task.evaluation_data.get("judge_engine_config")
                if judge_engine_config is not None:
                    from are.simulation.agents.are_simulation_agent_config import (  # type: ignore[import-not-found]
                        LLMEngineConfig,
                    )
                    from are.simulation.validation.configs import create_judge_engine  # type: ignore[import-not-found]

                    llm_engine_config = LLMEngineConfig(
                        model_name=judge_engine_config.model_name,
                        provider=judge_engine_config.provider,
                        endpoint=judge_engine_config.endpoint,
                    )
                    engine = create_judge_engine(llm_engine_config)
                    judge_cfg = GraphPerEventJudgeConfig(engine=engine)
                else:
                    judge_cfg = GraphPerEventJudgeConfig()

                judge = JudgeFactory()(judge_cfg)
                judge.initialize_state(scenario)

            # Ensure intermediate turns are judged before final validation.
            #
            # ARE's validate() checks (turn_idx + 1) == (nb_turns - 1) to
            # confirm it's on the last turn before running final validation.
            # ARE simulation/validation/base.py:104
            #
            # MASEval uses a fixed run-then-evaluate cycle with no hook to
            # call the judge between turns. ARE's online-mode trigger
            # conditions run in the background event loop and MAY have
            # already advanced turn_idx, but MASEval does not control this.
            # We check turn_idx and only judge turns that haven't been
            # processed yet — this is idempotent regardless of whether
            # trigger conditions fired.
            nb_turns = judge.state.nb_turns
            last_intermediate_turn = nb_turns - 2  # turns 0..N-2 are intermediate
            while judge.state.turn_idx < last_intermediate_turn:
                turn = judge.state.turn_idx + 1
                judgment = judge(are_env)
                if not judgment.success:
                    logger.info("Intermediate turn %d/%d failed: %s", turn, nb_turns - 1, judgment.failure)
                    break  # validate() will return failure via last_turn_success check

            # Run ARE's judge validation for the final turn
            result = judge.validate(are_env)

            # Convert ARE ScenarioValidationResult to MASEval format
            passed = bool(result.success)
            gsr = 1.0 if passed else 0.0

            # Extract partial GSR from judge result if available
            # ARE's judge can produce partial scores based on event matching
            partial_gsr = getattr(result, "partial_success_rate", gsr)
            if partial_gsr is None:
                partial_gsr = gsr

            return {
                "gsr": gsr,
                "partial_gsr": partial_gsr,
                "passed": passed,
                "status": "success" if passed else "failed",
                "rationale": getattr(result, "rationale", None),
                "capability": self.task.metadata.get("capability"),
                "tool_call_count": len(traces.get("tool_invocations", [])),
                "final_simulation_time": traces.get("simulation_time", 0),
            }

        except Exception as e:
            # Evaluation error: return None score (excluded from scoring)
            # ARE benchmark/hf_upload_utils.py:42-46: exceptions get score=None
            return {
                "gsr": None,
                "partial_gsr": None,
                "passed": False,
                "status": "exception",
                "error": str(e),
                "capability": self.task.metadata.get("capability"),
                "tool_call_count": len(traces.get("tool_invocations", [])),
                "final_simulation_time": traces.get("simulation_time", 0),
            }


def compute_gaia2_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all Gaia2 benchmark results.

    Matches ARE's scoring logic:
    - Only validated runs (non-null GSR) count toward success rate
    - Exceptions and no_validation results are excluded from scoring
    - ARE benchmark/report_stats.py: success_rate calculated only from validated runs

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with metrics including total_tasks, scored_tasks, GSR, and per-capability breakdown.
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

        evals = res.get("eval") or []

        for entry in evals:
            gsr = entry.get("gsr")
            partial_gsr = entry.get("partial_gsr")
            capability = entry.get("capability", "unknown")

            # Skip entries with None score (exceptions, no_validation)
            # ARE benchmark/report_stats.py: only validated runs count
            if gsr is None:
                continue

            scored_tasks += 1
            total_gsr += gsr
            total_partial_gsr += partial_gsr if partial_gsr is not None else gsr

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
            by_capability[capability]["partial_gsr_sum"] += partial_gsr if partial_gsr is not None else gsr
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
