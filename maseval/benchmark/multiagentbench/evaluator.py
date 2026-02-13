"""MultiAgentBench evaluator implementation.

This module provides evaluation metrics matching MARBLE's evaluation methodology.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from maseval import Evaluator, ModelAdapter

# Sentinel value for "evaluation not performed" or "evaluation failed"
# Using None instead of -1 to avoid confusion with valid scores
SCORE_NOT_EVALUATED: None = None


@dataclass
class MultiAgentBenchMetrics:
    """Metrics collected during MultiAgentBench evaluation.

    Attributes:
        task_completion: Whether the task was completed
        token_consumption: Total tokens used
        communication_score: Score for inter-agent communication (1-5), None if not evaluated
        task_evaluation: Domain-specific evaluation results
        agent_kpis: Per-agent key performance indicators
        total_milestones: Number of milestones achieved
    """

    task_completion: bool = False
    token_consumption: int = 0
    communication_score: Optional[float] = None
    task_evaluation: Dict[str, Any] = field(default_factory=dict)
    agent_kpis: Dict[str, int] = field(default_factory=dict)
    total_milestones: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_completion": self.task_completion,
            "token_consumption": self.token_consumption,
            "communication_score": self.communication_score,
            "task_evaluation": self.task_evaluation,
            "agent_kpis": self.agent_kpis,
            "total_milestones": self.total_milestones,
        }


class MultiAgentBenchEvaluator(Evaluator):
    """Evaluator for MultiAgentBench tasks matching MARBLE's methodology.

    This evaluator implements MARBLE's LLM-based evaluation metrics:
    - Task completion assessment
    - Communication quality scoring
    - Planning/coordination scoring
    - Domain-specific task evaluation (research, bargaining, etc.)

    Attributes:
        domain: The benchmark domain (research, bargaining, etc.)
        model_adapter: Model adapter for LLM-based evaluation
        metrics_config: Configuration for metrics to evaluate
    """

    DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"

    def __init__(
        self,
        domain: str,
        model_adapter: ModelAdapter,
        metrics_config: Optional[Dict[str, Any]] = None,
        output_format: str = "",
    ):
        """Initialize the evaluator.

        Args:
            domain: Benchmark domain (research, bargaining, etc.)
            model_adapter: Model adapter for LLM evaluation
            metrics_config: Configuration for evaluation metrics
            output_format: Expected output format for task evaluation
        """
        self.domain = domain.lower()
        self.model_adapter = model_adapter
        self.metrics_config = metrics_config or {}
        self.output_format = output_format
        self._evaluation_prompts = self._load_evaluation_prompts()

    def _load_template(self, filename: str) -> str:
        """Load a prompt template from file.

        Args:
            filename: Template filename (without directory)

        Returns:
            Template content as string
        """
        template_path = self.DEFAULT_TEMPLATES_DIR / filename
        return template_path.read_text()

    def _load_evaluation_prompts(self) -> Dict[str, Any]:
        """Load evaluation prompts from template files."""
        return {
            "communication": {
                "prompt": self._load_template("communication.txt"),
            },
            "research": {
                "task_evaluation": {
                    "prompt": self._load_template("research.txt"),
                }
            },
            "bargaining": {
                "task_evaluation": {
                    "buyer_prompt": self._load_template("bargaining_buyer.txt"),
                    "seller_prompt": self._load_template("bargaining_seller.txt"),
                }
            },
            "coding": {
                "task_evaluation": {
                    "prompt": self._load_template("coding.txt"),
                }
            },
        }

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter traces for evaluation.

        Args:
            traces: All collected traces

        Returns:
            Filtered traces relevant for evaluation
        """
        return {
            "agents": traces.get("agents", {}),
            "environment": traces.get("environment", {}),
            "communications": self._extract_communications(traces),
        }

    def _extract_communications(self, traces: Dict[str, Any]) -> str:
        """Extract communication logs from traces.

        Args:
            traces: Execution traces

        Returns:
            Formatted communication string
        """
        communications: List[str] = []

        # Extract from agent traces
        agent_traces = traces.get("agents", {})
        for agent_id, agent_trace in agent_traces.items():
            comm_log = agent_trace.get("communication_log", [])
            for entry in comm_log:
                comm = entry.get("communication", "")
                if comm:
                    communications.append(f"[{agent_id}]: {comm}")

        return "\n".join(communications) if communications else "No communications recorded."

    def _extract_results(self, traces: Dict[str, Any]) -> str:
        """Extract agent results from traces.

        Args:
            traces: Execution traces

        Returns:
            Formatted results string
        """
        results: List[str] = []

        agent_traces = traces.get("agents", {})
        for agent_id, agent_trace in agent_traces.items():
            action_log = agent_trace.get("action_log", [])
            for entry in action_log:
                result = entry.get("result", "")
                if result:
                    results.append(f"[{agent_id}]: {result}")

        return "\n".join(results) if results else "No results recorded."

    def __call__(  # type: ignore[override]
        self, traces: Dict[str, Any], final_answer: Any
    ) -> Dict[str, Any]:
        """Evaluate the task execution.

        Args:
            traces: Filtered execution traces
            final_answer: Final output from agents (dict, list, str, or None)

        Returns:
            Evaluation results dictionary
        """
        metrics = MultiAgentBenchMetrics()

        # Extract filtered data
        filtered = self.filter_traces(traces)
        communications = filtered["communications"]

        # Calculate token consumption
        metrics.token_consumption = self._calculate_token_consumption(traces)

        # Evaluate communication if present
        if communications != "No communications recorded.":
            metrics.communication_score = self._evaluate_communication(self._get_task_description(traces), communications)

        # Domain-specific evaluation
        task_desc = self._get_task_description(traces)
        final_result = self._format_final_answer(final_answer)

        if self.domain == "research":
            metrics.task_evaluation = self._evaluate_research(task_desc, final_result)
        elif self.domain in ("bargaining", "worldsimulation"):
            metrics.task_evaluation = self._evaluate_bargaining(task_desc, final_result)
        elif self.domain == "coding":
            metrics.task_evaluation = self._evaluate_coding(task_desc, final_result)
        elif self.domain == "database":
            metrics.task_evaluation = self._evaluate_database(task_desc, final_result)
        else:
            # Default: check if task has a completion marker
            metrics.task_completion = bool(final_result)

        # Set task completion based on evaluation
        metrics.task_completion = self._determine_completion(metrics)

        return {
            "passed": metrics.task_completion,
            "metrics": metrics.to_dict(),
            "domain": self.domain,
        }

    def _get_task_description(self, traces: Dict[str, Any]) -> str:
        """Get task description from traces."""
        env_traces = traces.get("environment", {})
        state = env_traces.get("marble_state", {})
        return state.get("task_description", "")

    def _format_final_answer(self, final_answer: Any) -> str:
        """Format final answer for evaluation."""
        if isinstance(final_answer, dict):
            # Handle structured output from run_agents
            results = final_answer.get("agent_results", [])
            if results:
                return "\n".join(f"[{r.get('agent_id', 'unknown')}]: {r.get('result', '')}" for r in results)
            return json.dumps(final_answer)
        elif isinstance(final_answer, list):
            return "\n".join(f"[{r.get('agent_id', 'unknown')}]: {r.get('result', '')}" for r in final_answer if isinstance(r, dict))
        return str(final_answer) if final_answer else ""

    def _calculate_token_consumption(self, traces: Dict[str, Any]) -> int:
        """Calculate total token consumption."""
        total = 0

        agent_traces = traces.get("agents", {})
        for agent_trace in agent_traces.values():
            token_usage = agent_trace.get("token_usage", 0)
            if isinstance(token_usage, int):
                total += token_usage

        return total

    def _evaluate_communication(self, task: str, communications: str) -> Optional[float]:
        """Evaluate communication quality using LLM."""
        prompt_template = self._evaluation_prompts["communication"]["prompt"]
        prompt = prompt_template.format(task=task, communications=communications)

        try:
            response = self.model_adapter.generate(prompt)
            return self._parse_score(response)
        except Exception:
            return None

    def _evaluate_research(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate research task output."""
        prompt_template = self._evaluation_prompts["research"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        try:
            response = self.model_adapter.generate(prompt)
            return self._parse_research_ratings(response)
        except Exception:
            return {"innovation": None, "safety": None, "feasibility": None}

    def _evaluate_bargaining(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate bargaining/world simulation task output."""
        # Evaluate both buyer and seller perspectives
        buyer_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["buyer_prompt"]
        seller_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["seller_prompt"]

        ratings = {"buyer": {}, "seller": {}}

        try:
            buyer_response = self.model_adapter.generate(buyer_prompt.format(task=task, result=result))
            ratings["buyer"] = self._parse_bargaining_ratings(buyer_response)
        except Exception:
            ratings["buyer"] = {
                "effectiveness_of_strategies": None,
                "progress_and_outcome": None,
                "interaction_dynamics": None,
            }

        try:
            seller_response = self.model_adapter.generate(seller_prompt.format(task=task, result=result))
            ratings["seller"] = self._parse_bargaining_ratings(seller_response)
        except Exception:
            ratings["seller"] = {
                "effectiveness_of_strategies": None,
                "progress_and_outcome": None,
                "interaction_dynamics": None,
            }

        return ratings

    def _evaluate_coding(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate coding task output."""
        prompt_template = self._evaluation_prompts["coding"]["task_evaluation"]["prompt"]

        # For coding, we need requirements and solution separately
        # If not available, use task as description and result as solution
        prompt = prompt_template.format(
            task_description=task,
            requirements="See task description",
            solution=result,
        )

        try:
            response = self.model_adapter.generate(prompt)
            return self._parse_coding_ratings(response)
        except Exception:
            return {
                "instruction_following": None,
                "executability": None,
                "consistency": None,
                "quality": None,
            }

    def _evaluate_database(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate database task output.

        Database tasks have ground truth labels that would be compared
        separately. Here we just store the prediction.
        """
        return {
            "predicted": result,
            "root_cause": [],  # Would be filled from task data
        }

    def _parse_score(self, response: str) -> Optional[float]:
        """Parse a single score from LLM response.

        Returns:
            Score as float (1-5), or None if parsing fails
        """
        try:
            content = response.strip()

            # Remove markdown code block markers
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Find JSON object
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                rating_data = json.loads(json_str)
                if isinstance(rating_data, dict) and "rating" in rating_data:
                    score = int(rating_data["rating"])
                    if 1 <= score <= 5:
                        return float(score)

            return None

        except Exception:
            return None

    def _parse_research_ratings(self, response: str) -> Dict[str, Optional[int]]:
        """Parse research evaluation ratings."""
        try:
            content = response.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                ratings = json.loads(json_str)
                return {k: int(v) for k, v in ratings.items()}
        except Exception:
            pass

        return {"innovation": None, "safety": None, "feasibility": None}

    def _parse_bargaining_ratings(self, response: str) -> Dict[str, Optional[int]]:
        """Parse bargaining evaluation ratings."""
        try:
            content = response.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                ratings = json.loads(json_str)
                return {
                    "effectiveness_of_strategies": int(ratings["effectiveness_of_strategies"])
                    if "effectiveness_of_strategies" in ratings
                    else None,
                    "progress_and_outcome": int(ratings["progress_and_outcome"]) if "progress_and_outcome" in ratings else None,
                    "interaction_dynamics": int(ratings["interaction_dynamics"]) if "interaction_dynamics" in ratings else None,
                }
        except Exception:
            pass

        return {
            "effectiveness_of_strategies": None,
            "progress_and_outcome": None,
            "interaction_dynamics": None,
        }

    def _parse_coding_ratings(self, response: str) -> Dict[str, Optional[int]]:
        """Parse coding evaluation ratings."""
        try:
            content = response.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                ratings = json.loads(json_str)
                return {
                    "instruction_following": int(ratings["instruction_following"]) if "instruction_following" in ratings else None,
                    "executability": int(ratings["executability"]) if "executability" in ratings else None,
                    "consistency": int(ratings["consistency"]) if "consistency" in ratings else None,
                    "quality": int(ratings["quality"]) if "quality" in ratings else None,
                }
        except Exception:
            pass

        return {
            "instruction_following": None,
            "executability": None,
            "consistency": None,
            "quality": None,
        }

    def _determine_completion(self, metrics: MultiAgentBenchMetrics) -> bool:
        """Determine if task was completed based on metrics.

        A task is considered completed if all required scores are present (not None)
        and positive (> 0).
        """
        eval_data = metrics.task_evaluation

        if not eval_data:
            return False

        def _all_scores_valid(scores: List[Any]) -> bool:
            """Check all scores are present and positive."""
            return all(s is not None and s > 0 for s in scores)

        if self.domain == "research":
            scores = [eval_data.get(k) for k in ["innovation", "safety", "feasibility"]]
            return _all_scores_valid(scores)

        elif self.domain in ("bargaining", "worldsimulation"):
            buyer = eval_data.get("buyer", {})
            seller = eval_data.get("seller", {})
            buyer_scores = [buyer.get(k) for k in ["effectiveness_of_strategies", "progress_and_outcome", "interaction_dynamics"]]
            seller_scores = [seller.get(k) for k in ["effectiveness_of_strategies", "progress_and_outcome", "interaction_dynamics"]]
            return _all_scores_valid(buyer_scores) and _all_scores_valid(seller_scores)

        elif self.domain == "coding":
            scores = [eval_data.get(k) for k in ["instruction_following", "executability", "consistency", "quality"]]
            return _all_scores_valid(scores)

        elif self.domain == "database":
            # Database completion is determined by comparing prediction to labels
            return bool(eval_data.get("predicted"))

        return False
