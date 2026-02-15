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

# Default per-agent result truncation length (in characters) before LLM summarization.
# Matches MARBLE's _summarize_results() in engine.py which truncates each formatted
# result line to 1000 characters. Set to None to disable truncation and LLM summarization.
DEFAULT_RESULT_TRUNCATION_LENGTH = 1000


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
        result_truncation_length: Optional[int] = DEFAULT_RESULT_TRUNCATION_LENGTH,
    ):
        """Initialize the evaluator.

        Args:
            domain: Benchmark domain (research, bargaining, etc.)
            model_adapter: Model adapter for LLM evaluation
            metrics_config: Configuration for evaluation metrics
            output_format: Expected output format for task evaluation
            result_truncation_length: Maximum characters per agent result before LLM
                summarization. Matches MARBLE's ``_summarize_results()`` which truncates
                each result to 1000 chars, then passes the truncated output through an
                LLM summarization call (``planner.summarize_output()``). Set to ``None``
                to disable both truncation and LLM summarization, passing raw agent
                results directly to the evaluator (not recommended for domains with
                large outputs like research).
        """
        self.domain = domain.lower()
        self.model_adapter = model_adapter
        self.metrics_config = metrics_config or {}
        self.output_format = output_format
        self.result_truncation_length = result_truncation_length
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
            "werewolf": {
                "task_evaluation": {
                    "prompt": self._load_template("werewolf.txt"),
                }
            },
            "minecraft": {
                "task_evaluation": {
                    "prompt": self._load_template("minecraft.txt"),
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

        # MARBLE-compatible result summarization: truncate per-agent results then
        # pass through an LLM summarization call before domain evaluation. This
        # prevents token explosions for domains with large outputs (e.g. research).
        # See MARBLE's Engine._summarize_results() and EnginePlanner.summarize_output().
        agent_results = self._extract_agent_results(final_answer)
        if self.result_truncation_length is not None and agent_results:
            truncated = self._summarize_results(agent_results)
            final_result = self._summarize_output(truncated, task_desc, self.output_format)
        else:
            final_result = self._format_final_answer(final_answer)

        if self.domain == "research":
            metrics.task_evaluation = self._evaluate_research(task_desc, final_result)
        elif self.domain == "bargaining":
            metrics.task_evaluation = self._evaluate_bargaining(task_desc, final_result)
        elif self.domain == "coding":
            metrics.task_evaluation = self._evaluate_coding(task_desc, final_result)
        elif self.domain == "database":
            metrics.task_evaluation = self._evaluate_database(task_desc, final_result)
        elif self.domain == "werewolf":
            metrics.task_evaluation = self._evaluate_werewolf(task_desc, final_result)
        elif self.domain == "minecraft":
            metrics.task_evaluation = self._evaluate_minecraft(task_desc, final_result)
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

    def _extract_agent_results(self, final_answer: Any) -> List[Dict[str, Any]]:
        """Extract the agent_results list from final_answer, if present.

        Args:
            final_answer: Final output from agents (dict, list, str, or None).

        Returns:
            List of agent result dicts, or empty list if not extractable.
        """
        if isinstance(final_answer, dict):
            results = final_answer.get("agent_results", [])
            if results and isinstance(results, list):
                return results
        elif isinstance(final_answer, list):
            return [r for r in final_answer if isinstance(r, dict)]
        return []

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

    def _summarize_results(self, agent_results: List[Dict[str, Any]]) -> str:
        """Truncate and concatenate agent results for LLM summarization.

        Matches MARBLE's ``Engine._summarize_results()`` (engine.py). Each agent's
        result is formatted as ``"- {result}"`` and truncated to
        ``self.result_truncation_length`` characters before concatenation.

        Args:
            agent_results: List of dicts with ``agent_id`` and ``result`` keys,
                as returned in ``final_answer["agent_results"]``.

        Returns:
            Concatenated summary string with truncated per-agent results.
        """
        summary = "Agents' Results Summary:\n"
        for entry in agent_results:
            result_str = str(entry.get("result", ""))
            line = f"- {result_str}"
            assert self.result_truncation_length is not None
            summary += f"{line[: self.result_truncation_length]}\n"
        return summary

    def _summarize_output(self, summary: str, task: str, output_format: str) -> str:
        """Summarize truncated agent results via LLM call.

        Matches MARBLE's ``EnginePlanner.summarize_output()`` (engine_planner.py).
        Uses temperature=0.0 and max_tokens=2048 for deterministic, compact output.

        Note:
            The prompt text preserves MARBLE's original wording (including the
            ``"thr"`` typo) to maintain reproduction fidelity with the reference
            implementation.

        Args:
            summary: Truncated results string from ``_summarize_results()``.
            task: Task description string.
            output_format: Expected JSON output format specification.

        Returns:
            LLM-generated summary string.
        """
        prompt = (
            f"Summarize the output of the agents for the task: {task}\n\n"
            f"Now here is some result of thr agent: {summary}, please analyze it. "
            f"Return the final output into a json following the format: {output_format}"
        )
        return self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 2048},
        )

    def _calculate_token_consumption(self, traces: Dict[str, Any]) -> int:
        """Calculate total token consumption."""
        total = 0

        agent_traces = traces.get("agents", {})
        for agent_trace in agent_traces.values():
            token_usage = agent_trace.get("token_usage", 0)
            if isinstance(token_usage, int):
                total += token_usage

        return total

    def _evaluate_communication(self, task: str, communications: str) -> float:
        """Evaluate communication quality using LLM."""
        prompt_template = self._evaluation_prompts["communication"]["prompt"]
        prompt = prompt_template.format(task=task, communications=communications)

        response = self.model_adapter.generate(prompt)
        return self._parse_score(response)

    def _evaluate_research(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate research task output."""
        prompt_template = self._evaluation_prompts["research"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        response = self.model_adapter.generate(prompt)
        return self._parse_research_ratings(response)

    def _evaluate_bargaining(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate bargaining/world simulation task output."""
        # Evaluate both buyer and seller perspectives
        buyer_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["buyer_prompt"]
        seller_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["seller_prompt"]

        buyer_response = self.model_adapter.generate(buyer_prompt.format(task=task, result=result))
        seller_response = self.model_adapter.generate(seller_prompt.format(task=task, result=result))

        return {
            "buyer": self._parse_bargaining_ratings(buyer_response),
            "seller": self._parse_bargaining_ratings(seller_response),
        }

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

        response = self.model_adapter.generate(prompt)
        return self._parse_coding_ratings(response)

    def _evaluate_database(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate database task output.

        Database tasks have ground truth labels that would be compared
        separately. Here we just store the prediction.
        """
        return {
            "predicted": result,
            "root_cause": [],  # Would be filled from task data
        }

    def _evaluate_werewolf(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate werewolf game output."""
        prompt_template = self._evaluation_prompts["werewolf"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        response = self.model_adapter.generate(prompt)
        return self._parse_werewolf_ratings(response)

    def _evaluate_minecraft(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate minecraft building task output.

        Warning:
            Minecraft evaluation is untested. It requires a running Minecraft
            Server (1.19.2) and Node.js/npm for Mineflayer bot dependencies.
        """
        prompt_template = self._evaluation_prompts["minecraft"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        response = self.model_adapter.generate(prompt)
        return self._parse_minecraft_ratings(response)

    def _parse_score(self, response: str) -> float:
        """Parse a single score from LLM response.

        Returns:
            Score as float (1-5)

        Raises:
            ValueError: If the response cannot be parsed into a valid score
        """
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

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        rating_data = json.loads(json_str)

        if not isinstance(rating_data, dict) or "rating" not in rating_data:
            raise ValueError(f"Expected {{'rating': ...}} in evaluator response, got: {rating_data!r}")

        score = int(rating_data["rating"])
        if not 1 <= score <= 5:
            raise ValueError(f"Score {score} out of valid range 1-5 in evaluator response")

        return float(score)

    def _parse_research_ratings(self, response: str) -> Dict[str, int]:
        """Parse research evaluation ratings.

        Raises:
            ValueError: If the response cannot be parsed into valid ratings
        """
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)
        return {k: int(v) for k, v in ratings.items()}

    def _parse_bargaining_ratings(self, response: str) -> Dict[str, int]:
        """Parse bargaining evaluation ratings.

        Raises:
            ValueError: If the response cannot be parsed into valid ratings
        """
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)
        return {
            "effectiveness_of_strategies": int(ratings["effectiveness_of_strategies"]),
            "progress_and_outcome": int(ratings["progress_and_outcome"]),
            "interaction_dynamics": int(ratings["interaction_dynamics"]),
        }

    def _parse_coding_ratings(self, response: str) -> Dict[str, int]:
        """Parse coding evaluation ratings.

        Raises:
            ValueError: If the response cannot be parsed into valid ratings
        """
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)
        return {
            "instruction_following": int(ratings["instruction_following"]),
            "executability": int(ratings["executability"]),
            "consistency": int(ratings["consistency"]),
            "quality": int(ratings["quality"]),
        }

    def _parse_werewolf_ratings(self, response: str) -> Dict[str, int]:
        """Parse werewolf evaluation ratings.

        Raises:
            ValueError: If the response cannot be parsed into valid ratings
        """
        keys = [
            "game_outcome",
            "deception_detection",
            "voting_strategy",
            "role_fulfillment",
            "information_usage",
            "collaboration",
            "survival_rate",
        ]
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)
        return {k: int(ratings[k]) for k in keys}

    def _parse_minecraft_ratings(self, response: str) -> Dict[str, int]:
        """Parse minecraft evaluation ratings.

        Raises:
            ValueError: If the response cannot be parsed into valid ratings
        """
        keys = [
            "structural_completeness",
            "blueprint_accuracy",
            "coordination",
            "efficiency",
        ]
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)
        return {k: int(ratings[k]) for k in keys}

    def _determine_completion(self, metrics: MultiAgentBenchMetrics) -> bool:
        """Determine if task was completed based on metrics.

        For LLM-evaluated domains, scores are always positive ints if we reach this
        point (parse failures raise before getting here), so this just checks > 0.
        """
        eval_data = metrics.task_evaluation

        if not eval_data:
            return False

        # Parse methods guarantee int scores or raise, so None checks are
        # prob dead code — remove later if no edge cases surface.
        def _all_scores_positive(scores: List[Any]) -> bool:
            return all(s is not None and s > 0 for s in scores)

        if self.domain == "research":
            return _all_scores_positive([eval_data[k] for k in ["innovation", "safety", "feasibility"]])

        elif self.domain == "bargaining":
            buyer = eval_data["buyer"]
            seller = eval_data["seller"]
            return _all_scores_positive(
                [buyer[k] for k in ["effectiveness_of_strategies", "progress_and_outcome", "interaction_dynamics"]]
            ) and _all_scores_positive([seller[k] for k in ["effectiveness_of_strategies", "progress_and_outcome", "interaction_dynamics"]])

        elif self.domain == "coding":
            return _all_scores_positive([eval_data[k] for k in ["instruction_following", "executability", "consistency", "quality"]])

        elif self.domain == "database":
            return bool(eval_data.get("predicted"))

        elif self.domain == "werewolf":
            return _all_scores_positive(
                [
                    eval_data[k]
                    for k in [
                        "game_outcome",
                        "deception_detection",
                        "voting_strategy",
                        "role_fulfillment",
                        "information_usage",
                        "collaboration",
                        "survival_rate",
                    ]
                ]
            )

        elif self.domain == "minecraft":
            return _all_scores_positive([eval_data[k] for k in ["structural_completeness", "blueprint_accuracy", "coordination", "efficiency"]])

        return False
