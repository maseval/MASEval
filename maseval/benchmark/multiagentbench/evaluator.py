"""MultiAgentBench evaluator implementation.

This module provides evaluation metrics matching MARBLE's evaluation methodology.

Original Repository: https://github.com/ulab-uiuc/MARBLE
Fork Used: https://github.com/cemde/MARBLE (contains bug fixes for MASEval integration)
Code License: MIT

Citation:
    Zhu, et al. (2025). MultiAgentBench: Evaluating the Collaboration and Competition
    of LLM agents. arXiv:2503.01935.
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
        communication_score: Single communication score (base evaluator, 1-5 or None)
        communication_scores: Per-iteration communication scores (MARBLE reproduction, 1-5 or -1)
        planning_scores: Per-iteration planning scores (MARBLE reproduction, 1-5 or -1)
        task_evaluation: Domain-specific evaluation results
        agent_kpis: Per-agent key performance indicators
        total_milestones: Number of milestones achieved
        code_quality: Code quality scores (coding domain only)
    """

    task_completion: bool = False
    token_consumption: int = 0
    communication_score: Optional[float] = None
    communication_scores: List[int] = field(default_factory=list)
    planning_scores: List[int] = field(default_factory=list)
    task_evaluation: Dict[str, Any] = field(default_factory=dict)
    agent_kpis: Dict[str, int] = field(default_factory=dict)
    total_milestones: int = 0
    code_quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_completion": self.task_completion,
            "token_consumption": self.token_consumption,
            "communication_score": self.communication_score,
            "communication_scores": self.communication_scores,
            "planning_scores": self.planning_scores,
            "task_evaluation": self.task_evaluation,
            "agent_kpis": self.agent_kpis,
            "total_milestones": self.total_milestones,
            "code_quality": self.code_quality,
        }


class MarbleReproductionEvaluator(Evaluator):
    """Thin evaluator for MARBLE reproduction mode.

    All LLM-based evaluation happens in the coordination loop via MARBLE's
    own ``Evaluator`` class (imported directly from the vendored package).
    This class only reformats the pre-computed metrics from
    ``final_answer["marble_evaluation"]`` into MASEval's evaluation result
    format. It makes NO LLM calls.
    """

    def __init__(self, domain: str):
        self.domain = domain.lower()

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through traces unchanged."""
        return traces

    def __call__(  # type: ignore[override]
        self,
        traces: Dict[str, Any],
        final_answer: Any = None,
    ) -> Dict[str, Any]:
        """Reformat pre-computed MARBLE metrics into MASEval result format.

        Args:
            traces: Execution traces (not used — evaluation is pre-computed)
            final_answer: Dict from coordination loop containing
                ``marble_evaluation`` key with MARBLE's ``evaluator.metrics``
                dict.

        Returns:
            Evaluation result dictionary with ``passed``, ``metrics``,
            ``marble_raw_metrics``, and ``domain`` keys.
        """
        if not isinstance(final_answer, dict) or "marble_evaluation" not in final_answer:
            raise ValueError(
                f"MarbleReproductionEvaluator expects final_answer to be a dict with "
                f"'marble_evaluation' key, got: {type(final_answer).__name__}"
            )
        marble_eval: Dict[str, Any] = final_answer["marble_evaluation"]

        # Direct dict access — MARBLE's Evaluator.__init__ (evaluator.py:31-39)
        # always initializes all keys. Using [] instead of .get() ensures we
        # fail loudly if a key is unexpectedly missing.
        metrics = MultiAgentBenchMetrics(
            # MARBLE's metrics["task_completion"] is always [] (evaluator.update()
            # is never called in any coordination mode). bool([]) == False.
            task_completion=bool(marble_eval["task_completion"]),
            communication_scores=marble_eval["communication_score"],
            planning_scores=marble_eval["planning_score"],
            task_evaluation=marble_eval["task_evaluation"],
            agent_kpis=marble_eval["agent_kpis"],
            total_milestones=marble_eval["total_milestones"],
            code_quality=marble_eval["code_quality"],
        )

        return {
            "passed": metrics.task_completion,
            "metrics": metrics.to_dict(),
            "marble_raw_metrics": marble_eval,
            "domain": self.domain,
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
        """Load evaluation prompts from MARBLE's evaluator_prompts.json.

        Communication, research, and bargaining prompts are loaded from the
        vendored ``evaluator_prompts.json`` to match MARBLE's exact prompts.
        Coding, werewolf, and minecraft use local templates (MARBLE handles
        these differently — coding prompt is hardcoded in evaluator.py:538-576,
        werewolf/minecraft are not in the JSON).
        """
        # Load MARBLE's canonical prompts
        prompts_path = Path(__file__).parent / "marble" / "marble" / "evaluator" / "evaluator_prompts.json"
        with open(prompts_path, "r", encoding="utf-8") as f:
            marble_prompts = json.load(f)

        return {
            "communication": {
                "prompt": marble_prompts["Graph"]["Communication"]["prompt"],
            },
            "research": {
                "task_evaluation": {
                    "prompt": marble_prompts["research"]["task_evaluation"]["prompt"],
                }
            },
            "bargaining": {
                "task_evaluation": {
                    # MARBLE evaluator.py:210 has buyer_prompt commented out (code
                    # regression). The paper requires both prompts to compute
                    # TS = (buyer_avg + seller_avg) / 2 * 20. Each prompt evaluates
                    # a single role and returns {"role": {metrics}}.
                    "seller_prompt": marble_prompts["world"]["task_evaluation"]["seller_prompt"],
                    "buyer_prompt": marble_prompts["world"]["task_evaluation"]["buyer_prompt"],
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

        Reads the ``communication_log`` key from each agent's trace dict.
        This key is populated by ``MarbleAgentAdapter`` (which captures it
        from ``BaseAgent.act()``'s return value) but is **not** automatically
        populated by other agent adapters. When ``communication_log`` is
        absent, ``communication_score`` will be ``None`` and this method
        returns ``"No communications recorded."``.

        To enable communication evaluation with custom agents, ensure each
        adapter's ``gather_traces()`` returns a ``"communication_log"`` key
        with entries of the form ``{"communication": str}``.

        Args:
            traces: Execution traces.

        Returns:
            Formatted communication string.
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
            # Extract root_causes from environment state if available.
            # MARBLE engine.py:473-479 passes self.config.task["root_causes"].
            env_state = traces.get("environment", {}).get("state", {})
            task_config = env_state.get("task_config", {})
            root_causes = task_config.get("root_causes") if isinstance(task_config, dict) else None
            metrics.task_evaluation = self._evaluate_database(task_desc, final_result, root_causes=root_causes)
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
        """Evaluate communication quality using LLM.

        Generation params match MARBLE evaluator.py:80-88:
        temperature=0.0, max_token_num=512.
        """
        prompt_template = self._evaluation_prompts["communication"]["prompt"]
        # Use .replace() instead of .format() because MARBLE's Communication
        # prompt contains literal {"rating": X} (single braces, unlike research/
        # bargaining which use {{ double braces). This is a bug in MARBLE's
        # evaluator_prompts.json that causes .format() to raise KeyError.
        prompt = prompt_template.replace("{task}", task).replace("{communications}", communications)

        # Params from evaluator.py:80-88
        response = self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 512},
        )
        return self._parse_score(response)

    def _evaluate_research(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate research task output.

        Generation params match MARBLE evaluator.py:183-191:
        temperature=0.0, max_token_num=512.
        """
        prompt_template = self._evaluation_prompts["research"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        # Params from evaluator.py:183-191
        response = self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 512},
        )
        return self._parse_research_ratings(response)

    def _evaluate_bargaining(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate bargaining task output using separate buyer and seller prompts.

        The MARBLE paper computes Task Score from both buyer and seller evaluations:
        ``TS = (buyer_avg + seller_avg) / 2 * 20``. Each prompt evaluates a single
        role and returns ``{"role": {"metric": score}}``.

        MARBLE evaluator.py:210 has the buyer_prompt commented out (code regression).
        This restores both calls to match the paper's published methodology.

        Generation params match evaluator.py:214-220: temperature=0.0,
        max_token_num=512.
        """
        gen_params = {"temperature": 0.0, "max_tokens": 512}

        seller_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["seller_prompt"]
        seller_response = self.model_adapter.generate(
            seller_prompt.format(task=task, result=result),
            generation_params=gen_params,
        )
        seller_ratings = self._parse_role_ratings(seller_response, "seller")

        buyer_prompt = self._evaluation_prompts["bargaining"]["task_evaluation"]["buyer_prompt"]
        buyer_response = self.model_adapter.generate(
            buyer_prompt.format(task=task, result=result),
            generation_params=gen_params,
        )
        buyer_ratings = self._parse_role_ratings(buyer_response, "buyer")

        # Compute scores matching MARBLE paper methodology:
        # Per-role score = mean of 3 metrics (1-5 scale) * 20 -> 0-100
        buyer_score = sum(buyer_ratings.values()) / len(buyer_ratings) * 20
        seller_score = sum(seller_ratings.values()) / len(seller_ratings) * 20

        return {
            "buyer": buyer_ratings,
            "seller": seller_ratings,
            "buyer_score": buyer_score,
            "seller_score": seller_score,
            "mean_score": (buyer_score + seller_score) / 2,
        }

    def _parse_role_ratings(self, response: str, role: str) -> Dict[str, int]:
        """Parse bargaining ratings for a single role from LLM response.

        Each bargaining prompt (buyer or seller) returns a JSON object with
        a single role key containing three metric scores on a 1-5 scale.

        Args:
            response: LLM response containing JSON like ``{"role": {"metric": score}}``.
            role: Expected role key (``"buyer"`` or ``"seller"``).

        Returns:
            Dict with ``effectiveness_of_strategies``, ``progress_and_outcome``,
            ``interaction_dynamics`` as int scores.

        Raises:
            ValueError: If the response cannot be parsed or the role key is missing.
        """
        content = response.strip()
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            raise ValueError(f"No JSON object found in evaluator response: {response!r}")

        json_str = content[json_start:json_end]
        ratings = json.loads(json_str)

        # The response may be {"role": {metrics}} or just {metrics}
        if role in ratings:
            role_data = ratings[role]
        elif "effectiveness_of_strategies" in ratings:
            role_data = ratings
        else:
            raise ValueError(f"Expected '{role}' key or metric keys in response, got: {ratings!r}")

        return {
            "effectiveness_of_strategies": int(role_data["effectiveness_of_strategies"]),
            "progress_and_outcome": int(role_data["progress_and_outcome"]),
            "interaction_dynamics": int(role_data["interaction_dynamics"]),
        }

    def _evaluate_coding(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate coding task output.

        Generation params match MARBLE evaluator.py:586-594:
        temperature=0.0, max_token_num=4096.
        """
        prompt_template = self._evaluation_prompts["coding"]["task_evaluation"]["prompt"]

        # For coding, we need requirements and solution separately
        # If not available, use task as description and result as solution
        prompt = prompt_template.format(
            task_description=task,
            requirements="See task description",
            solution=result,
        )

        # Params from evaluator.py:586-594 (coding uses 4096 tokens)
        response = self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 4096},
        )
        return self._parse_coding_ratings(response)

    def _evaluate_database(self, task: str, result: str, root_causes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate database task output.

        Matches MARBLE evaluator.py:284-300. Database evaluation stores the
        predicted result alongside ground-truth root causes for offline
        comparison (no LLM call).

        Args:
            task: Task description
            result: Predicted root cause analysis
            root_causes: Ground truth root cause labels from task config
                (``task.environment_data["task"]["root_causes"]``)
        """
        # Matching evaluator.py:297-300
        return {
            "root_cause": root_causes if root_causes is not None else [],
            "predicted": result,
        }

    def _evaluate_werewolf(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate werewolf game output."""
        prompt_template = self._evaluation_prompts["werewolf"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        response = self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 512},
        )
        return self._parse_werewolf_ratings(response)

    def _evaluate_minecraft(self, task: str, result: str) -> Dict[str, Any]:
        """Evaluate minecraft building task output.

        Warning:
            Minecraft evaluation is untested. It requires a running Minecraft
            Server (1.19.2) and Node.js/npm for Mineflayer bot dependencies.
        """
        prompt_template = self._evaluation_prompts["minecraft"]["task_evaluation"]["prompt"]
        prompt = prompt_template.format(task=task, result=result)

        response = self.model_adapter.generate(
            prompt,
            generation_params={"temperature": 0.0, "max_tokens": 512},
        )
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
