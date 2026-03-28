"""MMLU Benchmark - Multiple Choice Question Answering Evaluation.

Implements MMLU evaluation compatible with lm-evaluation-harness output format,
with anchor point-based task selection for DISCO prediction.

Reference: Based on disco-public evaluation methodology.
Dataset: MMLU (Massive Multitask Language Understanding)

Usage:
    from maseval.benchmark.mmlu import (
        DefaultMMLUBenchmark, load_tasks,
    )
    from maseval.core.task import DISCOQueue

    # Load tasks (optionally filtered to anchor points)
    tasks = load_tasks(
        data_path="/path/to/mmlu_prompts_examples.json",
        anchor_points_path="/path/to/anchor_points.pkl",
    )

    # Run with the HuggingFace concrete implementation
    benchmark = DefaultMMLUBenchmark(model_id="meta-llama/Llama-2-7b-hf")
    results = benchmark.run(tasks=tasks, agent_data={"model_id": "meta-llama/Llama-2-7b-hf"})
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

from maseval.core.agent import AgentAdapter
from maseval.core.benchmark import Benchmark
from maseval.core.environment import Environment
from maseval.core.evaluator import Evaluator
from maseval.core.history import MessageHistory
from maseval.core.model import ModelAdapter
from maseval.core.seeding import SeedGenerator
from maseval.core.task import DISCOQueue, SequentialTaskQueue, Task
from maseval.core.user import User


# =============================================================================
# Constants (configurable defaults)
# =============================================================================

DEFAULT_CHOICES = ["A", "B", "C", "D"]
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 8
DEFAULT_AGENT_NAME = "mmlu_agent"
DEFAULT_MODEL_REGISTER_NAME = "mmlu_model"
TARGET_DELIMITER = " "  # lm-eval convention for MCQ
MMLU_TASK_NAME = "mmlu_prompts"
TASK_TYPE_MMLU = "mmlu"
STATUS_SUCCESS = "success"


# =============================================================================
# Agent adapter for scorer-based evaluation
# =============================================================================


class _ScorerBackedAdapter(AgentAdapter):
    """Agent adapter for benchmarks that use scorer-based evaluation.

    This adapter is a message container for tracing — the benchmark's
    ``run_agents()`` drives evaluation via a ``ModelScorer`` and records
    results here.  Calling ``agent.run()`` directly is an error because
    there is no generation model behind this adapter.
    """

    def __init__(self, scorer: Any, name: str) -> None:
        super().__init__(agent_instance=scorer, name=name)
        self._messages: List[Dict[str, Any]] = []

    def record_message(self, message: Dict[str, Any]) -> None:
        """Record a message for tracing purposes."""
        self._messages.append(message)

    def _run_agent(self, query: str) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} is backed by a ModelScorer, not a generation model. "
            "Use benchmark.run_agents() instead of calling agent.run() directly."
        )

    def get_messages(self) -> MessageHistory:
        return MessageHistory(self._messages)


# =============================================================================
# Environment
# =============================================================================


class MMLUEnvironment(Environment):
    """Simple environment for MMLU multiple choice evaluation.

    MMLU tasks don't require tools - the environment just holds
    the task context (question, choices, etc.).
    """

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state from task data.

        Args:
            task_data: Must contain ``"query"`` (str) and ``"environment_data"``
                (dict with ``"choices"``, ``"full_prompt"``, ``"use_full_prompt"``).
        """
        env_data = task_data["environment_data"]
        use_full_prompt = env_data["use_full_prompt"]
        if use_full_prompt and "full_prompt" not in env_data:
            raise ValueError(
                "use_full_prompt=True but 'full_prompt' is missing from environment_data. "
                "Ensure the dataset includes few-shot prompts or set use_full_prompt=False."
            )
        state: Dict[str, Any] = {
            "query": task_data["query"],
            "choices": env_data["choices"],
            "use_full_prompt": use_full_prompt,
        }
        if "full_prompt" in env_data:
            state["full_prompt"] = env_data["full_prompt"]
        return state

    def create_tools(self) -> Dict[str, Any]:
        """MMLU doesn't use tools."""
        return {}

    def get_prompt(self) -> str:
        """Get the prompt to send to the model.

        Returns ``full_prompt`` if ``use_full_prompt`` is True, otherwise ``query``.
        """
        if self.state["use_full_prompt"]:
            return self.state["full_prompt"]
        return self.state["query"]


# =============================================================================
# Evaluator
# =============================================================================


class MMLUEvaluator(Evaluator):
    """Evaluator for MMLU multiple choice questions.

    Computes accuracy metrics (acc and acc_norm) by comparing model predictions
    with gold answers.
    """

    def __init__(
        self,
        task: Task,
        environment: Environment,
        user: Optional[Any] = None,
    ):
        """Initialize MMLU evaluator.

        Args:
            task: Task being evaluated. Must have ``evaluation_data["gold"]`` (int)
                with the correct answer index.
            environment: Environment (provides choices).
            user: Unused for MMLU.
        """
        self.task = task
        self.environment = environment
        self.gold = task.evaluation_data["gold"]
        self.choices = task.environment_data["choices"]

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for evaluation.

        For MMLU, we need the model's response from agent traces.
        """
        # Get agent traces
        agents = traces.get("agents", {})
        if agents:
            # Get first agent's messages
            first_agent = next(iter(agents.values()), {})
            messages = first_agent.get("messages", [])
            return {"messages": messages}
        return {"messages": []}

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the model's response.

        Args:
            traces: Filtered traces with messages.
            final_answer: The model's final answer.

        Returns:
            Dict with acc, acc_norm, predicted, gold, correct, parse_failed, and optionally logprobs fields.
        """
        # Parse the model's answer
        predicted = self._parse_answer(final_answer or "")
        parse_failed = predicted is None

        # Check if correct — unparseable responses are never correct
        correct = (not parse_failed) and predicted == self.gold

        result = {
            "acc": 1.0 if correct else 0.0,
            "acc_norm": 1.0 if correct else 0.0,
            "predicted": predicted,
            "gold": self.gold,
            "correct": correct,
            "parse_failed": parse_failed,
            "doc_id": self.task.metadata["doc_id"],
        }

        # Extract logprobs from traces if available (for logprobs-based evaluation)
        messages = traces["messages"]
        for msg in messages:
            if isinstance(msg, dict) and "logprobs" in msg:
                result["logprobs"] = msg["logprobs"]
                break

        return result

    def _parse_answer(self, response: str) -> Optional[int]:
        """Parse model response to extract answer choice.

        Handles various formats:
        - Direct letter: "A", "B", "C", "D"
        - With period: "A."
        - As part of sentence: "The answer is A"

        Args:
            response: Model's response string.

        Returns:
            Index of the predicted choice (0-3), or None if unparseable.
        """
        if not response:
            return None

        response = response.strip().upper()

        # Direct letter match
        for i, choice in enumerate(DEFAULT_CHOICES):
            if response == choice or response.startswith(f"{choice}."):
                return i

        # Look for "answer is X" pattern
        for i, choice in enumerate(DEFAULT_CHOICES):
            if f"ANSWER IS {choice}" in response:
                return i
            if f"ANSWER: {choice}" in response:
                return i

        # Last character check
        last_char = response.rstrip(".")[-1] if response else ""
        for i, choice in enumerate(DEFAULT_CHOICES):
            if last_char == choice:
                return i

        return None


# =============================================================================
# Benchmark
# =============================================================================


class MMLUBenchmark(Benchmark):
    """MMLU Benchmark - Framework-agnostic base class.

    Evaluates language models on MMLU multiple choice questions.
    Supports anchor point-based evaluation for DISCO prediction.

    Subclasses must implement:

    - ``setup_agents()`` - create agents for MCQ evaluation
    - ``get_model_adapter()`` - provide model adapters

    For a ready-to-use implementation, see ``DefaultMMLUBenchmark``.
    """

    def __init__(
        self,
        use_full_prompt: bool = False,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            use_full_prompt: If True, use full_prompt (with few-shot examples)
                instead of just the query.
            callbacks: Benchmark callbacks.
            n_task_repeats: Repetitions per task.
        """
        super().__init__(callbacks=callbacks, n_task_repeats=n_task_repeats, max_invocations=1, **kwargs)
        self.use_full_prompt = use_full_prompt

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> MMLUEnvironment:
        """Create environment for a task."""
        task_data = {
            "query": task.query,
            "environment_data": {
                **task.environment_data,
                "use_full_prompt": self.use_full_prompt,
            },
        }
        return MMLUEnvironment(task_data)

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        """Create MMLU evaluator."""
        return [MMLUEvaluator(task, environment)]

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Any:
        """Execute agent on the MMLU prompt."""
        mmlu_env = cast(MMLUEnvironment, environment)
        prompt = mmlu_env.get_prompt()

        # Run the agent
        agent = agents[0]
        return agent.run(prompt)

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate model response."""
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results


class DefaultMMLUBenchmark(MMLUBenchmark):
    """MMLU Benchmark using HuggingFace transformers models.

    This concrete implementation uses log-likelihood based MCQ evaluation
    via ``HuggingFaceModelScorer``, with the same optimisations as
    lm-evaluation-harness:

    1. Single forward pass per question (one-token continuation optimisation)
    2. Efficient log-softmax computation
    3. Proper left-padding for batch processing

    Agents are created using a scorer-backed adapter (see ``_ScorerBackedAdapter``).
    """

    def __init__(
        self,
        model_id: str,
        device: str = DEFAULT_DEVICE,
        trust_remote_code: bool = True,
        use_full_prompt: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs: Any,
    ):
        """Initialize HuggingFace MMLU benchmark.

        Args:
            model_id: HuggingFace model identifier.
            device: Device to run model on.
            trust_remote_code: Trust remote code when loading model (default True).
            use_full_prompt: Use full prompt with few-shot examples (default True).
            batch_size: Batch size for lm-eval batching (number of questions per batch).
            **kwargs: Additional arguments passed to ``MMLUBenchmark``.
        """
        super().__init__(use_full_prompt=use_full_prompt, **kwargs)
        self._model_id = model_id
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._batch_size = batch_size
        self._precomputed_logprobs: Optional[Dict[Any, List[float]]] = None

        from maseval.interface.inference.huggingface_scorer import HuggingFaceModelScorer

        self._scorer = HuggingFaceModelScorer(
            model_id=model_id,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create scorer-backed agent for MCQ evaluation.

        The returned adapter is a tracing container — actual evaluation is
        driven by ``self._scorer`` in ``run_agents()``.

        Args:
            agent_data: Agent config (unused; model is set at ``__init__``).
            environment: MMLU environment.
            task: Current task.
            user: Unused.
            seed_generator: Seed generator (unused for MMLU).

        Returns:
            Tuple of (agents_to_run, agents_dict).
        """
        adapter = _ScorerBackedAdapter(self._scorer, DEFAULT_AGENT_NAME)
        return [adapter], {DEFAULT_AGENT_NAME: adapter}

    def precompute_all_logprobs_lmeval(self, tasks: Sequence[Task]) -> Dict[Any, List[float]]:
        """Precompute log-likelihoods for ALL tasks using lm-eval's batching.

        CRITICAL: lm-evaluation-harness batches ALL requests together and uses
        its Collator class to reorder/group them. This affects floating-point
        precision for some edge cases. To get EXACT matches, we must process
        ALL requests together in a single batch.

        This method:
        1. Creates Instance objects for all task/choice combinations
        2. Calls lm-eval's HFLM.loglikelihood() with ALL instances
        3. Returns a mapping from doc_id to logprobs

        Args:
            tasks: Iterable of Task objects with prompt and choices.

        Returns:
            Dict mapping doc_id -> list of log-likelihoods for each choice.
        """
        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance

        # Create HFLM model (this handles model loading internally)
        print(f"Loading HFLM model for {self._model_id}")
        lm = HFLM(
            pretrained=self._model_id,
            trust_remote_code=self._trust_remote_code,
            batch_size=self._batch_size,
            device=self._device,
        )

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = TARGET_DELIMITER
        choices = DEFAULT_CHOICES
        continuations = [f"{target_delimiter}{c}" for c in choices]

        # Build ALL instances like lm-eval task system does
        all_instances = []
        instance_map = {}  # (doc_id, choice_idx) -> position in results

        for task in tasks:
            doc_id = task.metadata["doc_id"]
            # Get prompt from task - use full_prompt from environment_data if available
            if self.use_full_prompt and "full_prompt" in task.environment_data:
                prompt = task.environment_data["full_prompt"]
            else:
                prompt = task.query

            for i, cont in enumerate(continuations):
                inst = Instance(
                    request_type="loglikelihood",
                    doc={"doc_id": doc_id},
                    arguments=(prompt, cont),
                    idx=i,
                    metadata=(MMLU_TASK_NAME, doc_id, 1),
                )
                instance_map[(doc_id, i)] = len(all_instances)
                all_instances.append(inst)

        print(f"Precomputing logprobs for {len(all_instances)} instances ({len(all_instances) // len(choices)} tasks)")

        # Call loglikelihood with ALL instances at once - this is the key!
        results = lm.loglikelihood(all_instances)

        # Map results back to doc_ids
        doc_logprobs = {}
        for task in tasks:
            doc_id = task.metadata["doc_id"]
            logprobs = []
            for i in range(len(choices)):
                pos = instance_map[(doc_id, i)]
                logprob, _ = results[pos]
                logprobs.append(logprob)
            doc_logprobs[doc_id] = logprobs

        # Store for later use
        self._precomputed_logprobs = doc_logprobs

        return doc_logprobs

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str = "",
    ) -> Any:
        """Execute log-likelihood based MCQ evaluation.

        Uses precomputed logprobs if available (for exact lm-eval match),
        otherwise delegates to ``HuggingFaceModelScorer.loglikelihood_choices()``
        which automatically picks single-token or multi-token scoring.
        """
        mmlu_env = cast(MMLUEnvironment, environment)
        prompt = mmlu_env.get_prompt()
        choices = mmlu_env.state["choices"]
        doc_id = task.metadata["doc_id"]
        agent = cast(_ScorerBackedAdapter, agents[0])

        if self._precomputed_logprobs is not None and doc_id in self._precomputed_logprobs:
            logprobs = self._precomputed_logprobs[doc_id]
        else:
            logprobs = self._scorer.loglikelihood_choices(prompt, choices, delimiter=TARGET_DELIMITER)

        best_idx = logprobs.index(max(logprobs))
        answer = choices[best_idx]
        mmlu_env.state["logprobs"] = logprobs
        mmlu_env.state["predicted_idx"] = best_idx

        agent.record_message({"role": "user", "content": prompt})
        agent.record_message({"role": "assistant", "content": answer, "logprobs": logprobs})
        return answer

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Not used — ``DefaultMMLUBenchmark`` uses ``HuggingFaceModelScorer``.

        Raises:
            NotImplementedError: Always. Use ``HuggingFaceModelScorer`` via
                ``self._scorer`` for log-likelihood evaluation.
        """
        raise NotImplementedError(
            "DefaultMMLUBenchmark uses HuggingFaceModelScorer for log-likelihood "
            "evaluation, not a generation ModelAdapter. Access the scorer via self._scorer."
        )


# =============================================================================
# Data Loading
# =============================================================================


def load_tasks(
    data_path: Union[str, Path],
    anchor_points_path: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
) -> Union[DISCOQueue, SequentialTaskQueue]:
    """Load MMLU tasks from JSON file.

    Args:
        data_path: Path to MMLU prompts JSON file (mmlu_prompts_examples.json format).
        anchor_points_path: Optional path to anchor points pickle file.
            If provided, returns an DISCOQueue that evaluates
            only the anchor tasks in order.
        limit: Optional limit on number of tasks to load.

    Returns:
        TaskQueue containing MMLU tasks.

    """
    data_path = Path(data_path)

    # Load JSON data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Apply limit before filtering
    if limit is not None:
        data = data[:limit]

    # Convert to Tasks
    tasks = []
    for i, item in enumerate(data):
        query = item.get("query") or item.get("example")
        if query is None:
            raise ValueError(f"MMLU task at index {i} has neither 'query' nor 'example' field")

        if "gold" not in item:
            raise ValueError(f"MMLU task at index {i} missing required 'gold' field (correct answer index)")

        if "choices" not in item:
            raise ValueError(f"MMLU task at index {i} missing required 'choices' field")

        task = Task(
            query=query,
            id=f"mmlu_{i}",
            environment_data={
                "choices": item["choices"],
                **({"full_prompt": item["full_prompt"]} if "full_prompt" in item else {}),
                **({"example": item["example"]} if "example" in item else {}),
            },
            evaluation_data={
                "gold": item["gold"],
            },
            metadata={
                "doc_id": i,
                "task_type": TASK_TYPE_MMLU,
            },
        )
        tasks.append(task)

    if anchor_points_path is not None:
        return DISCOQueue(tasks, anchor_points_path=anchor_points_path)
    else:
        return SequentialTaskQueue(tasks)


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    Args:
        results: List of result dicts from benchmark.run().

    Returns:
        Dict with accuracy metrics and task counts.
    """
    if not results:
        return {
            "total_tasks": 0,
            "acc": 0.0,
            "acc_norm": 0.0,
        }

    total_tasks = len(results)
    correct_count = 0
    parse_failed_count = 0
    acc_sum = 0.0
    acc_norm_sum = 0.0

    for res in results:
        if res["status"] != STATUS_SUCCESS:
            continue

        evals = res["eval"] if res["eval"] is not None else []
        for entry in evals:
            acc_sum += entry["acc"]
            acc_norm_sum += entry["acc_norm"]
            if entry["correct"]:
                correct_count += 1
            if entry.get("parse_failed", False):
                parse_failed_count += 1

    return {
        "total_tasks": total_tasks,
        "correct_count": correct_count,
        "parse_failed_count": parse_failed_count,
        "acc": acc_sum / total_tasks if total_tasks > 0 else 0.0,
        "acc_norm": acc_norm_sum / total_tasks if total_tasks > 0 else 0.0,
    }
