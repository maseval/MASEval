"""Data loading utilities for Gaia2 benchmark.

This module provides functions to:
1. Load Gaia2 scenarios from HuggingFace
2. Convert scenarios to MASEval Task objects
3. Configure model IDs and judge engine for benchmark components

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

No side effects on import. Data download/processing must be explicitly called.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from maseval import Task, TaskQueue
from maseval.core.task import TaskProtocol


# =============================================================================
# Judge Engine Configuration
# =============================================================================


@dataclass
class Gaia2JudgeEngineConfig:
    """Configuration for the ARE judge's LLM engine used in semantic comparison.

    ARE's ``GraphPerEventJudge`` uses an LLM to semantically compare tool arguments
    (e.g., email content, calendar event descriptions) between agent actions and oracle
    (expected) actions. This config controls which model and provider the judge uses.

    Defaults match ARE's built-in defaults.

    ARE's ``LLMEngineConfig`` only supports ``model_name``, ``provider``, and
    ``endpoint``. Provider-specific parameters (e.g., OpenRouter's ``fallbacks``
    or ``route``) are not supported by ARE's engine pipeline.

    ARE ``validation/configs.py:28-29``

    Attributes:
        model_name: LLM model identifier for the judge engine.
        provider: LLM provider name (e.g., ``"huggingface"``, ``"openrouter"``, ``"openai"``).
            Passed to LiteLLM as ``custom_llm_provider``.
        endpoint: Optional custom API endpoint URL.

    Example::

        from maseval.benchmark.gaia2 import (
            load_tasks, configure_model_ids, Gaia2JudgeEngineConfig,
        )

        tasks = load_tasks(capability="execution", limit=5)

        # Use OpenRouter instead of HuggingFace for judge LLM
        configure_model_ids(
            tasks,
            judge_engine_config=Gaia2JudgeEngineConfig(
                provider="openrouter",
            ),
        )
    """

    # ARE validation/configs.py:28
    model_name: str = "meta-llama/Meta-Llama-3.3-70B-Instruct"
    # ARE validation/configs.py:29
    provider: str = "huggingface"
    endpoint: Optional[str] = None


# =============================================================================
# Constants
# =============================================================================

# HuggingFace config names that exist on the dataset.
# Each config corresponds to a capability and contains ~160 scenarios.
# ARE simulation/types.py: CapabilityTag enum values
VALID_CAPABILITIES: Tuple[str, ...] = (
    "execution",
    "search",
    "adaptability",
    "time",
    "ambiguity",
)

VALID_SPLITS: Tuple[str, ...] = ("validation",)  # Only validation has oracle events

# ARE scenarios/config.py:20: DEFAULT_SCENARIO_TIMEOUT = 1860
DEFAULT_TIMEOUT_SECONDS = 1860.0  # 31 minutes per task (matching ARE)
DEFAULT_MAX_RETRIES = 1

# HuggingFace dataset info
HF_DATASET_ID = "meta-agents-research-environments/gaia2"
HF_DATASET_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"


# =============================================================================
# Task Loading Functions
# =============================================================================


def load_tasks(
    capability: Optional[str] = None,
    split: str = "validation",
    limit: Optional[int] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> TaskQueue:
    """Load Gaia2 tasks from HuggingFace.

    Each HuggingFace config corresponds to a capability (execution, search,
    adaptability, time, ambiguity). When ``capability`` is None, all
    capabilities are loaded and combined.

    GAIA2 is event-driven: the task query is delivered to agents via the
    notification system at runtime (first ``send_message_to_agent`` event),
    not as a static field. ``task.query`` is left empty.

    Args:
        capability: Filter by capability type. None loads all capabilities.
        split: Dataset split (currently only "validation" available)
        limit: Maximum number of tasks to load (across all capabilities)
        timeout_seconds: Maximum execution time per task. Default 1860 (31 minutes,
            matching ARE's DEFAULT_SCENARIO_TIMEOUT). Set to None to disable timeout.
        max_retries: Maximum retry attempts. Default 1 (skip on failure).

    Returns:
        TaskQueue with Task objects.

    Raises:
        ValueError: If capability or split is invalid
        ImportError: If required dependencies are not installed

    Example:
        >>> tasks = load_tasks(capability="execution", limit=5)
        >>> len(tasks)
        5

        >>> # Load all capabilities
        >>> tasks = load_tasks(limit=10)
    """
    if capability is not None and capability not in VALID_CAPABILITIES:
        raise ValueError(f"Invalid capability '{capability}'. Must be one of {VALID_CAPABILITIES}")

    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {VALID_SPLITS}")

    # Import dependencies (optional)
    try:
        from datasets import Dataset, load_dataset
    except ImportError as e:
        raise ImportError("HuggingFace datasets library is required for loading Gaia2 tasks.\nInstall with: pip install datasets") from e

    try:
        from are.simulation.data_handler.importer import JsonScenarioImporter  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "ARE (Agent Research Environments) is required for Gaia2 benchmark.\n"
            "Install with: pip install meta-agents-research-environments\n"
            "Or: uv add --optional gaia2 meta-agents-research-environments"
        ) from e

    # When no capability specified, load all capabilities and combine
    capabilities = [capability] if capability else list(VALID_CAPABILITIES)

    importer = JsonScenarioImporter()
    tasks: List[Task] = []

    for cap in capabilities:
        # Each capability is a HuggingFace config name
        # Passing `split` guarantees the return type is Dataset (not DatasetDict)
        dataset = load_dataset(
            HF_DATASET_ID,
            name=cap,
            split=split,
            revision=HF_DATASET_REVISION,
        )
        assert isinstance(dataset, Dataset)

        for row in dataset:
            # Parse scenario from JSON
            # import_from_json_to_benchmark returns (scenario, completed_events, _)
            # completed_events are from previous runs, not oracle events.
            # Oracle events are generated at runtime by preprocess_scenario().
            scenario, _, _ = importer.import_from_json_to_benchmark(json_str=row["data"])

            task = _convert_gaia2_to_maseval(
                row=row,
                scenario=scenario,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                config_capability=cap,
            )
            tasks.append(task)

            if limit and len(tasks) >= limit:
                break

        if limit and len(tasks) >= limit:
            break

    return TaskQueue(tasks)


def _convert_gaia2_to_maseval(
    row: Dict[str, Any],
    scenario: Any,
    timeout_seconds: Optional[float],
    max_retries: int,
    config_capability: str,
) -> Task:
    """Convert Gaia2 scenario to MASEval Task.

    GAIA2 is event-driven: the task query is delivered via the notification
    system at runtime (first ``send_message_to_agent`` event). There is no
    static query field on ARE scenario objects.
    ARE agents/default_agent/are_simulation_main.py:79-102

    Oracle events are generated at runtime by ``preprocess_scenario()`` during
    environment setup, not at data-load time.

    Args:
        row: Raw row from HuggingFace dataset
        scenario: ARE BenchmarkScenario object
        timeout_seconds: Maximum execution time per task
        max_retries: Maximum retry attempts
        config_capability: The capability from the HuggingFace config name

    Returns:
        MASEval Task object
    """
    scenario_id = getattr(scenario, "scenario_id", None)

    # Build environment_data
    # Duration is NOT set here — ARE's preprocess_scenario() sets it during
    # environment setup based on capability (1800s standard, 420s for Time).
    # ARE scenarios/config.py:18-19, scenarios/scenario_imported_from_json/utils.py:69-76
    environment_data: Dict[str, Any] = {
        "scenario": scenario,
        "capability": config_capability,
    }

    # Evaluation uses scenario.judge.validate() at runtime (created by
    # preprocess_scenario). No static evaluation data needed at load time.
    # ARE scenarios/scenario_imported_from_json/utils.py:110-112
    evaluation_data: Dict[str, Any] = {
        "judge_type": "graph_per_event",
    }

    metadata: Dict[str, Any] = {
        "scenario_id": scenario_id or row.get("scenario_id") or row.get("id"),
        "capability": config_capability,
    }

    protocol = TaskProtocol(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        tags={"capability": config_capability, "benchmark": "gaia2"},
    )

    task_id = row.get("id") or row.get("scenario_id")
    if not task_id:
        raise ValueError("HuggingFace row missing both 'id' and 'scenario_id' fields")

    return Task(
        id=str(task_id),
        query="",  # Event-driven: real query comes from notification system at runtime
        environment_data=environment_data,
        evaluation_data=evaluation_data,
        user_data={},
        metadata=metadata,
        protocol=protocol,
    )


# =============================================================================
# Model Configuration
# =============================================================================


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    evaluator_model_id: Optional[str] = None,
    judge_engine_config: Optional[Gaia2JudgeEngineConfig] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure model IDs and judge engine for benchmark components.

    Gaia2's ``GraphPerEventJudge`` uses an LLM for semantic comparison of tool
    arguments (email content, calendar descriptions, etc.). By default it uses
    ARE's built-in defaults (``meta-llama/Meta-Llama-3.3-70B-Instruct`` via
    HuggingFace). Pass ``judge_engine_config`` to override the model/provider.

    Note: Unlike Tau2, Gaia2 doesn't have a user simulator (interactions
    happen through scheduled events), so there's no user_model_id.

    Args:
        tasks: TaskQueue or list of Tasks to configure.
        evaluator_model_id: Optional model ID for LLM-based evaluation.
        judge_engine_config: Optional judge engine configuration. Controls
            which LLM model and provider the ARE judge uses for semantic
            comparison. When ``None``, ARE's defaults are used.

    Returns:
        The same collection (mutated in place for convenience).

    Example::

        >>> tasks = load_tasks(capability="execution", limit=5)
        >>> configure_model_ids(
        ...     tasks,
        ...     judge_engine_config=Gaia2JudgeEngineConfig(
        ...         provider="openrouter",
        ...     ),
        ... )
    """
    for task in tasks:
        # Evaluation data: evaluator model ID (optional)
        if evaluator_model_id is not None:
            if "model_id" in task.evaluation_data and task.evaluation_data["model_id"] != evaluator_model_id:
                raise ValueError(
                    f"Task {task.id} already has evaluator `model_id` "
                    f"set to '{task.evaluation_data['model_id']}', cannot override with '{evaluator_model_id}'"
                )
            task.evaluation_data["model_id"] = evaluator_model_id

        # Evaluation data: judge engine configuration (optional)
        if judge_engine_config is not None:
            task.evaluation_data["judge_engine_config"] = judge_engine_config

    return tasks


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Loading Gaia2 tasks from HuggingFace...")
    try:
        tasks = load_tasks(limit=5)
        print(f"Loaded {len(tasks)} tasks")
        for task in tasks:
            print(f"  - {task.id} (capability={task.metadata.get('capability')})")
    except ImportError as e:
        print(f"Error: {e}")
