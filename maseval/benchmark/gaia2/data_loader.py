"""Data loading utilities for Gaia2 benchmark.

This module provides functions to:
1. Load Gaia2 scenarios from HuggingFace
2. Convert scenarios to MASEval Task objects
3. Configure model IDs for benchmark components

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

No side effects on import. Data download/processing must be explicitly called.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from maseval import Task, TaskQueue
from maseval.core.task import TaskProtocol


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

VALID_CAPABILITIES: Tuple[str, ...] = (
    "execution",
    "search",
    "adaptability",
    "time",
    "ambiguity",
    "agent2agent",
    "noise",
)

VALID_SPLITS: Tuple[str, ...] = ("validation",)  # Only validation has oracle events

DEFAULT_CONFIG = "validation"  # Full dataset
DEFAULT_TIMEOUT_SECONDS = 600.0  # 10 minutes per task
DEFAULT_MAX_RETRIES = 1

# HuggingFace dataset info
HF_DATASET_ID = "meta-agents-research-environments/gaia2"


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

    Args:
        capability: Filter by capability type (execution, search, adaptability,
            time, ambiguity, agent2agent, noise). None loads all.
        split: Dataset split (currently only "validation" available)
        limit: Maximum number of tasks to load
        timeout_seconds: Maximum execution time per task. Default 600 (10 minutes).
            Set to None to disable timeout.
        max_retries: Maximum retry attempts. Default 1 (skip on failure).

    Returns:
        TaskQueue with Task objects containing:
            - id: Unique scenario identifier
            - query: Initial task instructions
            - environment_data: {"scenario": BenchmarkScenario, "capability": str, ...}
            - evaluation_data: {"oracle_events": [...], "judge_type": str}
            - user_data: {}  # Gaia2 uses event-based simulation, not user turns
            - metadata: {"capability": str, "universe_id": str, ...}
            - protocol: TaskProtocol with timeout and tags

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
        from datasets import load_dataset
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

    # Determine HuggingFace config name
    config_name = capability if capability else DEFAULT_CONFIG

    # Load dataset from HuggingFace
    dataset = load_dataset(
        HF_DATASET_ID,
        name=config_name,
        split=split,
    )

    # Apply limit
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to MASEval Task objects
    importer = JsonScenarioImporter()
    tasks = []

    for row in dataset:
        # Parse scenario from JSON
        scenario, oracle_events, _ = importer.import_from_json_to_benchmark(json_str=row["data"])

        task = _convert_gaia2_to_maseval(
            row=row,
            scenario=scenario,
            oracle_events=oracle_events,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        tasks.append(task)

    return TaskQueue(tasks)


def _get_scenario_metadata(scenario: Any, key: str, default: Any = None) -> Any:
    """Safely get metadata from an ARE scenario object.

    Args:
        scenario: ARE BenchmarkScenario object
        key: Metadata key to retrieve
        default: Default value if key not found

    Returns:
        The metadata value or default
    """
    metadata = getattr(scenario, "metadata", None)
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    # Try attribute access as fallback
    return getattr(metadata, key, default)


def _convert_gaia2_to_maseval(
    row: Dict[str, Any],
    scenario: Any,
    oracle_events: List[Any],
    timeout_seconds: Optional[float],
    max_retries: int,
) -> Task:
    """Convert Gaia2 scenario to MASEval Task.

    Args:
        row: Raw row from HuggingFace dataset
        scenario: ARE BenchmarkScenario object
        oracle_events: List of oracle events for evaluation
        timeout_seconds: Maximum execution time per task
        max_retries: Maximum retry attempts

    Returns:
        MASEval Task object
    """
    # Extract query from scenario's task definition
    query = getattr(scenario, "task_instruction", "")

    # Parse capability from scenario metadata or row
    capability = row.get("category") or _get_scenario_metadata(scenario, "capability", "unknown")

    # Build environment_data
    environment_data: Dict[str, Any] = {
        "scenario": scenario,
        "capability": capability,
        "universe_id": _get_scenario_metadata(scenario, "universe_id"),
        "duration": getattr(scenario, "duration", 86400),
    }

    # Build evaluation_data with oracle events
    evaluation_data: Dict[str, Any] = {
        "oracle_events": oracle_events,
        "judge_type": _get_scenario_metadata(scenario, "judge_type", "graph_per_event"),
    }

    # Build metadata
    metadata: Dict[str, Any] = {
        "scenario_id": row.get("scenario_id") or row.get("id"),
        "capability": capability,
        "universe_id": environment_data.get("universe_id"),
    }

    # Build protocol
    protocol = TaskProtocol(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        tags={"capability": capability, "benchmark": "gaia2"},
    )

    # Build task kwargs
    task_kwargs: Dict[str, Any] = {
        "query": query,
        "environment_data": environment_data,
        "evaluation_data": evaluation_data,
        "user_data": {},  # Gaia2 uses event-based simulation
        "metadata": metadata,
        "protocol": protocol,
    }

    # Include id if provided
    task_id = row.get("id") or row.get("scenario_id")
    if task_id:
        task_kwargs["id"] = str(task_id)

    return Task(**task_kwargs)


# =============================================================================
# Model Configuration
# =============================================================================


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure model IDs for benchmark components in task data.

    Gaia2 uses ARE's deterministic judge by default, but can optionally
    use an LLM-based judge for complex assertions.

    Note: Unlike Tau2, Gaia2 doesn't have a user simulator (interactions
    happen through scheduled events), so there's no user_model_id.

    Args:
        tasks: TaskQueue or list of Tasks to configure
        evaluator_model_id: Optional model ID for LLM-based evaluation

    Returns:
        The same collection (mutated in place for convenience)

    Example:
        >>> tasks = load_tasks(capability="execution", limit=5)
        >>> configure_model_ids(
        ...     tasks,
        ...     evaluator_model_id="gpt-4o",  # Optional, for LLM-based judge
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
            print(f"  - {task.id}: {task.query[:50]}...")
    except ImportError as e:
        print(f"Error: {e}")
