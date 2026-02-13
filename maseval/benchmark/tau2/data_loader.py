"""Data loading utilities for Tau 2 benchmark.

This module provides functions to:
1. Download domain data from tau2-bench GitHub repository
2. Load tasks, database state, and policies for each domain
3. Configure model IDs for benchmark components

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

No side effects on import. Data download/processing must be explicitly called.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from maseval import Task, TaskQueue
from maseval.core.task import TaskProtocol


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
VALID_DOMAINS: Tuple[str, ...] = ("retail", "airline", "telecom")
TASK_SPLITS: Tuple[str, ...] = ("base", "hard", "all")

# Task counts per domain in v0.2.0 "base" split
BASE_SPLIT_COUNTS = {
    "airline": 50,
    "retail": 114,
    "telecom": 114,
}

# Default execution protocol settings
DEFAULT_TIMEOUT_SECONDS = 600.0  # 10 minutes per task
DEFAULT_MAX_RETRIES = 1  # Skip on first failure

# GitHub raw content URLs for v0.2.0 tag
GITHUB_BASE = "https://raw.githubusercontent.com/sierra-research/tau2-bench"
DEFAULT_VERSION = "v0.2.0"

URLS = {
    "retail": {
        "db": "{base}/{version}/data/tau2/domains/retail/db.json",
        "tasks": "{base}/{version}/data/tau2/domains/retail/tasks.json",
        "policy": "{base}/{version}/data/tau2/domains/retail/policy.md",
    },
    "airline": {
        "db": "{base}/{version}/data/tau2/domains/airline/db.json",
        "tasks": "{base}/{version}/data/tau2/domains/airline/tasks.json",
        "policy": "{base}/{version}/data/tau2/domains/airline/policy.md",
    },
    "telecom": {
        "db": "{base}/{version}/data/tau2/domains/telecom/db.toml",
        "tasks": "{base}/{version}/data/tau2/domains/telecom/tasks.json",
        "policy": "{base}/{version}/data/tau2/domains/telecom/main_policy.md",
    },
}


# =============================================================================
# Download Functions
# =============================================================================


def download_file(url: str, timeout: int = 30) -> str:
    """Download a file from URL and return as text.

    Args:
        url: URL to download from
        timeout: Request timeout in seconds

    Returns:
        File content as string

    Raises:
        RuntimeError: If download fails
    """
    try:
        with urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def download_json(url: str) -> object:
    """Download and parse JSON from URL.

    Args:
        url: URL to download from

    Returns:
        Parsed JSON object

    Raises:
        RuntimeError: If download or parsing fails
    """
    text = download_file(url)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from {url}: {e}") from e


def download_domain_data(
    data_dir: Optional[Path] = None,
    domain: Optional[str] = None,
    version: str = DEFAULT_VERSION,
    verbose: int = 1,
) -> Path:
    """Download domain data from tau2-bench GitHub repository.

    Downloads tasks.json, db.json/db.toml, and policy.md for specified domain(s).

    Args:
        data_dir: Base data directory (default: module's data/)
        domain: Specific domain to download, or None for all domains
        version: Git tag/ref to download from (default: v0.2.0)
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the data directory

    Raises:
        ValueError: If domain is invalid
        RuntimeError: If download fails

    Example:
        >>> download_domain_data(domain="retail")
        PosixPath('.../maseval/benchmark/tau2/data')
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    domains = [domain] if domain else list(VALID_DOMAINS)

    for d in domains:
        if d not in URLS:
            raise ValueError(f"Unknown domain: {d}. Must be one of {VALID_DOMAINS}")

        domain_dir = data_dir / d
        domain_dir.mkdir(parents=True, exist_ok=True)

        for name, url_template in URLS[d].items():
            url = url_template.format(base=GITHUB_BASE, version=version)

            if verbose >= 2:
                print(f"Downloading {url}...")

            content = download_file(url)

            # Determine output filename
            if name == "db":
                ext = ".toml" if d == "telecom" else ".json"
                out_path = domain_dir / f"db{ext}"
            elif name == "tasks":
                out_path = domain_dir / "tasks.json"
            elif name == "policy":
                out_path = domain_dir / "policy.md"
            else:
                out_path = domain_dir / f"{name}.txt"

            out_path.write_text(content, encoding="utf-8")

            if verbose >= 2:
                print(f"  -> {out_path}")

    if verbose >= 1:
        print(f"Downloaded tau2 domain data to {data_dir}")

    return data_dir


def ensure_data_exists(
    data_dir: Optional[Path] = None,
    domain: Optional[str] = None,
    force_download: bool = False,
    verbose: int = 1,
) -> Path:
    """Ensure domain data exists, downloading if needed.

    Args:
        data_dir: Base data directory (default: module's data/)
        domain: Specific domain to check/download, or None for all
        force_download: If True, re-download even if data exists
        verbose: 0=silent, 1=summary, 2=detailed

    Returns:
        Path to the data directory

    Example:
        >>> ensure_data_exists(domain="retail")
        PosixPath('.../maseval/benchmark/tau2/data')
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    domains = [domain] if domain else list(VALID_DOMAINS)

    # Check if required files exist for all domains
    all_exist = True
    for d in domains:
        domain_dir = data_dir / d
        db_ext = ".toml" if d == "telecom" else ".json"
        required_files = [
            domain_dir / f"db{db_ext}",
            domain_dir / "tasks.json",
            domain_dir / "policy.md",
        ]
        if not all(f.exists() for f in required_files):
            all_exist = False
            break

    if all_exist and not force_download:
        if verbose >= 1:
            print(f"Data already exists at {data_dir}")
        return data_dir

    return download_domain_data(data_dir, domain, verbose=verbose)


# =============================================================================
# Task Loading Functions
# =============================================================================


def _get_split_indices(domain: str, split: str, total_tasks: int) -> List[int]:
    """Get task indices for a given split.

    The "base" split matches the v0.2.0 task counts.
    The "hard" split would be the remaining tasks (if any).
    The "all" split includes all tasks.

    Args:
        domain: Domain name
        split: Split name ("base", "hard", "all")
        total_tasks: Total number of tasks available

    Returns:
        List of task indices to include
    """
    if split == "all":
        return list(range(total_tasks))

    base_count = BASE_SPLIT_COUNTS.get(domain, total_tasks)

    if split == "base":
        return list(range(min(base_count, total_tasks)))
    elif split == "hard":
        return list(range(base_count, total_tasks))
    else:
        raise ValueError(f"Unknown split: {split}. Must be one of {TASK_SPLITS}")


def load_tasks(
    domain: str,
    split: str = "base",
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> TaskQueue:
    """Load tasks for a tau2 domain.

    Args:
        domain: One of "airline", "retail", "telecom"
        split: One of "base", "hard", "all" (base recommended for reproducibility)
        data_dir: Base data directory (default: module's data/)
        limit: Maximum number of tasks to load
        timeout_seconds: Maximum execution time per task in seconds. Default 600 (10 minutes).
            Set to None to disable timeout.
        max_retries: Maximum retry attempts for transient failures. Default 1 (skip on failure).

    Returns:
        TaskQueue containing Task objects with:
            - id: Task identifier from tau2 data
            - query: Initial user message (from user_scenario)
            - environment_data: Domain tools, database state, policies
            - evaluation_data: Assertions, expected outcomes
            - user_data: User profile, instructions
            - metadata: domain, split, description
            - protocol: Execution settings (timeout, retries, tags)

    Raises:
        ValueError: If domain or split is invalid
        FileNotFoundError: If tasks.json doesn't exist

    Example:
        >>> tasks = load_tasks("retail", split="base", limit=5)
        >>> len(tasks)
        5

        >>> # Custom timeout and retries
        >>> tasks = load_tasks("retail", timeout_seconds=300, max_retries=2)
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {VALID_DOMAINS}")

    if split not in TASK_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {TASK_SPLITS}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    tasks_path = data_dir / domain / "tasks.json"

    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}. Run ensure_data_exists(domain='{domain}') first.")

    with tasks_path.open() as f:
        raw_tasks: List[Dict[str, Any]] = json.load(f)

    # Apply split filtering
    indices = _get_split_indices(domain, split, len(raw_tasks))
    raw_tasks = [raw_tasks[i] for i in indices]

    # Apply limit
    if limit:
        raw_tasks = raw_tasks[:limit]

    # Load domain config once and embed in each task
    domain_config = load_domain_config(domain, data_dir)

    # Convert to MASEval Task objects
    tasks = []
    for raw_task in raw_tasks:
        task = _convert_tau2_task_to_maseval(raw_task, domain, split, domain_config, timeout_seconds, max_retries)
        tasks.append(task)

    return TaskQueue(tasks)


def _convert_tau2_task_to_maseval(
    raw_task: Dict[str, Any],
    domain: str,
    split: str,
    domain_config: Dict[str, Any],
    timeout_seconds: Optional[float],
    max_retries: int,
) -> Task:
    """Convert a tau2-bench task dict to MASEval Task.

    Args:
        raw_task: Raw task dict from tasks.json
        domain: Domain name
        split: Split name
        domain_config: Domain configuration with policy and db_path
        timeout_seconds: Maximum execution time per task in seconds
        max_retries: Maximum retry attempts for transient failures

    Returns:
        MASEval Task object
    """
    # Extract initial query from user_scenario
    user_scenario = raw_task.get("user_scenario", {})
    instructions = user_scenario.get("instructions", {})

    if isinstance(instructions, str):
        query = instructions
    elif isinstance(instructions, dict):
        # Structured instructions - build query from task_instructions
        query = instructions.get("task_instructions", "")
    else:
        query = ""

    # Build environment_data with embedded domain config
    environment_data: Dict[str, Any] = {
        "domain": domain,
        "initial_state": raw_task.get("initial_state"),
        "policy": domain_config["policy"],
        "db_path": str(domain_config["db_path"]),
    }

    # Build evaluation_data from evaluation_criteria
    eval_criteria = raw_task.get("evaluation_criteria", {})
    evaluation_data: Dict[str, Any] = {
        "actions": eval_criteria.get("actions"),
        "env_assertions": eval_criteria.get("env_assertions"),
        "communicate_info": eval_criteria.get("communicate_info"),
        "nl_assertions": eval_criteria.get("nl_assertions"),
        "reward_basis": eval_criteria.get("reward_basis", ["DB", "COMMUNICATE"]),
    }

    # Build user_data from user_scenario
    user_data: Dict[str, Any] = {
        "persona": user_scenario.get("persona"),
        "instructions": instructions,
    }

    # Build metadata
    metadata: Dict[str, Any] = {
        "domain": domain,
        "split": split,
        "description": raw_task.get("description"),
        "ticket": raw_task.get("ticket"),  # For solo mode (not used)
    }

    # Build execution protocol with timeout, retries, and tags
    protocol = TaskProtocol(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        tags={"domain": domain, "split": split},
    )

    # Build task kwargs, only include id if provided in raw task
    task_kwargs: Dict[str, Any] = {
        "query": query,
        "environment_data": environment_data,
        "evaluation_data": evaluation_data,
        "user_data": user_data,
        "metadata": metadata,
        "protocol": protocol,
    }
    if raw_task.get("id"):
        task_kwargs["id"] = str(raw_task["id"])

    return Task(**task_kwargs)


def load_domain_config(
    domain: str,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load domain configuration (policy, database schema).

    Args:
        domain: One of "airline", "retail", "telecom"
        data_dir: Base data directory (default: module's data/)

    Returns:
        Dict with:
            - policy: Markdown policy text
            - db_path: Path to database file

    Raises:
        ValueError: If domain is invalid
        FileNotFoundError: If required files don't exist
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {VALID_DOMAINS}")

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    domain_dir = data_dir / domain

    policy_path = domain_dir / "policy.md"
    db_ext = ".toml" if domain == "telecom" else ".json"
    db_path = domain_dir / f"db{db_ext}"

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    policy = policy_path.read_text(encoding="utf-8")

    return {
        "policy": policy,
        "db_path": db_path,
    }


# =============================================================================
# Model Configuration
# =============================================================================


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    user_model_id: Optional[str] = None,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure model IDs for benchmark components in task data.

    Tau2 tools execute real business logic and don't need a tool_model_id.
    Only user simulation and evaluation use LLMs.

    Args:
        tasks: TaskQueue or list of Tasks to configure
        user_model_id: Model ID for user simulator (stored in user_data)
        evaluator_model_id: Model ID for evaluators (stored in evaluation_data)

    Returns:
        The same collection (mutated in place for convenience)

    Example:
        >>> tasks = load_tasks("retail", limit=5)
        >>> configure_model_ids(
        ...     tasks,
        ...     user_model_id="gpt-4o",
        ...     evaluator_model_id="gpt-4o",
        ... )
    """
    for task in tasks:
        # User data: user model ID
        if user_model_id is not None:
            if "model_id" in task.user_data and task.user_data["model_id"] != user_model_id:
                raise ValueError(
                    f"Task {task.id} already has user `model_id` set to '{task.user_data['model_id']}', cannot override with '{user_model_id}'"
                )
            task.user_data["model_id"] = user_model_id

        # Evaluation data: evaluator model ID
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
    print("Downloading tau2 domain data...")
    download_domain_data(verbose=2)
    print("\nDone!")
