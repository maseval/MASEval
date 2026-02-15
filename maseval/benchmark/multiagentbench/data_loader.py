"""Data loading utilities for MultiAgentBench tasks.

This module provides functions for loading and configuring tasks from
MARBLE's MultiAgentBench JSONL files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import git

from maseval import Task

logger = logging.getLogger(__name__)

# MARBLE repository configuration
# Original: https://github.com/ulab-uiuc/MARBLE (where the work was done)
# Using fork: https://github.com/cemde/MARBLE (contains bug fixes)
MARBLE_REPO_URL = "https://github.com/cemde/MARBLE.git"
MARBLE_DEFAULT_COMMIT = None  # Unpinned while bug fixes are being added; will pin once stable

# Valid domain names
VALID_DOMAINS: FrozenSet[str] = frozenset(
    {
        "coding",
        "database",
        "minecraft",
        "research",
        "bargaining",
        "werewolf",
    }
)

# Domains requiring external infrastructure
INFRASTRUCTURE_DOMAINS: FrozenSet[str] = frozenset({"database", "minecraft"})


def _get_marble_dir() -> Path:
    """Get the default MARBLE installation directory.

    Returns:
        Path to marble/ directory relative to this module
    """
    return Path(__file__).parent / "marble"


def download_marble(
    target_dir: Optional[Path] = None,
    commit: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Clone MARBLE repository to the specified directory.

    Args:
        target_dir: Directory to clone into. Defaults to marble/ relative to this module.
        commit: Specific commit hash to checkout. Defaults to MARBLE_DEFAULT_COMMIT or latest.
        force: If True, remove existing directory and re-clone.

    Returns:
        Path to the cloned MARBLE directory

    Raises:
        RuntimeError: If git clone fails
        FileExistsError: If directory exists and force=False
    """
    if target_dir is None:
        target_dir = _get_marble_dir()

    target_dir = Path(target_dir)

    # Check if already exists
    if target_dir.exists():
        if not force:
            logger.info(f"MARBLE already exists at {target_dir}")
            return target_dir

        # Remove existing directory
        import shutil

        logger.info(f"Removing existing MARBLE directory: {target_dir}")
        shutil.rmtree(target_dir)

    # Clone repository
    logger.info(f"Cloning MARBLE from {MARBLE_REPO_URL} to {target_dir}")

    try:
        repo = git.Repo.clone_from(MARBLE_REPO_URL, str(target_dir))
    except (git.GitCommandError, git.exc.GitCommandNotFound) as e:
        raise RuntimeError(f"Failed to clone MARBLE: {e}") from e

    # Checkout specific commit if requested
    checkout_commit = commit or MARBLE_DEFAULT_COMMIT
    if checkout_commit:
        logger.info(f"Checking out commit: {checkout_commit}")
        try:
            repo.git.checkout(checkout_commit)
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to checkout commit {checkout_commit}: {e}") from e

    # Create __init__.py at clone root so Python can traverse it as a package.
    # The actual MARBLE Python package lives at marble/marble/ inside the clone,
    # but we need the clone root to be importable for relative imports to work.
    init_file = target_dir / "__init__.py"
    if target_dir.exists() and not init_file.exists():
        init_file.write_text("")

    logger.info(f"MARBLE successfully installed at {target_dir}")
    return target_dir


def ensure_marble_exists(auto_download: bool = True) -> Path:
    """Ensure MARBLE is available, optionally downloading it.

    This function checks if MARBLE is installed and optionally downloads it
    if not present.

    Args:
        auto_download: If True, automatically download MARBLE if not found.
            If False, raise an error if MARBLE is not found.

    Returns:
        Path to the MARBLE directory

    Raises:
        FileNotFoundError: If MARBLE is not found and auto_download=False

    Example:
        >>> marble_dir = ensure_marble_exists()
        >>> # MARBLE is now available at marble_dir
    """
    marble_dir = _get_marble_dir()

    # Check if MARBLE exists and has the expected structure
    if marble_dir.exists() and (marble_dir / "multiagentbench").exists():
        # Verify pinned commit if set
        if MARBLE_DEFAULT_COMMIT:
            try:
                repo = git.Repo(marble_dir)
                current_commit = repo.head.commit.hexsha
                if current_commit != MARBLE_DEFAULT_COMMIT:
                    logger.info(f"MARBLE at {current_commit[:12]} but pinned to {MARBLE_DEFAULT_COMMIT[:12]}, checking out...")
                    repo.git.checkout(MARBLE_DEFAULT_COMMIT)
            except (git.InvalidGitRepositoryError, git.GitCommandError):
                logger.warning("Could not verify MARBLE commit (not a git repo or checkout failed)")
        return marble_dir

    if not auto_download:
        raise FileNotFoundError(
            f"MARBLE not found at {marble_dir}.\n"
            "Run `ensure_marble_exists(auto_download=True)` to download automatically,\n"
            "or manually clone: git clone https://github.com/cemde/MARBLE.git marble"
        )

    return download_marble(marble_dir)


def _resolve_data_dir(data_dir: Optional[Path] = None) -> Path:
    """Resolve the MARBLE data directory.

    Searches for MARBLE data in the following order:
    1. Provided data_dir argument
    2. MARBLE_DATA_DIR environment variable
    3. marble/multiagentbench/ relative to this module
    4. Current working directory / marble/multiagentbench/

    Args:
        data_dir: Optional explicit data directory

    Returns:
        Path to the MARBLE multiagentbench data directory

    Raises:
        FileNotFoundError: If no valid data directory is found
    """
    if data_dir is not None:
        path = Path(data_dir)
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified data_dir does not exist: {data_dir}")

    # Check environment variable
    env_dir = os.environ.get("MARBLE_DATA_DIR")

    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    # Relative to this module (vendored MARBLE)
    candidates.append(Path(__file__).parent / "marble" / "multiagentbench")

    # Current working directory
    candidates.append(Path.cwd() / "marble" / "multiagentbench")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "MARBLE data directory not found. Either:\n"
        "1. Clone MARBLE to maseval/benchmark/multiagentbench/marble/\n"
        "2. Set MARBLE_DATA_DIR environment variable\n"
        "See multiagentbench/README.md for setup instructions."
    )


def _parse_task_entry(entry: Dict[str, Any], domain: str, idx: int) -> Task:
    """Parse a JSONL entry into a MASEval Task.

    Args:
        entry: Raw JSONL entry dict
        domain: Domain name
        idx: Entry index (for error messages)

    Returns:
        MASEval Task object

    Raises:
        ValueError: If required fields are missing
    """
    # Required fields - fail if missing
    REQUIRED_FIELDS = ["scenario", "task_id", "task", "agents", "relationships"]
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"Task entry {idx} missing required fields: {missing}\nEntry keys: {list(entry.keys())}")

    # Validate agent specifications
    for i, agent_spec in enumerate(entry["agents"]):
        if "agent_id" not in agent_spec:
            raise ValueError(f"Agent {i} in task {entry['task_id']} missing 'agent_id'\nAgent spec: {agent_spec}")

    # Extract task content
    task_content = entry["task"]
    if isinstance(task_content, dict):
        query = task_content.get("content", "")
        output_format = task_content.get("output_format", "")
    else:
        query = str(task_content)
        output_format = ""

    if not query:
        raise ValueError(f"Task {entry['task_id']} has empty query/content")

    # Extract environment config
    env_config = entry.get("environment", {})

    # Extract coordination mode
    coordinate_mode = entry.get("coordinate_mode", "")
    if not coordinate_mode:
        # Default based on domain if not specified
        coordinate_mode = "star"

    # Build Task object
    return Task(
        id=f"{domain}_{entry['task_id']}",
        query=query,
        environment_data={
            "scenario": entry["scenario"],
            "coordinate_mode": coordinate_mode,
            "relationships": entry["relationships"],
            "environment": env_config,
            "task": entry["task"],
            "agents": entry["agents"],
            "max_iterations": env_config.get("max_iterations") or 10,
            "engine_planner": entry.get("engine_planner", {}),
            "memory": entry.get("memory", {}),
            "output": entry.get("output", {}),
            # Store raw entry for MARBLE compatibility
            "raw_marble_config": entry,
        },
        evaluation_data={
            "metrics": entry.get("metrics", {}),
            "output_format": output_format,
        },
        metadata={
            "domain": domain,
            "task_id": entry["task_id"],
            "scenario": entry["scenario"],
        },
    )


def _load_werewolf_tasks(
    data_dir: Path,
    limit: Optional[int] = None,
) -> List[Task]:
    """Load werewolf tasks from MARBLE config files.

    Werewolf uses a config-based game engine rather than JSONL task data.
    This function finds werewolf config YAMLs in the MARBLE configs directory
    and constructs Task objects from them.

    Args:
        data_dir: Resolved MARBLE data directory (typically marble/multiagentbench/)
        limit: Maximum number of tasks to load (None for all)

    Returns:
        List of Task objects for werewolf games

    Raises:
        FileNotFoundError: If no werewolf configs found
    """
    # Navigate from data_dir (marble/multiagentbench/) to MARBLE root (marble/)
    marble_root = data_dir.parent

    # Search for werewolf config YAMLs
    configs_dir = marble_root / "marble" / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(
            f"MARBLE configs directory not found: {configs_dir}\n"
            "Ensure MARBLE is cloned to multiagentbench/marble/\n"
            "See multiagentbench/README.md for setup instructions."
        )

    config_paths = sorted(configs_dir.glob("**/werewolf_config*.yaml"))

    if not config_paths:
        raise FileNotFoundError(f"No werewolf config files found in {configs_dir}\nExpected files matching: **/werewolf_config*.yaml")

    tasks = []
    for idx, config_path in enumerate(config_paths):
        if limit is not None and idx >= limit:
            break

        # Parse the YAML config to extract game setup
        try:
            import yaml

            with config_path.open(encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except ImportError:
            # Fall back to basic parsing if PyYAML not available
            config = _parse_werewolf_config_basic(config_path)

        roles = config.get("roles", [])
        cooperation_mode = config.get("cooperation_mode", "cooperative")

        # Build agent specs from roles
        agents = []
        for role_idx, role in enumerate(roles):
            agents.append(
                {
                    "agent_id": f"player_{role_idx}",
                    "profile": f"Werewolf game player with role: {role}",
                    "type": "WerewolfAgent",
                    "role": role,
                }
            )

        task = Task(
            id=f"werewolf_{idx}",
            query="Play a Werewolf social deduction game",
            environment_data={
                "scenario": "werewolf",
                "coordinate_mode": cooperation_mode,
                "relationships": [],
                "environment": {
                    "type": "Werewolf",
                    "description": "Werewolf social deduction game",
                },
                "task": {
                    "content": "Play a Werewolf social deduction game",
                },
                "agents": agents,
                "max_iterations": 20,
                "engine_planner": {},
                "memory": {},
                "output": {},
                "werewolf_config_path": str(config_path),
                "raw_marble_config": config,
            },
            evaluation_data={
                "metrics": {},
                "output_format": "",
            },
            metadata={
                "domain": "werewolf",
                "task_id": idx,
                "scenario": "werewolf",
                "config_path": str(config_path),
            },
        )
        tasks.append(task)

    return tasks


def _parse_werewolf_config_basic(config_path: Path) -> Dict[str, Any]:
    """Parse a werewolf YAML config without PyYAML.

    Basic key-value and list parsing for werewolf configs.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Parsed config dictionary
    """
    config: Dict[str, Any] = {}
    current_key = ""
    current_list: List[str] = []

    with config_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("- "):
                # List item
                current_list.append(stripped[2:].strip())
                config[current_key] = current_list
            elif ":" in stripped:
                # Key-value pair
                key, _, value = stripped.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if value:
                    config[key] = value
                else:
                    # Start of a list or nested object
                    current_key = key
                    current_list = []

    return config


def load_tasks(
    domain: str,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Task]:
    """Load MultiAgentBench tasks for a domain.

    Most domains load from JSONL files. Werewolf uses config-based
    task loading since it has no JSONL data (it uses a game engine).

    Args:
        domain: Domain name (one of: coding, database, minecraft, research,
            bargaining, werewolf)
        data_dir: Optional path to MARBLE data directory
        limit: Maximum number of tasks to load (None for all)

    Returns:
        List of Task objects

    Raises:
        ValueError: If domain is invalid
        FileNotFoundError: If data files not found

    Example:
        >>> tasks = load_tasks("research", limit=5)
        >>> len(tasks)
        5
        >>> tasks[0].metadata["domain"]
        'research'
    """
    # Normalize and validate domain
    domain_lower = domain.lower()
    if domain_lower not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}")

    # Find data directory
    resolved_data_dir = _resolve_data_dir(data_dir)

    # Werewolf uses config-based loading (no JSONL data)
    if domain_lower == "werewolf":
        return _load_werewolf_tasks(resolved_data_dir, limit)

    jsonl_path = resolved_data_dir / domain_lower / f"{domain_lower}_main.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Task data not found: {jsonl_path}\n"
            f"Ensure MARBLE is cloned to multiagentbench/marble/\n"
            f"See multiagentbench/README.md for setup instructions."
        )

    tasks = []
    with jsonl_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break

            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            task = _parse_task_entry(entry, domain_lower, idx)
            tasks.append(task)

    return tasks


def configure_model_ids(
    tasks: List[Task],
    *,
    agent_model_id: str,
    evaluator_model_id: Optional[str] = None,
) -> List[Task]:
    """Configure model IDs for MARBLE agents and evaluator.

    Modifies tasks in-place to set the LLM model IDs used by agents
    and optionally the evaluator.

    Args:
        tasks: List of Tasks to configure
        agent_model_id: Model ID for all MARBLE agents (e.g., "gpt-4o")
        evaluator_model_id: Optional model ID for LLM-based evaluation

    Returns:
        The input tasks (modified in-place)

    Example:
        >>> tasks = load_tasks("research", limit=5)
        >>> configure_model_ids(tasks, agent_model_id="gpt-4o")
        >>> tasks[0].environment_data["llm"]
        'gpt-4o'
    """
    for task in tasks:
        # Set agent model
        task.environment_data["llm"] = agent_model_id

        # Set evaluator model if provided
        if evaluator_model_id:
            task.evaluation_data["model_id"] = evaluator_model_id
        else:
            # Default to agent model for evaluation
            task.evaluation_data["model_id"] = agent_model_id

    return tasks


def get_domain_info(domain: str) -> Dict[str, Any]:
    """Get information about a domain.

    Args:
        domain: Domain name

    Returns:
        Dict with domain information including:
        - requires_infrastructure: Whether external services needed
        - description: Brief domain description
        - coordination_mode: Default coordination mode

    Raises:
        ValueError: If domain is invalid
    """
    domain_lower = domain.lower()
    if domain_lower not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of: {sorted(VALID_DOMAINS)}")

    domain_info = {
        "coding": {
            "requires_infrastructure": False,
            "description": "Software development collaboration tasks",
            "coordination_mode": "tree",
        },
        "database": {
            "requires_infrastructure": True,
            "description": "Database manipulation and querying tasks (requires Docker)",
            "coordination_mode": "star",
        },
        "minecraft": {
            "requires_infrastructure": True,
            "description": "Collaborative building in Minecraft (untested; requires Minecraft Server 1.19.2, Node.js, npm)",
            "coordination_mode": "cooperative",
        },
        "research": {
            "requires_infrastructure": False,
            "description": "Research idea generation and collaboration",
            "coordination_mode": "cooperative",
        },
        "bargaining": {
            "requires_infrastructure": False,
            "description": "Negotiation and bargaining scenarios",
            "coordination_mode": "cooperative",
        },
        "werewolf": {
            "requires_infrastructure": False,
            "description": "Adversarial social deduction game with roles (wolf, villager, seer, witch, guard)",
            "coordination_mode": "cooperative",
        },
    }

    return domain_info[domain_lower]
