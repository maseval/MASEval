"""Data loading utilities for the CONVERSE benchmark.

CONVERSE data files are fetched on demand from the upstream benchmark repository.
Raw dataset files are not bundled in the package.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from maseval import Task, TaskProtocol, TaskQueue

ConverseDomain = Literal["travel", "real_estate", "insurance"]

GITHUB_BASE = "https://raw.githubusercontent.com/amrgomaaelhady/ConVerse"
DEFAULT_VERSION = "main"
REPO_BASE_URL = f"{GITHUB_BASE}/{DEFAULT_VERSION}/resources"
LOCAL_DATA_DIR = Path(__file__).parent / "data"
DOMAIN_MAP: Dict[ConverseDomain, str] = {
    "travel": "travel_planning_usecase",
    "real_estate": "real_estate_usecase",
    "insurance": "insurance_usecase",
}
PERSONAS: List[int] = [1, 2, 3, 4]


def download_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download a file from *url* to *dest_path*, skipping if it already exists.

    Args:
        url: Remote URL to fetch.
        dest_path: Local filesystem destination.
        timeout: HTTP timeout in seconds.

    Raises:
        RuntimeError: If the download fails.
    """
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=timeout) as response:
            dest_path.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download CONVERSE data from {url}: {exc}") from exc


def ensure_data_exists(
    domain: ConverseDomain,
    data_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Ensure local CONVERSE data exists for the selected domain.

    Downloads benchmark files the first time they are needed.

    Args:
        domain: CONVERSE domain to load.
        data_dir: Optional override for the local data cache directory.
        force_download: Re-download files even if they already exist.

    Returns:
        Path to the local data root directory.
    """
    if domain not in DOMAIN_MAP:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {sorted(DOMAIN_MAP)}")

    data_root = Path(data_dir) if data_dir is not None else LOCAL_DATA_DIR
    use_case_dir = DOMAIN_MAP[domain]

    paths = [(f"{REPO_BASE_URL}/{use_case_dir}/options.txt", data_root / domain / "options.txt")]

    for persona_id in PERSONAS:
        paths.extend(
            [
                (
                    f"{REPO_BASE_URL}/{use_case_dir}/env_persona{persona_id}.txt",
                    data_root / domain / f"env_persona{persona_id}.txt",
                ),
                (
                    f"{REPO_BASE_URL}/{use_case_dir}/privacy_attacks/privacy_attacks_persona{persona_id}.json",
                    data_root / domain / "privacy" / f"attacks_p{persona_id}.json",
                ),
                (
                    f"{REPO_BASE_URL}/{use_case_dir}/security_attacks/security_attacks_persona{persona_id}.json",
                    data_root / domain / "security" / f"attacks_p{persona_id}.json",
                ),
            ]
        )

    for url, local_path in paths:
        if force_download and local_path.exists():
            local_path.unlink()
        download_file(url, local_path)

    return data_root


def _slugify(value: str) -> str:
    """Lowercase *value*, collapse whitespace, and join with underscores."""
    return "_".join(value.strip().lower().split())


def parse_privacy_attacks(file_path: Path, persona_id: int, env_text: str, options_text: str, domain: str) -> List[Task]:
    """Parse privacy attacks from the official CONVERSE JSON structure.

    Args:
        file_path: Path to a ``attacks_p<N>.json`` file in the privacy directory.
        persona_id: Persona number (1–4).
        env_text: Persona-specific environment description text.
        options_text: Shared domain options text.
        domain: Domain name (``"travel"``, ``"real_estate"``, or ``"insurance"``).

    Returns:
        List of :class:`Task` objects, one per privacy attack item.
    """
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tasks: List[Task] = []
    categories = data.get("categories", {})

    for category_name, category_data in categories.items():
        items = category_data.get("items", [])
        for item in items:
            data_item = str(item.get("data_item", "unknown"))
            task_id = f"{domain}_p{persona_id}_privacy_{_slugify(category_name)}_{_slugify(data_item)}"

            # Build evaluation_data with fields matching the original ConVerse
            # privacy_judge.py:199-221 routing logic.
            evaluation_data: Dict[str, Any] = {
                "type": "privacy",
                "category": category_name,
                "data_item": data_item,
                "attack_action": item.get("attack_action", ""),
            }

            if category_name == "related_and_useful":
                # Abstraction pathway: ConVerse/judge/privacy_judge.py:211-218
                evaluation_data["raw_data"] = item.get("raw_data", [])
                evaluation_data["acceptable_abstraction"] = item.get("acceptable_abstraction", "")
            else:
                # Information leak pathway: ConVerse/judge/privacy_judge.py:203-210
                evaluation_data["ground_truth_snippets"] = item.get("extraction_snippets", [])

            tasks.append(
                Task(
                    id=task_id,
                    query=item.get("user_task", "Please help me plan this task based on my profile."),
                    environment_data={
                        "persona_text": env_text,
                        "options_text": options_text,
                        "domain": domain,
                    },
                    user_data={
                        "persona": "Service Provider",
                        "attack_type": "privacy",
                        "attack_goal": f"Extract: {data_item}",
                        "attack_strategy": item.get("attack_action", ""),
                        "attack_rationale": item.get("attack_rationale", ""),
                    },
                    evaluation_data=evaluation_data,
                    protocol=TaskProtocol(timeout_seconds=600),
                )
            )

    return tasks


def parse_security_attacks(file_path: Path, persona_id: int, env_text: str, options_text: str, domain: str) -> List[Task]:
    """Parse security attacks from old and new CONVERSE JSON structures.

    Supports both the ``security_attacks`` format and the legacy
    ``toolkit_attacks`` / ``final_package_attacks`` format.

    Args:
        file_path: Path to a ``attacks_p<N>.json`` file in the security directory.
        persona_id: Persona number (1–4).
        env_text: Persona-specific environment description text.
        options_text: Shared domain options text.
        domain: Domain name (``"travel"``, ``"real_estate"``, or ``"insurance"``).

    Returns:
        List of :class:`Task` objects, one per security attack item.
    """
    if not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_attacks: List[Dict[str, Any]] = []

    if "security_attacks" in data:
        for category_name, category_attacks in data.get("security_attacks", {}).items():
            if not isinstance(category_attacks, dict):
                continue
            for attack_key, attack_data in category_attacks.items():
                if isinstance(attack_data, dict):
                    raw_attacks.append(
                        {
                            "category": category_name,
                            "attack_key": attack_key,
                            "payload": attack_data,
                        }
                    )
    else:
        merged = {
            **data.get("toolkit_attacks", {}),
            **data.get("final_package_attacks", {}),
        }
        for attack_key, attack_data in merged.items():
            if isinstance(attack_data, dict):
                raw_attacks.append(
                    {
                        "category": "legacy",
                        "attack_key": attack_key,
                        "payload": attack_data,
                    }
                )

    tasks: List[Task] = []
    for attack in raw_attacks:
        item = attack["payload"]
        attack_action = str(item.get("attack_action", ""))
        responsibility_flag = str(item.get("responsibility_flag", ""))

        task_id = f"{domain}_p{persona_id}_security_{_slugify(str(attack['category']))}_{_slugify(str(attack['attack_key']))}"
        tasks.append(
            Task(
                id=task_id,
                query=item.get("user_task", "Help me with my request."),
                environment_data={
                    "persona_text": env_text,
                    "options_text": options_text,
                    "domain": domain,
                },
                user_data={
                    "persona": "Service Provider",
                    "attack_type": "security",
                    "attack_goal": item.get("external_attack_goal", attack_action),
                    "attack_strategy": attack_action,
                    "attack_rationale": item.get("attack_rationale", ""),
                },
                # Store original ConVerse fields for LLM-based security evaluation.
                # ConVerse/judge/security_judge.py:124-167 routes by responsibility_flag.
                evaluation_data={
                    "type": "security",
                    "responsibility_flag": responsibility_flag,
                    "attack_action": attack_action,
                    "attack_rationale": item.get("attack_rationale", ""),
                    "legitimate_action": item.get("legitimate_action", ""),
                    "attack_name": item.get("name", attack["attack_key"]),
                },
                protocol=TaskProtocol(timeout_seconds=600),
            )
        )

    return tasks


def load_tasks(
    domain: ConverseDomain,
    split: Literal["privacy", "security", "all"] = "all",
    limit: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> TaskQueue:
    """Load CONVERSE tasks for a domain and attack split.

    Downloads benchmark data on first call via :func:`ensure_data_exists`.

    Args:
        domain: CONVERSE domain to load.
        split: Attack type filter — ``"privacy"``, ``"security"``, or ``"all"``.
        limit: Maximum number of tasks to return (``None`` for all).
        data_dir: Optional override for the local data cache directory.

    Returns:
        A :class:`TaskQueue` containing the loaded tasks.

    Raises:
        ValueError: If *domain* is not one of the supported domains.
    """
    if domain not in DOMAIN_MAP:
        raise ValueError(f"Invalid domain '{domain}'. Must be one of {sorted(DOMAIN_MAP)}")

    data_root = ensure_data_exists(domain=domain, data_dir=data_dir)

    with (data_root / domain / "options.txt").open("r", encoding="utf-8") as handle:
        options_text = handle.read()

    all_tasks: List[Task] = []

    for persona_id in PERSONAS:
        with (data_root / domain / f"env_persona{persona_id}.txt").open("r", encoding="utf-8") as handle:
            env_text = handle.read()

        if split in ["privacy", "all"]:
            privacy_path = data_root / domain / "privacy" / f"attacks_p{persona_id}.json"
            all_tasks.extend(parse_privacy_attacks(privacy_path, persona_id, env_text, options_text, domain))

        if split in ["security", "all"]:
            security_path = data_root / domain / "security" / f"attacks_p{persona_id}.json"
            all_tasks.extend(parse_security_attacks(security_path, persona_id, env_text, options_text, domain))

        if limit is not None and len(all_tasks) >= limit:
            break

    if limit is not None:
        all_tasks = all_tasks[:limit]

    return TaskQueue(all_tasks)


# Mapping from MASEval domain names to original ConVerse use_case keys
# used in USE_CASE_CONFIGS (ConVerse/judge/generic_prompts.py:246-306).
DOMAIN_TO_USE_CASE: Dict[str, str] = {
    "travel": "travel_planning",
    "real_estate": "real_estate",
    "insurance": "insurance",
}


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure the evaluator model ID for LLM-based evaluation.

    The original ConVerse uses LLM judges for both privacy evaluation
    (``ConVerse/judge/privacy_judge.py``) and security evaluation
    (``ConVerse/judge/security_judge.py``).  Setting *evaluator_model_id*
    enables LLM-based evaluation for both; without it, privacy evaluation
    falls back to substring matching and security evaluation is skipped.

    Only sets values if not already present, allowing task-specific
    overrides in the original data to take precedence.

    Args:
        tasks: TaskQueue or list of Tasks to configure.
        evaluator_model_id: Model ID for the LLM judge used by both
            :class:`PrivacyEvaluator` and :class:`SecurityEvaluator`.

    Returns:
        The same queue or list (mutated in place for convenience).

    Example:
        ```python
        tasks = load_tasks("travel", split="all", limit=5)
        configure_model_ids(tasks, evaluator_model_id="gpt-4o")
        benchmark = MyConverseBenchmark(agent_data=agent_config)
        results = benchmark.run(tasks)
        ```
    """
    for task in tasks:
        if evaluator_model_id is not None:
            task.evaluation_data.setdefault("evaluator_model_id", evaluator_model_id)
    return tasks
