"""Data loading utilities for the CONVERSE benchmark."""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from maseval import Task, TaskProtocol, TaskQueue

REPO_BASE_URL = "https://raw.githubusercontent.com/amrgomaaelhady/ConVerse/main/resources"
LOCAL_DATA_DIR = Path(__file__).parent / "data"
DOMAIN_MAP: Dict[str, str] = {
    "travel": "travel_planning_usecase",
    "real_estate": "real_estate_usecase",
    "insurance": "insurance_usecase",
}
PERSONAS: List[int] = [1, 2, 3, 4]


def download_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    """Download a file from URL if it does not already exist."""
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=timeout) as response:
            dest_path.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download CONVERSE data from {url}: {exc}") from exc


def ensure_data_exists(
    domain: Literal["travel", "real_estate", "insurance"],
    data_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """Ensure local CONVERSE data exists for the selected domain."""
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
    return "_".join(value.strip().lower().split())


def _infer_forbidden_tools(attack_action: str, responsibility_flag: str) -> List[str]:
    if responsibility_flag == "EXTERNAL_AGENT_OPTIONS":
        return []

    action = attack_action.lower()
    tool_map = {
        "delete_calendar_event": ["calendar", "delete", "schedule"],
        "send_email": ["email", "forward", "send"],
        "update_insurance_policy": ["insurance", "coverage", "policy"],
        "create_financial_product": ["loan", "credit", "bank", "financial", "account"],
    }

    selected_tools: List[str] = []
    for tool_name, keywords in tool_map.items():
        if any(keyword in action for keyword in keywords):
            selected_tools.append(tool_name)

    return selected_tools


def parse_privacy_attacks(file_path: Path, persona_id: int, env_text: str, options_text: str, domain: str) -> List[Task]:
    """Parse privacy attacks from official CONVERSE JSON structure."""
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
                    evaluation_data={
                        "type": "privacy",
                        "target_info": data_item,
                        "ground_truth_snippets": item.get("extraction_snippets", []),
                    },
                    protocol=TaskProtocol(timeout_seconds=600),
                )
            )

    return tasks


def parse_security_attacks(file_path: Path, persona_id: int, env_text: str, options_text: str, domain: str) -> List[Task]:
    """Parse security attacks from old and new CONVERSE JSON structures."""
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
        forbidden_tools = _infer_forbidden_tools(attack_action, responsibility_flag)

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
                evaluation_data={
                    "type": "security",
                    "forbidden_tools": forbidden_tools,
                    "forbidden_action_text": attack_action,
                    "responsibility_flag": responsibility_flag,
                },
                protocol=TaskProtocol(timeout_seconds=600),
            )
        )

    return tasks


def load_tasks(
    domain: Literal["travel", "real_estate", "insurance"],
    split: Literal["privacy", "security", "all"] = "all",
    limit: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> TaskQueue:
    """Load CONVERSE tasks for a domain and split."""
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
