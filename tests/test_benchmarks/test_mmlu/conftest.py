"""Shared fixtures for MMLU benchmark tests."""

import json
from typing import Any, Dict, List, Optional

import pytest

from maseval.core.task import Task


def make_mmlu_task(
    query: str = "What is 2+2?",
    gold: int = 0,
    choices: Optional[List[str]] = None,
    doc_id: int = 0,
    full_prompt: str = "Few-shot examples...\n\nWhat is 2+2?",
) -> Task:
    """Create a Task with MMLU-shaped data."""
    if choices is None:
        choices = ["A", "B", "C", "D"]
    return Task(
        query=query,
        id=f"mmlu_{doc_id}",
        environment_data={
            "choices": choices,
            "full_prompt": full_prompt,
            "example": query,
        },
        evaluation_data={"gold": gold},
        metadata={"doc_id": doc_id, "task_type": "mmlu"},
    )


def make_mmlu_json_item(
    query: str = "What is 2+2?",
    gold: int = 0,
    choices: Optional[List[str]] = None,
    full_prompt: str = "Few-shot examples...\n\nWhat is 2+2?",
) -> Dict[str, Any]:
    """Create a raw JSON dict matching the MMLU prompts file format."""
    if choices is None:
        choices = ["A", "B", "C", "D"]
    return {
        "query": query,
        "gold": gold,
        "choices": choices,
        "full_prompt": full_prompt,
        "example": query,
    }


@pytest.fixture
def sample_mmlu_task():
    """Single MMLU task with gold=0 (answer A)."""
    return make_mmlu_task()


@pytest.fixture
def sample_mmlu_tasks():
    """Three MMLU tasks with different gold answers."""
    return [
        make_mmlu_task(query="Q1", gold=0, doc_id=0),
        make_mmlu_task(query="Q2", gold=1, doc_id=1),
        make_mmlu_task(query="Q3", gold=2, doc_id=2),
    ]


@pytest.fixture
def mmlu_json_path(tmp_path):
    """Write a 5-item MMLU JSON file and return its path."""
    items = [make_mmlu_json_item(query=f"Question {i}", gold=i % 4, full_prompt=f"Full prompt {i}") for i in range(5)]
    path = tmp_path / "mmlu_prompts.json"
    path.write_text(json.dumps(items))
    return path
