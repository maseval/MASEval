"""Utility evaluation prompts for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Prompts are copied from the original ConVerse implementation to ensure
faithful reproduction of the utility evaluation methodology.

Sources:
    ConVerse/judge/generic_prompts.py:186-214 — generic_ratings_prompt
    ConVerse/judge/generic_prompts.py:218-243 — generic_utility_questions_prompt
    ConVerse/judge/generic_prompts.py:246-306 — USE_CASE_CONFIGS
    ConVerse/judge/utility_prompts.py          — use-case wrappers
"""

from ..config import USE_CASE_CONFIGS
from ._loader import load_prompt

# ConVerse/judge/generic_prompts.py:218-243 — verbatim
UTILITY_COVERAGE_PROMPT = load_prompt("judge", "utility_coverage")

# ConVerse/judge/generic_prompts.py:186-214 — verbatim
UTILITY_RATINGS_PROMPT = load_prompt("judge", "utility_ratings")


def format_coverage_prompt(
    use_case: str,
    final_package_text: str,
    user_task: str,
) -> str:
    """Format the utility coverage evaluation prompt.

    Mirrors ``ConVerse/judge/utility_prompts.py:6-23`` which calls
    ``get_use_case_prompt(generic_utility_questions_prompt, use_case, ...)``.

    Args:
        use_case: Use case key (``"travel_planning"``, ``"insurance"``,
            or ``"real_estate"``).
        final_package_text: The final package text from the assistant.
        user_task: The original user task description.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return UTILITY_COVERAGE_PROMPT.format(
        task_description=config["task_description"],
        output_type=config["output_type"],
        package_format=config["package_format"],
        final_package_text=final_package_text,
        user_task=user_task,
    )


def format_ratings_prompt(
    use_case: str,
    final_package_text: str,
    ground_truth_ratings: str,
) -> str:
    """Format the utility ratings evaluation prompt.

    Mirrors ``ConVerse/judge/utility_prompts.py:25-36`` which calls
    ``get_use_case_prompt(generic_ratings_prompt, use_case, ...)``.

    Args:
        use_case: Use case key (``"travel_planning"``, ``"insurance"``,
            or ``"real_estate"``).
        final_package_text: The final package text from the assistant.
        ground_truth_ratings: JSON string of ground-truth ratings.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return UTILITY_RATINGS_PROMPT.format(
        task_description=config["task_description"],
        output_type=config["output_type"],
        final_package_text=final_package_text,
        ground_truth_ratings=ground_truth_ratings,
    )
