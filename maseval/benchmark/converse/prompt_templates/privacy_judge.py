"""Privacy evaluation prompts for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Prompts and use-case configurations are copied from the original ConVerse
implementation to ensure faithful reproduction of the evaluation methodology.

Sources:
    ConVerse/judge/generic_prompts.py:6-98   — prompt templates
    ConVerse/judge/generic_prompts.py:246-306 — USE_CASE_CONFIGS
    ConVerse/judge/generic_prompts.py:308-328 — get_use_case_prompt()
"""

from ..config import USE_CASE_CONFIGS
from ._loader import load_prompt

# ConVerse/judge/generic_prompts.py:6-50
PRIVACY_LEAK_PROMPT = load_prompt("judge", "privacy_leak")

# ConVerse/judge/generic_prompts.py:52-98
PRIVACY_ABSTRACTION_PROMPT = load_prompt("judge", "privacy_abstraction")


def format_leak_prompt(
    use_case: str,
    data_item: str,
    attack_action: str,
    extraction_snippets: str,
    conversation: str,
) -> str:
    """Format the privacy leak evaluation prompt for a specific use case.

    Mirrors ``ConVerse/judge/privacy_prompts.py:4-15`` which calls
    ``get_use_case_prompt(generic_privacy_leak_prompt, use_case, ...)``.

    Args:
        use_case: Use case key (``"travel_planning"``, ``"insurance"``,
            or ``"real_estate"``).
        data_item: Name of the data being evaluated (e.g. ``"Email"``).
        attack_action: The attack action text.
        extraction_snippets: Pre-formatted snippet list.
        conversation: The conversation between assistant and external agent.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return PRIVACY_LEAK_PROMPT.format(
        **config,
        data_item=data_item,
        attack_action=attack_action,
        extraction_snippets=extraction_snippets,
        conversation=conversation,
    )


def format_abstraction_prompt(
    use_case: str,
    data_item: str,
    attack_action: str,
    raw_data: str,
    acceptable_abstraction: str,
    conversation: str,
) -> str:
    """Format the abstraction appropriateness evaluation prompt.

    Mirrors ``ConVerse/judge/privacy_prompts.py:17-30`` which calls
    ``get_use_case_prompt(generic_privacy_abstraction_prompt, use_case, ...)``.

    Args:
        use_case: Use case key.
        data_item: Name of the data being evaluated.
        attack_action: The attack action text.
        raw_data: Pre-formatted raw data list.
        acceptable_abstraction: The acceptable abstraction description.
        conversation: The conversation between assistant and external agent.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return PRIVACY_ABSTRACTION_PROMPT.format(
        **config,
        data_item=data_item,
        attack_action=attack_action,
        raw_data=raw_data,
        acceptable_abstraction=acceptable_abstraction,
        conversation=conversation,
    )
