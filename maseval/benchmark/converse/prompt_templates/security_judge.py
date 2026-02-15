"""Security evaluation prompts for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Prompts are copied from the original ConVerse implementation to ensure
faithful reproduction of the security evaluation methodology.

Sources:
    ConVerse/judge/generic_prompts.py:101-137 — final package prompt
    ConVerse/judge/generic_prompts.py:139-183 — toolkit actions prompt
    ConVerse/judge/security_prompts.py:6-32   — use-case wrappers
"""

from ..config import USE_CASE_CONFIGS
from ._loader import load_prompt

# ConVerse/judge/generic_prompts.py:101-137
SECURITY_FINAL_PACKAGE_PROMPT = load_prompt("judge", "security_final_package")

# ConVerse/judge/generic_prompts.py:139-183
SECURITY_TOOLKIT_ACTIONS_PROMPT = load_prompt("judge", "security_toolkit_actions")


def format_final_package_prompt(
    use_case: str,
    attack_name: str,
    attack_action: str,
    legitimate_action: str,
    final_package: str,
) -> str:
    """Format the security final-package evaluation prompt.

    Mirrors ``ConVerse/judge/security_prompts.py:6-17`` which calls
    ``get_use_case_prompt(generic_security_final_package_prompt, use_case, ...)``.

    Args:
        use_case: Use case key (``"travel_planning"``, ``"insurance"``,
            or ``"real_estate"``).
        attack_name: Name of the security attack.
        attack_action: The attack action/goal text.
        legitimate_action: What the AI assistant should legitimately do.
        final_package: The final package output (string or JSON).

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return SECURITY_FINAL_PACKAGE_PROMPT.format(
        use_case_context=config["use_case_context"],
        package_type=config["package_type"],
        attack_name=attack_name,
        attack_action=attack_action,
        legitimate_action=legitimate_action,
        final_package=final_package,
    )


def format_toolkit_actions_prompt(
    use_case: str,
    legitimate_action: str,
    attack_name: str,
    attack_action: str,
    attack_rationale: str,
    assistant_env_communication: str,
) -> str:
    """Format the security toolkit-actions evaluation prompt.

    Mirrors ``ConVerse/judge/security_prompts.py:21-32`` which calls
    ``get_use_case_prompt(generic_security_toolkit_actions_prompt, use_case, ...)``.

    Args:
        use_case: Use case key.
        legitimate_action: What the AI assistant should legitimately do.
        attack_name: Name of the security attack.
        attack_action: The attack action/goal text.
        attack_rationale: Rationale used by external agent to justify the attack.
        assistant_env_communication: Pre-formatted assistant-environment
            communication text.

    Returns:
        Fully formatted prompt string.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    config = USE_CASE_CONFIGS[use_case]
    return SECURITY_TOOLKIT_ACTIONS_PROMPT.format(
        use_case_context=config["use_case_context"],
        legitimate_action=legitimate_action,
        attack_name=attack_name,
        attack_action=attack_action,
        attack_rationale=attack_rationale,
        assistant_env_communication=assistant_env_communication,
    )
