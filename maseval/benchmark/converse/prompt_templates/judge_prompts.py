"""Judge evaluation prompts for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Prompt templates for all three ConVerse evaluation dimensions (privacy,
security, utility).  Each ``format_*`` function loads a template from the
corresponding ``.txt`` file, injects use-case configuration from
:data:`~maseval.benchmark.converse.config.USE_CASE_CONFIGS`, and returns
a fully formatted prompt string ready for the LLM judge.

Sources:
    ConVerse/judge/generic_prompts.py   — prompt templates & USE_CASE_CONFIGS
    ConVerse/judge/privacy_prompts.py   — privacy use-case wrappers
    ConVerse/judge/security_prompts.py  — security use-case wrappers
    ConVerse/judge/utility_prompts.py   — utility use-case wrappers
"""

from ..config import USE_CASE_CONFIGS, UseCaseConfig
from ._loader import load_prompt

# -- Privacy templates -------------------------------------------------------
PRIVACY_LEAK_PROMPT = load_prompt("judge", "privacy_leak")
PRIVACY_ABSTRACTION_PROMPT = load_prompt("judge", "privacy_abstraction")

# -- Security templates -------------------------------------------------------
SECURITY_FINAL_PACKAGE_PROMPT = load_prompt("judge", "security_final_package")
SECURITY_TOOLKIT_ACTIONS_PROMPT = load_prompt("judge", "security_toolkit_actions")

# -- Utility templates --------------------------------------------------------
UTILITY_COVERAGE_PROMPT = load_prompt("judge", "utility_coverage")
UTILITY_RATINGS_PROMPT = load_prompt("judge", "utility_ratings")


def _get_config(use_case: str) -> UseCaseConfig:
    """Look up and validate a use-case configuration.

    Raises:
        ValueError: If *use_case* is not a recognised key.
    """
    if use_case not in USE_CASE_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(USE_CASE_CONFIGS)}")
    return USE_CASE_CONFIGS[use_case]


# -- Privacy formatters -------------------------------------------------------


def format_leak_prompt(
    use_case: str,
    data_item: str,
    attack_action: str,
    extraction_snippets: str,
    conversation: str,
) -> str:
    """Format the privacy leak evaluation prompt for a specific use case.

    Mirrors ``ConVerse/judge/privacy_prompts.py:4-15``.
    """
    config = _get_config(use_case)
    return PRIVACY_LEAK_PROMPT.format(
        **vars(config),
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

    Mirrors ``ConVerse/judge/privacy_prompts.py:17-30``.
    """
    config = _get_config(use_case)
    return PRIVACY_ABSTRACTION_PROMPT.format(
        **vars(config),
        data_item=data_item,
        attack_action=attack_action,
        raw_data=raw_data,
        acceptable_abstraction=acceptable_abstraction,
        conversation=conversation,
    )


# -- Security formatters ------------------------------------------------------


def format_final_package_prompt(
    use_case: str,
    attack_name: str,
    attack_action: str,
    legitimate_action: str,
    final_package: str,
) -> str:
    """Format the security final-package evaluation prompt.

    Mirrors ``ConVerse/judge/security_prompts.py:6-17``.
    """
    config = _get_config(use_case)
    return SECURITY_FINAL_PACKAGE_PROMPT.format(
        use_case_context=config.use_case_context,
        package_type=config.package_type,
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

    Mirrors ``ConVerse/judge/security_prompts.py:21-32``.
    """
    config = _get_config(use_case)
    return SECURITY_TOOLKIT_ACTIONS_PROMPT.format(
        use_case_context=config.use_case_context,
        legitimate_action=legitimate_action,
        attack_name=attack_name,
        attack_action=attack_action,
        attack_rationale=attack_rationale,
        assistant_env_communication=assistant_env_communication,
    )


# -- Utility formatters -------------------------------------------------------


def format_coverage_prompt(
    use_case: str,
    final_package_text: str,
    user_task: str,
) -> str:
    """Format the utility coverage evaluation prompt.

    Mirrors ``ConVerse/judge/utility_prompts.py:6-23``.
    """
    config = _get_config(use_case)
    return UTILITY_COVERAGE_PROMPT.format(
        task_description=config.task_description,
        output_type=config.output_type,
        package_format=config.package_format,
        final_package_text=final_package_text,
        user_task=user_task,
    )


def format_ratings_prompt(
    use_case: str,
    final_package_text: str,
    ground_truth_ratings: str,
) -> str:
    """Format the utility ratings evaluation prompt.

    Mirrors ``ConVerse/judge/utility_prompts.py:25-36``.
    """
    config = _get_config(use_case)
    return UTILITY_RATINGS_PROMPT.format(
        task_description=config.task_description,
        output_type=config.output_type,
        final_package_text=final_package_text,
        ground_truth_ratings=ground_truth_ratings,
    )
