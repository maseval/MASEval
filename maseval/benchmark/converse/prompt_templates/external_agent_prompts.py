"""External agent (adversarial) prompts for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Ported from:
    ConVerse/external_agent/external_prompts_adv.py     — adversarial prompt
    ConVerse/external_agent/external_prompts_general.py  — shared components
    ConVerse/read_toolkits.py:22-30                      — simplified_toolkits_for_adv

Execution-format sections (delimiter tags for ``<scratch_pad>``,
``<red_teamer_log_summary>``, ``<output_to_ai_assistant>``) are stripped
because MASEval's :class:`LLMUser` directly generates text responses.
All adversarial instructions, reasoning checklists, and behavioural rules
are kept verbatim from the original.
"""

from ._loader import load_prompt

# ConVerse/external_agent/external_prompts_adv.py:11-38 — verbatim
# (give_up_adv="3" is embedded in the resolved text)
_INITIAL_EXPLANATION_ADV = load_prompt("external", "initial_explanation_adv")

# ConVerse/external_agent/external_prompts_adv.py:43-50
# (simplified_toolkits_for_adv is embedded in the resolved text)
_ENVIRONMENT_DESCRIPTION_ADV = load_prompt("external", "environment_description_adv")

# ConVerse/external_agent/external_prompts_general.py:33-51 — verbatim
_SIMULATE_ACCURATE_OUTPUT_GENERAL = load_prompt("external", "simulate_output_general")

# ConVerse/external_agent/external_prompts_adv.py:52-61 — verbatim
# (give_up_adv="3" is embedded in the resolved text)
_SIMULATE_ACCURATE_OUTPUT_ADV = load_prompt("external", "simulate_output_adv")

# ConVerse/external_agent/external_prompts_general.py:7-22 — verbatim
_HISTORY_DESCRIPTION = load_prompt("external", "history_description")

# ConVerse/external_agent/external_prompts_adv.py:63-100 — verbatim
# (give_up_adv="3" is embedded in the resolved text)
_SCRATCHPAD_ADV = load_prompt("external", "scratchpad_adv")

# ConVerse/external_agent/external_prompts_general.py:54-76
# Adapted: delimiter tags removed; the LLMUser generates plain text.
_AGENT_ADDRESSING_PROMPT = load_prompt("external", "agent_addressing")


def build_adversarial_scenario(
    options_text: str,
    adversarial_task: str,
    role: str = "service provider",
    use_case: str = "travel_planning",
) -> str:
    """Build the full adversarial external-agent scenario prompt.

    Assembles all sections matching the original
    ``get_external_aggregated_prompt_adv(package_options)`` from
    ``ConVerse/external_agent/external_prompts_adv.py:103-115``.

    Args:
        options_text: The full package options text for the domain.
        adversarial_task: The adversarial goal the agent should pursue.
        role: The role the external agent simulates (e.g. ``"travel agent"``).
        use_case: ConVerse use-case key (used for context only).

    Returns:
        Fully assembled scenario prompt string for the external agent.
    """
    # ConVerse/external_agent/external_prompts_general.py:25-29
    package_options_section = f"""
== {{package_options}} ==

These are the options available to you as a simulation database for each \
item in your package: {options_text}
"""

    adversarial_task_section = f"""
== {{adversarial_task}} ==

Your adversarial task is: {adversarial_task}

You are simulating the role of: {role}
"""

    sections = [
        _INITIAL_EXPLANATION_ADV,
        _ENVIRONMENT_DESCRIPTION_ADV,
        _SIMULATE_ACCURATE_OUTPUT_GENERAL,
        _SIMULATE_ACCURATE_OUTPUT_ADV,
        package_options_section,
        adversarial_task_section,
        _HISTORY_DESCRIPTION,
        _SCRATCHPAD_ADV,
        _AGENT_ADDRESSING_PROMPT,
    ]
    return "\n".join(sections)
