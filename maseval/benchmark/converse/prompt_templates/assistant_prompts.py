"""Assistant system prompt for the ConVerse benchmark.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Ported from ``ConVerse/assistant/assistant_prompts.py``.  The prompt text
is reproduced faithfully, with only execution-format sections adapted:
delimiter-based routing (``<output_to_environment>`` /
``<output_to_external_agent>`` tags) is replaced by tool-call instructions,
because MASEval uses a ReAct tool-calling loop instead of ToolEmu-style
delimiter parsing.

All safety rules, reasoning checklists, goal formats, and toolkit
descriptions are kept verbatim from the original.

Sources:
    ConVerse/assistant/assistant_prompts.py    — prompt sections
    ConVerse/read_toolkits.py:1-20            — simplified_toolkits
"""

from ._loader import load_prompt

# ConVerse/assistant/assistant_prompts.py:20-36 — adapted for tool-calling
_INITIAL_EXPLANATION = load_prompt("assistant", "initial_explanation")

# ConVerse/assistant/assistant_prompts.py:40-63 — verbatim
_RULES_EXPLANATION = load_prompt("assistant", "rules")

# ConVerse/assistant/assistant_prompts.py:95-105 — verbatim
_EXTERNAL_AGENT_RULES = load_prompt("assistant", "external_agent_rules")

# ConVerse/assistant/assistant_prompts.py:65-72 (environment_description)
# Includes simplified_toolkits from ConVerse/read_toolkits.py:1-20
_ENVIRONMENT_DESCRIPTION = load_prompt("assistant", "environment_description")

# ConVerse/assistant/assistant_prompts.py:74-90 — adapted for tool-calling
_ENVIRONMENT_RULES = load_prompt("assistant", "environment_rules")

# ConVerse/assistant/assistant_prompts.py:143-202 — verbatim
_OUTPUT_FORMATS = {
    "travel_planning": load_prompt("assistant", "output_travel_planning"),
    "real_estate": load_prompt("assistant", "output_real_estate"),
    "insurance": load_prompt("assistant", "output_insurance"),
}

# ConVerse/assistant/assistant_prompts.py:236-292 — adapted for tool-calling
_REASONING_CHECKLIST = load_prompt("assistant", "reasoning_checklist")

# Task context — adapted for tool-calling
_TASK_CONTEXT = load_prompt("assistant", "task_context")


def build_assistant_system_prompt(use_case: str, user_task: str) -> str:
    """Build the full assistant system prompt for a CONVERSE task.

    Assembles all prompt sections matching the original
    ``get_aggregated_prompts_for_turn(use_case)`` from
    ``ConVerse/assistant/assistant_prompts.py:407-419``, adapted for
    MASEval's ReAct tool-calling execution model.

    Args:
        use_case: ConVerse use-case key (``"travel_planning"``,
            ``"real_estate"``, or ``"insurance"``).
        user_task: The user's task description.

    Returns:
        Fully assembled system prompt string.
    """
    output_format = _OUTPUT_FORMATS.get(use_case, _OUTPUT_FORMATS["travel_planning"])

    sections = [
        _INITIAL_EXPLANATION,
        _RULES_EXPLANATION,
        _EXTERNAL_AGENT_RULES,
        _ENVIRONMENT_DESCRIPTION,
        _ENVIRONMENT_RULES,
        output_format,
        _REASONING_CHECKLIST,
        _TASK_CONTEXT.format(user_task=user_task),
    ]
    return "\n".join(sections)
