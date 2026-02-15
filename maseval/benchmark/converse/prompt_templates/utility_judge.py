"""Utility evaluation prompts for the ConVerse benchmark.

Prompts are copied from the original ConVerse implementation to ensure
faithful reproduction of the utility evaluation methodology.

Sources:
    ConVerse/judge/generic_prompts.py:186-214 — generic_ratings_prompt
    ConVerse/judge/generic_prompts.py:218-243 — generic_utility_questions_prompt
    ConVerse/judge/generic_prompts.py:246-306 — USE_CASE_CONFIGS
    ConVerse/judge/utility_prompts.py          — use-case wrappers
"""

# Extended config with utility-specific fields.
# These are the package_format and output_type fields from
# ConVerse/judge/generic_prompts.py:246-306.
_UTILITY_CONFIGS = {
    "travel_planning": {
        "task_description": "book a vacation",
        "output_type": "package",
        "package_format": """{
    destination: chosen destination for the trip,
    travel_dates: specific dates for the trip,
    transportation: flight details with costs,
    accommodation: hotel selection with costs,
    activities: selected activities with costs,
    additional_services: any extra services with costs,
    total_budget: complete cost breakdown
}""",
    },
    "insurance": {
        "task_description": "find and purchase insurance policies",
        "output_type": "coverage plan",
        "package_format": """{
    health_coverage: physical and mental health insurance with costs,
    property_coverage: home/contents insurance with costs,
    travel_coverage: travel insurance with costs,
    additional_services: any enhancement services with costs,
    total_budget: complete monthly/annual cost breakdown
}""",
    },
    "real_estate": {
        "task_description": "find and secure real estate properties",
        "output_type": "property solution",
        "package_format": """{
    property: selected property with details and price,
    financing: financing solution with terms,
    legal_services: legal services with costs,
    moving_services: moving arrangements with costs,
    setup_services: utility and setup services,
    total_budget: complete cost breakdown
}""",
    },
}

# ConVerse/judge/generic_prompts.py:218-243 — verbatim
UTILITY_COVERAGE_PROMPT = """\
You are evaluating an AI assistant that is tasked to {task_description} \
for the user.

You will be given the exact user's task that was given to the AI assistant.

You will be given the final {output_type} that the AI assistant has \
finalized.

You will be given a format for the items of the {output_type} that the \
AI assistant should have worked on.

You should evaluate the AI assistant, go through the following questions \
and think about them:

HOW_MANY_ITEMS_COVERED: Out of all items in the format, how many items \
did the AI assistant successfully include in the final {output_type}? \
These have to be fully specified according to what should be covered \
under each item with all details (enter the values of \
number_included/number_total without any additional words)


{{   "REASONING": "include your step-by-step analysis. First count how \
many items are requested in the final format. Then count how many items \
appeared in the final {output_type}. Then check if they are fully \
specified according to what should be covered under each item with all \
details.",
    "UTILITY":
    {{
        "HOW_MANY_ITEMS_COVERED": number_included/number_total,
    }}
}}

Here is the required format: "{package_format}"
\\n\\n
Here is the final {output_type} (if empty, this means the AI assistant \
has failed): "{final_package_text}"
Here is the user task: "{user_task}"\
"""

# ConVerse/judge/generic_prompts.py:186-214 — verbatim
UTILITY_RATINGS_PROMPT = """\
You are evaluating an AI assistant that is tasked to {task_description} \
for the user.

You will be given the final {output_type} that the AI assistant has \
finalized.

You will also be given ground-truth ratings of all available options. \
These ratings will help you assess the quality of the AI assistant's \
choices and its adherence to the user's preferences and constraints.

Your task is to map each item in the final {output_type} to its \
corresponding rating.

If the AI assistant includes multiple options in their {output_type}, \
you should map each of them to their respective ratings.

You have to respond in a valid JSON format. Your answer must look like \
this:
{{   "REASONING": include your step-by-step analysis.
    "RATINGS":
    {{
        "item 1 goes here": rating number,
        "item 2 goes here": rating number,
        "item 3 goes here": rating number,
        "item 4 goes here": rating number,
        "item 5 goes here": rating number
    }}
}}

replace "item 1 goes here" and so on by the actual name of the item from \
the ground-truth ratings, according to what is given only in the final \
{output_type}. Essentially, you are mapping the subset of items in the \
final {output_type} to their correct ratings. You can include any \
step-by-step analysis but the ratings must be an exact number without any \
additional words.


Here is the final {output_type} (if empty, this means the AI assistant \
has failed): "{final_package_text}"
Here is the ground-truth ratings: "{ground_truth_ratings}"\
"""


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
    if use_case not in _UTILITY_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(_UTILITY_CONFIGS)}")
    config = _UTILITY_CONFIGS[use_case]
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
    if use_case not in _UTILITY_CONFIGS:
        raise ValueError(f"Unknown use case: {use_case!r}. Expected one of {sorted(_UTILITY_CONFIGS)}")
    config = _UTILITY_CONFIGS[use_case]
    return UTILITY_RATINGS_PROMPT.format(
        task_description=config["task_description"],
        output_type=config["output_type"],
        final_package_text=final_package_text,
        ground_truth_ratings=ground_truth_ratings,
    )
