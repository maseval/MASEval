"""Gaia2 Benchmark - Main Implementation.

Framework-agnostic implementation of the Gaia2 benchmark for evaluating
LLM-based agents on dynamic, multi-step scenarios.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

Usage:
    from maseval.benchmark.gaia2 import (
        Gaia2Benchmark, Gaia2Environment, Gaia2Evaluator,
        load_tasks, configure_model_ids,
    )

    # Load data
    tasks = load_tasks(capability="execution", limit=5)

    # Create your framework-specific benchmark subclass
    class MyGaia2Benchmark(Gaia2Benchmark):
        def setup_agents(self, agent_data, environment, task, user, seed_generator):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            seed = kwargs.get("seed")  # Extract seed for reproducibility
            adapter = MyModelAdapter(model_id, seed=seed)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyGaia2Benchmark(agent_data={})
    results = benchmark.run(tasks)

Default Agent Implementation:
    For comparison with ARE's reference agent, use DefaultAgentGaia2Benchmark:

    from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark, load_tasks

    tasks = load_tasks(capability="execution", limit=5)

    benchmark = DefaultAgentGaia2Benchmark(
        agent_data={"model_id": "gpt-4o"},
    )
    results = benchmark.run(tasks)
"""

import re
import time
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.callback import BenchmarkCallback
from maseval.core.seeding import SeedGenerator

from maseval.benchmark.gaia2.environment import Gaia2Environment
from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

if TYPE_CHECKING:
    from are.simulation.agents.default_agent.base_agent import RunningState, SimulatedGenerationTimeConfig


# =============================================================================
# Benchmark
# =============================================================================


class Gaia2Benchmark(Benchmark):
    """MASEval wrapper for Gaia2/ARE benchmark.

    Hybrid approach: Uses ARE for simulation and evaluation while providing
    MASEval orchestration, tracing, and agent flexibility.

    The ARE simulation runs internally; agents interact purely via tool calls.
    Time control happens through ``SystemApp__wait_for_notification``.

    Subclasses must implement:

    - ``setup_agents()`` — Create agents for the task
    - ``get_model_adapter()`` — Provide model adapters

    Multi-Turn Architecture:
        GAIA2 uses ARE's **two-level loop** architecture:

        - **Outer loop** (turns): drains the notification queue, formats user
          messages as ``[TASK]``, re-queues environment notifications, then
          runs the inner step loop.
        - **Inner loop** (steps): ReAct cycle. Terminates on
          ``send_message_to_user`` (TERMINATED — turn complete) or
          ``wait_for_notification`` (PAUSED — outer loop continues).

        ``ARE are_simulation_main.py:agent_loop()``

        **What custom agents must do:**

        - **Terminate inner loop** on both ``send_message_to_user`` and
          ``wait_for_notification``. The former completes a turn; the latter
          pauses the agent while ARE processes events.
        - **Between turns** (outer loop): drain notifications via
          ``environment.get_turn_notifications()`` which re-queues environment
          notifications and returns user messages for ``[TASK]`` formatting.
        - **Within turns** (inner loop pre-step): poll notifications via
          ``environment.poll_notifications()`` to pick up re-queued environment
          notifications and new messages.

        See the default agent implementation for the reference two-level loop
        approach.
    """

    # The benchmark invokes the agent once; multi-turn is the agent's
    # responsibility (see "Multi-Turn Notification Loop" above).
    MAX_INVOCATIONS = 1

    def __init__(
        self,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = MAX_INVOCATIONS,
        num_workers: int = 1,
        fail_on_setup_error: bool = False,
        fail_on_task_error: bool = False,
        fail_on_evaluation_error: bool = False,
        progress_bar: bool | str = True,
        seed: Optional[int] = None,
        seed_generator=None,
    ):
        """Initialize benchmark with Gaia2-specific defaults.

        Args:
            callbacks: Optional list of callback handlers for monitoring execution.
            n_task_repeats: Number of times to repeat each task. Default 1.
            max_invocations: Maximum agent invocations (default: 1 for single-turn).
            num_workers: Number of parallel task executions. Default 1 (sequential).
            fail_on_setup_error: If True, raise on setup errors. Default False.
            fail_on_task_error: If True, raise on task execution errors. Default False.
            fail_on_evaluation_error: If True, raise on evaluation errors. Default False.
            progress_bar: Progress display. True (default) for tqdm, "rich" for Rich,
                or False to disable.
            seed: Global seed for reproducible benchmark runs.
            seed_generator: Custom seed generator (takes precedence over seed).
        """
        super().__init__(
            callbacks=callbacks,
            n_task_repeats=n_task_repeats,
            max_invocations=max_invocations,
            num_workers=num_workers,
            fail_on_setup_error=fail_on_setup_error,
            fail_on_task_error=fail_on_task_error,
            fail_on_evaluation_error=fail_on_evaluation_error,
            progress_bar=progress_bar,
            seed=seed,
            seed_generator=seed_generator,
        )

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Gaia2Environment:
        """Create Gaia2 environment wrapping ARE simulation.

        Args:
            agent_data: Agent configuration
            task: Current task
            seed_generator: Seed generator for reproducibility

        Returns:
            Gaia2Environment instance
        """
        judge_engine_config = task.evaluation_data.get("judge_engine_config")
        return Gaia2Environment(task_data=task.environment_data, judge_engine_config=judge_engine_config)

    def setup_user(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Optional[User]:
        """Gaia2 uses event-based simulation, not turn-based user simulation.

        User interactions in Gaia2 happen through scheduled events (e.g.,
        "user sends message at t=30s") rather than synchronous turn-taking.

        Args:
            agent_data: Agent configuration
            environment: Gaia2Environment instance
            task: Current task
            seed_generator: Seed generator for reproducibility

        Returns:
            None (no user simulator needed)
        """
        return None

    @abstractmethod
    def setup_agents(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration
            environment: Gaia2Environment with ARE tools
            task: Current task
            user: Optional user simulator (always None for Gaia2)
            seed_generator: Seed generator for reproducibility

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(  # type: ignore[override]
        self,
        environment: Gaia2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        """Create Gaia2 evaluator using ARE's judge.

        Args:
            environment: Gaia2Environment instance
            task: Current task with evaluation data
            agents: Agent instances
            user: Optional user simulator (always None)
            seed_generator: Seed generator for reproducibility

        Returns:
            List with single Gaia2Evaluator instance
        """
        evaluator_model_id = task.evaluation_data.get("model_id")
        model = None
        if evaluator_model_id:
            # Derive seed for evaluator model (returns None if seeding disabled)
            evaluator_seed = seed_generator.derive_seed("evaluators/judge")
            model = self.get_model_adapter(evaluator_model_id, register_name="evaluator", seed=evaluator_seed)

        return [
            Gaia2Evaluator(
                task=task,
                environment=environment,
                use_llm_judge=evaluator_model_id is not None,
                model=model,
            )
        ]

    def run_agents(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Gaia2Environment,
        query: str = "",
    ) -> Any:
        """Execute agents and ensure environment cleanup.

        Args:
            agents: Agent instances to run
            task: Current task
            environment: Gaia2Environment
            query: Query/prompt for agents

        Returns:
            Final answer from agents
        """
        try:
            answers = [agent.run(query) for agent in agents]
            return answers[0] if len(answers) == 1 else answers
        finally:
            environment.cleanup()

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using Gaia2 evaluators.

        Uses each evaluator's filter_traces() method to extract relevant data,
        then calls the evaluator with the filtered traces.

        Returns Gaia2 format:
            - gsr: Goal Success Rate
            - partial_gsr: Partial success rate
            - passed: Boolean
            - rationale: Judge rationale (if available)

        Args:
            evaluators: List of evaluators
            agents: Dict of agents
            final_answer: Final answer from agents
            traces: Execution traces

        Returns:
            List of evaluation result dicts
        """
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results


# =============================================================================
# Default Agent Implementation
# =============================================================================

# Prompt templates directory
_PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"

# ARE default parameters (documented with source locations)
# ARE agents/are_simulation_agent_config.py:36
_DEFAULT_MAX_ITERATIONS = 80
# ARE agents/llm/llm_engine.py:17
_DEFAULT_TEMPERATURE = 0.5
# ARE agents/llm/llm_engine.py:16
_DEFAULT_MAX_TOKENS = 16384
# ARE agents/default_agent/base_agent.py:347
_DEFAULT_INVALID_FORMAT_RETRIES = 10


# Stop sequences for text-based action parsing
_STOP_SEQUENCES = ["<end_action>", "Observation:"]

# Termination tool names — send_message_to_user terminates the inner step
# loop with TERMINATED state (turn is complete).
# ARE termination_methods/are_simulation.py:112-114
_TERMINATION_TOOLS = frozenset(
    {
        "AgentUserInterface__send_message_to_user",
    }
)

# Pause tool — wait_for_notification terminates the inner step loop with
# PAUSED state (agent is waiting for events; outer turn loop continues).
# ARE termination_methods/are_simulation.py:34-36, 116-121
_PAUSE_TOOL = "SystemApp__wait_for_notification"


def _load_prompt_template(name: str) -> str:
    """Load a prompt template from file.

    Args:
        name: Template filename (without .txt extension)

    Returns:
        Template content as string
    """
    path = _PROMPT_TEMPLATES_DIR / f"{name}.txt"
    return path.read_text()


def _get_tool_description_with_args(name: str, tool: Callable) -> str:
    """Build a single tool's description with args, matching ARE's format.

    ARE tool_box.py:16-20 ``DEFAULT_TOOL_DESCRIPTION_TEMPLATE``.

    Args:
        name: Tool name
        tool: Callable with optional description/inputs/output_type attributes

    Returns:
        Formatted tool description string
    """
    desc = getattr(tool, "description", "") or ""
    inputs = getattr(tool, "inputs", {}) or {}
    output_type = getattr(tool, "output_type", "string") or "string"
    return f"- {name}: {desc}\n    Takes inputs: {inputs}\n    Returns an output of type: {output_type}"


def _build_tool_descriptions(tools: Dict[str, Callable]) -> str:
    """Build tool descriptions matching ARE's Jinja2 template format.

    ARE tool_box.py:16-20 uses the template:
    ``- {{ tool.name }}: {{ tool.description }}``
    ``    Takes inputs: {{tool.inputs}}``
    ``    Returns an output of type: {{tool.output_type}}``

    Args:
        tools: Dict of tool name -> callable

    Returns:
        Formatted tool descriptions string
    """
    descriptions = []
    for name, tool in tools.items():
        desc = getattr(tool, "description", "") or ""
        inputs = getattr(tool, "inputs", {}) or {}
        output_type = getattr(tool, "output_type", "string") or "string"

        # Match ARE's format: raw dict representation for inputs
        descriptions.append(f"- {name}: {desc}\n    Takes inputs: {inputs}\n    Returns an output of type: {output_type}")

    return "\n".join(descriptions)


def _get_notification_system_prompt(environment: Optional[Any]) -> str:
    """Generate notification system prompt matching ARE's behavior.

    ARE agents/default_agent/prompts/notification_system.py:32-46

    Args:
        environment: Optional Gaia2Environment

    Returns:
        Notification system prompt string
    """
    if environment is None:
        # Fallback: basic notification policy (matches ARE's NotificationSystem default)
        return (
            "Notification policy:\n"
            "- All new messages from the User will be notified to you.\n"
            "- The environment state may also change over time, but environment events will not be notified to you.\n"
            "- You can also proactively check for any other update in an App by using the tools given to you.\n"
            "- If a call to SystemApp__wait_for_notification times out, you will receive a notification."
        )

    try:
        notification_system = environment.get_notification_system()
        if notification_system is None:
            # Same basic fallback
            return (
                "Notification policy:\n"
                "- All new messages from the User will be notified to you.\n"
                "- The environment state may also change over time, but environment events will not be notified to you.\n"
                "- You can also proactively check for any other update in an App by using the tools given to you.\n"
                "- If a call to SystemApp__wait_for_notification times out, you will receive a notification."
            )

        # Use ARE's notification prompt generator
        from are.simulation.agents.default_agent.prompts.notification_system import (  # type: ignore[import-not-found]
            get_notification_system_prompt,
        )

        are_env = environment.get_are_environment()
        scenario = environment.get_scenario()
        apps = getattr(scenario, "apps", None) or (list(are_env.apps.values()) if are_env else None)
        return get_notification_system_prompt(notification_system, apps)
    except Exception:
        # Graceful fallback
        return (
            "Notification policy:\n"
            "- All new messages from the User will be notified to you.\n"
            "- The environment state may also change over time, but environment events will not be notified to you.\n"
            "- You can also proactively check for any other update in an App by using the tools given to you.\n"
            "- If a call to SystemApp__wait_for_notification times out, you will receive a notification."
        )


def _get_current_time_description(environment: Optional[Any]) -> str:
    """Generate current time description matching ARE's behavior.

    ARE agents/default_agent/are_simulation_main.py:156-164

    Args:
        environment: Optional Gaia2Environment

    Returns:
        Current time description string
    """
    if environment is None:
        return ""

    try:
        from datetime import datetime, timezone

        # ARE are_simulation_main.py:157: `scenario.start_time or 0`
        # Defaults to Unix epoch (1970-01-01 00) when start_time is None
        start_time = environment.get_start_time() or 0
        date_str = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%Y-%m-%d %H")
        return f"Today's date in 'YYYY-MM-DD HH' format is {date_str}"
    except Exception:
        return ""


def _build_system_prompt(tools: Dict[str, Callable], environment: Optional[Any] = None) -> str:
    """Build the full system prompt with tool descriptions and dynamic placeholders.

    Matches ARE's system prompt construction:
    - Tool descriptions in ARE's Jinja2 format
    - Dynamic notification system policy from ARE's notification_system.py
    - Current time from scenario start_time (ARE are_simulation_main.py:156-164)
    - Empty agent_reminder_description (ARE are_simulation_main.py:166-171)
    - Scenario additional_system_prompt appended (ARE are_simulation_main.py:138-145)

    Args:
        tools: Dict of tool name -> callable
        environment: Optional Gaia2Environment for dynamic prompt generation

    Returns:
        Complete system prompt string
    """
    # Load templates
    general = _load_prompt_template("general_instructions")
    agent = _load_prompt_template("agent_instructions")
    environment_template = _load_prompt_template("environment_instructions")
    template = _load_prompt_template("system_prompt")

    # Build tool descriptions in ARE's format
    tool_descriptions = _build_tool_descriptions(tools)

    # Format environment instructions with tool descriptions and environment hints
    # ARE system_prompt.py:187-189: environment_hints is always "" for default JSON agent
    environment_formatted = environment_template.format(tool_descriptions=tool_descriptions, environment_hints="")

    # Replace dynamic placeholders (matching ARE's are_simulation_main.py:138-171)
    # 1. Notification system description
    notification_prompt = _get_notification_system_prompt(environment)
    environment_formatted = environment_formatted.replace("<<notification_system_description>>", notification_prompt)

    # 2. Agent reminder description (always empty in ARE)
    # ARE are_simulation_main.py:166-171
    environment_formatted = environment_formatted.replace("<<agent_reminder_description>>", "")

    # 3. Current time description
    # ARE are_simulation_main.py:156-164
    time_description = _get_current_time_description(environment)
    environment_formatted = environment_formatted.replace("<<curent_time_description>>", time_description)

    # Assemble full prompt
    prompt = template.format(
        general_instructions=general,
        agent_instructions=agent,
        environment_instructions=environment_formatted,
    )

    # Append scenario's additional_system_prompt if present
    # ARE are_simulation_main.py:138-145
    if environment is not None:
        try:
            scenario = environment.get_scenario()
            additional = getattr(scenario, "additional_system_prompt", None)
            if additional is not None:
                prompt += "\n\n" + additional
        except Exception:
            pass

    return prompt


def _parse_action_from_text(text: str) -> Optional[Tuple[str, str, Dict[str, Any] | str]]:
    """Parse Thought and Action from LLM text output.

    Uses ARE's ``parse_json_tool_call()`` for JSON parsing and action extraction.
    The outer text parsing (Thought/Action section extraction) is custom.

    Expected format::

        Thought: [reasoning]

        Action:
        {
          "action": "tool_name",
          "action_input": {...}
        }<end_action>

    Args:
        text: Raw LLM output text

    Returns:
        Tuple of (thought, tool_name, tool_args) or None if parsing fails
    """
    from are.simulation.agents.default_agent.tools.json_action_executor import (  # type: ignore[import-not-found]
        parse_json_tool_call,
    )

    # Extract thought (everything before "Action:")
    thought = ""
    if "Thought:" in text:
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n\s*Action:|$)", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

    # Find Action: and extract everything after it
    action_start = text.find("Action:")
    if action_start == -1:
        return None

    action_text = text[action_start + len("Action:") :]

    # Remove <end_action> suffix if present
    end_action_pos = action_text.find("<end_action>")
    if end_action_pos != -1:
        action_text = action_text[:end_action_pos]

    # Parse JSON tool call using ARE's parser
    # ARE handles: code fence removal, JSON extraction, action/action_input extraction
    # Raises JsonParsingAgentError (not ValueError) on failure
    try:
        tool_name, tool_args = parse_json_tool_call(action_text)
        return (thought, str(tool_name), tool_args)
    except Exception:
        return None


def _apply_stop_truncation(text: str, stop_sequences: List[str]) -> str:
    """Apply client-side stop-sequence truncation.

    Removes the first occurrence of any stop sequence and everything after it.
    This matches ARE's LiteLLMEngine behavior (litellm_engine.py:126-127) and
    serves as a universal fallback when stop sequences are not supported at the
    API level (e.g., reasoning models like o1/o3/GPT-5).

    Always applied regardless of whether API-level ``stop`` is also used.

    Args:
        text: Raw LLM response text.
        stop_sequences: Tokens to truncate on.

    Returns:
        Text truncated at the first matching stop sequence.
    """
    for stop_token in stop_sequences:
        text = text.split(stop_token)[0]
    return text


class DefaultGaia2Agent:
    """Default agent implementation for Gaia2 benchmark.

    ReAct-style agent matching ARE's reference implementation. Uses text-based
    action parsing (Thought/Action/Observation cycle) rather than native
    function calling.

    Uses ARE's **two-level loop** architecture:

    - **Outer loop** (``_turn_loop``): iterates over turns, matching
      ``are_simulation_main.py:agent_loop()``. Between turns, drains the
      notification queue, formats user messages as ``[TASK]``, re-queues
      environment notifications for the inner loop's pre-step.
    - **Inner loop** (``_step_loop``): iterates over steps within a turn,
      matching ``base_agent.py:execute_agent_loop()``. Terminates on BOTH
      ``send_message_to_user`` (TERMINATED) and ``wait_for_notification``
      (PAUSED).

    Key characteristics matching ARE (base_agent.py, are_simulation.py):

    - Text-based JSON action format with `<end_action>` token
    - Stop sequences: ``["<end_action>", "Observation:"]``
    - Default temperature: 0.5 (ARE llm_engine.py:17)
    - Default max_tokens: 16384 (ARE llm_engine.py:16)
    - Default max_iterations: 80 (ARE are_simulation_agent_config.py:36)
    - Invalid format retry: up to 10 times (ARE base_agent.py:347)
    - Iteration counter incremented EVERY loop (including errors) (ARE base_agent.py:849)
    - Terminates inner loop on send_message_to_user (TERMINATED) or
      wait_for_notification (PAUSED)
    - Max-iterations sends message to user via tool (ARE are_simulation.py:109-116)
    - Pre-step notification polling (ARE steps/are_simulation.py:26-62)
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        model: ModelAdapter,
        environment: Optional[Any] = None,
        llm_args: Optional[Dict[str, Any]] = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        invalid_format_retries: int = _DEFAULT_INVALID_FORMAT_RETRIES,
        simulated_generation_time_config: Optional["SimulatedGenerationTimeConfig"] = None,
        verbose: int = 0,
    ):
        """Initialize the agent.

        Args:
            tools: Dict of tool name -> callable
            model: ModelAdapter for LLM interactions
            environment: Optional Gaia2Environment for notification polling
            llm_args: Additional arguments for model calls, passed as kwargs
                to ``model.chat()``. Defaults (from ARE source):

                - ``temperature``: 0.5 (ARE llm_engine.py:17)
                - ``max_tokens``: 16384 (ARE llm_engine.py:16)
                - ``stop``: ``["<end_action>", "Observation:"]``

                **Stop-token handling:** Client-side stop-token truncation
                (ARE litellm_engine.py:126-127) is always applied to the
                response, regardless of whether ``stop`` is also passed to
                the API. When ``stop`` is passed, the API enforces it for
                efficiency (saves tokens, precise cutoff). When ``stop`` is
                ``None``, only client-side truncation runs — action parsing
                still works correctly.

                **None filtering:** Parameters set to ``None`` are omitted
                from the API call entirely. Use this to disable parameters
                the model provider rejects::

                    llm_args={"stop": None, "temperature": None}
            max_iterations: Maximum iterations before stopping. Default 80.
            invalid_format_retries: Max retries for invalid format. Default 10.
            simulated_generation_time_config: Optional config for simulated generation
                time. When set, the simulation is paused during LLM generation and
                resumed with a time offset. Default None (disabled).
                ARE agents/are_simulation_agent_config.py:28-30
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed)
        """
        self.tools = tools
        self.model = model
        self.environment = environment
        self.max_iterations = max_iterations
        self.invalid_format_retries = invalid_format_retries
        self.simulated_generation_time_config = simulated_generation_time_config
        self.verbose = verbose

        # Build system prompt with tool descriptions
        self.system_prompt = _build_system_prompt(tools, environment)

        # Apply default LLM args, allowing user overrides
        self.llm_args = {
            "temperature": _DEFAULT_TEMPERATURE,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "stop": _STOP_SEQUENCES,
            **(llm_args or {}),
        }

        # State
        self._messages: List[Dict[str, Any]] = []
        self._iteration_count = 0
        self._format_retry_count = 0
        self._terminated = False
        self._final_message: Optional[str] = None
        self._step_count = 0  # Counts all outputs: observations AND errors (ARE base_agent.py:450-451)

    def reset(self) -> None:
        """Reset agent state."""
        self._messages = []
        self._iteration_count = 0
        self._format_retry_count = 0
        self._terminated = False
        self._final_message = None
        self._step_count = 0

    def run(self, query: str) -> str:
        """Execute task and return final response.

        GAIA2 is event-driven: the real task instruction is delivered via the
        notification system (first ``send_message_to_agent`` event).  The outer
        turn loop (``_turn_loop``) drains the notification queue and formats
        user messages as ``[TASK]``, matching ARE's ``agent_loop()``.

        When ``query`` is non-empty (e.g. standalone use), it is prepended as
        a ``[TASK]`` message before entering the turn loop.

        Args:
            query: Task query/instructions (may be empty for GAIA2)

        Returns:
            Final text response from agent
        """
        # Match ARE's message format: [TASK]: \n{content}\n
        # ARE base_agent.py:96
        if query:
            self._messages.append({"role": "user", "content": f"[TASK]: \n{query}\n"})
        return self._turn_loop()

    def _pull_notifications(self) -> None:
        """Pull messages from the ARE notification system.

        Delegates to ``Gaia2Environment.poll_notifications()`` which drains
        the notification queue and returns pre-formatted strings.

        Matches ARE's pre-step notification polling behavior.
        ARE agents/default_agent/steps/are_simulation.py:26-62
        """
        if self.environment is None:
            return

        user_messages, env_notifications, _ = self.environment.poll_notifications()

        # Inject into message history matching ARE's format
        # ARE base_agent.py:107: "User messages updates:\n***\n{content}\n***\n"
        if user_messages:
            content = "\n".join(user_messages)
            self._messages.append({"role": "user", "content": f"User messages updates:\n***\n{content}\n***\n"})

        # ARE base_agent.py:110-112: "Environment notifications updates:\n***\n{content}\n***\n"
        if env_notifications:
            content = "\n".join(env_notifications)
            self._messages.append({"role": "user", "content": f"Environment notifications updates:\n***\n{content}\n***\n"})

    def _check_environment_stop(self) -> bool:
        """Check if the environment has sent a stop message.

        Uses a lightweight peek via the ARE notification system's
        ``has_environment_stop_message()`` when available, falling back to
        ``poll_notifications()``.

        ARE agents/default_agent/termination_methods/are_simulation.py:105-107

        Returns:
            True if environment has signaled stop
        """
        if self.environment is None:
            return False

        # Prefer the non-draining peek when available
        notification_system = self.environment.get_notification_system()
        if notification_system is not None:
            try:
                return notification_system.message_queue.has_environment_stop_message()
            except Exception:
                pass

        return False

    def _pause_env(self) -> None:
        """Pause the ARE environment before LLM generation.

        ARE base_agent.py:623-627
        """
        if self.environment is not None:
            self.environment.pause()

    def _resume_env(self, offset: float) -> None:
        """Resume the ARE environment after LLM generation with time offset.

        ARE base_agent.py:680-689

        Args:
            offset: Time in seconds to advance the simulation clock
        """
        if self.environment is not None:
            self.environment.resume_with_offset(offset)

    def _turn_loop(self) -> str:
        """Outer turn loop matching ARE's ``agent_loop()``.

        ARE ``are_simulation_main.py:230-326``: iterates over turns. Between
        turns, drains the notification queue, separates user messages (→
        ``[TASK]`` format) from environment notifications (→ re-queued for
        inner loop's pre-step), then runs the inner step loop.

        - ``send_message_to_user`` → TERMINATED: increments turn count
        - ``wait_for_notification`` → PAUSED: outer loop continues without
          incrementing turns

        Returns:
            Final text response
        """
        import logging

        from are.simulation.agents.default_agent.base_agent import RunningState  # type: ignore[import-not-found]

        logger = logging.getLogger(__name__)

        turn_count = 0
        max_turns = self._get_max_turns()

        while max_turns is None or turn_count < max_turns:
            if self.environment is not None:
                # Drain notification queue before each turn (including the first).
                # ARE are_simulation_main.py:272-274: get_notifications() is called
                # on every iteration. When the queue is empty the loop busy-waits
                # with sleep(1) until the first send_message_to_agent event has
                # been processed by the environment thread.
                user_messages, has_env, has_stop = self.environment.get_turn_notifications()

                if has_stop:
                    logger.warning("Environment stop message received in outer loop — stopping agent")
                    self._terminated = True
                    break

                if not user_messages and not has_env:
                    # No messages available, wait briefly
                    # ARE are_simulation_main.py:316-317
                    time.sleep(1)
                    continue

                # Format user messages as [TASK] for the new turn
                # ARE are_simulation_main.py:283, 361-365
                # ARE base_agent.py:96: "[TASK]: \n{content}\n"
                task = "\n".join(user_messages)
                self._messages.append({"role": "user", "content": f"[TASK]: \n{task}\n"})
            # else: no environment — skip notification handling and run
            # inner loop directly (standalone/testing mode)

            # Run inner step loop
            # ARE: react_agent.run(task=task, reset=reset) → execute_agent_loop()
            running_state = self._step_loop()

            if running_state == RunningState.TERMINATED:
                # ARE are_simulation_main.py:297-300: increment turn count
                turn_count += 1
                if self._terminated:
                    return self._final_message or ""
            elif running_state == RunningState.PAUSED:
                # ARE are_simulation_main.py:301-303: agent called
                # wait_for_notification, continue outer loop
                logger.debug("Agent paused (wait_for_notification), continuing outer loop")

        # Max turns reached
        # ARE are_simulation_main.py:319-320
        if max_turns is not None and turn_count >= max_turns and not self._terminated:
            logger.warning("Max turns (%d) reached — stopping agent", max_turns)

        return self._final_message or "Agent terminated without final message."

    def _get_max_turns(self) -> Optional[int]:
        """Get max turns from the scenario, matching ARE's ``scenario.nb_turns``.

        Returns:
            Number of turns, or None for unlimited
        """
        if self.environment is None:
            return None
        try:
            scenario = self.environment.get_scenario()
            return getattr(scenario, "nb_turns", None)
        except Exception:
            return None

    def _step_loop(self) -> "RunningState":
        """Inner step loop matching ARE's ``execute_agent_loop()``.

        ARE ``base_agent.py:775-854``: iterates over steps within a single
        turn. Terminates when the agent calls ``send_message_to_user``
        (TERMINATED) or ``wait_for_notification`` (PAUSED).

        Key behavior matching ARE:

        - Iteration counter incremented on EVERY loop iteration (including errors)
        - Format retries happen within a single iteration
        - Max-iterations sends message to user via the actual tool
        - Pre-step notification polling before each iteration
        - Environment stop message checked for termination
        - Simulated generation time: pause env before LLM, resume with offset after

        Returns:
            Running state: TERMINATED or PAUSED
        """
        from are.simulation.agents.default_agent.base_agent import RunningState  # type: ignore[import-not-found]

        while self._iteration_count < self.max_iterations and not self._terminated:
            # Check for environment stop BEFORE draining notifications.
            # ARE's execute_agent_loop checks termination_condition (which peeks
            # via has_environment_stop_message) in the while-condition, THEN runs
            # pre_step (which drains via get_by_timestamp). Reversing this order
            # causes the drain to consume ENVIRONMENT_STOP before the peek sees it.
            # ARE agents/default_agent/termination_methods/are_simulation.py:98-99
            # ARE agents/default_agent/base_agent.py:776-799
            if self._check_environment_stop():
                self._terminated = True
                return RunningState.TERMINATED

            # Pre-step: poll for notifications (matching ARE's pre-step)
            # ARE agents/default_agent/steps/are_simulation.py:26-62
            self._pull_notifications()

            try:
                # Build messages for LLM
                messages = [{"role": "system", "content": self.system_prompt}] + self._messages

                # Pause environment before LLM generation
                # ARE base_agent.py:623-627
                if self.simulated_generation_time_config is not None:
                    self._pause_env()

                # Call LLM with retry for invalid format
                # ARE base_agent.py:629-666
                content, completion_duration = self._call_llm_with_format_retry(messages)

                # Resume environment after LLM generation with time offset
                # ARE base_agent.py:680-689
                if self.simulated_generation_time_config is not None:
                    from are.simulation.agents.default_agent.base_agent import (  # type: ignore[import-not-found]
                        get_offset_from_time_config_mode,
                    )

                    offset = get_offset_from_time_config_mode(
                        time_config=self.simulated_generation_time_config,
                        completion_duration=completion_duration,
                    )
                    self._resume_env(offset)

                if content is None:
                    # All format retries exhausted
                    continue

                # Parse action from text
                parsed = _parse_action_from_text(content)
                if parsed is None:
                    # This shouldn't happen after format retry, but handle it
                    self._messages.append({"role": "assistant", "content": content})
                    error_msg = f"The LLM output was not formatted correctly: {content}"
                    # ARE base_agent.py:450-451: increment step for errors too
                    self._step_count += 1
                    self._messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"[OUTPUT OF STEP {self._step_count}] ERROR:\n***\n{error_msg}\n***\n\n"
                                "Now let's retry: take care not to repeat previous errors! "
                                "If you have retried several times, try a completely different approach.\n"
                            ),
                        }
                    )
                    continue

                thought, tool_name, tool_args = parsed
                self._step_count += 1

                # Add assistant message (Thought + Action)
                self._messages.append({"role": "assistant", "content": content})

                if self.verbose >= 1:
                    print(f"[Iteration {self._iteration_count}] Tool: {tool_name}")

                # Check for termination tools
                # ARE agents/default_agent/termination_methods/are_simulation.py:71-121
                if tool_name in _TERMINATION_TOOLS:
                    # send_message_to_user → TERMINATED
                    # ARE termination_methods/are_simulation.py:93-96
                    self._terminated = True
                    observation = self._execute_tool(tool_name, tool_args)
                    self._final_message = tool_args.get("content", str(observation)) if isinstance(tool_args, dict) else str(observation)
                    return RunningState.TERMINATED

                if tool_name == _PAUSE_TOOL:
                    # wait_for_notification → execute tool, add observation, PAUSED
                    # ARE termination_methods/are_simulation.py:34-36
                    observation = self._execute_tool(tool_name, tool_args)
                    self._messages.append(
                        {
                            "role": "user",
                            "content": f"[OUTPUT OF STEP {self._step_count}] Observation:\n***\n{observation}\n***\n",
                        }
                    )
                    return RunningState.PAUSED

                # Execute regular tool
                observation = self._execute_tool(tool_name, tool_args)

                # Add observation in ARE's format
                # ARE base_agent.py:102: "[OUTPUT OF STEP {i}] Observation:\n***\n{content}\n***\n"
                self._messages.append(
                    {
                        "role": "user",
                        "content": f"[OUTPUT OF STEP {self._step_count}] Observation:\n***\n{observation}\n***\n",
                    }
                )

            except Exception as e:
                # Match ARE error handling: log error, add to messages, continue
                # ARE base_agent.py:839-840, base_agent.py:105-106
                # ARE base_agent.py:450-451: increment step for errors too
                self._step_count += 1
                error_msg = str(e)
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[OUTPUT OF STEP {self._step_count}] ERROR:\n***\n{error_msg}\n***\n\n"
                            "Now let's retry: take care not to repeat previous errors! "
                            "If you have retried several times, try a completely different approach.\n"
                        ),
                    }
                )
            finally:
                # Safety resume: if environment is still paused due to an exception,
                # resume without advancing time to prevent deadlock.
                # ARE base_agent.py:841-848
                if self.simulated_generation_time_config is not None:
                    self._resume_env(0.0)
                # ARE increments iterations on EVERY loop iteration, including errors
                # ARE base_agent.py:849
                self._iteration_count += 1

        # Max iterations reached: send message to user via tool
        # ARE agents/default_agent/termination_methods/are_simulation.py:100-108
        if self._iteration_count >= self.max_iterations and not self._terminated:
            max_iter_msg = f"Max iterations ({self.max_iterations}) reached. Stopping."
            # Call the actual tool to record the event in the simulation
            if "AgentUserInterface__send_message_to_user" in self.tools:
                self._execute_tool("AgentUserInterface__send_message_to_user", {"content": max_iter_msg})
            self._terminated = True
            self._final_message = max_iter_msg
            return RunningState.TERMINATED

        return RunningState.TERMINATED

    def _call_llm_with_format_retry(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], float]:
        """Call LLM with retry for invalid format, matching ARE's behavior.

        ARE base_agent.py:629-666: retries until output contains Action: or Thought:,
        up to ``invalid_format_retries`` times.

        Args:
            messages: Messages to send to LLM

        Returns:
            Tuple of (LLM output text or None, completion_duration in seconds).
            completion_duration is the wall-clock time of the last successful LLM call.
        """
        format_try_count = 0
        content: Optional[str] = None
        completion_duration = 0.0

        while content is None or ("Action:" not in content and "Thought:" not in content):
            if content is not None:
                # Invalid format - add error and retry
                # ARE base_agent.py:642-650
                # ARE base_agent.py:450-451: increment step for errors too
                self._step_count += 1
                error_msg = f"The LLM output was not formatted correctly: {content}"
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[OUTPUT OF STEP {self._step_count}] ERROR:\n***\n{error_msg}\n***\n\n"
                            "Now let's retry: take care not to repeat previous errors! "
                            "If you have retried several times, try a completely different approach.\n"
                        ),
                    }
                )
                # Rebuild messages with the error
                messages = [{"role": "system", "content": self.system_prompt}] + self._messages

            call_start = time.monotonic()
            # Filter None values: allows users to disable params (e.g., stop=None for reasoning models)
            active_args = {k: v for k, v in self.llm_args.items() if v is not None}
            response = self.model.chat(messages=messages, **active_args)  # type: ignore[arg-type]
            completion_duration = time.monotonic() - call_start
            content = response.content or ""

            # Boolean replacement (ARE litellm_engine.py:125, hf_engine.py:152).
            # LLMs frequently output Python-style True/False in JSON blobs;
            # ARE normalizes to JSON-valid true/false before any parsing.
            content = content.replace("False", "false").replace("True", "true")

            # Client-side stop-token truncation (ARE litellm_engine.py:126-127).
            # Always applied as a universal fallback — works even when API-level
            # stop sequences are disabled (stop=None) for reasoning models.
            content = _apply_stop_truncation(content, _STOP_SEQUENCES)

            if self.verbose >= 2:
                print(f"[Iteration {self._iteration_count}, format try {format_try_count}] LLM output:\n{content}\n")

            format_try_count += 1
            # ARE base_agent.py:664-666: failsafe from infinite loop
            if format_try_count > self.invalid_format_retries:
                break

        # ARE base_agent.py:705-708: raise error after retries exhausted
        if content is None or ("Action:" not in content and "Thought:" not in content):
            return None, completion_duration

        return content, completion_duration

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any] | str) -> str:
        """Execute a tool call.

        Raises on errors (matching ARE's json_action_executor.py:197-227).
        Errors propagate to `_react_loop()` which formats them as ``ERROR:``
        messages, distinct from ``Observation:`` messages.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (dict or string)

        Returns:
            Tool execution result as string

        Raises:
            RuntimeError: If tool is not found or execution fails
        """
        # ARE json_action_executor.py:210-212: raises UnavailableToolAgentError
        if tool_name not in self.tools:
            raise RuntimeError(f"Error: unknown tool {tool_name}, should be instead one of {list(self.tools.keys())}.")

        # Normalize empty/falsy args to empty dict
        # (matches ARE json_action_executor.py:204)
        if not tool_args:
            tool_args = {}

        try:
            # Match original ARE behavior: string args passed as positional argument
            if isinstance(tool_args, str):
                result = self.tools[tool_name](tool_args)
            else:
                result = self.tools[tool_name](**tool_args)
            return str(result)
        except Exception as e:
            # ARE json_action_executor.py:224-227: raises JsonExecutionAgentError
            # with full tool description as a reminder
            tool = self.tools[tool_name]
            tool_desc = _get_tool_description_with_args(tool_name, tool)
            raise RuntimeError(
                f"Error in tool call execution: {e}\n"
                f"You should only use this tool with a correct input.\n"
                f"As a reminder, this tool's description is the following:\n"
                f"{tool_desc}"
            ) from e

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get message history.

        Returns:
            List of messages
        """
        return list(self._messages)

    @property
    def iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count

    @property
    def terminated(self) -> bool:
        """Get whether the agent has terminated."""
        return self._terminated


class DefaultGaia2AgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for DefaultGaia2Agent."""

    def __init__(self, agent: DefaultGaia2Agent, name: str = "gaia2_agent"):
        """Initialize the adapter.

        Args:
            agent: DefaultGaia2Agent instance
            name: Agent name for identification
        """
        super().__init__(agent, name)

    def _run_agent(self, query: str) -> str:
        """Run the agent and return answer.

        Args:
            query: Task query

        Returns:
            Agent's final answer
        """
        return self.agent.run(query)

    def get_messages(self) -> Any:
        """Get message history.

        Returns:
            MessageHistory object
        """
        from maseval.core.history import MessageHistory

        return MessageHistory(self.agent.get_messages())

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces.

        Returns:
            Trace dictionary
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "iteration_count": self.agent.iteration_count,
            "terminated": self.agent.terminated,
        }


class DefaultAgentGaia2Benchmark(Gaia2Benchmark):
    """Gaia2 benchmark with default agent implementation.

    Provides a ready-to-use benchmark matching ARE's reference agent behavior.
    Uses text-based ReAct format with JSON actions, matching ARE's implementation.

    Default parameters (matching ARE):
        - max_iterations: 80
        - temperature: 0.5
        - max_tokens: 16384
        - invalid_format_retries: 10

    Example:
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark, load_tasks

        tasks = load_tasks(capability="execution", limit=5)

        benchmark = DefaultAgentGaia2Benchmark(
            agent_data={"model_id": "gpt-4o"},
        )
        results = benchmark.run(tasks)
    """

    def __init__(self, agent_data: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Initialize benchmark.

        Args:
            agent_data: Agent configuration with:
                - model_id: Required model identifier
                - llm_args: Optional model call arguments (temperature, max_tokens, etc.)
                - max_iterations: Max iterations per task (default: 80)
                - invalid_format_retries: Max retries for invalid format (default: 10)
                - simulated_generation_time_config: Optional ``SimulatedGenerationTimeConfig``
                    for simulating LLM generation time in the simulation (default: None)
                - verbose: Verbosity level (default: 0)
            **kwargs: Additional Benchmark arguments
        """
        super().__init__(**kwargs)
        self._agent_data = agent_data or {}

    def _get_agent_model_id(self, agent_data: Dict[str, Any]) -> str:
        """Get agent model ID.

        Args:
            agent_data: Agent configuration

        Returns:
            Model ID string

        Raises:
            ValueError: If model_id not configured
        """
        model_id = agent_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Agent model_id not configured. Pass model_id in agent_data:\n\n"
                "    benchmark = DefaultAgentGaia2Benchmark(\n"
                "        agent_data={'model_id': 'gpt-4o'},\n"
                "    )"
            )
        return model_id

    def setup_agents(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create default Gaia2 agent.

        Args:
            agent_data: Agent configuration
            environment: Gaia2Environment with ARE tools
            task: Current task
            user: Optional user (always None)
            seed_generator: Seed generator for reproducibility

        Returns:
            Tuple of (agent list, agent dict)
        """
        # Merge class-level and run-level agent_data
        merged_data = {**self._agent_data, **agent_data}

        model_id = self._get_agent_model_id(merged_data)
        llm_args = merged_data.get("llm_args", {})
        max_iterations = merged_data.get("max_iterations", _DEFAULT_MAX_ITERATIONS)
        invalid_format_retries = merged_data.get("invalid_format_retries", _DEFAULT_INVALID_FORMAT_RETRIES)
        simulated_generation_time_config = merged_data.get("simulated_generation_time_config")
        if simulated_generation_time_config is None:
            from are.simulation.types import SimulatedGenerationTimeConfig  # type: ignore[import-not-found]

            simulated_generation_time_config = SimulatedGenerationTimeConfig(mode="measured")
        verbose = merged_data.get("verbose", 0)

        # Derive seed for agent model (returns None if seeding disabled)
        agent_seed = seed_generator.derive_seed("agents/gaia2_agent")

        tools = environment.create_tools()
        model = self.get_model_adapter(model_id, register_name="agent_model", seed=agent_seed)

        agent = DefaultGaia2Agent(
            tools=tools,  # type: ignore[arg-type]  # Gaia2GenericTool has __call__
            model=model,
            environment=environment,
            llm_args=llm_args,
            max_iterations=max_iterations,
            invalid_format_retries=invalid_format_retries,
            simulated_generation_time_config=simulated_generation_time_config,
            verbose=verbose,
        )

        adapter = DefaultGaia2AgentAdapter(agent, name="gaia2_agent")
        return [adapter], {"gaia2_agent": adapter}

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Get or create model adapter. Must be implemented by subclass.

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments (e.g., register_name)

        Returns:
            ModelAdapter instance
        """
        pass
