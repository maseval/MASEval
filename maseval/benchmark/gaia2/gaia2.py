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

import json
import re
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.callback import BenchmarkCallback
from maseval.core.seeding import SeedGenerator

from maseval.benchmark.gaia2.environment import Gaia2Environment
from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator


# =============================================================================
# Benchmark
# =============================================================================


class Gaia2Benchmark(Benchmark):
    """MASEval wrapper for Gaia2/ARE benchmark.

    Hybrid approach: Uses ARE for simulation and evaluation while providing
    MASEval orchestration, tracing, and agent flexibility.

    The ARE simulation runs internally; agents interact purely via tool calls.
    Time control happens through SystemApp.wait_for_notification().

    Subclasses must implement:
        - setup_agents(): Create agents for the task
        - get_model_adapter(): Provide model adapters
    """

    # Single-turn by default (ARE handles time internally via tools)
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


@dataclass
class Gaia2SimulatedGenerationTimeConfig:
    """Configuration for simulating the LLM's generation time.

    Matches ARE's ``SimulatedGenerationTimeConfig`` (ARE types.py:1574-1592).

    In "measured" mode, the simulation clock advances by the actual LLM
    generation wall-clock time. In "fixed" mode, it advances by a fixed
    number of seconds.

    Attributes:
        mode: "measured" uses actual LLM wall-clock time; "fixed" uses ``seconds``.
        seconds: Fixed offset in seconds. Used when mode is "fixed".
    """

    # ARE types.py:1586
    mode: Literal["fixed", "measured"] = "measured"
    # ARE types.py:1587
    seconds: Optional[float] = 1.0

    def __post_init__(self) -> None:
        # ARE types.py:1589-1592
        if self.mode == "fixed":
            if self.seconds is None:
                raise ValueError("When mode is 'fixed', seconds must be provided and cannot be None.")


def _get_offset_from_time_config_mode(
    time_config: Gaia2SimulatedGenerationTimeConfig,
    completion_duration: float,
) -> float:
    """Determine the time offset based on the config mode.

    Matches ARE's ``get_offset_from_time_config_mode`` (base_agent.py:990-1006).

    Args:
        time_config: Simulated generation time configuration
        completion_duration: Actual wall-clock LLM generation time in seconds

    Returns:
        Time offset in seconds to advance the simulation clock
    """
    if time_config.mode == "fixed" and time_config.seconds is not None:
        return time_config.seconds
    elif time_config.mode == "measured":
        return completion_duration
    else:
        raise ValueError(f"Invalid simulated_generation_time_config: {time_config}")


# Stop sequences for text-based action parsing
_STOP_SEQUENCES = ["<end_action>", "Observation:"]

# Termination tool names - agent terminates when these are called
_TERMINATION_TOOLS = frozenset(
    {
        "AgentUserInterface__send_message_to_user",
        "SystemApp__wait_for_notification",
    }
)


def _load_prompt_template(name: str) -> str:
    """Load a prompt template from file.

    Args:
        name: Template filename (without .txt extension)

    Returns:
        Template content as string
    """
    path = _PROMPT_TEMPLATES_DIR / f"{name}.txt"
    return path.read_text()


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

        start_time = environment.get_start_time()
        if start_time is None:
            return ""

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

    # Format environment instructions with tool descriptions
    environment_formatted = environment_template.format(tool_descriptions=tool_descriptions)

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
    return template.format(
        general_instructions=general,
        agent_instructions=agent,
        environment_instructions=environment_formatted,
    )


def _parse_json_blob(json_blob: str) -> Dict[str, Any]:
    """Parse JSON blob matching ARE's ``parse_json_blob()``.

    ARE agents/default_agent/tools/json_action_executor.py:33-57.
    Finds the first '{' and last '}' to handle nested JSON. Raises on failure
    (no lenient fallbacks — matching ARE's error propagation).

    Args:
        json_blob: String potentially containing JSON

    Returns:
        Parsed dict

    Raises:
        ValueError: If JSON parsing fails
    """
    try:
        first_brace = json_blob.find("{")
        last_brace_positions = [m.start() for m in re.finditer(r"}", json_blob)]
        if first_brace == -1 or not last_brace_positions:
            raise ValueError(f"No JSON object found in: {json_blob[:200]}")

        last_brace = last_brace_positions[-1]
        json_str = json_blob[first_brace : last_brace + 1]

        # Handle escaped quotes (ARE json_action_executor.py:38)
        json_str = json_str.replace('\\"', "'")

        # Handle triple quotes (ARE json_action_executor.py:42)
        json_str = re.sub(r'"""(.*?)"""', r"'\1'", json_str, flags=re.DOTALL)

        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        # ARE json_action_executor.py:45-55: raises JsonParsingAgentError
        place = e.pos or 0
        if json_str[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            ) from e
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_str}, decoding failed on that specific part of the blob:\n"
            f"'{json_str[place - 4 : place + 5]}'."
        ) from e
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}") from e


def _parse_action_from_text(text: str) -> Optional[Tuple[str, str, Dict[str, Any] | str]]:
    """Parse Thought and Action from LLM text output.

    Matches ARE's ``JsonActionExecutor.extract_action()`` and ``parse_json_tool_call()``.

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

    # Remove markdown code fences (ARE json_action_executor.py:61)
    action_text = action_text.replace("```json", "").replace("```", "")

    # Parse JSON (raises ValueError on failure, matching ARE)
    try:
        action_data = _parse_json_blob(action_text)
    except ValueError:
        return None

    tool_name = action_data.get("action")
    if tool_name is None:
        return None
    tool_name = str(tool_name)

    # ARE json_action_executor.py:70: missing action_input defaults to ""
    tool_args = action_data.get("action_input", "")

    return (thought, tool_name, tool_args)


class DefaultGaia2Agent:
    """Default agent implementation for Gaia2 benchmark.

    ReAct-style agent matching ARE's reference implementation. Uses text-based
    action parsing (Thought/Action/Observation cycle) rather than native
    function calling.

    Key characteristics matching ARE (base_agent.py, are_simulation.py):

    - Text-based JSON action format with `<end_action>` token
    - Stop sequences: ``["<end_action>", "Observation:"]``
    - Default temperature: 0.5 (ARE llm_engine.py:17)
    - Default max_tokens: 16384 (ARE llm_engine.py:16)
    - Default max_iterations: 80 (ARE are_simulation_agent_config.py:36)
    - Invalid format retry: up to 10 times (ARE base_agent.py:347)
    - Iteration counter incremented EVERY loop (including errors) (ARE base_agent.py:849)
    - Terminates on send_message_to_user or wait_for_notification
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
        simulated_generation_time_config: Optional[Gaia2SimulatedGenerationTimeConfig] = None,
        verbose: int = 0,
    ):
        """Initialize the agent.

        Args:
            tools: Dict of tool name -> callable
            model: ModelAdapter for LLM interactions
            environment: Optional Gaia2Environment for notification polling
            llm_args: Additional arguments for model calls. Defaults are applied
                for temperature (0.5), max_tokens (16384), and stop sequences.
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
        self._step_count = 0  # Counts successful tool executions (for observation formatting)

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
        notification system (first ``send_message_to_agent`` event), polled in
        ``_pull_notifications()`` at the start of each iteration. When the
        query is empty (normal for GAIA2), we skip the ``[TASK]`` message to
        avoid wasting a turn on an empty prompt.

        Args:
            query: Task query/instructions (may be empty for GAIA2)

        Returns:
            Final text response from agent
        """
        # Match ARE's message format: [TASK]: \n{content}\n
        # ARE base_agent.py:97
        if query:
            self._messages.append({"role": "user", "content": f"[TASK]: \n{query}\n"})
        return self._react_loop()

    def _pull_notifications(self) -> None:
        """Pull messages from the ARE notification system.

        Matches ARE's pre-step notification polling behavior.
        ARE agents/default_agent/steps/are_simulation.py:26-62
        """
        if self.environment is None:
            return

        notification_system = self.environment.get_notification_system()
        if notification_system is None:
            return

        try:
            from datetime import datetime, timezone

            timestamp = datetime.now(tz=timezone.utc)
            unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

            if not unhandled:
                return

            # Separate user messages from environment notifications
            # ARE steps/are_simulation.py:34-61
            user_messages = []
            env_notifications = []
            for notif in unhandled:
                msg_type = getattr(notif, "message_type", None)
                if msg_type is not None:
                    # Import MessageType at call time to avoid import errors
                    from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

                    if msg_type == MessageType.USER_MESSAGE:
                        user_messages.append(notif.message)
                    elif msg_type == MessageType.ENVIRONMENT_NOTIFICATION:
                        ts = notif.timestamp.strftime("%Y-%m-%d %H:%M:%S") if notif.timestamp else ""
                        env_notifications.append(f"[{ts}] {notif.message}")

            # Inject into message history matching ARE's format
            # ARE base_agent.py:107: "User messages updates:\n***\n{content}\n***\n"
            if user_messages:
                content = "\n".join(user_messages)
                self._messages.append({"role": "user", "content": f"User messages updates:\n***\n{content}\n***\n"})

            # ARE base_agent.py:110-112: "Environment notifications updates:\n***\n{content}\n***\n"
            if env_notifications:
                content = "\n".join(env_notifications)
                self._messages.append({"role": "user", "content": f"Environment notifications updates:\n***\n{content}\n***\n"})
        except Exception:
            pass  # Notification polling is best-effort

    def _check_environment_stop(self) -> bool:
        """Check if the environment has sent a stop message.

        Matches ARE's termination_condition_are_simulation check.
        ARE agents/default_agent/termination_methods/are_simulation.py:105-107

        Returns:
            True if environment has signaled stop
        """
        if self.environment is None:
            return False

        notification_system = self.environment.get_notification_system()
        if notification_system is None:
            return False

        try:
            return notification_system.message_queue.has_environment_stop_message()
        except Exception:
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

    def _react_loop(self) -> str:
        """ReAct loop: Thought -> Action -> Observation -> repeat.

        Matches ARE's ``execute_agent_loop()`` and ``step()`` methods.
        Key behavior matching ARE (base_agent.py:776-854):

        - Iteration counter incremented on EVERY loop iteration (including errors)
        - Format retries happen within a single iteration
        - Max-iterations sends message to user via the actual tool
        - Pre-step notification polling before each iteration
        - Environment stop message checked for termination
        - Simulated generation time: pause env before LLM, resume with offset after

        Returns:
            Final text response
        """
        while self._iteration_count < self.max_iterations and not self._terminated:
            # Pre-step: poll for notifications (matching ARE's pre-step)
            # ARE agents/default_agent/steps/are_simulation.py:26-62
            self._pull_notifications()

            # Check for environment stop message
            # ARE agents/default_agent/termination_methods/are_simulation.py:105-107
            if self._check_environment_stop():
                self._terminated = True
                break

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
                    offset = _get_offset_from_time_config_mode(
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
                # ARE agents/default_agent/termination_methods/are_simulation.py:76-118
                if tool_name in _TERMINATION_TOOLS:
                    self._terminated = True

                    # Execute the termination tool
                    observation = self._execute_tool(tool_name, tool_args)

                    # For send_message_to_user, capture the message
                    if tool_name == "AgentUserInterface__send_message_to_user":
                        self._final_message = tool_args.get("content", str(observation)) if isinstance(tool_args, dict) else str(observation)
                        return self._final_message

                    # For wait_for_notification, return the observation
                    return str(observation)

                # Execute tool
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
        # ARE agents/default_agent/termination_methods/are_simulation.py:109-116
        if self._iteration_count >= self.max_iterations and not self._terminated:
            max_iter_msg = f"Max iterations ({self.max_iterations}) reached. Stopping."
            # Call the actual tool to record the event in the simulation
            if "AgentUserInterface__send_message_to_user" in self.tools:
                self._execute_tool("AgentUserInterface__send_message_to_user", {"content": max_iter_msg})
            self._terminated = True
            return max_iter_msg

        return self._final_message or "Agent terminated without final message."

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
            response = self.model.chat(messages=messages, **self.llm_args)  # type: ignore[arg-type]
            completion_duration = time.monotonic() - call_start
            content = response.content or ""

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

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (dict or string)

        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        try:
            # Match original ARE behavior: string args passed as positional argument
            if isinstance(tool_args, str):
                result = self.tools[tool_name](tool_args)
            else:
                result = self.tools[tool_name](**tool_args)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

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
                - simulated_generation_time_config: Optional Gaia2SimulatedGenerationTimeConfig
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
