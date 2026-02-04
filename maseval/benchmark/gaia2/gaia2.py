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
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
        return Gaia2Environment(task_data=task.environment_data)

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
            - event_results: Per-event results

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

# ARE default parameters
_DEFAULT_MAX_ITERATIONS = 80
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_MAX_TOKENS = 16384
_DEFAULT_INVALID_FORMAT_RETRIES = 10

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
    """Build tool descriptions for embedding in system prompt.

    Args:
        tools: Dict of tool name -> callable

    Returns:
        Formatted tool descriptions string
    """
    descriptions = []
    for name, tool in tools.items():
        desc = getattr(tool, "description", "") or ""
        inputs = getattr(tool, "inputs", {}) or {}

        # Format parameters
        params = []
        properties = inputs.get("properties", {})
        required = set(inputs.get("required", []))

        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            req_marker = " (required)" if param_name in required else " (optional)"
            params.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

        params_str = "\n".join(params) if params else "    (no parameters)"

        descriptions.append(f"Tool: {name}\nDescription: {desc}\nParameters:\n{params_str}")

    return "\n\n".join(descriptions)


def _build_system_prompt(tools: Dict[str, Callable]) -> str:
    """Build the full system prompt with tool descriptions.

    Args:
        tools: Dict of tool name -> callable

    Returns:
        Complete system prompt string
    """
    # Load templates
    general = _load_prompt_template("general_instructions")
    agent = _load_prompt_template("agent_instructions")
    environment = _load_prompt_template("environment_instructions")
    template = _load_prompt_template("system_prompt")

    # Build tool descriptions
    tool_descriptions = _build_tool_descriptions(tools)

    # Format environment instructions with tool descriptions
    environment_formatted = environment.format(tool_descriptions=tool_descriptions)

    # Assemble full prompt
    return template.format(
        general_instructions=general,
        agent_instructions=agent,
        environment_instructions=environment_formatted,
    )


def _parse_json_blob(json_blob: str) -> Optional[Dict[str, Any]]:
    """Parse JSON blob using the same approach as original ARE.

    Finds the first '{' and last '}' to correctly handle nested JSON.

    Args:
        json_blob: String potentially containing JSON

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        first_brace = json_blob.find("{")
        if first_brace == -1:
            return None

        # Find all closing braces and use the last one (handles nested JSON)
        brace_positions = [m.start() for m in re.finditer(r"}", json_blob)]
        if not brace_positions:
            return None

        last_brace = brace_positions[-1]
        json_str = json_blob[first_brace : last_brace + 1]

        # Handle escaped quotes
        json_str = json_str.replace('\\"', "'")

        # Handle triple quotes
        json_str = re.sub(r'"""(.*?)"""', r"'\1'", json_str, flags=re.DOTALL)

        return json.loads(json_str, strict=False)
    except json.JSONDecodeError:
        # Try to fix common issues
        try:
            # Remove trailing commas
            fixed = re.sub(r",\s*}", "}", json_str)
            fixed = re.sub(r",\s*]", "]", fixed)
            return json.loads(fixed, strict=False)
        except (json.JSONDecodeError, UnboundLocalError):
            return None
    except Exception:
        return None


def _parse_action_from_text(text: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Parse Thought and Action from LLM text output.

    Expected format:
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

    # Parse JSON using the robust method (matching original ARE)
    action_data = _parse_json_blob(action_text)
    if action_data is None:
        return None

    tool_name = action_data.get("action", "")
    tool_args = action_data.get("action_input", {})

    # Handle string action_input - pass through as-is (matching original ARE behavior)
    # Original ARE passes string args directly to tools, we convert to dict with single arg
    if isinstance(tool_args, str):
        # Keep string as-is for tools that accept positional string args
        pass

    return (thought, tool_name, tool_args)


class DefaultGaia2Agent:
    """Default agent implementation for Gaia2 benchmark.

    ReAct-style agent matching ARE's reference implementation. Uses text-based
    action parsing (Thought/Action/Observation cycle) rather than native
    function calling.

    Key characteristics matching ARE:
        - Text-based JSON action format with <end_action> token
        - Stop sequences: ["<end_action>", "Observation:"]
        - Default temperature: 0.5
        - Default max_tokens: 16384
        - Default max_iterations: 80
        - Invalid format retry: up to 10 times
        - Terminates on send_message_to_user or wait_for_notification
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        model: ModelAdapter,
        llm_args: Optional[Dict[str, Any]] = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        invalid_format_retries: int = _DEFAULT_INVALID_FORMAT_RETRIES,
        verbose: int = 0,
    ):
        """Initialize the agent.

        Args:
            tools: Dict of tool name -> callable
            model: ModelAdapter for LLM interactions
            llm_args: Additional arguments for model calls. Defaults are applied
                for temperature (0.5), max_tokens (16384), and stop sequences.
            max_iterations: Maximum iterations before stopping. Default 80.
            invalid_format_retries: Max retries for invalid format. Default 10.
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed)
        """
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.invalid_format_retries = invalid_format_retries
        self.verbose = verbose

        # Build system prompt with tool descriptions
        self.system_prompt = _build_system_prompt(tools)

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

    def reset(self) -> None:
        """Reset agent state."""
        self._messages = []
        self._iteration_count = 0
        self._format_retry_count = 0
        self._terminated = False
        self._final_message = None

    def run(self, query: str) -> str:
        """Execute task and return final response.

        Args:
            query: Task query/instructions

        Returns:
            Final text response from agent
        """
        self._messages.append({"role": "user", "content": query})
        return self._react_loop()

    def _react_loop(self) -> str:
        """ReAct loop: Thought -> Action -> Observation -> repeat.

        Returns:
            Final text response
        """
        while self._iteration_count < self.max_iterations and not self._terminated:
            # Build messages for LLM
            messages = [{"role": "system", "content": self.system_prompt}] + self._messages

            # Call LLM (text completion, no tools parameter)
            response = self.model.chat(messages=messages, **self.llm_args)  # type: ignore[arg-type]
            content = response.content or ""

            if self.verbose >= 2:
                print(f"[Iteration {self._iteration_count + 1}] LLM output:\n{content}\n")

            # Parse action from text
            parsed = _parse_action_from_text(content)

            if parsed is None:
                # Invalid format - retry
                self._format_retry_count += 1
                if self._format_retry_count >= self.invalid_format_retries:
                    self._messages.append({"role": "assistant", "content": content})
                    return f"Failed to parse action after {self.invalid_format_retries} retries. Last output: {content}"

                # Add error observation and retry
                error_msg = (
                    "Error: Invalid action format. Please use the correct format:\n"
                    "Thought: [your reasoning]\n\n"
                    "Action:\n"
                    '{"action": "tool_name", "action_input": {...}}<end_action>'
                )
                self._messages.append({"role": "assistant", "content": content})
                self._messages.append({"role": "user", "content": f"Observation: {error_msg}"})
                continue

            thought, tool_name, tool_args = parsed
            self._iteration_count += 1
            self._format_retry_count = 0  # Reset on successful parse

            # Add assistant message (Thought + Action)
            self._messages.append({"role": "assistant", "content": content})

            if self.verbose >= 1:
                print(f"[Iteration {self._iteration_count}] Tool: {tool_name}")

            # Check for termination tools
            if tool_name in _TERMINATION_TOOLS:
                self._terminated = True

                # Execute the termination tool
                observation = self._execute_tool(tool_name, tool_args)

                # For send_message_to_user, capture the message
                if tool_name == "AgentUserInterface__send_message_to_user":
                    self._final_message = tool_args.get("content", str(observation))
                    return self._final_message

                # For wait_for_notification, return the observation
                return str(observation)

            # Execute tool
            observation = self._execute_tool(tool_name, tool_args)

            # Add observation
            self._messages.append({"role": "user", "content": f"Observation: {observation}"})

        if self._iteration_count >= self.max_iterations:
            return f"Max iterations ({self.max_iterations}) reached."

        return self._final_message or "Agent terminated without final message."

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
        verbose = merged_data.get("verbose", 0)

        # Derive seed for agent model (returns None if seeding disabled)
        agent_seed = seed_generator.derive_seed("agents/gaia2_agent")

        tools = environment.create_tools()
        model = self.get_model_adapter(model_id, register_name="agent_model", seed=agent_seed)

        agent = DefaultGaia2Agent(
            tools=tools,  # type: ignore[arg-type]  # AREToolWrapper is Callable
            model=model,
            llm_args=llm_args,
            max_iterations=max_iterations,
            invalid_format_retries=invalid_format_retries,
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
