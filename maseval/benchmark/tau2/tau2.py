"""Tau 2 Benchmark - Main Implementation.

Framework-agnostic implementation of the tau2-bench benchmark for evaluating
LLM-based agents on customer service tasks across multiple domains.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Reference Paper: "Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
https://arxiv.org/abs/2406.12045

Usage:
    from maseval.benchmark.tau2 import (
        Tau2Benchmark, Tau2Environment, Tau2Evaluator, Tau2User,
        load_tasks, configure_model_ids,
    )

    # Load data and configure model IDs
    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(
        tasks,
        user_model_id="gpt-4o",
        evaluator_model_id="gpt-4o",  # Optional - only for NL assertions
    )

    # Create your framework-specific benchmark subclass
    class MyTau2Benchmark(Tau2Benchmark):
        def setup_agents(self, agent_data, environment, task, user, seed_generator):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            # Create and optionally register model adapters
            seed = kwargs.get("seed")  # Extract seed for reproducibility
            adapter = MyModelAdapter(model_id, seed=seed)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyTau2Benchmark(agent_data={})
    results = benchmark.run(tasks)

Default Agent Implementation:
    For comparison with the original tau2-bench results, use DefaultAgentTau2Benchmark
    which implements their agent logic exactly:

    from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark, load_tasks, configure_model_ids

    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(tasks, user_model_id="gpt-4o")

    benchmark = DefaultAgentTau2Benchmark(
        agent_data={"model_id": "gpt-4o"},
    )
    results = benchmark.run(tasks)
"""

import inspect
import json
import textwrap
from abc import abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

from pydantic import BaseModel, Field, create_model

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.callback import BenchmarkCallback
from maseval.core.exceptions import UserExhaustedError
from maseval.core.seeding import DefaultSeedGenerator, SeedGenerator

from maseval.benchmark.tau2.environment import Tau2Environment
from maseval.benchmark.tau2.evaluator import Tau2Evaluator

# Initial greeting from the original tau2-bench orchestrator.
# The orchestrator adds this as the first AssistantMessage in the trajectory.
# The user simulator sees it; the agent does not.
# Source: tau2-bench orchestrator.py:L34-36
INITIAL_GREETING = "Hi! How can I help you today?"


# =============================================================================
# User Simulator
# =============================================================================


class Tau2User(User):
    """Tau2-specific user simulator matching original tau2-bench UserSimulator.

    Uses chat API with role-flipped messages, matching the original's architecture:
    - System message: simulation_guidelines + scenario
    - Messages: role-flipped (user->assistant, assistant->user) matching
      original's UserState.flip_roles()
    - Tools: native OpenAI function calling for user tools
    - Stop: exact case match for ###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###
      (tokens kept in content, skipped if message has tool_calls)

    Adapted from: tau2-bench src/tau2/user/user_simulator.py
    """

    STOP = "###STOP###"
    TRANSFER = "###TRANSFER###"
    OUT_OF_SCOPE = "###OUT-OF-SCOPE###"

    GUIDELINES_DIR = Path(__file__).parent / "prompt_templates"

    SYSTEM_PROMPT_TEMPLATE = """\
{global_user_sim_guidelines}

<scenario>
{instructions}
</scenario>"""

    def __init__(
        self,
        model: ModelAdapter,
        scenario: str,
        initial_query: str,
        tools: Optional[Dict[str, Callable]] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        llm_args: Optional[Dict[str, Any]] = None,
        max_turns: int = 50,
        exhausted_response: Optional[str] = None,
    ):
        """Initialize Tau2 user simulator.

        Args:
            model: ModelAdapter for LLM-based response generation
            scenario: Full scenario text containing user instructions
            initial_query: The initial query to the agent
            tools: Optional dictionary of user tools (name -> callable)
            tool_definitions: Optional OpenAI-format tool definitions for LLM
            llm_args: Optional additional args for model.chat() (e.g. temperature)
            max_turns: Maximum conversation turns
            exhausted_response: Message to return when ``respond()`` is called
                after the user is done. If ``None`` (default), raises
                ``UserExhaustedError`` instead.
        """
        self.model = model
        self.scenario = scenario
        self._initial_query = initial_query
        self.tools = tools or {}
        self.tool_definitions = tool_definitions
        self.llm_args = llm_args or {}
        self.max_turns = max_turns
        self.exhausted_response = exhausted_response

        # Load guidelines matching original tau2-bench
        use_tools = bool(self.tools)
        if use_tools:
            guidelines_path = self.GUIDELINES_DIR / "simulation_guidelines_tools.md"
        else:
            guidelines_path = self.GUIDELINES_DIR / "simulation_guidelines.md"
        guidelines = guidelines_path.read_text()

        # Build system prompt matching original format
        self._system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            global_user_sim_guidelines=guidelines,
            instructions=self.scenario,
        )

        # Message history in original role format (before flipping)
        # Contains: {"role": "user"/"assistant"/"tool", "content": ..., ...}
        self._messages: List[Dict[str, Any]] = []
        self._turn_count = 0
        self._stopped = False

    def get_initial_query(self) -> str:
        """Return the initial query to start the conversation."""
        return self._initial_query

    def respond(self, message: str) -> str:
        """Respond to an agent message.

        Matches original tau2-bench UserSimulator._generate_next_message:
        1. Add agent message to history (as AssistantMessage)
        2. Flip roles and generate via model.chat()
        3. If tool_calls: execute, add results, generate again
        4. Return final text response (with stop tokens kept in content)

        Args:
            message: The agent's message

        Returns:
            The user's response text
        """
        self._turn_count += 1
        if self._stopped or self._turn_count > self.max_turns:
            self._stopped = True
            self._last_respond_steps = 0
            if self.exhausted_response is not None:
                return self.exhausted_response
            raise UserExhaustedError(
                f"Tau2User has no more turns (max_turns={self.max_turns}, turn_count={self._turn_count}, stopped={self._stopped})",
                component="user",
            )

        # Add agent's message as assistant message
        self._messages.append({"role": "assistant", "content": message})

        # Generate response (may loop for tool calls)
        # C8: Track steps (messages added during generation) for step counting
        pre_count = len(self._messages)
        result = self._generate_response()
        self._last_respond_steps = len(self._messages) - pre_count
        return result

    def _generate_response(self) -> str:
        """Generate user response via chat API with role flipping.

        Matches original's generate flow:
        - Flip roles (user<->assistant)
        - Call model.chat() with system prompt + flipped messages + tools
        - If tool_calls returned: execute, feed results, generate again
        - If text returned: create user message and return
        """
        while True:
            # Build flipped messages for the LLM
            flipped = self._flip_roles()
            messages = [{"role": "system", "content": self._system_prompt}] + flipped

            # Generate via model.chat with tools (native function calling)
            # H2: tool_choice="auto" matching original llm_utils.py
            response = self.model.chat(
                messages=messages,
                tools=self.tool_definitions,
                tool_choice="auto" if self.tool_definitions else None,
                **self.llm_args,
            )

            content = response.content or ""
            tool_calls = response.tool_calls or []

            if tool_calls:
                # User wants to make tool calls
                # Add as user message with tool_calls (original format)
                self._messages.append(
                    {
                        "role": "user",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )

                # Execute each tool call and add results
                for tc in tool_calls:
                    tool_result = self._execute_tool(tc)
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": _to_json_str(tool_result),
                            "requestor": "user",
                        }
                    )

                # Continue loop to generate next response
                continue
            else:
                # Text response - add to history and return
                # Original keeps stop tokens in content (no stripping)
                self._messages.append({"role": "user", "content": content})

                # Check stop (exact case match, skip if tool_calls)
                # Matching original: STOP in message.content, returns False if is_tool_call()
                if self.STOP in content or self.TRANSFER in content or self.OUT_OF_SCOPE in content:
                    self._stopped = True

                return content

    def _flip_roles(self) -> List[Dict[str, Any]]:
        """Flip message roles matching original's UserState.flip_roles().

        Original behavior:
        - UserMessage -> AssistantMessage (with tool_calls if present)
        - AssistantMessage -> UserMessage (only if NOT a tool call)
        - ToolMessage with requestor="user" -> ToolMessage (kept)
        - AssistantMessage with tool_calls -> SKIPPED (raises error in original)
        """
        flipped = []
        for msg in self._messages:
            role = msg.get("role")
            if role == "user":
                # User -> Assistant (for the LLM, user plays assistant role)
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                }
                if "tool_calls" in msg:
                    entry["tool_calls"] = msg["tool_calls"]
                flipped.append(entry)
            elif role == "assistant":
                # Assistant -> User (only non-tool-call messages)
                # Original skips assistant messages that have tool_calls
                if not msg.get("tool_calls"):
                    flipped.append(
                        {
                            "role": "user",
                            "content": msg.get("content", ""),
                        }
                    )
            elif role == "tool":
                # Keep tool messages for user tool results only
                if msg.get("requestor") == "user":
                    flipped.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "content": msg.get("content", ""),
                        }
                    )
        return flipped

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a user tool call.

        Args:
            tool_call: Tool call dict in OpenAI format

        Returns:
            Tool execution result
        """
        if "function" in tool_call:
            name = tool_call["function"].get("name", "")
            arguments = tool_call["function"].get("arguments", {})
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

        # As in environment.py:get_response, tool errors return "Error: ..." strings (not exceptions)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid or missing arguments for tool '{name}'"

        if name not in self.tools:
            return f"Error: Tool '{name}' not found"

        try:
            return self.tools[name](**arguments)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def is_done(self) -> bool:
        """Check if the user interaction should terminate."""
        return self._stopped or self._turn_count >= self.max_turns

    def inject_greeting(self, greeting: str) -> None:
        """Inject the agent's initial greeting into message history.

        Must be called AFTER get_initial_query() returns.
        In the original tau2-bench, the orchestrator adds
        "Hi! How can I help you today?" as the first AssistantMessage
        before the user's initial query.

        Args:
            greeting: The greeting message to inject
        """
        self._messages.insert(0, {"role": "assistant", "content": greeting})

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces with Tau2-specific information."""
        return {
            "type": type(self).__name__,
            "max_turns": self.max_turns,
            "turns_used": self._turn_count,
            "stopped_by_user": self._stopped,
            "messages": list(self._messages),
        }


# =============================================================================
# Benchmark
# =============================================================================


class Tau2Benchmark(Benchmark):
    """Tau2 Benchmark - Framework-agnostic base class.

    This base class handles:
    - Environment setup with Tau2Environment (real tools)
    - Deterministic evaluation via database state comparison
    - Optional user simulation for multi-turn tasks

    Users must subclass and implement:
    - setup_agents() for their agent framework
    - get_model_adapter() to provide model adapters

    Model IDs for components are read from task data:
    - task.user_data["model_id"] for user simulator
    - task.evaluation_data["model_id"] for NL assertion evaluator (optional)

    Use configure_model_ids() to set these values after loading tasks.

    Example:
        class MyTau2Benchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                # Setup your agents here
                ...

            def get_model_adapter(self, model_id, **kwargs):
                seed = kwargs.get("seed")  # Extract seed for reproducibility
                return MyModelAdapter(model_id, seed=seed)

        tasks = load_tasks("retail")
        configure_model_ids(tasks, user_model_id="gpt-4o")

        benchmark = MyTau2Benchmark()
        benchmark.run(tasks)
    """

    # C8: Maximum steps matching original DEFAULT_MAX_STEPS=200 (config.py)
    MAX_INVOCATIONS = 200

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
        seed_generator: Optional[SeedGenerator] = None,
    ):
        """Initialize benchmark with tau2-specific defaults.

        Args:
            callbacks: Optional list of callback handlers for monitoring execution.
            n_task_repeats: Number of times to repeat each task. Default 1.
            max_invocations: Maximum steps (default: 200, matching original DEFAULT_MAX_STEPS).
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

    def _get_user_model_id(self, task: Task) -> str:
        """Get user simulator model ID from task.user_data.

        Raises:
            ValueError: If model_id not configured in task.user_data
        """
        model_id = task.user_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "User simulator model_id not configured in task.user_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.tau2 import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('retail')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        user_model_id='gpt-4o',\n"
                "    )"
            )
        return model_id

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator,
    ) -> Tau2Environment:
        """Create environment for a task.

        Creates a Tau2Environment with real tool implementations
        for the task's domain.

        Args:
            agent_data: Agent configuration
            task: Current task

        Returns:
            Tau2Environment instance
        """
        return Tau2Environment(task_data=task.environment_data)

    def setup_user(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        seed_generator: DefaultSeedGenerator,
    ) -> Optional[User]:
        """Create Tau2 user simulator.

        Creates a Tau2User with scenario from the task.
        Model ID is read from task.user_data["model_id"].

        Scenario text is formatted to match original tau2-bench's
        ``str(task.user_scenario)`` chain:
        - ``StructuredUserInstructions.__str__()`` for dict instructions
        - ``UserScenario.__str__()`` wrapping persona + instructions

        Args:
            agent_data: Agent configuration
            environment: The task environment
            task: Current task with user scenario

        Returns:
            Tau2User instance
        """
        # Build scenario matching original tau2-bench str(task.user_scenario)
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        # Format instructions matching StructuredUserInstructions.__str__()
        if isinstance(instructions, str):
            instructions_str = instructions
        elif isinstance(instructions, dict):
            tab = "\t"
            lines = []
            if instructions.get("domain"):
                lines.append(f"Domain: {instructions['domain']}")
            if instructions.get("reason_for_call"):
                lines.append(f"Reason for call:\n{textwrap.indent(instructions['reason_for_call'], tab)}")
            if instructions.get("known_info") is not None:
                lines.append(f"Known info:\n{textwrap.indent(instructions['known_info'], tab)}")
            if instructions.get("unknown_info") is not None:
                lines.append(f"Unknown info:\n{textwrap.indent(instructions['unknown_info'], tab)}")
            if instructions.get("task_instructions"):
                lines.append(f"Task instructions:\n{textwrap.indent(instructions['task_instructions'], tab)}")
            instructions_str = "\n".join(lines)
        else:
            instructions_str = ""

        # Format scenario matching UserScenario.__str__()
        scenario_lines = []
        persona = user_data.get("persona")
        if persona is not None:
            scenario_lines.append("Persona:")
            scenario_lines.append(textwrap.indent(persona, "\t"))
        scenario_lines.append("Instructions:")
        scenario_lines.append(textwrap.indent(instructions_str, "\t"))
        scenario = "\n".join(scenario_lines)

        user_model_id = self._get_user_model_id(task)

        # Derive seed for user simulator (returns None if seeding disabled)
        sim_gen = seed_generator.child("simulators")
        user_seed = sim_gen.derive_seed("user")

        # Get user tools from environment
        user_tools = environment.create_user_tools()

        # Build OpenAI-format tool definitions for the LLM
        # Matches original tau2-bench Tool.openai_schema format
        tool_definitions = _build_tool_definitions(user_tools) if user_tools else None

        # D19: Default temperature=0.0 matching original tau2-bench
        # D20: Disable Claude thinking for Claude models
        user_llm_args: Dict[str, Any] = {"temperature": 0.0}
        if user_model_id.startswith("claude"):
            user_llm_args["thinking"] = {"type": "disabled"}

        return Tau2User(
            model=self.get_model_adapter(
                user_model_id,
                register_name="user_simulator",
                seed=user_seed,
            ),
            scenario=scenario,
            initial_query=task.query,
            tools=user_tools,
            tool_definitions=tool_definitions,
            llm_args=user_llm_args,
        )

    @abstractmethod
    def setup_agents(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
        seed_generator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration
            environment: Tau2Environment with real tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(  # type: ignore[override]
        self,
        environment: Tau2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator,
    ) -> Sequence[Evaluator]:
        """Create evaluator for the task.

        Creates a Tau2Evaluator with optional NL assertion model.
        NL model ID is read from task.evaluation_data["model_id"].

        Args:
            environment: Tau2Environment instance
            task: Current task with evaluation criteria
            agents: Agent instances
            user: Optional user simulator

        Returns:
            List with single Tau2Evaluator instance
        """
        # C6: Create NL assertion model if configured
        nl_model = None
        nl_model_id = task.evaluation_data.get("model_id")
        if nl_model_id:
            nl_model = self.get_model_adapter(nl_model_id, register_name="nl_evaluator")

        return [
            Tau2Evaluator(
                task=task,
                environment=environment,
                nl_model=nl_model,
            )
        ]

    def run_agents(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Tau2Environment,
        query: str = "",
    ) -> Any:
        """Execute agents and return final answer.

        Args:
            agents: Agent instances to run
            task: Current task
            environment: Tau2Environment
            query: Query/prompt for agents

        Returns:
            Final answer from agents
        """
        answers = [agent.run(query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using Tau2 evaluators.

        Uses each evaluator's filter_traces() method to extract relevant data,
        then calls the evaluator with the filtered traces.

        Returns tau2 format:
        - reward: Float [0.0, 1.0]
        - passed: Boolean
        - reward_breakdown: Per-evaluator scores
        - env_check, action_check, communicate_check: Detailed results

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

    def execution_loop(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Tau2Environment,
        user: Optional[Tau2User],
    ) -> Any:
        """Execute agents with user-generated initial query.

        C7: Matches original tau2-bench orchestrator.initialize():
        The orchestrator sends the greeting to the user simulator, and the
        user LLM-generates the initial query (not pre-set from task.query).
        The agent never sees the greeting — only the user's first message.

        Source: tau2-bench orchestrator.py:L34-36, L223-229

        Args:
            agents: Agents to execute.
            task: The task being solved.
            environment: The Tau2Environment providing tools and state.
            user: Optional Tau2 user simulator.

        Returns:
            Final answer from the last agent execution.
        """
        final_answer = None

        if user is not None:
            # C7: User LLM-generates initial query in response to greeting.
            # respond() adds the greeting as an assistant message to user's
            # history, then generates via LLM. No inject_greeting needed.
            query_text = user.respond(INITIAL_GREETING)
        else:
            query_text = task.query

        for _ in range(self.max_invocations):
            final_answer = self.run_agents(agents, task, environment, query_text)

            if user is None:
                break

            user_response = user.respond(str(final_answer) if final_answer else "")

            if user.is_done():
                break

            query_text = user_response

        return final_answer


# =============================================================================
# Default Agent Implementation
# =============================================================================

# Agent system prompt constants (matching original tau2-bench)
_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

_SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def _flatten_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a JSON schema by inlining ``$ref`` references and removing unsupported keys.

    Pydantic v2's ``model_json_schema()`` emits ``$ref`` / ``$defs`` for nested
    models and ``anyOf`` for ``Optional`` fields.  Google GenAI rejects all of
    these.  This helper recursively resolves them so the resulting schema is
    self-contained and compatible with every provider.

    Args:
        schema: A JSON schema dict (typically from ``Model.model_json_schema()``).

    Returns:
        A flattened copy with ``$ref``, ``$defs``, ``anyOf``, ``title``,
        ``default``, and ``additionalProperties`` removed / inlined.
    """
    _STRIP_KEYS = {"$defs", "additionalProperties", "title", "default"}

    def _resolve(node: Any, defs: Dict[str, Any]) -> Any:
        if not isinstance(node, dict):
            return node

        # Inline $ref
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            if ref_name in defs:
                return _resolve(dict(defs[ref_name]), defs)
            return node

        # Simplify anyOf (Optional[X] → X with nullable)
        if "anyOf" in node:
            variants = node["anyOf"]
            non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
            if len(non_null) == 1:
                resolved = _resolve(non_null[0], defs)
                resolved["nullable"] = True
                if "description" in node and "description" not in resolved:
                    resolved["description"] = node["description"]
                return resolved
            if non_null:
                return _resolve(non_null[0], defs)

        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRIP_KEYS or key == "anyOf":
                continue
            if isinstance(value, dict):
                out[key] = _resolve(value, defs)
            elif isinstance(value, list):
                out[key] = [_resolve(v, defs) if isinstance(v, dict) else v for v in value]
            else:
                out[key] = value
        return out

    return _resolve(schema, schema.get("$defs", {}))


def _build_tool_definitions(tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Build OpenAI-format tool definitions from a dict of callables.

    Matches the original tau2-bench Tool.openai_schema format:
    uses docstring_parser + Pydantic create_model for parameter schemas.

    Args:
        tools: Dictionary mapping tool names to callables

    Returns:
        List of tool definitions in OpenAI function calling format
    """
    from docstring_parser import parse as parse_docstring
    from typing import Any as TypingAny

    definitions = []
    for name, func in tools.items():
        sig = inspect.signature(func)
        doc = parse_docstring(func.__doc__ or "")

        # Build tool description from parsed docstring (short + long)
        if doc.short_description:
            description = doc.short_description
            if doc.long_description:
                description += "\n\n" + doc.long_description
        else:
            description = name

        # Build Pydantic model from signature + docstring params
        doc_params = {p.arg_name: p for p in doc.params}
        model_fields = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            anno = param.annotation
            default = param.default

            if default is param.empty:
                default = ...  # required

            if param_name in doc_params:
                default = Field(default, description=doc_params[param_name].description)
                if (anno is param.empty) and (doc_params[param_name].type_name is not None):
                    anno = doc_params[param_name].type_name

            if anno is param.empty:
                anno = TypingAny

            model_fields[param_name] = (anno, default)

        params_model = create_model("parameters", **model_fields)  # type: ignore[call-overload]

        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": _flatten_schema(params_model.model_json_schema()),
                },
            }
        )

    return definitions


def _to_json_str(resp: Any) -> str:
    """Convert a tool response to a JSON string.

    Matches the serialization from the original tau2-bench Environment.to_json_str
    (environment.py:338-366). Pydantic models are serialized via model_dump(),
    dates via isoformat(), and the result is passed through json.dumps().

    Args:
        resp: Tool response (Pydantic model, dict, list, primitive, etc.)

    Returns:
        JSON string representation
    """

    def _process(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return obj
        elif isinstance(obj, (int, float, bool)):
            return str(obj)
        elif isinstance(obj, list):
            return [_process(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_process(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: _process(v) for k, v in obj.items()}
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        else:
            raise ValueError(f"Unsupported type: {type(obj)}")

    if isinstance(resp, str):
        return resp
    return json.dumps(_process(resp), default=str)


class DefaultTau2Agent:
    """Default agent implementation matching original tau2-bench LLMAgent.

    This agent mirrors the behavior of the original tau2-bench LLMAgent class,
    enabling direct comparison with the original benchmark results.

    The agent uses a simple ReAct-style loop:
    1. Receives user message
    2. Generates response (text or tool call)
    3. If tool call: executes tool and loops back to step 2
    4. If text: returns text as response

    Original implementation: tau2-bench/src/tau2/agent/llm_agent.py

    Attributes:
        tools: Dictionary mapping tool names to callables
        policy: Domain policy text (markdown)
        model: ModelAdapter for LLM calls
        llm_args: Additional arguments for LLM calls
        max_tool_calls: Maximum tool calls per turn (prevents infinite loops)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed)
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        policy: str,
        model: ModelAdapter,
        llm_args: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 50,
        verbose: int = 0,
    ):
        """Initialize the default tau2 agent.

        Args:
            tools: Dictionary mapping tool names to callable implementations
            policy: Domain policy text (markdown format)
            model: ModelAdapter for making LLM calls
            llm_args: Optional additional arguments passed to model.generate()
            max_tool_calls: Maximum number of tool calls per agent turn
            verbose: Verbosity level for debugging output:
                - 0: Silent (no output)
                - 1: Basic (tool calls and responses)
                - 2: Detailed (full message contents, tool arguments and results)
        """
        self.tools = tools
        self.policy = policy
        self.model = model
        self.llm_args = llm_args or {}
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose

        # Build system prompt
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=_AGENT_INSTRUCTION,
            domain_policy=self.policy,
        )

        # Message history for the conversation
        self._messages: List[Dict[str, Any]] = []
        self._tool_call_count = 0

    def reset(self) -> None:
        """Reset the agent state for a new conversation."""
        self._messages = []
        self._tool_call_count = 0

    def _log(self, level: int, message: str) -> None:
        """Print message if verbosity level is high enough.

        Args:
            level: Minimum verbosity level required (1 or 2)
            message: Message to print
        """
        if self.verbose >= level:
            print(message)

    def run(self, query: str) -> str:
        """Process a user query and return the agent's response.

        This method handles the full agent turn:
        1. Adds user message to history
        2. Generates LLM response with tool access
        3. If tool call: executes tools and continues generating
        4. Returns final text response to user

        Args:
            query: The user's message/query

        Returns:
            Agent's text response to the user
        """
        # Reset tool call counter for this turn (prevents accumulation across turns).
        # In the original tau2-bench, each generate_next_message() call is independent.
        # Source: tau2-bench agent/llm_agent.py
        self._tool_call_count = 0

        self._log(1, f"[Agent] Received query: {query[:100]}{'...' if len(query) > 100 else ''}")

        # Add user message to history
        self._messages.append({"role": "user", "content": query})

        # Generate response with potential tool calls
        # C8: Track steps (messages added during generation) for step counting
        pre_count = len(self._messages)
        result = self._generate_with_tools()
        self._last_turn_steps = len(self._messages) - pre_count
        return result

    def _generate_with_tools(self) -> str:
        """Generate response, handling any tool calls.

        Implements the agent's ReAct loop:
        - Generate LLM response with tools available
        - If response includes tool calls, execute them and continue
        - If response is text only, return it

        Returns:
            Final text response from the agent
        """
        while self._tool_call_count < self.max_tool_calls:
            # Build messages for LLM call
            messages = [{"role": "system", "content": self.system_prompt}] + self._messages
            self._log(2, f"[Agent] Generating response (messages: {len(messages)}, tools: {len(self.tools)})")

            # Generate response with tool access using chat() method
            # H2: tool_choice="auto" matching original llm_utils.py
            tool_defs = self._get_tool_definitions()
            response = self.model.chat(
                messages=messages,
                tools=tool_defs,
                tool_choice="auto" if tool_defs else None,
                **self.llm_args,
            )

            # Parse response from ChatResponse
            content = response.content or ""
            tool_calls = response.tool_calls or []

            if tool_calls:
                self._log(1, f"[Agent] Tool calls: {[self._get_tool_name(tc) for tc in tool_calls]}")

                # Add assistant message with tool calls
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )

                # Execute each tool call
                for tool_call in tool_calls:
                    self._tool_call_count += 1
                    tool_result = self._execute_tool_call(tool_call)

                    # Add tool result to history
                    # Serialize via _to_json_str to match original tau2-bench
                    # (environment.py:408), which uses model_dump() + json.dumps()
                    # instead of Python's str()/repr().
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": _to_json_str(tool_result),
                        }
                    )

                # Continue loop to generate next response
                continue
            else:
                # Text response - add to history and return
                self._log(1, f"[Agent] Response: {content[:100]}{'...' if len(content) > 100 else ''}")
                self._messages.append({"role": "assistant", "content": content})
                return content

        # Max tool calls reached - return empty or error message
        self._log(1, f"[Agent] Max tool calls ({self.max_tool_calls}) reached")
        return "I apologize, but I've encountered an issue processing your request. Please try again."

    def _get_tool_name(self, tool_call: Dict[str, Any]) -> str:
        """Extract tool name from a tool call dict."""
        if "function" in tool_call:
            return tool_call["function"].get("name", "unknown")
        return tool_call.get("name", "unknown")

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Dict in OpenAI format with 'function.name' and 'function.arguments',
                or flat format with 'name' and 'arguments' keys.

        Returns:
            Tool execution result
        """
        # Handle both flat format and nested 'function' format (OpenAI/ChatResponse style)
        if "function" in tool_call:
            name = tool_call["function"].get("name", "")
            arguments = tool_call["function"].get("arguments", {})
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

        # As in environment.py:get_response, tool errors return "Error: ..." strings (not exceptions)
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid or missing arguments for tool '{name}'"

        if name not in self.tools:
            self._log(1, f"[Agent] Tool not found: {name}")
            return f"Error: Tool '{name}' not found"

        self._log(2, f"[Agent] Executing {name}({arguments})")
        try:
            result = self.tools[name](**arguments)
            result_str = str(result)
            self._log(2, f"[Agent] Result: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")
            return result
        except Exception as e:
            self._log(1, f"[Agent] Tool error: {name} - {e}")
            return f"Error executing tool '{name}': {str(e)}"

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions for the LLM.

        Uses docstring_parser and Pydantic create_model to build parameter
        schemas, matching the original tau2-bench Tool.openai_schema approach.

        Returns:
            List of tool definitions in OpenAI function calling format
        """
        import inspect
        from typing import Any as TypingAny

        from docstring_parser import parse as parse_docstring
        from pydantic import Field, create_model

        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = parse_docstring(func.__doc__ or "")

            # Build tool description from parsed docstring (short + long)
            if doc.short_description:
                description = doc.short_description
                if doc.long_description:
                    description += "\n\n" + doc.long_description
            else:
                description = name

            # Build Pydantic model from signature + docstring params
            doc_params = {p.arg_name: p for p in doc.params}
            model_fields = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                anno = param.annotation
                default = param.default

                if default is param.empty:
                    default = ...  # required

                if param_name in doc_params:
                    default = Field(default, description=doc_params[param_name].description)
                    if (anno is param.empty) and (doc_params[param_name].type_name is not None):
                        anno = doc_params[param_name].type_name

                if anno is param.empty:
                    anno = TypingAny

                model_fields[param_name] = (anno, default)

            params_model = create_model("parameters", **model_fields)  # type: ignore[call-overload]

            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": _flatten_schema(params_model.model_json_schema()),
                    },
                }
            )

        return definitions

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the current message history.

        Returns:
            List of message dictionaries
        """
        return list(self._messages)


class DefaultTau2AgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for DefaultTau2Agent.

    Provides the standard MASEval AgentAdapter interface for DefaultTau2Agent.
    """

    def __init__(self, agent: DefaultTau2Agent, name: str = "default_agent"):
        """Initialize the adapter.

        Args:
            agent: DefaultTau2Agent instance to wrap
            name: Name for the agent adapter
        """
        super().__init__(agent, name)
        self._agent = agent

    def _run_agent(self, query: str) -> str:
        """Execute the agent with a query.

        Args:
            query: User query string

        Returns:
            Agent's response string
        """
        return self._agent.run(query)

    def get_messages(self) -> Any:
        """Get the agent's message history.

        Returns:
            Message history from the underlying agent
        """
        return self._agent.get_messages()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this agent.

        Overrides base implementation to handle list-based message history.
        """
        history = self.get_messages()
        # history is already a list, not a MessageHistory object
        messages = history if isinstance(history, list) else []
        return {
            "type": type(self).__name__,
            "gathered_at": __import__("datetime").datetime.now().isoformat(),
            "name": self.name,
            "agent_type": type(self.agent).__name__,
            "adapter_type": type(self).__name__,
            "message_count": len(messages),
            "messages": messages,
            "callbacks": [type(cb).__name__ for cb in self.callbacks],
            "logs": self.logs,
        }


class DefaultAgentTau2Benchmark(Tau2Benchmark):
    """Tau2 benchmark with default agent implementation.

    This benchmark uses the DefaultTau2Agent which mirrors the original
    tau2-bench LLMAgent implementation for direct comparison.

    Configuration via agent_data:
        - model_id: LLM model identifier (required)
        - llm_args: Optional dict of additional LLM arguments
        - max_tool_calls: Maximum tool calls per turn (default: 50)
        - verbose: Verbosity level for debugging (0=silent, 1=basic, 2=detailed)

    Example:
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark, load_tasks, configure_model_ids

        tasks = load_tasks("retail", split="base", limit=5)
        configure_model_ids(tasks, user_model_id="gpt-4o")

        benchmark = DefaultAgentTau2Benchmark(
            agent_data={"model_id": "gpt-4o", "verbose": 1},
        )
        results = benchmark.run(tasks)
    """

    def execution_loop(  # type: ignore[override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Tau2Environment,
        user: Optional[Tau2User],
    ) -> Any:
        """Execute with step counting matching original orchestrator.

        C8: The original counts steps per-message-appended:
        - Each agent LLM generation = 1 step
        - Each tool result = 1 step
        - Each user LLM generation = 1 step
        Steps during initialization (greeting + initial query) don't count.

        Args:
            agents: Agents to execute.
            task: The task being solved.
            environment: The Tau2Environment providing tools and state.
            user: Optional Tau2 user simulator.

        Returns:
            Final answer from the last agent execution.
        """
        final_answer = None

        if user is not None:
            # C7: User LLM-generates initial query (steps don't count)
            query_text = user.respond(INITIAL_GREETING)
        else:
            query_text = task.query

        # Access underlying DefaultTau2Agent for step tracking
        agent: DefaultTau2Agent = agents[0]._agent  # type: ignore[attr-defined]
        steps = 0

        while steps < self.max_invocations:
            final_answer = agent.run(query_text)
            steps += agent._last_turn_steps

            if user is None or steps >= self.max_invocations:
                break

            user_response = user.respond(str(final_answer) if final_answer else "")
            steps += user._last_respond_steps

            if user.is_done() or steps >= self.max_invocations:
                break

            query_text = user_response

        return final_answer

    def _get_agent_model_id(self, agent_data: Dict[str, Any]) -> str:
        """Get agent model ID from agent_data.

        Args:
            agent_data: Agent configuration dict

        Returns:
            Model ID string

        Raises:
            ValueError: If model_id not configured
        """
        model_id = agent_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Agent model_id not configured in agent_data.\n"
                "Pass model_id when creating the benchmark:\n\n"
                "    benchmark = DefaultAgentTau2Benchmark(\n"
                "        agent_data={'model_id': 'gpt-4o'},\n"
                "    )"
            )
        return model_id

    def setup_agents(  # type: ignore[invalid-method-override]
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
        seed_generator: DefaultSeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create the default tau2 agent.

        Args:
            agent_data: Agent configuration with model_id
            environment: Tau2Environment with real tools
            task: Current task
            user: Optional user simulator
            seed_generator: Seed generator for deriving agent seeds

        Returns:
            Tuple of (agent list, agent dict)
        """
        # Get configuration
        model_id = self._get_agent_model_id(agent_data)
        llm_args = dict(agent_data.get("llm_args", {}))
        max_tool_calls = agent_data.get("max_tool_calls", 50)
        verbose = agent_data.get("verbose", 0)

        # D19: Default temperature=0.0 matching original tau2-bench
        # (config.py: DEFAULT_LLM_TEMPERATURE_AGENT = 0.0)
        llm_args.setdefault("temperature", 0.0)

        # D20: Disable Claude thinking matching original tau2-bench
        # (llm_utils.py: kwargs["thinking"] = {"type": "disabled"} for Claude)
        if model_id.startswith("claude"):
            llm_args.setdefault("thinking", {"type": "disabled"})

        # Get tools and policy from environment
        tools = environment.create_tools()
        policy = environment.policy

        # Derive seed for agent model (returns None if seeding disabled)
        agent_gen = seed_generator.child("agents")
        agent_seed = agent_gen.derive_seed("default_agent")

        # Create model adapter
        model = self.get_model_adapter(model_id, register_name="agent_model", seed=agent_seed)

        # Create agent
        agent = DefaultTau2Agent(
            tools=tools,
            policy=policy,
            model=model,
            llm_args=llm_args,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )

        # Wrap in adapter
        adapter = DefaultTau2AgentAdapter(agent, name="default_agent")

        return [adapter], {"default_agent": adapter}

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Get or create a model adapter.

        Must be implemented by subclass to provide the actual ModelAdapter
        implementation for the desired LLM provider.

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments (e.g., register_name for tracing)

        Returns:
            ModelAdapter instance

        Note:
            DefaultAgentTau2Benchmark uses lazy initialization for model caching.
            Access via `getattr(self, '_model_cache', {})` in subclass implementations.
        """
        pass
