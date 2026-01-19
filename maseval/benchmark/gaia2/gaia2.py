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
        def setup_agents(self, agent_data, environment, task, user):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            adapter = MyModelAdapter(model_id)
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
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from maseval import AgentAdapter, Benchmark, Evaluator, ModelAdapter, Task, User
from maseval.core.callback import BenchmarkCallback

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
    ) -> Gaia2Environment:
        """Create Gaia2 environment wrapping ARE simulation.

        Args:
            agent_data: Agent configuration
            task: Current task

        Returns:
            Gaia2Environment instance
        """
        return Gaia2Environment(task_data=task.environment_data)

    def setup_user(  # type: ignore[override]
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
    ) -> Optional[User]:
        """Gaia2 uses event-based simulation, not turn-based user simulation.

        User interactions in Gaia2 happen through scheduled events (e.g.,
        "user sends message at t=30s") rather than synchronous turn-taking.

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
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration
            environment: Gaia2Environment with ARE tools
            task: Current task
            user: Optional user simulator (always None for Gaia2)

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
    ) -> Sequence[Evaluator]:
        """Create Gaia2 evaluator using ARE's judge.

        Args:
            environment: Gaia2Environment instance
            task: Current task with evaluation data
            agents: Agent instances
            user: Optional user simulator (always None)

        Returns:
            List with single Gaia2Evaluator instance
        """
        evaluator_model_id = task.evaluation_data.get("model_id")
        model = None
        if evaluator_model_id:
            model = self.get_model_adapter(evaluator_model_id, register_name="evaluator")

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

# System prompt for Gaia2 agent
_GAIA2_SYSTEM_PROMPT = """
You are an AI assistant helping a user with tasks in a mobile environment.
You have access to various apps including Calendar, Email, Messaging, Contacts,
Shopping, Cab, Browser, and a FileSystem.

Key behaviors:
1. Use get_current_time() to check the current time when relevant.
2. For tasks requiring waiting (e.g., "wait for response"), use wait_for_notification(timeout_seconds).
3. Execute tasks step by step, using the appropriate tools.
4. When the task is complete, provide a final response summarizing what was done.

Available tools will be provided to you. Use them to accomplish the user's task.
""".strip()


class DefaultGaia2Agent:
    """Default agent implementation for Gaia2 benchmark.

    ReAct-style agent that interacts with ARE's simulation through tool calls.
    Supports temporal reasoning via SystemApp tools.

    This agent follows the pattern from ARE's reference implementation,
    using a simple ReAct loop with tool calling.
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        model: ModelAdapter,
        llm_args: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 100,
        verbose: int = 0,
    ):
        """Initialize the agent.

        Args:
            tools: Dict of tool name -> callable
            model: ModelAdapter for LLM interactions
            llm_args: Additional arguments for model calls
            max_tool_calls: Maximum tool calls before stopping
            verbose: Verbosity level (0=quiet, 1=basic, 2=detailed)
        """
        self.tools = tools
        self.model = model
        self.llm_args = llm_args or {}
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose
        self.system_prompt = _GAIA2_SYSTEM_PROMPT

        self._messages: List[Dict[str, Any]] = []
        self._tool_call_count = 0

    def reset(self) -> None:
        """Reset agent state."""
        self._messages = []
        self._tool_call_count = 0

    def run(self, query: str) -> str:
        """Execute task and return final response.

        Args:
            query: Task query/instructions

        Returns:
            Final text response from agent
        """
        self._messages.append({"role": "user", "content": query})
        return self._generate_with_tools()

    def _generate_with_tools(self) -> str:
        """ReAct loop: generate -> execute tools -> repeat or return.

        Returns:
            Final text response
        """
        while self._tool_call_count < self.max_tool_calls:
            messages = [{"role": "system", "content": self.system_prompt}] + self._messages

            response = self.model.chat(
                messages=messages,
                tools=self._get_tool_definitions(),
                **self.llm_args,
            )

            content = response.content or ""
            tool_calls = response.tool_calls or []

            if tool_calls:
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

                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": str(tool_result),
                        }
                    )

                continue
            else:
                # Text response - done
                self._messages.append({"role": "assistant", "content": content})
                return content

        return "Max tool calls reached."

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dict with name and arguments

        Returns:
            Tool execution result
        """
        # Handle both function format and direct format
        if "function" in tool_call:
            name = tool_call["function"].get("name", "")
            arguments = tool_call["function"].get("arguments", {})
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if name not in self.tools:
            return f"Error: Tool '{name}' not found"

        try:
            return self.tools[name](**arguments)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions in OpenAI format.

        Returns:
            List of tool definitions
        """
        definitions = []

        for name, tool in self.tools.items():
            # Try to get schema from tool wrapper
            schema: Dict[str, Any] = {}
            description = ""

            if hasattr(tool, "inputs"):
                schema = tool.inputs or {}
            if hasattr(tool, "description"):
                description = tool.description or ""

            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", []),
                        },
                    },
                }
            )

        return definitions

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get message history.

        Returns:
            List of messages
        """
        return list(self._messages)


class DefaultGaia2AgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for DefaultGaia2Agent."""

    def __init__(self, agent: DefaultGaia2Agent, name: str = "gaia2_agent"):
        """Initialize the adapter.

        Args:
            agent: DefaultGaia2Agent instance
            name: Agent name for identification
        """
        super().__init__(agent, name)
        self._agent = agent

    def _run_agent(self, query: str) -> str:
        """Run the agent and return answer.

        Args:
            query: Task query

        Returns:
            Agent's final answer
        """
        return self._agent.run(query)

    def get_messages(self) -> Any:
        """Get message history.

        Returns:
            List of messages
        """
        return self._agent.get_messages()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces.

        Returns:
            Trace dictionary
        """
        history = self.get_messages()
        return {
            **super().gather_traces(),
            "name": self.name,
            "message_count": len(history),
            "messages": history,
            "tool_call_count": self._agent._tool_call_count,
        }


class DefaultAgentGaia2Benchmark(Gaia2Benchmark):
    """Gaia2 benchmark with default agent implementation.

    Provides a ready-to-use benchmark matching ARE's reference agent behavior.

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
                - llm_args: Optional model call arguments
                - max_tool_calls: Max tool calls per task (default: 100)
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
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create default Gaia2 agent.

        Args:
            agent_data: Agent configuration
            environment: Gaia2Environment with ARE tools
            task: Current task
            user: Optional user (always None)

        Returns:
            Tuple of (agent list, agent dict)
        """
        # Merge class-level and run-level agent_data
        merged_data = {**self._agent_data, **agent_data}

        model_id = self._get_agent_model_id(merged_data)
        llm_args = merged_data.get("llm_args", {})
        max_tool_calls = merged_data.get("max_tool_calls", 100)
        verbose = merged_data.get("verbose", 0)

        tools = environment.create_tools()
        model = self.get_model_adapter(model_id, register_name="agent_model")

        agent = DefaultGaia2Agent(
            tools=tools,
            model=model,
            llm_args=llm_args,
            max_tool_calls=max_tool_calls,
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
