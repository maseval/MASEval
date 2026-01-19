# Gaia2/ARE Integration Plan for MASEval

## Overview

This plan implements **Option C (Hybrid Wrapper)** from STRATEGY.md to integrate Meta's ARE (Agent Research Environments) with the Gaia2 benchmark into MASEval. The approach uses ARE as a dependency for simulation and evaluation while providing MASEval-native wrappers for orchestration, tracing, and agent flexibility.

**Key architectural insight**: ARE's time control is tool-based (`SystemApp.wait_for_notification`), so no MASEval core execution loop changes are needed. The simulation complexity is fully encapsulated within `Environment`.

---

## Scientific Integrity and Reproducibility

> **WARNING: This is a scientific benchmark.**
>
> The primary purpose of this integration is to enable rigorous, reproducible evaluation of agent systems against the Gaia2 benchmark. Results produced by this implementation must be **semantically equivalent** to those produced by ARE's reference implementation. Deviations that affect benchmark scores invalidate scientific comparisons.

### Core Principle

**Preserve Benchmark Semantics**

- The benchmark measures specific agent capabilities (execution, search, adaptability, time, ambiguity, agent2agent, noise)
- Any implementation choice that changes what is being measured corrupts the benchmark

---

## Directory Structure

```
maseval/benchmark/gaia2/
├── __init__.py              # Public API exports
├── gaia2.py                 # Gaia2Benchmark, Gaia2User, DefaultGaia2Agent
├── environment.py           # Gaia2Environment wrapping ARE
├── evaluator.py             # Gaia2Evaluator using ARE judge
├── tool_wrapper.py          # AREToolWrapper for MASEval tracing
├── data_loader.py           # load_tasks(), configure_model_ids()
├── data/                    # Downloaded scenario data (gitignored)
└── prompt_templates/
    └── user_simulator.txt   # Prompt for Gaia2User

examples/gaia2_benchmark/
├── __init__.py
└── gaia2_benchmark.py       # Example script with CLI
```

---

## Phase 1: Core Infrastructure

### 1.1 Tool Wrapper (`tool_wrapper.py`)

Wraps ARE's AppTool instances for MASEval compatibility and tracing.

```python
class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    """Wraps ARE AppTool for MASEval tracing and compatibility.

    Records all tool invocations with inputs, outputs, timestamps,
    and simulation time for post-hoc analysis.
    """

    def __init__(self, are_tool, environment: "Gaia2Environment"):
        self.are_tool = are_tool
        self.environment = environment
        self.name = are_tool.name
        self.description = are_tool.description
        self.inputs = self._extract_schema(are_tool)  # Convert ARE schema to MASEval format
        self.history = ToolInvocationHistory()

    def __call__(self, **kwargs) -> Any:
        """Execute tool and record invocation."""
        start_time = datetime.now()
        sim_time_before = self._get_simulation_time()

        result = self.are_tool(**kwargs)

        sim_time_after = self._get_simulation_time()

        self.history.add_invocation(
            inputs=kwargs,
            outputs=result,
            meta={
                "wall_time": start_time.isoformat(),
                "simulation_time_before": sim_time_before,
                "simulation_time_after": sim_time_after,
                "simulation_time_elapsed": sim_time_after - sim_time_before,
            }
        )
        return result

    def _get_simulation_time(self) -> float:
        """Get current simulation time from ARE environment."""
        ...

    def _extract_schema(self, are_tool) -> Dict[str, Any]:
        """Convert ARE tool schema to MASEval input format."""
        ...

    def gather_traces(self) -> Dict[str, Any]: ...
    def gather_config(self) -> Dict[str, Any]: ...
```

**Design decisions**:

- Records both wall clock time and simulation time
- Captures simulation time delta for temporal scenario analysis
- Preserves ARE's native return types (no forced string conversion)

---

### 1.2 Environment (`environment.py`)

Wraps ARE's simulation environment with MASEval's Environment interface.

```python
class Gaia2Environment(Environment):
    """MASEval Environment wrapping ARE's simulation.

    The ARE simulation runs its own internal event loop. Agent interaction
    happens purely through tool calls - including time control via
    SystemApp.wait_for_notification(). No special execution loop needed.

    Exposes all ARE app tools (Calendar, Email, Messaging, Contacts, Shopping,
    Cab, City, FileSystem, Browser, ChatsApp, SystemApp, Timer) to agents.
    """

    def __init__(self, task_data: Dict[str, Any], callbacks=None):
        self._scenario = task_data["scenario"]
        self._are_env = None
        self._tool_wrappers: Dict[str, AREToolWrapper] = {}
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data) -> Dict[str, Any]:
        """Initialize ARE scenario and start simulation."""
        from are.simulation.environment import Environment as AREEnvironment
        from are.simulation.environment import EnvironmentConfig

        scenario = task_data["scenario"]

        # Create ARE environment
        config = EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
        )
        self._are_env = AREEnvironment(config)

        # Initialize scenario (loads apps, events, state)
        self._are_env.initialize_scenario(scenario)

        return {
            "scenario_id": scenario.scenario_id,
            "duration": scenario.duration,
            "capability": task_data.get("capability"),
            "universe_id": task_data.get("universe_id"),
        }

    def create_tools(self) -> Dict[str, AREToolWrapper]:
        """Wrap all ARE app tools for MASEval tracing.

        Includes critical tools:
        - SystemApp.get_current_time(): Query simulation time
        - SystemApp.wait_for_notification(timeout): Advance simulation time
        - All domain app tools (calendar, email, messaging, etc.)
        """
        tools = {}

        for app in self._are_env.apps.values():
            for tool in app.get_tools():
                wrapper = AREToolWrapper(tool, self)
                tools[tool.name] = wrapper
                self._tool_wrappers[tool.name] = wrapper

        return tools

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._are_env.time_manager.current_time

    def get_scenario(self):
        """Get the ARE scenario object."""
        return self._scenario

    def get_are_environment(self):
        """Get the underlying ARE Environment (for evaluator)."""
        return self._are_env

    def cleanup(self) -> None:
        """Stop ARE simulation when task completes."""
        if self._are_env:
            self._are_env.stop()

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces from environment and all tools."""
        tool_traces = {}
        for name, wrapper in self._tool_wrappers.items():
            tool_traces[name] = wrapper.gather_traces()

        return {
            "type": self.__class__.__name__,
            "gathered_at": datetime.now().isoformat(),
            "scenario_id": self.state.get("scenario_id"),
            "final_simulation_time": self.get_simulation_time(),
            "tool_count": len(self._tool_wrappers),
            "tools": tool_traces,
        }
```

**Design decisions**:

- `cleanup()` method ensures ARE simulation is stopped after task completion
- Exposes `get_simulation_time()` for evaluator access
- Exposes `get_are_environment()` for evaluator to use ARE's judge

---

### 1.3 Data Loader (`data_loader.py`)

Loads Gaia2 scenarios from HuggingFace and converts to MASEval Tasks.

```python
# Constants
VALID_CAPABILITIES = ("execution", "search", "adaptability", "time", "ambiguity", "agent2agent", "noise")
VALID_SPLITS = ("validation",)  # Only validation has oracle events
DEFAULT_CONFIG = "validation"  # Full dataset
DEFAULT_TIMEOUT_SECONDS = 600.0

def load_tasks(
    capability: Optional[str] = None,
    split: str = "validation",
    limit: Optional[int] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 1,
) -> TaskQueue:
    """Load Gaia2 tasks from HuggingFace.

    Args:
        capability: Filter by capability type (execution, search, adaptability,
            time, ambiguity, agent2agent, noise). None loads all.
        split: Dataset split (currently only "validation" available)
        limit: Maximum number of tasks to load
        timeout_seconds: Maximum execution time per task
        max_retries: Maximum retry attempts

    Returns:
        TaskQueue with Task objects containing:
            - id: Unique scenario identifier
            - query: Initial task instructions
            - environment_data: {"scenario": BenchmarkScenario, "capability": str, ...}
            - evaluation_data: {"oracle_events": [...], "judge_config": {...}}
            - user_data: {}  # Gaia2 uses event-based simulation, not user turns
            - metadata: {"capability": str, "universe_id": str, ...}
            - protocol: TaskProtocol with timeout and tags
    """
    from datasets import load_dataset
    from are.simulation.data_handler.importer import JsonScenarioImporter

    # Determine HuggingFace config name
    config_name = capability if capability else DEFAULT_CONFIG

    dataset = load_dataset(
        "meta-agents-research-environments/gaia2",
        name=config_name,
        split=split,
    )

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    importer = JsonScenarioImporter()
    tasks = []

    for row in dataset:
        scenario, oracle_events, world_logs = importer.import_from_json_to_benchmark(
            json_str=row["data"]
        )

        task = _convert_gaia2_to_maseval(
            row=row,
            scenario=scenario,
            oracle_events=oracle_events,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        tasks.append(task)

    return TaskQueue(tasks)


def _convert_gaia2_to_maseval(
    row: Dict[str, Any],
    scenario,
    oracle_events: List,
    timeout_seconds: Optional[float],
    max_retries: int,
) -> Task:
    """Convert Gaia2 scenario to MASEval Task."""
    # Extract query from scenario's task definition
    query = scenario.task_instruction

    # Parse capability from scenario metadata or row
    capability = row.get("category") or scenario.metadata.get("capability", "unknown")

    # Build environment_data
    environment_data = {
        "scenario": scenario,
        "capability": capability,
        "universe_id": scenario.metadata.get("universe_id"),
        "duration": scenario.duration,
    }

    # Build evaluation_data with oracle events
    evaluation_data = {
        "oracle_events": oracle_events,
        "judge_type": scenario.metadata.get("judge_type", "graph_per_event"),
    }

    # Build metadata
    metadata = {
        "scenario_id": row["scenario_id"],
        "capability": capability,
        "universe_id": scenario.metadata.get("universe_id"),
    }

    # Build protocol
    protocol = TaskProtocol(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        tags={"capability": capability, "benchmark": "gaia2"},
    )

    return Task(
        id=row["id"],
        query=query,
        environment_data=environment_data,
        evaluation_data=evaluation_data,
        user_data={},  # Gaia2 uses event-based simulation
        metadata=metadata,
        protocol=protocol,
    )


def configure_model_ids(
    tasks: Union[TaskQueue, List[Task]],
    *,
    evaluator_model_id: Optional[str] = None,
) -> Union[TaskQueue, List[Task]]:
    """Configure model IDs for benchmark components.

    Gaia2 uses ARE's deterministic judge by default, but can optionally
    use an LLM-based judge for complex assertions.

    Args:
        tasks: Tasks to configure
        evaluator_model_id: Optional model ID for LLM-based evaluation

    Returns:
        The same collection (mutated in place)
    """
    for task in tasks:
        if evaluator_model_id:
            task.evaluation_data["model_id"] = evaluator_model_id
    return tasks
```

**Design decisions**:

- Uses HuggingFace datasets library for loading (already a MASEval dependency)
- Imports ARE's `JsonScenarioImporter` for scenario parsing
- Scenario object is passed directly in `environment_data` (no serialization)
- Oracle events are passed in `evaluation_data` for the evaluator
- No `user_model_id` needed - Gaia2 uses event-based simulation, not turn-based user simulation

---

## Phase 2: Benchmark Implementation

### 2.1 Evaluator (`evaluator.py`)

Evaluates using ARE's judge system with MASEval trace integration.

```python
class Gaia2Evaluator(Evaluator):
    """Evaluates Gaia2 scenarios using ARE's judge system.

    Uses ARE's GraphPerEventJudge for deterministic evaluation based on
    the event DAG. Supports optional LLM-based judge for complex assertions.
    """

    def __init__(
        self,
        task: Task,
        environment: Gaia2Environment,
        user: Optional[User] = None,
        use_llm_judge: bool = False,
        model: Optional[ModelAdapter] = None,
    ):
        self.task = task
        self.environment = environment
        self.oracle_events = task.evaluation_data.get("oracle_events", [])
        self.judge_type = task.evaluation_data.get("judge_type", "graph_per_event")
        self.use_llm_judge = use_llm_judge
        self.model = model

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool invocations and environment state for evaluation.

        Returns:
            Dict with:
                - tool_invocations: List of all tool calls with timing
                - simulation_time: Final simulation time
                - scenario_id: For correlation
        """
        tool_traces = traces.get("environment", {}).get("tools", {})

        # Flatten all tool invocations
        invocations = []
        for tool_name, tool_data in tool_traces.items():
            for inv in tool_data.get("invocations", []):
                invocations.append({
                    "tool": tool_name,
                    "inputs": inv.get("inputs", {}),
                    "outputs": inv.get("outputs"),
                    "simulation_time": inv.get("meta", {}).get("simulation_time_after"),
                    "wall_time": inv.get("meta", {}).get("wall_time"),
                })

        return {
            "tool_invocations": invocations,
            "simulation_time": traces.get("environment", {}).get("final_simulation_time", 0),
            "scenario_id": self.task.metadata.get("scenario_id"),
        }

    def __call__(self, traces: Dict[str, Any], final_answer=None) -> Dict[str, Any]:
        """Evaluate using ARE's judge system.

        Returns:
            Dict with:
                - gsr: Goal Success Rate (0.0 or 1.0)
                - partial_gsr: Partial success rate
                - passed: Boolean indicating full success
                - event_results: Per-event evaluation results
                - capability: Task capability type
        """
        from are.simulation.validation import JudgeFactory
        from are.simulation.validation.config import GraphPerEventJudgeConfig

        # Create ARE judge
        judge_config = GraphPerEventJudgeConfig()
        judge = JudgeFactory.create(judge_config)

        # Get ARE environment and collected events
        are_env = self.environment.get_are_environment()
        completed_events = are_env.get_completed_events()

        # Run ARE's judge
        result = judge.evaluate(
            oracle_events=self.oracle_events,
            completed_events=completed_events,
            scenario=self.environment.get_scenario(),
        )

        # Convert ARE result to MASEval format
        gsr = 1.0 if result.passed else 0.0
        partial_gsr = result.partial_score if hasattr(result, "partial_score") else gsr

        return {
            "gsr": gsr,
            "partial_gsr": partial_gsr,
            "passed": result.passed,
            "event_results": result.event_results if hasattr(result, "event_results") else [],
            "capability": self.task.metadata.get("capability"),
            "tool_call_count": len(traces.get("tool_invocations", [])),
            "final_simulation_time": traces.get("simulation_time", 0),
        }
```

**Design decisions**:

- Delegates to ARE's `JudgeFactory` for evaluation fidelity
- Extracts `completed_events` from ARE environment after agent execution
- Preserves ARE's event-level results for debugging
- Adds MASEval-specific metadata (tool_call_count, final_simulation_time)

---

### 2.2 Benchmark (`gaia2.py`)

The main benchmark class following MASEval's lifecycle pattern.

```python
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
        """Create Gaia2 environment wrapping ARE simulation."""
        return Gaia2Environment(task_data=task.environment_data)

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
    ) -> Optional[User]:
        """Gaia2 uses event-based simulation, not turn-based user simulation.

        User interactions in Gaia2 happen through scheduled events (e.g.,
        "user sends message at t=30s") rather than synchronous turn-taking.
        Returns None.
        """
        return None

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass."""
        pass

    def setup_evaluators(
        self,
        environment: Gaia2Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create Gaia2 evaluator using ARE's judge."""
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

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Gaia2Environment,
        query: str = "",
    ) -> Any:
        """Execute agents and ensure environment cleanup."""
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
        """Evaluate using Gaia2 evaluators."""
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results
```

---

## Phase 3: Default Agent Implementation

### 3.1 Default Agent (`gaia2.py`)

A ReAct-style agent matching ARE's reference implementation.

```python
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
    """

    def __init__(
        self,
        tools: Dict[str, Callable],
        model: ModelAdapter,
        llm_args: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 100,
        verbose: int = 0,
    ):
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
        """Execute task and return final response."""
        self._messages.append({"role": "user", "content": query})
        return self._generate_with_tools()

    def _generate_with_tools(self) -> str:
        """ReAct loop: generate -> execute tools -> repeat or return."""
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
                self._messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                })

                # Execute each tool call
                for tool_call in tool_calls:
                    self._tool_call_count += 1
                    tool_result = self._execute_tool_call(tool_call)

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "content": str(tool_result),
                    })

                continue
            else:
                # Text response - done
                self._messages.append({"role": "assistant", "content": content})
                return content

        return "Max tool calls reached."

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a single tool call."""
        if "function" in tool_call:
            name = tool_call["function"].get("name", "")
            arguments = tool_call["function"].get("arguments", {})
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})

        if isinstance(arguments, str):
            import json
            arguments = json.loads(arguments)

        if name not in self.tools:
            return f"Error: Tool '{name}' not found"

        return self.tools[name](**arguments)

    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions in OpenAI format."""
        # Similar to DefaultTau2Agent._get_tool_definitions()
        ...

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get message history."""
        return list(self._messages)


class DefaultGaia2AgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for DefaultGaia2Agent."""

    def __init__(self, agent: DefaultGaia2Agent, name: str = "gaia2_agent"):
        super().__init__(agent, name)
        self._agent = agent

    def _run_agent(self, query: str) -> str:
        return self._agent.run(query)

    def get_messages(self) -> Any:
        return self._agent.get_messages()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces."""
        history = self.get_messages()
        return {
            "type": type(self).__name__,
            "gathered_at": datetime.now().isoformat(),
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

    def __init__(self, agent_data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._agent_data = agent_data or {}

    def _get_agent_model_id(self, agent_data: Dict[str, Any]) -> str:
        """Get agent model ID."""
        model_id = agent_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Agent model_id not configured. Pass model_id in agent_data:\n\n"
                "    benchmark = DefaultAgentGaia2Benchmark(\n"
                "        agent_data={'model_id': 'gpt-4o'},\n"
                "    )"
            )
        return model_id

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Gaia2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create default Gaia2 agent."""
        model_id = self._get_agent_model_id(agent_data)
        llm_args = agent_data.get("llm_args", {})
        max_tool_calls = agent_data.get("max_tool_calls", 100)
        verbose = agent_data.get("verbose", 0)

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
    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """Get or create model adapter. Must be implemented by subclass."""
        pass
```

---

## Phase 4: Example Script

### 4.1 Example (`examples/gaia2_benchmark/gaia2_benchmark.py`)

Following the Tau2 example pattern with CLI support.

```python
"""Gaia2 Benchmark Example.

Demonstrates running the Gaia2 benchmark with:
- default: MASEval's DefaultAgentGaia2Benchmark
- smolagents: HuggingFace smolagents framework
- langgraph: LangChain's LangGraph framework

The Gaia2 benchmark evaluates agents on dynamic, multi-step scenarios across
7 capability dimensions: Execution, Search, Ambiguity, Adaptability, Time,
Agent2Agent, and Noise.

Reference:
    Paper: https://arxiv.org/abs/2509.17158
    Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

Usage:
    # Run with default agent on execution capability
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --framework default --capability execution --limit 5

    # Run with smolagents on search capability
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --framework smolagents --capability search --limit 5

    # Run all capabilities with a specific model
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --framework default --model gpt-4o --limit 10
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from maseval.benchmark.gaia2 import (
    DefaultAgentGaia2Benchmark,
    Gaia2Benchmark,
    Gaia2Environment,
    load_tasks,
    configure_model_ids,
)
from maseval.core.callbacks.result_logger import FileResultLogger
# ... framework-specific imports


# Framework-specific implementations (GoogleGenAIGaia2Benchmark, SmolagentsGaia2Benchmark, etc.)
# Following the same pattern as tau2_benchmark.py


def run_benchmark(
    framework: Literal["default", "smolagents", "langgraph"],
    capability: Optional[str] = None,
    model_id: str = "gemini-2.5-flash",
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the Gaia2 benchmark."""
    output_dir = output_dir or Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gaia2 tasks (capability={capability or 'all'})...")
    tasks = load_tasks(capability=capability, limit=limit)
    print(f"Loaded {len(tasks)} tasks")

    configure_model_ids(tasks, evaluator_model_id=model_id)

    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"gaia2_{capability or 'all'}_{framework}_{{timestamp}}.jsonl",
    )

    BenchmarkClass = get_benchmark_class(framework, model_id)

    benchmark = BenchmarkClass(
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
        fail_on_setup_error=True,
        fail_on_evaluation_error=True,
        agent_data={"model_id": model_id},
    )

    print(f"\nRunning {framework} benchmark...")
    results = benchmark.run(tasks=tasks)

    # Compute and print summary
    summary = compute_gaia2_metrics(results)
    print_summary(summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Gaia2 benchmark")
    parser.add_argument("--framework", required=True, choices=["default", "smolagents", "langgraph"])
    parser.add_argument("--capability", choices=["execution", "search", "adaptability", "time", "ambiguity"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()
    run_benchmark(
        framework=args.framework,
        capability=args.capability,
        model_id=args.model,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
```

---

## Phase 5: Package Integration

### 5.1 Dependencies (`pyproject.toml`)

```toml
[project.optional-dependencies]
gaia2 = [
    "meta-agents-research-environments>=1.2.0",
]

# Update 'all' and 'examples' to include gaia2
all = [
    "maseval[smolagents,langgraph,gaia2]",
]
examples = [
    "maseval[all]",
]
```

### 5.2 Module Exports (`__init__.py`)

```python
"""Gaia2/ARE Benchmark for MASEval.

Provides MASEval integration for Meta's ARE (Agent Research Environments)
platform and the Gaia2 benchmark.

Example:
    from maseval.benchmark.gaia2 import (
        Gaia2Benchmark, Gaia2Environment, Gaia2Evaluator,
        DefaultAgentGaia2Benchmark,
        load_tasks, configure_model_ids,
    )

    tasks = load_tasks(capability="execution", limit=5)

    benchmark = DefaultAgentGaia2Benchmark(
        agent_data={"model_id": "gpt-4o"},
    )
    results = benchmark.run(tasks)
"""

from maseval.benchmark.gaia2.gaia2 import (
    Gaia2Benchmark,
    DefaultGaia2Agent,
    DefaultGaia2AgentAdapter,
    DefaultAgentGaia2Benchmark,
)
from maseval.benchmark.gaia2.environment import Gaia2Environment
from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator
from maseval.benchmark.gaia2.tool_wrapper import AREToolWrapper
from maseval.benchmark.gaia2.data_loader import (
    load_tasks,
    configure_model_ids,
    VALID_CAPABILITIES,
)

__all__ = [
    # Benchmark
    "Gaia2Benchmark",
    "DefaultAgentGaia2Benchmark",
    # Environment
    "Gaia2Environment",
    # Evaluator
    "Gaia2Evaluator",
    # Agent
    "DefaultGaia2Agent",
    "DefaultGaia2AgentAdapter",
    # Tools
    "AREToolWrapper",
    # Data loading
    "load_tasks",
    "configure_model_ids",
    "VALID_CAPABILITIES",
]
```

---

## Implementation Order

1. **Phase 1**: Core Infrastructure (tool_wrapper.py, environment.py, data_loader.py)
2. **Phase 2**: Benchmark Implementation (evaluator.py, gaia2.py base classes)
3. **Phase 3**: Default Agent (DefaultGaia2Agent, DefaultAgentGaia2Benchmark)
4. **Phase 4**: Example Script (gaia2_benchmark.py)
5. **Phase 5**: Package Integration (pyproject.toml)

---

## Testing Strategy

Tests will be added in the next step after implementation. Key test areas:

1. **Unit tests** for AREToolWrapper (tracing, schema conversion)
2. **Unit tests** for data_loader (task conversion, HuggingFace loading)
3. **Integration tests** for Gaia2Environment (tool creation, cleanup)
4. **Integration tests** for Gaia2Evaluator (judge invocation)
5. **Contract tests** ensuring MASEval interface compliance
6. **Smoke tests** running small scenarios end-to-end

---

## Error Handling Philosophy

Per the guidelines: **Fail loudly, no defensive defaults.**

- If `scenario` is missing from `task.environment_data`, crash with AttributeError
- If ARE's `JsonScenarioImporter.import_from_json_to_benchmark` fails, let it propagate
- If `oracle_events` is empty when evaluation runs, let the judge fail explicitly
- No silent fallbacks, no default values for required fields

This ensures bugs surface immediately rather than producing silently incorrect results.
