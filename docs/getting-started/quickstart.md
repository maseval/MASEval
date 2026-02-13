# Getting Started

This guide introduces the core concepts of MASEval and helps you get started quickly.

## Installation

MASEval is designed with a modular architecture:

- **Core**: Framework-agnostic benchmark infrastructure (tasks, evaluation, simulation)
- **Interface**: Optional adapters for specific agent frameworks (smolagents, langgraph, etc.)

Install only what you need to keep dependencies minimal.

### Core Installation

Install the base package from PyPI:

```bash
pip install maseval
```

This includes all core functionality for defining benchmarks, tasks, and evaluators.

### Optional Dependencies

Install additional integrations based on your agent framework. For example,

```bash
# Agent framework integrations
pip install "maseval[smolagents]"  # SmolAgents integration

# LLM providers
pip install "maseval[openai]"      # OpenAI models

# Combine multiple extras
pip install "maseval[smolagents,openai]"

# Install everything (for examples or development)
pip install "maseval[all]"
pip install "maseval[examples]"
```

---

## Using the Library

### Philosophy

MASEval follows a clear separation of concerns:

1. **You implement your agents** using any framework (LangChain, AutoGen, smolagents, custom code, etc.)
2. **MASEval provides the evaluation infrastructure** — benchmarks, tasks, environments, and metrics
3. **Adapters bridge the gap** — thin wrappers that connect your agent to MASEval's interface

Think of MASEval like pytest for agents: you bring the code, MASEval runs the tests.

### Key Concepts

| Term             | Description                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| **Benchmark**    | Orchestrates the evaluation lifecycle: setup, execution, and measurement across a collection of tasks. |
| **Task**         | A single evaluation unit with a query, expected outcome, and evaluation criteria.                      |
| **Environment**  | The context in which agents operate (e.g., simulated tools, databases, file systems).                  |
| **AgentAdapter** | Wraps your agent to provide a unified interface for MASEval.                                           |
| **Evaluator**    | Measures agent performance by comparing outputs or states to expected results.                         |
| **Callback**     | Hooks into the evaluation lifecycle for logging, tracing, or custom metrics.                           |

### Implementing a Benchmark

To create your own benchmark, subclass `Benchmark` and implement the required abstract methods. Here's the typical workflow:

1. **Agents and environment** Define your agents and environment using any tool you prefer to use. Wrap them in `Environment` and `AgentAdapter`.
2. **Create your tasks** as `Task` objects with queries and evaluation data
3. **Subclass `Benchmark`** and implement the abstract setup/run/evaluate methods
4. **Call `Benchmark.run(tasks)`** to execute the complete benchmark

```python
from maseval import Benchmark, AgentAdapter, Environment, Evaluator, Task

class MyBenchmark(Benchmark):
    """Custom benchmark for evaluating agents on my tasks."""

    def setup_environment(self, agent_data, task) -> Environment:
        # Initialize the environment for this task
        # e.g., set up tools, databases, or simulated systems
        ...

    def setup_user(self, agent_data, environment, task):
        # Optional: create a user simulator for interactive tasks
        # Return None if not needed
        return None

    def setup_agents(self, agent_data, environment, task, user):
        # Create your agent(s) and wrap them in AgentAdapter
        # Returns a tuple: (agents_to_run, agents_dict)
        #   - agents_to_run: list of agents to invoke in run_agents()
        #   - agents_dict: dict mapping names to all agents for tracing
        ...

    def setup_evaluators(self, environment, task, agents, user):
        # Define how success is measured
        # Return: list of Evaluator instances
        ...

    def run_agents(self, agents, task, environment):
        # Execute your agent system to solve the task
        # Return the final answer (message traces are captured automatically)
        ...

    def evaluate(self, evaluators, agents, final_answer, traces):
        # Run each evaluator with the execution data
        # Return: list of evaluation result dicts
        ...
```

Once implemented, run your benchmark:

```python
# Define your tasks
tasks = TaskQueue([Task(query="..."), ...])

# Configure your agents (e.g., model parameters, tool settings)
agent_config = {"model": "gpt-4", "temperature": 0.7}

# Instantiate and run the evaluation
benchmark = MyBenchmark(agent_data=agent_config)
reports = benchmark.run(tasks)
```

For the complete interface and lifecycle details, see the [Benchmark reference](../reference/benchmark.md).

### Adapters

Adapters are lightweight wrappers that connect your agent implementation to MASEval. They provide:

- A unified `run()` method for executing agents
- Message history tracking for tracing
- Callback hooks for monitoring

**Creating an adapter:**

```python
from maseval import AgentAdapter

class MyAgentAdapter(AgentAdapter):
    """Adapter for my custom agent framework."""

    def _run_agent(self, query: str):
        # Call your agent's execution method
        result = self.agent.execute(query)

        # Return the final answer (message history is tracked separately)
        return result

    def get_messages(self):
        # Return the conversation history from your agent
        return self.agent.get_conversation_history()
```

MASEval provides built-in adapters for popular frameworks in `maseval.interface.agents`. For example:

- `SmolAgentsAdapter` — for HuggingFace smolagents
- `LangGraphAdapter` — for LangGraph agents

See the [Agent Adapters](../interface/agents/smolagents.md) documentation for the full list.

### Existing Benchmarks

MASEval includes pre-built benchmarks for established evaluation suites. See the [Benchmarks](../benchmark/index.md) section for the full list.

**Using a default agent:** For quick evaluation or baseline comparisons, use the default benchmark class directly:

```python
from maseval.benchmark.tau2 import (
    DefaultAgentTau2Benchmark, load_tasks, ensure_data_exists,
)

ensure_data_exists(domain="retail")
tasks = load_tasks("retail", split="base", limit=5)

benchmark = DefaultAgentTau2Benchmark(
    agent_data={"model_id": "gpt-4o"},
    n_task_repeats=4,
)
results = benchmark.run(tasks)
```

**Plugging in your own agent:** Subclass the base benchmark to use your own agent implementation:

```python
from maseval.benchmark.tau2 import Tau2Benchmark, load_tasks

class MyTau2Benchmark(Tau2Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        tools = environment.tools
        # Create your agent with these tools
        ...

benchmark = MyTau2Benchmark(agent_data={}, n_task_repeats=4)
results = benchmark.run(tasks)
```

The base class handles environment setup, user simulation, and evaluation—you only implement `setup_agents()` and `run_agents()`.

---

## Using the Documentation

This documentation is organized to help you find what you need quickly:

### Examples

End-to-end walkthroughs demonstrating complete evaluation pipelines. Start here to see MASEval in action with real agent implementations.

- [Tiny Tutorial](../examples/tutorial.ipynb)
- [5-A-Day Benchmark](../examples/five_a_day_benchmark.ipynb)

### Guides

Topic-based discussions covering specific features and best practices:

- [Message Tracing](../guides/message-tracing.md) — Capture and analyze agent conversations
- [Configuration Gathering](../guides/config-gathering.md) — Collect reproducible experiment configurations
- [Seeding](../guides/seeding.md) — Enable reproducible benchmark runs

### Reference

Formal API documentation for all MASEval components. The reference is split into two sections:

**Core** — The fundamental building blocks (no optional dependencies):

- [Benchmark](../reference/benchmark.md) — Evaluation orchestration
- [Task](../reference/task.md) — Individual evaluation units
- [Environment](../reference/environment.md) — Agent execution context
- [AgentAdapter](../reference/agent.md) — Agent interface wrappers
- [Evaluator](../reference/evaluator.md) — Performance measurement
- [Callback](../reference/callback.md) — Lifecycle hooks
- [MessageHistory](../reference/history.md) — Conversation tracking

**Interface** — Optional integrations for specific frameworks:

- [Agent Adapters](../interface/agents/smolagents.md) — Pre-built adapters (smolagents, langgraph, etc.)
- [Inference Providers](../interface/inference/openai.md) — LLM provider integrations

## Next Steps

Work through the examples listed above. 3. **Explore the [`examples/five_a_day_benchmark/`](https://github.com/parameterlab/MASEval/tree/main/examples/five_a_day_benchmark) folder** for tool implementations, evaluators, and the CLI script (`five_a_day_benchmark.py`) 4. **Build your own benchmark** using the patterns you've learned
