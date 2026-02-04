# MultiAgentBench: Multi-Agent Collaboration Benchmark

The **MultiAgentBench** benchmark evaluates multi-agent collaboration and competition in LLM-based systems across diverse scenarios including research, negotiation, coding, and more.

## Overview

[MultiAgentBench](https://github.com/ulab-uiuc/MARBLE) (from the MARBLE framework) is designed to evaluate how multiple LLM-based agents collaborate and compete to solve complex tasks. The benchmark features:

- **7 diverse domains**: research, bargaining, coding, database, web, worldsimulation, minecraft
- **Multiple coordination modes**: cooperative, star, tree, hierarchical
- **LLM-based evaluation**: Matches MARBLE's evaluation methodology
- **Framework-agnostic**: Use with any agent framework or MARBLE's native agents

Reference Paper: [MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents](https://arxiv.org/abs/2503.01935)

Check out the [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md) file for more information including licenses.

## Quick Start

```python
from maseval.benchmark.multiagentbench import (
    MultiAgentBenchBenchmark,
    MultiAgentBenchEnvironment,
    MultiAgentBenchEvaluator,
    load_tasks,
    configure_model_ids,
    ensure_marble_exists,
)

# Ensure MARBLE is installed (auto-downloads if needed)
ensure_marble_exists()

# Load and configure tasks
tasks = load_tasks("research", limit=5)
configure_model_ids(tasks, agent_model_id="gpt-4o")

# Create your framework-specific benchmark subclass
class MyMultiAgentBenchmark(MultiAgentBenchBenchmark):
    def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
        # Your framework-specific agent creation
        agent_configs = task.environment_data.get("agents", [])
        # Create agents based on configs...
        ...

    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

# Run benchmark
benchmark = MyMultiAgentBenchmark()
results = benchmark.run(tasks, agent_data={})
```

## MARBLE Reproduction Mode

For exact reproduction of MARBLE's published results, use `MarbleMultiAgentBenchBenchmark` which wraps MARBLE's native agents:

```python
from maseval.benchmark.multiagentbench import (
    MarbleMultiAgentBenchBenchmark,
    load_tasks,
    configure_model_ids,
    ensure_marble_exists,
)

# Ensure MARBLE is installed
ensure_marble_exists()

# Load tasks
tasks = load_tasks("research", limit=5)
configure_model_ids(tasks, agent_model_id="gpt-4o")

# Create benchmark with model adapter
class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
    def get_model_adapter(self, model_id, **kwargs):
        from maseval.interface.openai import OpenAIModelAdapter
        adapter = OpenAIModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

benchmark = MyMarbleBenchmark()
results = benchmark.run(tasks, agent_data={})
```

## Available Domains

| Domain | Description | Infrastructure |
|--------|-------------|----------------|
| `research` | Research idea generation and collaboration | None |
| `bargaining` | Negotiation scenarios (buyer/seller) | None |
| `coding` | Software development collaboration | Filesystem |
| `database` | Database manipulation and querying | Docker + PostgreSQL |
| `web` | Web-based task completion | Network |
| `worldsimulation` | World simulation and interaction | None |
| `minecraft` | Collaborative building | External server |

## API Reference

::: maseval.benchmark.multiagentbench.MultiAgentBenchBenchmark

::: maseval.benchmark.multiagentbench.MarbleMultiAgentBenchBenchmark

::: maseval.benchmark.multiagentbench.MultiAgentBenchEnvironment

::: maseval.benchmark.multiagentbench.MultiAgentBenchEvaluator

::: maseval.benchmark.multiagentbench.MarbleAgentAdapter

::: maseval.benchmark.multiagentbench.load_tasks

::: maseval.benchmark.multiagentbench.configure_model_ids

::: maseval.benchmark.multiagentbench.ensure_marble_exists

::: maseval.benchmark.multiagentbench.download_marble

::: maseval.benchmark.multiagentbench.get_domain_info
