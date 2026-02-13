# Gaia2: Dynamic Multi-Step Scenario Benchmark

The **Gaia2 Benchmark** evaluates LLM-based agents on dynamic, multi-step scenarios using Meta's ARE (Agent Research Environments) platform. It tests agents across multiple capability dimensions in a simulated mobile environment.

## Overview

[Gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2) is designed to evaluate agents in realistic, time-sensitive scenarios. The benchmark features:

- **ARE simulation environment** with real-time dynamics and event scheduling
- **Tool-based time control** via `wait_for_notification()` for temporal reasoning
- **7 capability dimensions**: execution, search, adaptability, time, ambiguity, agent2agent, noise
- **Deterministic evaluation** via GraphPerEventJudge comparing completed vs expected events
- **12 app tools**: Calendar, Email, Messaging, Contacts, Shopping, Cab, City, FileSystem, Browser, ChatsApp, SystemApp, Timer

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"

Check out the [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md) file for more information including licenses.

## Installation

Gaia2 requires additional dependencies:

```bash
pip install maseval[gaia2]
```

Or with uv:

```bash
uv add maseval --extra gaia2
```

## Quick Start

```python
from maseval.benchmark.gaia2 import (
    Gaia2Benchmark, Gaia2Environment, Gaia2Evaluator,
    load_tasks, configure_model_ids, compute_gaia2_metrics,
)

# Load tasks (downloads from HuggingFace automatically)
tasks = load_tasks(capability="execution", limit=5)

# Optionally configure LLM-based judge
configure_model_ids(tasks, evaluator_model_id="gpt-4o")

# Create your framework-specific benchmark subclass
class MyGaia2Benchmark(Gaia2Benchmark):
    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        tools = environment.create_tools()
        # Create your agent with these tools
        ...

    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

# Run benchmark
benchmark = MyGaia2Benchmark()
results = benchmark.run(tasks)

# Compute metrics
metrics = compute_gaia2_metrics(results)
print(f"GSR: {metrics['gsr']:.2%}")
print(f"By capability: {metrics['by_capability']}")
```

For baseline comparisons, use `DefaultAgentGaia2Benchmark` which provides a ReAct-style reference agent:

```python
from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

# Note: You must subclass to provide get_model_adapter()
class MyDefaultGaia2Benchmark(DefaultAgentGaia2Benchmark):
    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

benchmark = MyDefaultGaia2Benchmark(
    agent_data={"model_id": "gpt-4o"},
)
results = benchmark.run(tasks)
```

## Capabilities

Gaia2 tasks are organized by capability dimension:

| Capability     | Description                                      |
| -------------- | ------------------------------------------------ |
| `execution`    | Basic task execution                             |
| `search`       | Information retrieval tasks                      |
| `adaptability` | Adapting to changing requirements                |
| `time`         | Temporal reasoning tasks                         |
| `ambiguity`    | Handling ambiguous instructions                  |
| `agent2agent`  | Multi-agent collaboration                        |
| `noise`        | Handling noisy inputs                            |

Load specific capabilities:

```python
# Load only time-related tasks
tasks = load_tasks(capability="time", limit=10)

# Load all capabilities
tasks = load_tasks(limit=50)
```

## Key Differences from Tau2

| Aspect           | Gaia2                                    | Tau2                              |
| ---------------- | ---------------------------------------- | --------------------------------- |
| Interaction      | Event-driven simulation                  | Turn-based user simulation        |
| Time Control     | Agent calls `wait_for_notification()`    | Fixed turns                       |
| Tools            | ARE app tools (12 apps)                  | Domain-specific tools (3 domains) |
| Evaluation       | Event DAG comparison                     | Database state comparison         |
| User Simulator   | None (events are scheduled)              | LLM-based customer simulator      |

## API Reference

::: maseval.benchmark.gaia2.Gaia2Benchmark

::: maseval.benchmark.gaia2.Gaia2Environment

::: maseval.benchmark.gaia2.Gaia2Evaluator

::: maseval.benchmark.gaia2.DefaultAgentGaia2Benchmark

::: maseval.benchmark.gaia2.DefaultGaia2Agent

::: maseval.benchmark.gaia2.AREToolWrapper

::: maseval.benchmark.gaia2.load_tasks

::: maseval.benchmark.gaia2.configure_model_ids

::: maseval.benchmark.gaia2.compute_gaia2_metrics
