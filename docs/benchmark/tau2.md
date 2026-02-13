# Tau2: Tool-Agent-User Interaction Benchmark

The **Tau2 Benchmark** evaluates LLM-based agents on customer service tasks across multiple real-world domains, testing their ability to use tools, follow policies, and interact with users.

## Overview

[Tau2-bench](https://github.com/sierra-research/tau2-bench) (Tool-Agent-User) is designed to evaluate single-agent customer service systems. The benchmark features:

- **Real tool implementations** that modify actual database state
- **Deterministic evaluation** via database state comparison
- **Three domains**: airline (50 tasks), retail (114 tasks), telecom (114 tasks)
- **Pass@k metrics** for robust evaluation with multiple runs

Reference Paper: [Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)

Check out the [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md) file for more information including licenses.

## Quick Start

```python
from maseval.benchmark.tau2 import (
    Tau2Benchmark, Tau2Environment, Tau2Evaluator, Tau2User,
    load_tasks, configure_model_ids, ensure_data_exists,
    compute_benchmark_metrics, compute_pass_at_k,
)

# Ensure domain data is downloaded
ensure_data_exists(domain="retail")

# Load tasks and configure model IDs
tasks = load_tasks("retail", split="base", limit=5)
configure_model_ids(
    tasks,
    user_model_id="gpt-4o",
    evaluator_model_id="gpt-4o",
)

# Create your framework-specific benchmark subclass
class MyTau2Benchmark(Tau2Benchmark):
    def setup_agents(self, agent_data, environment, task, user):
        tools = environment.tools
        # Create your agent with these tools
        ...

    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

# Run benchmark
benchmark = MyTau2Benchmark(agent_data={}, n_task_repeats=4)
results = benchmark.run(tasks)

# Compute metrics
metrics = compute_benchmark_metrics(results)
pass_k = compute_pass_at_k(results, k_values=[1, 2, 3, 4])
```

For baseline comparisons, use `DefaultAgentTau2Benchmark` which mirrors the original tau2-bench implementation:

```python
from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark

benchmark = DefaultAgentTau2Benchmark(
    agent_data={"model_id": "gpt-4o"},
    n_task_repeats=4,
)
results = benchmark.run(tasks)
```

::: maseval.benchmark.tau2.Tau2Benchmark

::: maseval.benchmark.tau2.Tau2User

::: maseval.benchmark.tau2.Tau2Environment

::: maseval.benchmark.tau2.Tau2Evaluator

::: maseval.benchmark.tau2.DefaultAgentTau2Benchmark

::: maseval.benchmark.tau2.DefaultTau2Agent

::: maseval.benchmark.tau2.load_tasks

::: maseval.benchmark.tau2.configure_model_ids

::: maseval.benchmark.tau2.ensure_data_exists

::: maseval.benchmark.tau2.compute_benchmark_metrics

::: maseval.benchmark.tau2.compute_pass_at_k

::: maseval.benchmark.tau2.compute_pass_hat_k
