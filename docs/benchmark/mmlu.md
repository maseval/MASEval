# MMLU: Massive Multitask Language Understanding (Beta)

!!! warning "Beta"
    This benchmark has been implemented carefully, but we have not yet validated the results against the original implementation. Use with caution when comparing with existing results or the original paper's numbers. Contributions and compute donations welcome!

The **MMLU Benchmark** evaluates language models on multiple-choice questions spanning 57 academic subjects. The MASEval integration supports anchor-point-based evaluation for [DISCO](https://arxiv.org/abs/2510.07959) prediction, enabling efficient estimation of full benchmark performance from a subset of tasks.

## Overview

[MMLU](https://arxiv.org/abs/2009.03300) (Hendrycks et al., 2021) is a widely used benchmark for measuring knowledge and reasoning across diverse domains. The MASEval implementation features:

- **Log-likelihood MCQ evaluation** matching [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) methodology
- **Anchor-point task selection** via `DISCOQueue` for DISCO-style subset evaluation
- **HuggingFace integration** with batched log-probability computation
- **lm-eval compatibility** mode for exact numerical reproduction

Check out the [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md) file for more information including licenses.

## Installation

Install MMLU with all dependencies needed to run the HuggingFace benchmark and example script:

```bash
pip install maseval[mmlu]
```

Or with uv:

```bash
uv sync --extra mmlu
```

This installs `transformers`, `torch`, `numpy`, and `huggingface_hub` (the latter two via `transformers`). You can then run the example:

```bash
python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full
```

For DISCO prediction support:

```bash
pip install maseval[disco]
```

For exact lm-evaluation-harness reproduction:

```bash
pip install maseval[lm-eval]
```

## Quick Start

```python
from maseval.benchmark.mmlu import (
    DefaultMMLUBenchmark,
    load_tasks,
    compute_benchmark_metrics,
)

# Load tasks (downloads from HuggingFace automatically)
tasks = load_tasks(data_path="/path/to/mmlu_prompts_examples.json")

# Create benchmark with HuggingFace model
benchmark = DefaultMMLUBenchmark(
    model_id="meta-llama/Llama-2-7b-hf",
    device="cuda:0",
)

# Run evaluation
results = benchmark.run(
    tasks=tasks,
    agent_data={"model_id": "meta-llama/Llama-2-7b-hf"},
)

# Compute metrics
metrics = compute_benchmark_metrics(results)
print(f"Accuracy: {metrics['acc']:.4f}")
```

### With Anchor Points (DISCO)

```python
from maseval.benchmark.mmlu import load_tasks

# Load tasks filtered to anchor points
tasks = load_tasks(
    data_path="/path/to/mmlu_prompts_examples.json",
    anchor_points_path="/path/to/anchor_points.json",
)

# tasks is a DISCOQueue — only anchor tasks are evaluated
print(f"Evaluating {len(tasks)} anchor tasks")
```

## Custom Benchmark Subclass

`MMLUBenchmark` is a framework-agnostic base class. To use a different model backend, subclass it and implement `setup_agents()` and `get_model_adapter()`:

```python
from maseval import AgentAdapter
from maseval.core.history import MessageHistory
from maseval.benchmark.mmlu import MMLUBenchmark

class MyAgentAdapter(AgentAdapter):
    def __init__(self, model, name):
        super().__init__(model, name)
        self._messages = []

    def _run_agent(self, query):
        self._messages.append({"role": "user", "content": query})
        response = self.agent.generate(query)
        self._messages.append({"role": "assistant", "content": response})
        return response

    def get_messages(self):
        return MessageHistory(self._messages)

class MyMMLUBenchmark(MMLUBenchmark):
    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        model = self.get_model_adapter(agent_data["model_id"])
        adapter = MyAgentAdapter(model, name="mmlu_agent")
        return [adapter], {"mmlu_agent": adapter}

    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        register_name = kwargs.get("register_name")
        if register_name:
            self.register("models", register_name, adapter)
        return adapter
```

## API Reference

::: maseval.benchmark.mmlu.MMLUBenchmark

::: maseval.benchmark.mmlu.DefaultMMLUBenchmark

::: maseval.benchmark.mmlu.MMLUEnvironment

::: maseval.benchmark.mmlu.MMLUEvaluator

::: maseval.benchmark.mmlu.load_tasks

::: maseval.benchmark.mmlu.compute_benchmark_metrics
