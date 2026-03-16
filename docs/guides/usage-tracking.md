# Usage & Cost Tracking

## Overview

MASEval tracks how much each benchmark run consumes (tokens, API calls, dollars) so you can compare models, stay within budget, and explain where money went.

!!! info "Usage vs Cost"

    **Usage** = Token counts and arbitrary resource units (API calls, data points, etc.)

    **Cost** = Monetary value computed from usage (USD, EUR, credits, etc.)

    Usage is always tracked automatically for LLM calls. Cost requires either a provider that reports it (e.g., LiteLLM) or a pluggable cost calculator.

## What Gets Tracked Automatically

**Model adapters** track every `chat()` call: input tokens, output tokens, cached tokens, reasoning tokens. No setup needed.

**Agent adapters** aggregate token usage from the underlying framework's execution. Cost is computed automatically when litellm is installed (see [Agent Cost Tracking](#agent-cost-tracking) below).

**Benchmarks** collect usage from all registered components after each task and include it in reports.

## Getting Started

### Reading Model Usage

```python
from maseval.interface.inference import OpenAIModelAdapter

model = OpenAIModelAdapter(client=client, model_id="gpt-4")

model.chat([{"role": "user", "content": "Hello"}])
model.chat([{"role": "user", "content": "How are you?"}])

# Accumulated usage across both calls
usage = model.gather_usage()
print(f"{usage.input_tokens} in, {usage.output_tokens} out")
print(f"Cost: ${usage.cost}")  # $0.0 if no cost calculator configured
```

### Reading Agent Usage

Agent adapters expose the same `gather_usage()` interface. Each adapter knows how to extract usage from its framework's internals:

```python
from maseval.interface.agents import SmolAgentAdapter

adapter = SmolAgentAdapter(agent, name="researcher")
adapter.run("What's the capital of France?")

# Usage is aggregated from the agent's memory steps
usage = adapter.gather_usage()
print(f"{usage.input_tokens} in, {usage.output_tokens} out")
```

This works across all supported frameworks (smolagents, CAMEL, LangGraph, and LlamaIndex). The adapter handles the framework-specific extraction; you always call `gather_usage()`.

### Agent Cost Tracking

Agent adapters compute cost automatically when litellm is installed. The adapter detects the model ID from the framework's agent object and uses `LiteLLMCostCalculator` behind the scenes. No configuration needed:

```python
# Cost tracking works automatically if litellm is installed
adapter = SmolAgentAdapter(agent, name="researcher")
adapter.run("What's the capital of France?")
print(f"Cost: ${adapter.gather_usage().cost:.4f}")
```

If auto-detection doesn't work for your setup (e.g., the adapter can't find the model ID), pass `model_id` explicitly:

```python
adapter = LangGraphAgentAdapter(
    compiled_graph, "agent",
    model_id="gpt-4o-mini",
)
```

To use custom pricing instead, pass `cost_calculator` and/or `model_id`:

```python
from maseval import StaticPricingCalculator

calculator = StaticPricingCalculator({
    "my-model": {"input": 0.001, "output": 0.002},
})

adapter = SmolAgentAdapter(
    agent, name="researcher",
    cost_calculator=calculator,
    model_id="my-model",
)
```

If litellm is not installed, auto-creation of the calculator is skipped and cost stays at `0.0`. Tokens are always tracked regardless.

### In Benchmarks

Usage is collected automatically alongside traces and configs after each task. Each report includes a `"usage"` key:

```python
results = benchmark.run()

for report in results:
    print(f"Task {report['task_id']}: {report['usage']}")
```

Live running totals are available during execution:

```python
benchmark.usage               # -> Usage (grand total across all tasks)
benchmark.usage_by_component  # -> Dict[str, Usage] (per-component totals)
```

## Adding Cost Tracking

Most LLM APIs return token counts but not cost. MASEval provides two built-in cost calculators.

### Quick Start: LiteLLM Pricing

The easiest path. Uses LiteLLM's [model pricing database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) covering OpenAI, Anthropic, Google, Mistral, and many more:

```python
from maseval.interface.usage import LiteLLMCostCalculator

calculator = LiteLLMCostCalculator()

model = OpenAIModelAdapter(
    client=client,
    model_id="gpt-4",
    cost_calculator=calculator,
)

response = model.chat([{"role": "user", "content": "Hello"}])
print(f"Cost: ${model.gather_usage().cost:.4f}")
```

!!! tip "LiteLLMModelAdapter already reports cost"

    If you're using `LiteLLMModelAdapter`, it extracts provider-reported cost automatically. You only need `LiteLLMCostCalculator` when using other adapters (OpenAI, Anthropic, Google) and want automatic pricing lookup.

If your model ID doesn't match LiteLLM's naming (e.g., using Google's OpenAI-compatible endpoint), remap it:

```python
calculator = LiteLLMCostCalculator(model_id_map={
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
})
```

You can also override pricing for specific models while using LiteLLM's database for the rest:

```python
calculator = LiteLLMCostCalculator(custom_pricing={
    "my-finetuned-gpt4": {
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
    },
})
```

### Manual Pricing

When you know your rates, use `StaticPricingCalculator`. Zero dependencies, fully explicit:

```python
from maseval import StaticPricingCalculator

calculator = StaticPricingCalculator({
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "claude-sonnet-4-5": {"input": 0.000003, "output": 0.000015},
})
```

Pricing is **per token** (not per 1K or 1M). For cached tokens, add a `"cached_input"` rate:

```python
calculator = StaticPricingCalculator({
    "claude-sonnet-4-5": {
        "input": 0.000003,
        "output": 0.000015,
        "cached_input": 0.0000003,  # 10x cheaper
    },
})
```

The cost unit is whatever your pricing represents (USD, EUR, university credits):

```python
calculator = StaticPricingCalculator({
    "llama-3-70b": {"input": 0.5, "output": 1.0},  # credits per token
})
```

### Sharing a Calculator Across Models

A single calculator instance works for multiple model adapters. The `model_id` is passed on each cost computation:

```python
calculator = StaticPricingCalculator({
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "claude-sonnet-4-5": {"input": 0.000003, "output": 0.000015},
})

model_a = OpenAIModelAdapter(client=client, model_id="gpt-4", cost_calculator=calculator)
model_b = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5", cost_calculator=calculator)
```

### How Cost Is Resolved

When a `ModelAdapter` records usage after a `chat()` call, cost is resolved in priority order:

1. **Provider-reported cost**: some providers (e.g., LiteLLM) include cost in the API response. This always wins.
2. **CostCalculator**: if no provider cost, the adapter calls `calculator.calculate_cost(token_usage, model_id)`.
3. **Zero**: if neither source provides cost, `usage.cost` stays `0.0`.

### Writing a Custom Calculator

Implement the `CostCalculator` protocol (a single method):

```python
from maseval import CostCalculator, TokenUsage
from typing import Optional

class MyCostCalculator:
    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
        rate = MY_PRICING_TABLE.get(model_id)
        if rate is None:
            return None
        return rate["input"] * usage.input_tokens + rate["output"] * usage.output_tokens
```

Return `None` if you don't have pricing for the given model.

## Post-hoc Analysis

After a benchmark completes, `UsageReporter` lets you slice usage by task, component, or both:

```python
from maseval import UsageReporter

reporter = UsageReporter.from_reports(benchmark.reports)

# Grand total
total = reporter.total()
print(f"Total cost: ${total.cost:.4f}")
print(f"Total tokens: {total.input_tokens + total.output_tokens}")

# Where did the money go?
for component, usage in reporter.by_component().items():
    print(f"  {component}: ${usage.cost:.4f}")

# Which tasks were expensive?
for task_id, usage in reporter.by_task().items():
    print(f"  {task_id}: ${usage.cost:.4f}")

# Full nested summary dict
summary = reporter.summary()
```

## How Usage Addition Works

`Usage` records can be added together with `+` or `sum()`. Understanding how fields combine helps you interpret aggregated results.

### Cost

`cost` defaults to `0.0`. Addition is straightforward numeric addition:

```python
from maseval import Usage

a = Usage(cost=0.05)
b = Usage(cost=0.03)
a + b  # cost=0.08

# Components without cost tracking default to 0.0, so they don't affect the total
agent_usage = Usage()  # cost=0.0 (default)
model_usage = Usage(cost=0.12)
agent_usage + model_usage  # cost=0.12

# sum() works with Usage() as the starting value
records = [Usage(cost=0.10), Usage(cost=0.20), Usage(cost=0.05)]
sum(records, Usage())  # cost=0.35
```

### Units

`units` dicts are merged by key. Matching keys are summed, new keys are added:

```python
a = Usage(units={"api_calls": 3, "data_points": 100})
b = Usage(units={"api_calls": 2, "images": 5})
total = a + b
# total.units == {"api_calls": 5, "data_points": 100, "images": 5}
```

### Grouping Fields

`provider`, `category`, `component_name`, and `kind` track where a record came from. When two records are added:

- **Same value** → preserved
- **Different values** → becomes `None` (meaning "aggregated across multiple")

```python
a = Usage(cost=0.05, provider="openai", kind="llm")
b = Usage(cost=0.03, provider="openai", kind="llm")
total = a + b
# total.provider == "openai"  (both match)
# total.kind == "llm"         (both match)

c = Usage(cost=0.10, provider="anthropic", kind="llm")
mixed = a + c
# mixed.provider is None  (openai ≠ anthropic → aggregated over)
# mixed.kind == "llm"     (both match)
```

This lets you tell at a glance whether a summed record came from one source or many.

## Tracking Non-LLM Resources

Tools, environments, and other components can track arbitrary usage by inheriting `UsageTrackableMixin` and overriding `gather_usage()`. Here's an example for a paid API:

```python
from maseval import Usage, UsageTrackableMixin

class BloombergEnvironment(Environment, UsageTrackableMixin):
    def __init__(self, task_data):
        super().__init__(task_data)
        self._usage_records = []

    def _call_bloomberg(self, query):
        result = bloomberg_client.query(query)
        self._usage_records.append(Usage(
            cost=result.billed_amount,
            units={"api_calls": 1, "data_points": result.count},
            provider="bloomberg",
            kind="service",
        ))
        return result

    def gather_usage(self) -> Usage:
        if not self._usage_records:
            return Usage()
        return sum(self._usage_records, Usage())
```

Non-LLM components set cost directly. There is no calculator involvement; each component knows its own billing model.

## Evaluator Usage

Evaluators that use LLM calls (LLM-as-judge) hold a `ModelAdapter`. Register it in the benchmark and its usage is collected separately from agent usage:

```python
class MyBenchmark(Benchmark):
    def setup_evaluators(self, task, environment):
        judge_model = OpenAIModelAdapter(client=client, model_id="gpt-4")
        self.register("evaluator_models", "judge", judge_model)
        return [MyLLMEvaluator(judge_model)]
```

The judge model's usage appears under `usage["evaluator_models"]["judge"]` in the report, so you can distinguish evaluation cost from agent cost.

## Tips

**For cost tracking**: Use `LiteLLMCostCalculator` for automatic pricing, or `StaticPricingCalculator` when you know your rates.

**For custom hosts**: Use `model_id_map` in `LiteLLMCostCalculator` when your adapter's model ID doesn't match LiteLLM's naming.

**For failed tasks**: Usage is collected before error status is determined, so partial usage from failed tasks is still tracked.

**For live monitoring**: Access `benchmark.usage` during execution to check running totals.
