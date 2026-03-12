# Usage & Cost Tracking

## Overview

MASEval provides first-class usage and cost tracking to monitor resource consumption during benchmark execution. This is useful for:

- **Cost control**: Track how much each benchmark run costs across providers
- **Budgeting**: Compare cost across models, tasks, and components
- **Billing**: Support custom credit systems (university clusters, internal APIs)
- **Analysis**: Understand token usage patterns per task, agent, or model

!!! info "Usage vs Cost"

    **Usage** = Token counts and arbitrary resource units (API calls, data points, etc.)

    **Cost** = Monetary value computed from usage (USD, EUR, credits, etc.)

    Usage is always tracked automatically for LLM calls. Cost requires either a provider that reports it (e.g., LiteLLM) or a pluggable cost calculator.

## Core Concepts

**`Usage`**: Generic usage record for any billable resource — cost, arbitrary units, and grouping metadata.

**`TokenUsage`**: LLM-specific extension of `Usage` with token fields (`input_tokens`, `output_tokens`, `cached_input_tokens`, etc.).

**`UsageTrackableMixin`**: Mixin that enables automatic usage collection for any component via `gather_usage()`.

**`CostCalculator`**: Protocol for pluggable cost computation from token counts.

## Automatic LLM Usage Tracking

All `ModelAdapter` subclasses track token usage automatically. No configuration needed — every `chat()` call records a `TokenUsage` entry internally.

```python
from maseval.interface.inference import OpenAIModelAdapter

model = OpenAIModelAdapter(client=client, model_id="gpt-4")

# Make some calls
model.chat([{"role": "user", "content": "Hello"}])
model.chat([{"role": "user", "content": "How are you?"}])

# Inspect accumulated usage
usage = model.gather_usage()
print(usage.input_tokens)   # e.g., 25
print(usage.output_tokens)  # e.g., 42
print(usage.cost)           # None (no cost calculator configured)
```

### In Benchmarks

Usage is collected automatically alongside traces and configs after each task repetition. Each report includes a `"usage"` key:

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

## Cost Calculation

Most LLM APIs return token counts but not cost. Cost is a client-side concern. MASEval provides two built-in cost calculators and a protocol for custom ones.

### Cost Priority

When a `ModelAdapter` records usage after a `chat()` call, cost is resolved in this order:

1. **Provider-reported cost** — e.g., LiteLLM sets `response._hidden_params.response_cost` directly. This always wins.
2. **CostCalculator** — if no provider cost, the adapter calls `calculator.calculate_cost(token_usage, model_id)`.
3. **None** — if neither source provides cost, `Usage.cost` stays `None`.

### StaticPricingCalculator

Zero-dependency calculator using user-supplied per-token rates. Lives in `maseval.core.usage`.

```python
from maseval import StaticPricingCalculator

calculator = StaticPricingCalculator({
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "claude-sonnet-4-5": {"input": 0.000003, "output": 0.000015},
})

model = OpenAIModelAdapter(
    client=client,
    model_id="gpt-4",
    cost_calculator=calculator,
)

response = model.chat([{"role": "user", "content": "Hello"}])
print(model.gather_usage().cost)  # e.g., 0.00234
```

Pricing is per token (not per 1K or 1M). Cached input tokens are handled automatically — set a `"cached_input"` rate to differentiate:

```python
calculator = StaticPricingCalculator({
    "claude-sonnet-4-5": {
        "input": 0.000003,
        "output": 0.000015,
        "cached_input": 0.0000003,  # 10x cheaper for cached tokens
    },
})
```

For custom unit systems (university credits, EUR, etc.), the "cost" unit is whatever your pricing represents:

```python
calculator = StaticPricingCalculator({
    "llama-3-70b": {"input": 0.5, "output": 1.0},  # credits per token
})
```

### LiteLLMCostCalculator

Uses LiteLLM's bundled [model pricing database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) for automatic cost calculation. Covers OpenAI, Anthropic, Google, Mistral, Cohere, and many more.

```python
from maseval.interface.usage import LiteLLMCostCalculator

calculator = LiteLLMCostCalculator()

model = OpenAIModelAdapter(
    client=client,
    model_id="gpt-4",
    cost_calculator=calculator,
)
```

!!! tip "LiteLLMModelAdapter already reports cost"

    If you're using the `LiteLLMModelAdapter`, it extracts provider-reported cost from `response._hidden_params.response_cost` automatically. You only need `LiteLLMCostCalculator` when using other adapters (OpenAI, Anthropic, Google) and want automatic pricing lookup.

#### Custom Pricing Overrides

Override pricing for specific models while using LiteLLM's database for the rest:

```python
calculator = LiteLLMCostCalculator(custom_pricing={
    "my-finetuned-gpt4": {
        "input_cost_per_token": 0.00006,
        "output_cost_per_token": 0.00012,
    },
})
```

#### Model ID Remapping

When your adapter's `model_id` doesn't match LiteLLM's naming convention (e.g., using Google's OpenAI-compatible endpoint), use `model_id_map` to remap:

```python
calculator = LiteLLMCostCalculator(model_id_map={
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "my-custom-gpt4": "gpt-4",
})
```

The map is applied before both custom pricing and LiteLLM lookup.

### Custom Cost Calculator

Implement the `CostCalculator` protocol for custom pricing logic:

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

The protocol requires a single method: `calculate_cost(usage, model_id) -> Optional[float]`. Return `None` if you don't have pricing for the given model.

### Sharing Calculators Across Adapters

A single calculator instance can be shared across multiple model adapters. The `model_id` is passed on each call, so the calculator can look up the right pricing:

```python
calculator = StaticPricingCalculator({
    "gpt-4": {"input": 0.00003, "output": 0.00006},
    "claude-sonnet-4-5": {"input": 0.000003, "output": 0.000015},
})

model_a = OpenAIModelAdapter(client=client, model_id="gpt-4", cost_calculator=calculator)
model_b = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5", cost_calculator=calculator)
```

## Non-LLM Usage Tracking

Tools, environments, and other components can track usage by inheriting `UsageTrackableMixin` and overriding `gather_usage()`:

```python
from maseval import Usage, UsageTrackableMixin
from maseval.core.tracing import TraceableMixin

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

Non-LLM components set cost directly in their `Usage` records — there is no calculator involvement. Each component knows its own billing model.

## Post-hoc Analysis with UsageReporter

`UsageReporter` provides sliced analysis across all benchmark reports:

```python
from maseval import UsageReporter

reporter = UsageReporter.from_reports(benchmark.reports)

# Grand total
total = reporter.total()
print(f"Total cost: ${total.cost:.4f}")
print(f"Total tokens: {total.input_tokens + total.output_tokens}")

# Per-task breakdown
for task_id, usage in reporter.by_task().items():
    print(f"  {task_id}: ${usage.cost:.4f}")

# Per-component breakdown
for component, usage in reporter.by_component().items():
    print(f"  {component}: ${usage.cost:.4f}")

# Full nested summary dict
summary = reporter.summary()
```

## Usage Data Model

### Usage

Generic record for any billable resource:

| Field | Type | Description |
|-------|------|-------------|
| `cost` | `Optional[float]` | Cost in USD (or custom unit). `None` = unknown. |
| `units` | `Dict[str, int\|float]` | Arbitrary countable units (e.g., `{"api_calls": 3}`). |
| `provider` | `Optional[str]` | Provider identifier (e.g., `"anthropic"`). |
| `category` | `Optional[str]` | Registry category (e.g., `"models"`, `"tools"`). |
| `component_name` | `Optional[str]` | Component name (e.g., `"main_model"`). |
| `kind` | `Optional[str]` | Component kind (e.g., `"llm"`, `"service"`). |

`Usage` supports addition: costs sum (both known) or become `None` (either unknown), units sum, grouping fields are preserved on match or set to `None` on mismatch.

### TokenUsage

Extends `Usage` with LLM-specific token counts:

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Input/prompt tokens. |
| `output_tokens` | `int` | Output/completion tokens. |
| `total_tokens` | `int` | Total tokens. |
| `cached_input_tokens` | `int` | Tokens served from cache. |
| `reasoning_tokens` | `int` | Reasoning/thinking tokens. |
| `audio_tokens` | `int` | Audio processing tokens. |

## Evaluator Usage

Evaluators that use LLM calls (LLM-as-judge) hold a `ModelAdapter`. Register the evaluator's model in the benchmark and its usage is collected automatically:

```python
class MyBenchmark(Benchmark):
    def setup_evaluators(self, task, environment):
        judge_model = OpenAIModelAdapter(client=client, model_id="gpt-4")
        self.register("evaluator_models", "judge", judge_model)
        return [MyLLMEvaluator(judge_model)]
```

The judge model's usage appears under `usage["evaluator_models"]["judge"]` in the report, separate from the agent's model usage.

## Tips

**For cost tracking**: Use `LiteLLMCostCalculator` for automatic pricing, or `StaticPricingCalculator` for custom rates.

**For custom hosts**: Use `model_id_map` in `LiteLLMCostCalculator` when your adapter's model ID doesn't match LiteLLM's naming.

**For failed tasks**: Usage is collected before error status is determined, so partial usage from failed tasks is still tracked.

**For live monitoring**: Access `benchmark.usage` during execution to check running totals.
