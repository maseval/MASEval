# Usage & Cost Tracking — Implementation Plan

## Motivation

Benchmarking multi-agent systems incurs real costs: LLM API calls (the primary driver), but also external service calls (e.g., Bloomberg data API, geocoding services, paid search APIs). MASEval currently extracts basic token counts into `ChatResponse.usage` but does not persist, enrich, or aggregate this data. We want first-class usage tracking that:

- Captures token usage and cost per LLM call with provider-specific detail
- Supports non-token costs (external service calls billed per-request or per-unit)
- Aggregates across provider, task, component role, and total
- Is queryable live during benchmark execution (not just post-hoc)
- Captures usage even for failed tasks
- Requires zero changes from benchmark implementers for the common LLM case

## Design Principles

1. **LLM-first, not LLM-only.** The base abstraction is generic (cost + arbitrary units), with an LLM-specific subclass that adds token semantics.
2. **No hardcoded prices.** Pricing changes constantly. Users supply pricing or rely on provider-reported cost (e.g., OpenRouter). If neither is available, cost is `None`.
3. **Automatic for models, opt-in for tools.** ModelAdapter tracks usage automatically via the base `chat()` method. Tool/environment authors opt in via `UsageTrackableMixin`.
4. **Non-breaking.** `ChatResponse.usage` stays a `Dict[str, int]` with additional optional keys. Existing code that reads `usage["input_tokens"]` continues to work.
5. **First-class collection axis.** Usage is collected via `gather_usage()` / `collect_usage()`, parallel to `gather_traces()` / `collect_traces()` and `gather_config()` / `collect_configs()`. It is not embedded inside traces.
6. **Live queryable.** The registry maintains a running usage total across repetitions, queryable at any time via `benchmark.usage`.

---

## Data Model

### `Usage` (base)

Generic usage record for any billable resource. Stored as a simple dataclass.

```
Usage
  cost: Optional[float]            # Total cost in USD (None = unknown)
  units: Dict[str, int | float]    # Countable units (e.g., {"api_calls": 3, "bytes": 1024})
  provider: Optional[str]          # e.g., "anthropic", "openai", "bloomberg"
  category: Optional[str]          # e.g., "models", "evaluator_models", "tools"
  component_name: Optional[str]    # e.g., "main_model", "judge", "bloomberg_api"
  kind: Optional[str]              # e.g., "llm", "service", "local"
```

Supports `__add__`: costs sum (if both known, else None), units sum. Grouping fields (`provider`, `category`, `component_name`, `kind`) are preserved when they match, set to `None` on mismatch. `None` means "aggregated over" — e.g., `provider=None, category="models"` represents all models summed across providers. A fully `None` grouping is a grand total.

### `TokenUsage(Usage)` (LLM-specific)

Extends `Usage` with token fields that every LLM provider reports.

```
TokenUsage(Usage)
  input_tokens: int
  output_tokens: int
  total_tokens: int
  # Optional provider-specific detail
  cached_input_tokens: int        # Anthropic cache_read, OpenAI cached_tokens
  reasoning_tokens: int           # OpenAI reasoning, Google thoughts
  audio_tokens: int               # OpenAI audio
```

`TokenUsage.__add__` sums all token fields plus delegates to `Usage.__add__` for cost/units.

Class method `TokenUsage.from_chat_response_usage(usage_dict) -> TokenUsage` maps the dict returned by adapters today into a `TokenUsage` instance, handling provider-specific key names.

---

## UsageTrackableMixin

Follows the established mixin pattern (`TraceableMixin`, `ConfigurableMixin`). Any component that inherits `UsageTrackableMixin` will have its usage automatically collected by the registry when registered.

```python
class UsageTrackableMixin:
    """Mixin that provides usage tracking capability to any component."""

    def gather_usage(self) -> Usage:
        """Return accumulated usage for this component.

        Subclasses must override this to return their accumulated Usage.
        Base implementation returns an empty Usage.
        """
        return Usage()
```

Components internally accumulate `Usage` records however they see fit (typically a list + sum). The mixin only defines the collection protocol — `gather_usage() -> Usage`.

### Usage in components

**ModelAdapter** (automatic):

```python
class ModelAdapter(ABC, TraceableMixin, ConfigurableMixin, UsageTrackableMixin):
    def __init__(self, seed=None):
        super().__init__()
        self._usage_records: List[Usage] = []

    def chat(self, messages, ...):
        response = self._chat_impl(messages, ...)
        if response.usage:
            self._usage_records.append(
                TokenUsage.from_chat_response_usage(response.usage)
            )
        return response

    def gather_usage(self) -> Usage:
        if not self._usage_records:
            return Usage()
        return sum(self._usage_records[1:], self._usage_records[0])
```

**Non-model components** (opt-in):

```python
class BloombergEnvironment(Environment, UsageTrackableMixin):
    def __init__(self, task_data):
        super().__init__(task_data)
        self._usage_records: List[Usage] = []

    def _call_bloomberg(self, query):
        result = bloomberg_client.query(query)
        self._usage_records.append(Usage(
            cost=result.billed_amount,
            units={"api_calls": 1, "data_points": result.count},
        ))
        return result

    def gather_usage(self) -> Usage:
        if not self._usage_records:
            return Usage()
        return sum(self._usage_records[1:], self._usage_records[0])
```

---

## Registry Integration

The `ComponentRegistry` gains a third collection axis for usage, parallel to traces and configs.

### Per-repetition collection

`collect_usage()` walks all registered `UsageTrackableMixin` components and calls `gather_usage()` on each. Returns a structured dict (same shape as `collect_traces()`/`collect_configs()`). This goes into `report["usage"]`.

```python
def collect_usage(self) -> Dict[str, Any]:
    """Collect usage from all registered UsageTrackableMixin components."""
    usage = {
        "metadata": {...},
        "agents": {},
        "models": {},
        "tools": {},
        ...
        "environment": None,
        "user": None,
    }

    for key, component in self._usage_registry.items():
        category, comp_name = key.split(":", 1)
        component_usage = component.gather_usage()

        # Store in structured dict (same pattern as traces/configs)
        ...

        # Accumulate into persistent aggregates
        self._usage_total += component_usage
        self._usage_by_component[key] += component_usage

    return usage
```

### Persistent aggregates (survive `clear()`)

The registry maintains running totals that persist across task repetitions:

```python
class ComponentRegistry:
    def __init__(self):
        # ... existing per-repetition state ...

        # Persistent usage aggregates (NOT cleared between repetitions)
        self._usage_total: Usage = Usage()
        self._usage_by_component: Dict[str, Usage] = {}

    def clear(self):
        # Clears per-repetition registrations
        # Does NOT clear _usage_total or _usage_by_component

    @property
    def total_usage(self) -> Usage:
        """Running total across all repetitions. Queryable at any time."""
        return self._usage_total

    @property
    def usage_by_component(self) -> Dict[str, Usage]:
        """Per-component running totals across all repetitions."""
        return dict(self._usage_by_component)
```

### Registration

The `register()` method gains an `isinstance(component, UsageTrackableMixin)` check, parallel to the existing `TraceableMixin` and `ConfigurableMixin` checks:

```python
def register(self, category, name, component):
    # ... existing trace/config registration ...

    if isinstance(component, UsageTrackableMixin):
        self._usage_registry[key] = component
        self._usage_component_id_map[component_id] = key
```

`RegisterableComponent` type alias is updated to include `UsageTrackableMixin`.

---

## Benchmark Integration

### Report structure

Each report gains a top-level `"usage"` key alongside `"traces"` and `"config"`:

```python
report = {
    "task_id": str(task.id),
    "repeat_idx": repeat_idx,
    "status": execution_status.value,
    "traces": execution_traces,
    "config": execution_configs,
    "usage": execution_usage,      # <-- new
    "eval": eval_results,
    "task": {...},
}
```

### Live usage access

```python
benchmark.usage        # -> Usage (running grand total, delegates to registry)
benchmark.usage_by_component  # -> Dict[str, Usage] (per-component totals)
```

### Failed task usage

`collect_usage()` is called alongside `collect_all_traces()` and `collect_all_configs()` — before error status is determined. If a task fails mid-execution, whatever usage was accumulated up to the failure point is still collected and aggregated.

---

## Adapter `_chat_impl` Enrichment (per-provider)

Each adapter enriches the `ChatResponse.usage` dict with provider-specific fields beyond the basic three. The base class `TokenUsage.from_chat_response_usage()` handles mapping.

| Adapter | Extra fields to extract |
|---------|------------------------|
| OpenAI | `reasoning_tokens` from `completion_tokens_details`, `cached_input_tokens` from `prompt_tokens_details.cached_tokens` |
| Anthropic | `cached_input_tokens` from `cache_read_input_tokens` |
| Google | `reasoning_tokens` from `thoughts_token_count` |
| LiteLLM | `reasoning_tokens` + `cached_input_tokens` from details; `cost` from `response._hidden_params` if available |
| HuggingFace | No change (local inference, no API cost) |

---

## UsageReporter (post-hoc)

Post-run utility that walks `report["usage"]` across all reports for sliced analysis.

```
UsageReporter
  @staticmethod from_reports(reports: List[Dict]) -> UsageReporter

  by_task() -> Dict[str, Usage]           # keyed by task_id
  by_component() -> Dict[str, Usage]      # keyed by registry key (e.g., "models:main_model")
  by_model() -> Dict[str, TokenUsage]     # keyed by model_id (LLM-only)
  total() -> Usage                        # grand total

  summary() -> Dict[str, Any]             # nested dict with all breakdowns
```

Unlike the registry's live aggregates, `UsageReporter` can slice by task (since it sees the full report list with task IDs).

---

## Evaluators

Evaluators that use LLM calls (LLM-as-judge) hold a `ModelAdapter`. That model should be registered in the benchmark via `self.register("evaluator_models", "judge", model)` inside `setup_evaluators()`. Since `ModelAdapter` now inherits `UsageTrackableMixin`, its usage is automatically collected under `usage.evaluator_models.judge`.

No changes to the `Evaluator` base class. This is a registration convention.

## LLMUser / AgenticLLMUser

These already hold a `ModelAdapter`. Their model's usage is collected automatically (since `ModelAdapter` inherits `UsageTrackableMixin` and `chat()` accumulates records). The model is already registered by the benchmark. No changes needed.

---

## File Plan

| File | Action | Content |
|------|--------|---------|
| `maseval/core/usage.py` | **Create** | `Usage`, `TokenUsage`, `UsageTrackableMixin` |
| `maseval/core/cost.py` | **Create** | `CostCalculator` protocol, `StaticPricingCalculator` |
| `maseval/core/registry.py` | **Edit** | Add `_usage_registry`, `_usage_total`, `_usage_by_component`, `collect_usage()`, `total_usage` property |
| `maseval/core/model.py` | **Edit** | Add `UsageTrackableMixin` to `ModelAdapter`, accumulate `TokenUsage` in `chat()`, implement `gather_usage()`, accept `cost_calculator` param |
| `maseval/core/benchmark.py` | **Edit** | Add `collect_all_usage()`, `usage` property, include `"usage"` in report dict |
| `maseval/core/reporting.py` | **Create** | `UsageReporter` post-hoc analysis utility |
| `maseval/interface/cost.py` | **Create** | `LiteLLMCostCalculator` (optional `litellm` dependency) |
| `maseval/interface/inference/openai.py` | **Edit** | Enrich `ChatResponse.usage` with `reasoning_tokens`, `cached_input_tokens`; accept `cost_calculator` |
| `maseval/interface/inference/anthropic.py` | **Edit** | Enrich with `cached_input_tokens`; accept `cost_calculator` |
| `maseval/interface/inference/google_genai.py` | **Edit** | Enrich with `reasoning_tokens`; accept `cost_calculator` |
| `maseval/interface/inference/litellm.py` | **Edit** | Enrich with detail tokens + provider-reported `cost`; accept `cost_calculator` |
| `maseval/interface/inference/huggingface.py` | **Edit** | Accept `cost_calculator` |
| `maseval/__init__.py` | **Edit** | Export `Usage`, `TokenUsage`, `UsageTrackableMixin`, `CostCalculator`, `StaticPricingCalculator`, `UsageReporter` |
| `tests/test_usage.py` | **Create** | Unit tests for data model, mixin, registry collection, aggregation, cost calculators |

No changes to: `evaluator.py`, `user.py`, `agent.py`, `environment.py`, `callback.py`, `tracing.py`, `config.py`.

---

## Cost Calculation

Most LLM APIs return token counts but **not** cost. Cost calculation is a client-side concern.

### CostCalculator protocol

A `CostCalculator` is a simple protocol with one method:

```python
class CostCalculator(Protocol):
    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]: ...
```

`ModelAdapter` accepts an optional `cost_calculator` parameter. After each `chat()` call, if the provider didn't report cost and a calculator is present, the calculator fills in `TokenUsage.cost`. Provider-reported cost always takes precedence.

### Built-in implementations

| Calculator | Location | Dependencies | Use case |
|-----------|----------|-------------|----------|
| `StaticPricingCalculator` | `maseval.core.cost` | None | User-supplied per-model rates. Supports custom units (USD, EUR, credits). |
| `LiteLLMCostCalculator` | `maseval.interface.cost` | `litellm` | Automatic pricing via LiteLLM's bundled model database. Covers OpenAI, Anthropic, Google, Mistral, etc. |

### Cost flow (priority order)

1. **Provider-reported cost** — e.g., LiteLLM's `response._hidden_params.response_cost`. Set directly in `ChatResponse.usage["cost"]`.
2. **CostCalculator** — if no provider cost, `ModelAdapter.chat()` calls `calculator.calculate_cost(token_usage, model_id)`.
3. **None** — if neither source provides cost, `Usage.cost` stays `None`.

### Examples

```python
# Static pricing for a university cluster (credits per token)
calculator = StaticPricingCalculator({
    "llama-3-70b": {"input": 0.5, "output": 1.0},
})

# Automatic pricing via LiteLLM's database
from maseval.interface.cost import LiteLLMCostCalculator
calculator = LiteLLMCostCalculator()

# Pass to any model adapter
model = OpenAIModelAdapter(client=client, model_id="gpt-4", cost_calculator=calculator)
```

### Non-LLM components

Non-LLM components (tools, environments) set cost directly in their `gather_usage()` implementation — there is no calculator involvement. Each component knows its own billing model.

---

## Non-goals

- **Hardcoded pricing tables** — prices change too often; delegated to LiteLLM or user-supplied.
- **Agent-internal model tracking** — models inside agent frameworks (AutoGen, LangGraph internals) are out of scope for now.
- **Billing integration** — no webhook/billing system integration.
- **Streaming usage** — not supported yet (usage is captured after completion).
- **Currency conversion** — `Usage.cost` is a bare float in whatever unit the calculator uses. Mixing units in one benchmark is a user error.

## Open Questions

1. **HuggingFace local inference**: Should we track compute-time as a "cost" proxy for local models? Probably not in v1.
