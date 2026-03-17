"""Core usage tracking infrastructure for API cost and resource monitoring.

This module provides the `Usage` and `TokenUsage` data classes for recording
billable resource consumption, the `UsageTrackableMixin` that enables
automatic usage collection through the component registry, pluggable
cost calculators (`CostCalculator`, `StaticPricingCalculator`) for translating
token counts into monetary cost, and `UsageReporter` for post-hoc analysis
of usage data from benchmark reports.

Usage tracking is a first-class collection axis alongside tracing
(`TraceableMixin`) and configuration (`ConfigurableMixin`). Components that
inherit `UsageTrackableMixin` have their usage automatically collected by the
registry via `gather_usage()`.

``Usage.cost`` defaults to ``0.0``, so ``Usage()`` works as a starting value
for accumulation (e.g., ``sum(records, Usage())``). Cost calculators are
optional — if no calculator is provided to a ``ModelAdapter``, cost stays
at ``0.0`` unless the provider reports it directly.
For automatic pricing via LiteLLM's bundled model database, see
``maseval.interface.usage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Usage:
    """Generic usage record for any billable resource.

    Represents accumulated cost and countable units for a component or
    aggregated group. All fields default to zero, so ``Usage()`` can be
    used as a starting value for accumulation with ``+`` and ``sum()``.

    Note:
        ``cost`` defaults to ``0.0``. This means adding a ``Usage()``
        to another record never changes the cost:
        ``Usage() + Usage(cost=0.05)`` gives ``cost=0.05``.
        Components that track cost start at ``0.0`` and accumulate upward.
        Components that *do not* track cost (e.g., agent adapters that only
        count tokens) also default to ``0.0`` — their cost simply has no
        effect when summed with components that do report cost.

    Grouping fields (``provider``, ``category``, ``component_name``, ``kind``)
    identify what scope the record covers. When two records are summed,
    matching grouping fields are preserved; mismatches become ``None``
    (meaning "aggregated over").

    Attributes:
        cost: Total cost in USD (or whatever unit your calculator uses).
            Defaults to ``0.0``.
        units: Arbitrary countable units (e.g., ``{"api_calls": 3}``).
        provider: Provider identifier (e.g., ``"anthropic"``, ``"bloomberg"``).
        category: Registry category (e.g., ``"models"``, ``"tools"``).
        component_name: Component name within category (e.g., ``"main_model"``).
        kind: Component kind (e.g., ``"llm"``, ``"service"``, ``"local"``).

    Example:
        ```python
        usage = Usage(cost=0.05, units={"api_calls": 1}, provider="bloomberg", kind="service")

        # Summing preserves matching fields
        total = usage + Usage(cost=0.03, units={"api_calls": 2}, provider="bloomberg", kind="service")
        assert total.cost == 0.08
        assert total.units == {"api_calls": 3}
        assert total.provider == "bloomberg"

        # Usage() is the zero element
        assert (usage + Usage()).cost == 0.05

        # Accumulate with sum()
        records = [Usage(cost=0.10), Usage(cost=0.20), Usage(cost=0.05)]
        assert sum(records, Usage()).cost == 0.35

        # Mismatched grouping fields become None
        mixed = usage + Usage(cost=0.10, provider="anthropic", kind="llm")
        assert mixed.provider is None  # aggregated over
        assert mixed.kind is None      # aggregated over
        ```
    """

    cost: float = 0.0
    units: Dict[str, int | float] = field(default_factory=dict)
    provider: Optional[str] = None
    category: Optional[str] = None
    component_name: Optional[str] = None
    kind: Optional[str] = None

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented

        # Delegate to TokenUsage.__add__ when the right operand is a
        # TokenUsage but self is a plain Usage, so token fields are preserved.
        if type(self) is Usage and isinstance(other, TokenUsage):
            return TokenUsage.__add__(other, self)

        cost = self.cost + other.cost

        # Sum units
        units: Dict[str, int | float] = dict(self.units)
        for key, value in other.units.items():
            units[key] = units.get(key, 0) + value

        # Grouping fields: preserve on match, None on mismatch
        provider = self.provider if self.provider == other.provider else None
        category = self.category if self.category == other.category else None
        component_name = self.component_name if self.component_name == other.component_name else None
        kind = self.kind if self.kind == other.kind else None

        return Usage(
            cost=cost,
            units=units,
            provider=provider,
            category=category,
            component_name=component_name,
            kind=kind,
        )

    def __radd__(self, other: object) -> Usage:
        """Support sum() by handling 0 + Usage."""
        if other == 0:
            return self
        if isinstance(other, Usage):
            return other.__add__(self)
        return NotImplemented

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "cost": self.cost,
            "units": dict(self.units),
            "provider": self.provider,
            "category": self.category,
            "component_name": self.component_name,
            "kind": self.kind,
        }


@dataclass
class TokenUsage(Usage):
    """LLM-specific usage record with token counts.

    Extends `Usage` with token fields reported by LLM providers. Use
    `from_chat_response_usage()` to create from the dict returned by
    model adapters.

    Attributes:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        total_tokens: Total tokens (input + output).
        cached_input_tokens: Tokens served from cache (Anthropic ``cache_read_input_tokens``,
            OpenAI ``cached_tokens``).
        cache_creation_input_tokens: Tokens used to create a new cache entry
            (Anthropic ``cache_creation_input_tokens``). Billed at a higher rate.
        reasoning_tokens: Tokens used for reasoning (OpenAI ``reasoning_tokens``,
            Google ``thoughts_token_count``).
        audio_tokens: Tokens for audio processing (OpenAI).

    Example:
        ```python
        token_usage = TokenUsage.from_chat_response_usage({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        })
        assert token_usage.input_tokens == 100
        ```
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        base = super().__add__(other)
        if not isinstance(base, Usage):
            return NotImplemented

        if isinstance(other, TokenUsage):
            return TokenUsage(
                cost=base.cost,
                units=base.units,
                provider=base.provider,
                category=base.category,
                component_name=base.component_name,
                kind=base.kind,
                input_tokens=self.input_tokens + other.input_tokens,
                output_tokens=self.output_tokens + other.output_tokens,
                total_tokens=self.total_tokens + other.total_tokens,
                cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
                cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
                reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
                audio_tokens=self.audio_tokens + other.audio_tokens,
            )

        # Adding TokenUsage + plain Usage: preserve token fields from self
        return TokenUsage(
            cost=base.cost,
            units=base.units,
            provider=base.provider,
            category=base.category,
            component_name=base.component_name,
            kind=base.kind,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            reasoning_tokens=self.reasoning_tokens,
            audio_tokens=self.audio_tokens,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            **super().to_dict(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_tokens": self.audio_tokens,
        }

    @classmethod
    def from_chat_response_usage(
        cls,
        usage_dict: Dict[str, Any],
        *,
        cost: float = 0.0,
        provider: Optional[str] = None,
        category: Optional[str] = None,
        component_name: Optional[str] = None,
        kind: str = "llm",
    ) -> TokenUsage:
        """Create a TokenUsage from a ChatResponse.usage dict.

        Maps provider-specific key names to the canonical fields.

        Args:
            usage_dict: The usage dict from ``ChatResponse.usage``.
            cost: Cost in USD (e.g., from provider-reported cost). Defaults to ``0.0``.
            provider: Provider identifier.
            category: Registry category.
            component_name: Component name.
            kind: Component kind, defaults to ``"llm"``.

        Returns:
            A TokenUsage instance with mapped fields.
        """
        return cls(
            cost=cost,
            provider=provider,
            category=category,
            component_name=component_name,
            kind=kind,
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            cached_input_tokens=usage_dict.get("cached_input_tokens", 0),
            cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens", 0),
            reasoning_tokens=usage_dict.get("reasoning_tokens", 0),
            audio_tokens=usage_dict.get("audio_tokens", 0),
        )


class UsageTrackableMixin:
    """Mixin that provides usage tracking capability to any component.

    Classes that inherit from UsageTrackableMixin can be registered with a
    Benchmark instance and will have their usage automatically collected
    by the registry via `collect_usage()`.

    The `gather_usage()` method provides a default implementation that returns
    an empty `Usage`. Subclasses should override this to return their
    accumulated usage data.

    How to use:
        For custom components that incur billable costs, inherit from
        UsageTrackableMixin and override `gather_usage()`:

        ```python
        class MyPaidService(TraceableMixin, UsageTrackableMixin):
            def __init__(self):
                self._usage_records: List[Usage] = []

            def call_api(self, query):
                result = api.call(query)
                self._usage_records.append(Usage(
                    cost=result.cost,
                    units={"api_calls": 1},
                ))
                return result

            def gather_usage(self) -> Usage:
                return sum(self._usage_records, Usage())
        ```

        Then register it with your benchmark:

        ```python
        service = MyPaidService()
        benchmark.register("tools", "my_service", service)
        ```

    Thread Safety:
        Usage collection happens synchronously in the main thread after
        task execution completes. Components should use thread-safe data
        structures when accumulating usage during concurrent execution,
        but `gather_usage()` itself is called sequentially.
    """

    def gather_usage(self) -> Usage:
        """Gather accumulated usage from this component.

        Provides a default implementation that returns an empty Usage.
        Subclasses should override this to return their accumulated
        usage data.

        Returns:
            Accumulated usage for this component.

        How to use:
            Override this method to return your component's usage:

            ```python
            def gather_usage(self) -> Usage:
                return sum(self._usage_records, Usage())
            ```
        """
        return Usage()


@runtime_checkable
class CostCalculator(Protocol):
    """Protocol for computing cost from token usage.

    Implementations receive a ``TokenUsage`` and the model ID, and return
    the cost in whatever unit the calculator declares (typically USD).

    Example:
        ```python
        class MyCostCalculator:
            def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
                rate = MY_PRICING.get(model_id)
                if rate is None:
                    return None
                return rate["input"] * usage.input_tokens + rate["output"] * usage.output_tokens
        ```
    """

    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
        """Compute cost for a single chat call.

        Args:
            usage: Token usage from the call.
            model_id: The model identifier (e.g., ``"gpt-4"``, ``"claude-sonnet-4-5"``).

        Returns:
            Cost as a float, or ``None`` if pricing is unknown for this model.
        """
        ...


class StaticPricingCalculator:
    """Cost calculator using user-supplied per-model pricing.

    Pricing is specified as cost per token (not per 1K or 1M tokens).
    If a model is not in the pricing table, ``calculate_cost`` returns ``None``.

    Args:
        pricing: Dict mapping model IDs to their per-token rates.
            Each value is a dict with keys:

            - ``"input"`` — cost per input token (required)
            - ``"output"`` — cost per output token (required)
            - ``"cached_input"`` — cost per cached input token (optional, defaults to ``"input"`` rate)
            - ``"cache_creation_input"`` — cost per cache creation token (optional, defaults to ``"input"`` rate)

    Example:
        ```python
        calculator = StaticPricingCalculator({
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "claude-sonnet-4-5": {"input": 0.000003, "output": 0.000015},
        })

        model = LiteLLMModelAdapter(model_id="gpt-4", cost_calculator=calculator)
        ```

    For university clusters or custom credit systems, the "cost" unit
    is whatever the pricing values represent (credits, EUR, etc.):

        ```python
        calculator = StaticPricingCalculator({
            "llama-3-70b": {"input": 0.5, "output": 1.0},  # credits per token
        })
        ```
    """

    def __init__(self, pricing: Dict[str, Dict[str, float]]):
        self._pricing = pricing

    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
        """Compute cost from static per-token rates.

        Args:
            usage: Token usage from the call.
            model_id: The model identifier to look up in the pricing table.

        Returns:
            Computed cost, or ``None`` if the model is not in the pricing table.
        """
        rates = self._pricing.get(model_id)
        if rates is None:
            return None

        input_rate = rates.get("input", 0.0)
        output_rate = rates.get("output", 0.0)
        cached_rate = rates.get("cached_input", input_rate)
        cache_creation_rate = rates.get("cache_creation_input", input_rate)

        # Non-cached input tokens = total input - cached - cache_creation
        non_cached_input = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_creation_input_tokens)

        cost = (
            non_cached_input * input_rate
            + usage.cached_input_tokens * cached_rate
            + usage.cache_creation_input_tokens * cache_creation_rate
            + usage.output_tokens * output_rate
        )

        return cost

    def add_model(self, model_id: str, rates: Dict[str, float]) -> None:
        """Add or update pricing for a model.

        Args:
            model_id: The model identifier.
            rates: Per-token rates (``"input"``, ``"output"``, optionally ``"cached_input"``).
        """
        self._pricing[model_id] = rates

    @property
    def models(self) -> List[str]:
        """List of model IDs with pricing configured."""
        return list(self._pricing.keys())

    def gather_config(self) -> Dict[str, Any]:
        """Return pricing configuration for reproducibility."""
        return {
            "type": type(self).__name__,
            "pricing": dict(self._pricing),
        }


class UsageReporter:
    """Post-hoc utility for analyzing usage across benchmark reports.

    Walks ``report["usage"]`` across all reports to produce breakdowns
    by task, component, model, etc.

    Example:
        ```python
        reporter = UsageReporter.from_reports(benchmark.reports)
        print(reporter.total())
        print(reporter.by_task())
        print(reporter.by_component())
        ```
    """

    def __init__(self, entries: List[Dict[str, Any]]):
        """Initialize with raw entries extracted from reports.

        Args:
            entries: List of dicts, each with ``"task_id"``, ``"repeat_idx"``,
                and ``"usage_items"`` (list of ``(key, usage_dict)`` tuples).
        """
        self._entries = entries

    @staticmethod
    def from_reports(reports: List[Dict[str, Any]]) -> UsageReporter:
        """Create a UsageReporter from benchmark reports.

        Args:
            reports: The ``benchmark.reports`` list.

        Returns:
            A UsageReporter ready for analysis.
        """
        entries = []
        for report in reports:
            usage_data = report.get("usage")
            if not usage_data or "error" in usage_data:
                continue

            usage_items = []
            for category, value in usage_data.items():
                if category == "metadata":
                    continue
                if isinstance(value, dict) and "cost" in value:
                    # Direct value (environment/user) — it's a usage dict
                    usage_items.append((category, value))
                elif isinstance(value, dict):
                    # Category dict with component names as keys
                    for comp_name, comp_usage in value.items():
                        if isinstance(comp_usage, dict) and "error" not in comp_usage:
                            usage_items.append((f"{category}:{comp_name}", comp_usage))

            entries.append(
                {
                    "task_id": report.get("task_id"),
                    "repeat_idx": report.get("repeat_idx"),
                    "usage_items": usage_items,
                }
            )

        return UsageReporter(entries)

    @staticmethod
    def _usage_from_dict(d: Dict[str, Any]) -> Usage:
        """Reconstruct a Usage (or TokenUsage) from a serialized dict."""
        has_tokens = "input_tokens" in d
        if has_tokens:
            return TokenUsage(
                cost=d.get("cost", 0.0),
                units=d.get("units", {}),
                provider=d.get("provider"),
                category=d.get("category"),
                component_name=d.get("component_name"),
                kind=d.get("kind"),
                input_tokens=d.get("input_tokens", 0),
                output_tokens=d.get("output_tokens", 0),
                total_tokens=d.get("total_tokens", 0),
                cached_input_tokens=d.get("cached_input_tokens", 0),
                cache_creation_input_tokens=d.get("cache_creation_input_tokens", 0),
                reasoning_tokens=d.get("reasoning_tokens", 0),
                audio_tokens=d.get("audio_tokens", 0),
            )
        return Usage(
            cost=d.get("cost", 0.0),
            units=d.get("units", {}),
            provider=d.get("provider"),
            category=d.get("category"),
            component_name=d.get("component_name"),
            kind=d.get("kind"),
        )

    def by_task(self) -> Dict[str, Usage]:
        """Aggregate usage by task_id across all repetitions."""
        result: Dict[str, Usage] = {}
        for entry in self._entries:
            task_id = entry["task_id"]
            for _key, usage_dict in entry["usage_items"]:
                usage = self._usage_from_dict(usage_dict)
                if task_id in result:
                    result[task_id] = result[task_id] + usage
                else:
                    result[task_id] = usage
        return result

    def by_component(self) -> Dict[str, Usage]:
        """Aggregate usage by registry key (e.g., ``"models:main_model"``)."""
        result: Dict[str, Usage] = {}
        for entry in self._entries:
            for key, usage_dict in entry["usage_items"]:
                usage = self._usage_from_dict(usage_dict)
                if key in result:
                    result[key] = result[key] + usage
                else:
                    result[key] = usage
        return result

    def total(self) -> Usage:
        """Grand total across all tasks and components."""
        all_usages = []
        for entry in self._entries:
            for _key, usage_dict in entry["usage_items"]:
                all_usages.append(self._usage_from_dict(usage_dict))
        if not all_usages:
            return Usage()
        result = all_usages[0]
        for u in all_usages[1:]:
            result = result + u
        return result

    def summary(self) -> Dict[str, Any]:
        """Nested dict with all breakdowns."""
        return {
            "total": self.total().to_dict(),
            "by_task": {k: v.to_dict() for k, v in self.by_task().items()},
            "by_component": {k: v.to_dict() for k, v in self.by_component().items()},
        }
