"""Pluggable cost calculation for usage records.

This module provides the ``CostCalculator`` protocol and a built-in
``StaticPricingCalculator`` that computes cost from token counts and
user-supplied pricing tables. For automatic pricing via LiteLLM's
bundled model database, see ``maseval.interface.cost``.

Cost calculators are optional — if no calculator is provided to a
``ModelAdapter``, cost is only set when the provider reports it directly
(e.g., LiteLLM's ``response._hidden_params.response_cost``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .usage import TokenUsage


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

        # Non-cached input tokens = total input - cached
        non_cached_input = max(0, usage.input_tokens - usage.cached_input_tokens)

        cost = non_cached_input * input_rate + usage.cached_input_tokens * cached_rate + usage.output_tokens * output_rate

        return cost

    def add_model(self, model_id: str, rates: Dict[str, float]) -> None:
        """Add or update pricing for a model.

        Args:
            model_id: The model identifier.
            rates: Per-token rates (``"input"``, ``"output"``, optionally ``"cached_input"``).
        """
        self._pricing[model_id] = rates

    @property
    def models(self) -> list[str]:
        """List of model IDs with pricing configured."""
        return list(self._pricing.keys())

    def gather_config(self) -> Dict[str, Any]:
        """Return pricing configuration for reproducibility."""
        return {
            "type": type(self).__name__,
            "pricing": dict(self._pricing),
        }
