"""Cost calculators that depend on optional third-party packages.

This module provides ``LiteLLMCostCalculator``, which uses LiteLLM's
bundled model pricing database to compute cost from token counts.

Requires: ``pip install litellm``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from maseval.core.cost import CostCalculator  # noqa: F401 — re-export protocol
from maseval.core.usage import TokenUsage


class LiteLLMCostCalculator:
    """Cost calculator using LiteLLM's bundled pricing database.

    LiteLLM maintains a comprehensive `model_prices_and_context_window.json
    <https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json>`_
    that covers most major LLM providers. This calculator delegates to
    ``litellm.cost_per_token`` for per-token rates and computes the total.

    This is the recommended calculator for most users — it covers OpenAI,
    Anthropic, Google, Mistral, Cohere, and many more without requiring
    manual pricing tables.

    Note:
        If you're already using the ``LiteLLMModelAdapter``, it extracts
        provider-reported cost from ``response._hidden_params.response_cost``
        automatically. This calculator is useful as a fallback when using
        other adapters (OpenAI, Anthropic, Google) directly.

    Example:
        ```python
        from maseval.interface.cost import LiteLLMCostCalculator
        from maseval.interface.inference import OpenAIModelAdapter

        calculator = LiteLLMCostCalculator()
        model = OpenAIModelAdapter(client=client, model_id="gpt-4", cost_calculator=calculator)

        # Cost is now computed automatically after each chat() call
        response = model.chat([{"role": "user", "content": "Hello"}])
        print(model.gather_usage().cost)  # e.g., 0.00123
        ```
    """

    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """Initialize the LiteLLM cost calculator.

        Args:
            custom_pricing: Optional overrides for specific models. Keys are
                model IDs, values are dicts with ``"input_cost_per_token"``
                and ``"output_cost_per_token"``. These take precedence over
                LiteLLM's built-in pricing.
        """
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError("LiteLLMCostCalculator requires litellm. Install it with: pip install litellm") from e

        self._custom_pricing = custom_pricing or {}

    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
        """Compute cost using LiteLLM's pricing database.

        Args:
            usage: Token usage from the call.
            model_id: The model identifier (must match LiteLLM's naming).

        Returns:
            Cost in USD, or ``None`` if LiteLLM doesn't have pricing for
            this model and no custom pricing was provided.
        """
        # Check custom overrides first
        if model_id in self._custom_pricing:
            rates = self._custom_pricing[model_id]
            input_cost = rates.get("input_cost_per_token", 0.0) * usage.input_tokens
            output_cost = rates.get("output_cost_per_token", 0.0) * usage.output_tokens
            return input_cost + output_cost

        # Fall back to LiteLLM's built-in pricing
        try:
            import litellm

            input_cost, output_cost = litellm.cost_per_token(
                model=model_id,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
            return input_cost + output_cost
        except Exception:
            # Model not in LiteLLM's pricing database
            return None

    def gather_config(self) -> Dict[str, Any]:
        """Return calculator configuration for reproducibility."""
        return {
            "type": type(self).__name__,
            "custom_pricing": dict(self._custom_pricing) if self._custom_pricing else None,
        }
