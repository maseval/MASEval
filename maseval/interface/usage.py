"""Usage and cost utilities that depend on optional third-party packages.

This module provides ``LiteLLMCostCalculator``, which uses LiteLLM's
bundled model pricing database to compute cost from token counts.

Requires: ``pip install litellm``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from maseval.core.usage import CostCalculator, TokenUsage  # noqa: F401 — re-export protocol


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
        from maseval.interface.usage import LiteLLMCostCalculator
        from maseval.interface.inference import OpenAIModelAdapter

        calculator = LiteLLMCostCalculator()
        model = OpenAIModelAdapter(client=client, model_id="gpt-4", cost_calculator=calculator)

        # Cost is now computed automatically after each chat() call
        response = model.chat([{"role": "user", "content": "Hello"}])
        print(model.gather_usage().cost)  # e.g., 0.00123
        ```
    """

    def __init__(
        self,
        custom_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        model_id_map: Optional[Dict[str, str]] = None,
    ):
        """Initialize the LiteLLM cost calculator.

        Args:
            custom_pricing: Optional overrides for specific models. Keys are
                model IDs, values are dicts with ``"input_cost_per_token"``
                and ``"output_cost_per_token"``. These take precedence over
                LiteLLM's built-in pricing.
            model_id_map: Optional mapping from adapter model IDs to LiteLLM
                model IDs. Use this when your adapter's ``model_id`` doesn't
                match LiteLLM's naming convention — e.g., when using Google's
                OpenAI-compatible endpoint where the adapter sees
                ``"gemini-2.0-flash"`` but LiteLLM expects
                ``"gemini/gemini-2.0-flash"``.

                Example::

                    LiteLLMCostCalculator(model_id_map={
                        "gemini-2.0-flash": "gemini/gemini-2.0-flash",
                    })
        """
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError("LiteLLMCostCalculator requires litellm. Install it with: pip install litellm") from e

        self._custom_pricing = custom_pricing or {}
        self._model_id_map = model_id_map or {}

    def calculate_cost(self, usage: TokenUsage, model_id: str) -> Optional[float]:
        """Compute cost using LiteLLM's pricing database.

        Args:
            usage: Token usage from the call.
            model_id: The model identifier. Remapped via ``model_id_map``
                if configured, then looked up in custom pricing and
                LiteLLM's database.

        Returns:
            Cost in USD, or ``None`` if LiteLLM doesn't have pricing for
            this model and no custom pricing was provided.
        """
        # Remap model ID if configured
        model_id = self._model_id_map.get(model_id, model_id)

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
                cache_read_input_tokens=usage.cached_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
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
            "model_id_map": dict(self._model_id_map) if self._model_id_map else None,
        }
