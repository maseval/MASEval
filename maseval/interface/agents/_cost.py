"""Shared cost-calculator auto-detection for agent adapters."""

from typing import Optional, Tuple

from maseval.core.usage import CostCalculator


def resolve_auto_cost_calculator(
    explicit: Optional[CostCalculator],
    cached: Optional[CostCalculator],
    attempted: bool,
) -> Tuple[Optional[CostCalculator], Optional[CostCalculator], bool]:
    """Resolve the cost calculator, auto-creating one if litellm is available.

    Args:
        explicit: The calculator passed explicitly by the user (may be ``None``).
        cached: The cached auto-calculator from a previous call (``None`` if
            not yet created or creation failed).
        attempted: Whether auto-creation has been attempted before.

    Returns:
        Tuple of ``(calculator_to_use, updated_cache, updated_attempted)``.
        Callers should store the second and third elements back into
        ``self._auto_calculator`` and ``self._auto_attempted``.
    """
    if explicit is not None:
        return explicit, cached, attempted

    if not attempted:
        attempted = True
        try:
            from maseval.interface.usage import LiteLLMCostCalculator

            cached = LiteLLMCostCalculator()
        except (ImportError, Exception):
            cached = None

    return cached, cached, attempted
