"""Tests for usage tracking and cost calculation correctness.

Verifies that:
- TokenUsage arithmetic produces correct results
- StaticPricingCalculator computes exact expected costs
- LiteLLMCostCalculator passes the right parameters to litellm
- Full pipeline (adapter → TokenUsage → CostCalculator → cost) is correct
- UsageReporter aggregates correctly from report dicts
- Serialization roundtrips preserve all fields
"""

import pytest

from maseval.core.usage import (
    Usage,
    TokenUsage,
    StaticPricingCalculator,
    UsageReporter,
)

pytestmark = [pytest.mark.core]


# =============================================================================
# TokenUsage — Construction & Serialization
# =============================================================================


class TestTokenUsageConstruction:
    """Verify TokenUsage fields map correctly from various sources."""

    def test_from_chat_response_basic(self):
        """Minimal usage dict maps to the right fields."""
        tu = TokenUsage.from_chat_response_usage({"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
        assert tu.input_tokens == 100
        assert tu.output_tokens == 50
        assert tu.total_tokens == 150
        assert tu.cached_input_tokens == 0
        assert tu.cache_creation_input_tokens == 0
        assert tu.reasoning_tokens == 0
        assert tu.audio_tokens == 0
        assert tu.cost == 0.0

    def test_from_chat_response_all_fields(self):
        """All optional fields are mapped when present."""
        tu = TokenUsage.from_chat_response_usage(
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "cached_input_tokens": 800,
                "cache_creation_input_tokens": 50,
                "reasoning_tokens": 100,
                "audio_tokens": 10,
            },
            cost=0.05,
            provider="anthropic",
        )
        assert tu.input_tokens == 1000
        assert tu.output_tokens == 200
        assert tu.cached_input_tokens == 800
        assert tu.cache_creation_input_tokens == 50
        assert tu.reasoning_tokens == 100
        assert tu.audio_tokens == 10
        assert tu.cost == 0.05
        assert tu.provider == "anthropic"

    def test_serialization_roundtrip(self):
        """to_dict → from_dict preserves every field."""
        original = TokenUsage(
            cost=0.123,
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            cached_input_tokens=200,
            cache_creation_input_tokens=50,
            reasoning_tokens=80,
            audio_tokens=5,
            provider="openai",
            category="models",
            component_name="main_model",
            kind="llm",
        )
        d = original.to_dict()

        # Verify dict has all expected keys
        assert d["input_tokens"] == 500
        assert d["output_tokens"] == 100
        assert d["total_tokens"] == 600
        assert d["cached_input_tokens"] == 200
        assert d["cache_creation_input_tokens"] == 50
        assert d["reasoning_tokens"] == 80
        assert d["audio_tokens"] == 5
        assert d["cost"] == 0.123
        assert d["provider"] == "openai"
        assert d["category"] == "models"
        assert d["component_name"] == "main_model"
        assert d["kind"] == "llm"

        # Reconstruct via UsageReporter's deserialization path
        reconstructed = UsageReporter._usage_from_dict(d)
        assert isinstance(reconstructed, TokenUsage)
        assert reconstructed.input_tokens == original.input_tokens
        assert reconstructed.output_tokens == original.output_tokens
        assert reconstructed.cached_input_tokens == original.cached_input_tokens
        assert reconstructed.cache_creation_input_tokens == original.cache_creation_input_tokens
        assert reconstructed.reasoning_tokens == original.reasoning_tokens
        assert reconstructed.audio_tokens == original.audio_tokens
        assert reconstructed.cost == original.cost


# =============================================================================
# TokenUsage — Arithmetic
# =============================================================================


class TestTokenUsageArithmetic:
    """Verify addition produces mathematically correct results."""

    def test_add_two_token_usages(self):
        """All token fields and cost sum correctly."""
        a = TokenUsage(cost=0.10, input_tokens=100, output_tokens=50, total_tokens=150, cached_input_tokens=20, cache_creation_input_tokens=10)
        b = TokenUsage(cost=0.05, input_tokens=200, output_tokens=30, total_tokens=230, cached_input_tokens=50, cache_creation_input_tokens=5)
        total = a + b

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.15)
        assert total.input_tokens == 300
        assert total.output_tokens == 80
        assert total.total_tokens == 380
        assert total.cached_input_tokens == 70
        assert total.cache_creation_input_tokens == 15

    def test_sum_multiple(self):
        """sum() over a list of TokenUsages works correctly."""
        records = [
            TokenUsage(cost=0.01, input_tokens=10, output_tokens=5, total_tokens=15),
            TokenUsage(cost=0.02, input_tokens=20, output_tokens=10, total_tokens=30),
            TokenUsage(cost=0.03, input_tokens=30, output_tokens=15, total_tokens=45),
        ]
        total = records[0]
        for r in records[1:]:
            total = total + r

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.06)
        assert total.input_tokens == 60
        assert total.output_tokens == 30
        assert total.total_tokens == 90

    def test_zero_cost_preserves_known(self):
        """Adding a zero-cost usage preserves the known cost."""
        a = TokenUsage(cost=0.10, input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(input_tokens=200, output_tokens=30, total_tokens=230)
        total = a + b

        assert total.cost == pytest.approx(0.10)
        # Token fields still sum correctly
        assert isinstance(total, TokenUsage)
        assert total.input_tokens == 300
        assert total.output_tokens == 80

    def test_both_zero_cost_stays_zero(self):
        """Summing two zero-cost usages gives zero cost."""
        a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(input_tokens=200, output_tokens=30, total_tokens=230)
        total = a + b

        assert total.cost == 0.0

    def test_grouping_fields_match(self):
        """Matching grouping fields are preserved."""
        a = TokenUsage(cost=0.10, provider="anthropic", kind="llm", input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(cost=0.05, provider="anthropic", kind="llm", input_tokens=200, output_tokens=30, total_tokens=230)
        total = a + b

        assert total.provider == "anthropic"
        assert total.kind == "llm"

    def test_grouping_fields_mismatch(self):
        """Mismatched grouping fields become None."""
        a = TokenUsage(cost=0.10, provider="anthropic", input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(cost=0.05, provider="openai", input_tokens=200, output_tokens=30, total_tokens=230)
        total = a + b

        assert total.provider is None

    def test_add_token_usage_plus_plain_usage(self):
        """TokenUsage + plain Usage preserves token fields from left operand."""
        token = TokenUsage(cost=0.10, input_tokens=100, output_tokens=50, total_tokens=150, cached_input_tokens=20)
        plain = Usage(cost=0.05, units={"api_calls": 1})
        total = token + plain

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.15)
        assert total.input_tokens == 100
        assert total.cached_input_tokens == 20
        assert total.units == {"api_calls": 1}


# =============================================================================
# StaticPricingCalculator — Cost Correctness
# =============================================================================


class TestStaticPricingCalculator:
    """Verify cost formulas with hand-calculated expected values."""

    def test_basic_cost(self):
        """Simple input + output cost with no caching.

        100 input * $0.01 = $1.00
        50 output * $0.02 = $1.00
        Total = $2.00
        """
        calc = StaticPricingCalculator(
            {
                "test-model": {"input": 0.01, "output": 0.02},
            }
        )
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        cost = calc.calculate_cost(usage, "test-model")

        assert cost == pytest.approx(2.00)

    def test_cached_input_tokens(self):
        """Cached tokens use the cheaper rate.

        input_tokens=1000, cached_input_tokens=800
        Non-cached: 200 * $0.003 = $0.60
        Cached: 800 * $0.0003 = $0.24
        Output: 100 * $0.015 = $1.50
        Total = $2.34
        """
        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5": {
                    "input": 0.003,
                    "output": 0.015,
                    "cached_input": 0.0003,
                },
            }
        )
        usage = TokenUsage(input_tokens=1000, output_tokens=100, total_tokens=1100, cached_input_tokens=800)
        cost = calc.calculate_cost(usage, "claude-sonnet-4-5")

        assert cost == pytest.approx(2.34)

    def test_cache_creation_tokens(self):
        """Cache creation tokens use the higher rate.

        input_tokens=1000, cached_input_tokens=600, cache_creation_input_tokens=200
        Non-cached: (1000 - 600 - 200) = 200 * $0.003 = $0.60
        Cached: 600 * $0.0003 = $0.18
        Cache creation: 200 * $0.00375 = $0.75
        Output: 100 * $0.015 = $1.50
        Total = $3.03
        """
        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5": {
                    "input": 0.003,
                    "output": 0.015,
                    "cached_input": 0.0003,
                    "cache_creation_input": 0.00375,
                },
            }
        )
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=100,
            total_tokens=1100,
            cached_input_tokens=600,
            cache_creation_input_tokens=200,
        )
        cost = calc.calculate_cost(usage, "claude-sonnet-4-5")

        assert cost == pytest.approx(3.03)

    def test_cache_creation_defaults_to_input_rate(self):
        """When cache_creation_input is not specified, it defaults to the input rate.

        input_tokens=1000, cache_creation_input_tokens=200
        Non-cached: 800 * $0.003 = $2.40
        Cache creation: 200 * $0.003 = $0.60 (uses input rate)
        Output: 100 * $0.015 = $1.50
        Total = $4.50
        """
        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
            }
        )
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=100,
            total_tokens=1100,
            cache_creation_input_tokens=200,
        )
        cost = calc.calculate_cost(usage, "claude-sonnet-4-5")

        assert cost == pytest.approx(4.50)

    def test_unknown_model_returns_none(self):
        """Model not in pricing table returns None, not zero."""
        calc = StaticPricingCalculator({"gpt-4": {"input": 0.01, "output": 0.02}})
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)

        assert calc.calculate_cost(usage, "unknown-model") is None

    def test_zero_tokens(self):
        """Zero tokens produces zero cost."""
        calc = StaticPricingCalculator({"m": {"input": 0.01, "output": 0.02}})
        usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)

        assert calc.calculate_cost(usage, "m") == pytest.approx(0.0)

    def test_real_world_anthropic_pricing(self):
        """Real Anthropic Sonnet 4 pricing: $3/$15 per 1M tokens.

        500 input * $0.000003 = $0.0015
        200 output * $0.000015 = $0.003
        Total = $0.0045
        """
        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5": {"input": 3e-6, "output": 15e-6},
            }
        )
        usage = TokenUsage(input_tokens=500, output_tokens=200, total_tokens=700)
        cost = calc.calculate_cost(usage, "claude-sonnet-4-5")

        assert cost == pytest.approx(0.0045)

    def test_real_world_openai_pricing(self):
        """Real GPT-4o pricing: $2.50/$10 per 1M tokens.

        1000 input * $0.0000025 = $0.0025
        500 output * $0.000010 = $0.005
        Total = $0.0075
        """
        calc = StaticPricingCalculator(
            {
                "gpt-4o": {"input": 2.5e-6, "output": 10e-6},
            }
        )
        usage = TokenUsage(input_tokens=1000, output_tokens=500, total_tokens=1500)
        cost = calc.calculate_cost(usage, "gpt-4o")

        assert cost == pytest.approx(0.0075)


# =============================================================================
# LiteLLMCostCalculator — Parameter Passing
# =============================================================================


class TestLiteLLMCostCalculator:
    """Verify LiteLLMCostCalculator passes the right params to litellm."""

    def test_passes_cache_tokens_to_cost_per_token(self):
        """Verify cache_read and cache_creation tokens are forwarded."""
        pytest.importorskip("litellm")
        from unittest.mock import patch
        from maseval.interface.usage import LiteLLMCostCalculator

        calc = LiteLLMCostCalculator()
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            cached_input_tokens=600,
            cache_creation_input_tokens=100,
        )

        with patch("litellm.cost_per_token", return_value=(0.003, 0.006)) as mock_cpt:
            cost = calc.calculate_cost(usage, "claude-sonnet-4-5-20250514")

        mock_cpt.assert_called_once_with(
            model="claude-sonnet-4-5-20250514",
            prompt_tokens=1000,
            completion_tokens=200,
            cache_read_input_tokens=600,
            cache_creation_input_tokens=100,
        )
        assert cost == pytest.approx(0.009)

    def test_model_id_map_remapping(self):
        """model_id_map remaps before calling litellm."""
        pytest.importorskip("litellm")
        from unittest.mock import patch
        from maseval.interface.usage import LiteLLMCostCalculator

        calc = LiteLLMCostCalculator(
            model_id_map={
                "gemini-2.0-flash": "gemini/gemini-2.0-flash",
            }
        )
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)

        with patch("litellm.cost_per_token", return_value=(0.001, 0.002)) as mock_cpt:
            calc.calculate_cost(usage, "gemini-2.0-flash")

        # Verify it called with the remapped ID
        assert mock_cpt.call_args.kwargs["model"] == "gemini/gemini-2.0-flash"

    def test_custom_pricing_overrides_litellm(self):
        """custom_pricing takes precedence over litellm database.

        100 input * $0.0001 = $0.01
        50 output * $0.0002 = $0.01
        Total = $0.02
        """
        pytest.importorskip("litellm")
        from unittest.mock import patch
        from maseval.interface.usage import LiteLLMCostCalculator

        calc = LiteLLMCostCalculator(
            custom_pricing={
                "my-model": {
                    "input_cost_per_token": 0.0001,
                    "output_cost_per_token": 0.0002,
                },
            }
        )
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)

        with patch("litellm.cost_per_token") as mock_cpt:
            cost = calc.calculate_cost(usage, "my-model")

        # litellm.cost_per_token should NOT be called
        mock_cpt.assert_not_called()
        assert cost == pytest.approx(0.02)

    def test_unknown_model_returns_none(self):
        """Model not in litellm's database returns None."""
        pytest.importorskip("litellm")
        from unittest.mock import patch
        from maseval.interface.usage import LiteLLMCostCalculator

        calc = LiteLLMCostCalculator()
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)

        with patch("litellm.cost_per_token", side_effect=Exception("not found")):
            cost = calc.calculate_cost(usage, "nonexistent-model-xyz")

        assert cost is None


# =============================================================================
# Full Pipeline — DummyModelAdapter + CostCalculator
# =============================================================================


class TestFullPipeline:
    """End-to-end: adapter → TokenUsage → CostCalculator → gather_usage().cost.

    Uses DummyModelAdapter from conftest with known usage dicts and a
    StaticPricingCalculator with known rates, then verifies the final cost
    matches hand-calculated values.
    """

    def test_basic_pipeline(self):
        """Single chat call → correct cost on gather_usage().

        100 input * $0.01 + 50 output * $0.02 = $2.00
        """
        from tests.conftest import DummyModelAdapter

        calc = StaticPricingCalculator(
            {
                "test-model": {"input": 0.01, "output": 0.02},
            }
        )
        adapter = DummyModelAdapter(
            model_id="test-model",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        adapter._cost_calculator = calc

        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.input_tokens == 100
        assert total.output_tokens == 50
        assert total.cost == pytest.approx(2.00)

    def test_pipeline_multiple_calls_accumulate(self):
        """Multiple chat calls accumulate usage correctly.

        Call 1: 100 input * $0.01 + 50 output * $0.02 = $2.00
        Call 2: 100 input * $0.01 + 50 output * $0.02 = $2.00
        Total = $4.00, 200 input, 100 output
        """
        from tests.conftest import DummyModelAdapter

        calc = StaticPricingCalculator(
            {
                "test-model": {"input": 0.01, "output": 0.02},
            }
        )
        adapter = DummyModelAdapter(
            model_id="test-model",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        adapter._cost_calculator = calc

        adapter.chat([{"role": "user", "content": "Hello"}])
        adapter.chat([{"role": "user", "content": "World"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.input_tokens == 200
        assert total.output_tokens == 100
        assert total.cost == pytest.approx(4.00)

    def test_pipeline_provider_cost_takes_precedence(self):
        """Provider-reported cost wins over calculator.

        Usage dict has cost=0.99 (provider-reported).
        Calculator would compute $2.00.
        Provider cost should win.
        """
        from tests.conftest import DummyModelAdapter

        calc = StaticPricingCalculator(
            {
                "test-model": {"input": 0.01, "output": 0.02},
            }
        )
        adapter = DummyModelAdapter(
            model_id="test-model",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.99},
        )
        adapter._cost_calculator = calc

        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert total.cost == pytest.approx(0.99)

    def test_pipeline_no_calculator_no_provider_cost(self):
        """Without calculator or provider cost, cost is None."""
        from tests.conftest import DummyModelAdapter

        adapter = DummyModelAdapter(
            model_id="test-model",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )

        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.input_tokens == 100
        assert total.cost == 0.0

    def test_pipeline_with_cached_tokens(self):
        """Pipeline correctly handles cached tokens in cost calculation.

        input_tokens=1000, cached_input_tokens=800
        Non-cached: 200 * $0.003 = $0.60
        Cached: 800 * $0.0003 = $0.24
        Output: 100 * $0.015 = $1.50
        Total = $2.34
        """
        from tests.conftest import DummyModelAdapter

        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5": {
                    "input": 0.003,
                    "output": 0.015,
                    "cached_input": 0.0003,
                },
            }
        )
        adapter = DummyModelAdapter(
            model_id="claude-sonnet-4-5",
            usage={
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
                "cached_input_tokens": 800,
            },
        )
        adapter._cost_calculator = calc

        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.cached_input_tokens == 800
        assert total.cost == pytest.approx(2.34)


# =============================================================================
# UsageReporter — Aggregation Correctness
# =============================================================================


class TestUsageReporter:
    """Verify UsageReporter produces correct aggregations from report dicts."""

    @pytest.fixture
    def sample_reports(self):
        """Two tasks, each with a model component."""
        return [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "models": {
                        "main_model": {
                            "cost": 0.10,
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": "openai",
                            "category": "models",
                            "component_name": "main_model",
                            "kind": "llm",
                        }
                    }
                },
            },
            {
                "task_id": "task_2",
                "repeat_idx": 0,
                "usage": {
                    "models": {
                        "main_model": {
                            "cost": 0.20,
                            "input_tokens": 200,
                            "output_tokens": 100,
                            "total_tokens": 300,
                            "cached_input_tokens": 50,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": "openai",
                            "category": "models",
                            "component_name": "main_model",
                            "kind": "llm",
                        }
                    }
                },
            },
        ]

    def test_total(self, sample_reports):
        reporter = UsageReporter.from_reports(sample_reports)
        total = reporter.total()

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.30)
        assert total.input_tokens == 300
        assert total.output_tokens == 150
        assert total.cached_input_tokens == 50

    def test_by_task(self, sample_reports):
        reporter = UsageReporter.from_reports(sample_reports)
        by_task = reporter.by_task()

        assert len(by_task) == 2
        assert isinstance(by_task["task_1"], TokenUsage)
        assert by_task["task_1"].cost == pytest.approx(0.10)
        assert by_task["task_1"].input_tokens == 100
        assert isinstance(by_task["task_2"], TokenUsage)
        assert by_task["task_2"].cost == pytest.approx(0.20)
        assert by_task["task_2"].input_tokens == 200

    def test_by_component(self, sample_reports):
        reporter = UsageReporter.from_reports(sample_reports)
        by_comp = reporter.by_component()

        assert len(by_comp) == 1
        assert "models:main_model" in by_comp
        total = by_comp["models:main_model"]
        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.30)
        assert total.input_tokens == 300

    def test_summary_structure(self, sample_reports):
        reporter = UsageReporter.from_reports(sample_reports)
        summary = reporter.summary()

        assert "total" in summary
        assert "by_task" in summary
        assert "by_component" in summary
        assert summary["total"]["cost"] == pytest.approx(0.30)
        assert summary["total"]["input_tokens"] == 300

    def test_empty_reports(self):
        reporter = UsageReporter.from_reports([])
        total = reporter.total()

        assert total.cost == 0.0
        assert isinstance(total, Usage)

    def test_skips_error_reports(self):
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {"error": "setup failed"},
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        total = reporter.total()
        assert total.cost == 0.0
        assert isinstance(total, Usage)

    def test_by_task_accumulates_repeats(self):
        """by_task sums usage when a task_id appears in multiple reports."""
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "models": {
                        "m": {
                            "cost": 0.10,
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": None,
                            "category": "models",
                            "component_name": "m",
                            "kind": "llm",
                        }
                    }
                },
            },
            {
                "task_id": "task_1",
                "repeat_idx": 1,
                "usage": {
                    "models": {
                        "m": {
                            "cost": 0.20,
                            "input_tokens": 200,
                            "output_tokens": 100,
                            "total_tokens": 300,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": None,
                            "category": "models",
                            "component_name": "m",
                            "kind": "llm",
                        }
                    }
                },
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        by_task = reporter.by_task()

        assert len(by_task) == 1
        assert isinstance(by_task["task_1"], TokenUsage)
        assert by_task["task_1"].cost == pytest.approx(0.30)
        assert by_task["task_1"].input_tokens == 300

    def test_plain_usage_fallback(self):
        """_usage_from_dict returns plain Usage when no token fields present."""
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "tools": {
                        "my_tool": {
                            "cost": 0.05,
                            "units": {"api_calls": 3},
                            "provider": None,
                            "category": "tools",
                            "component_name": "my_tool",
                            "kind": "tool",
                        }
                    }
                },
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        total = reporter.total()

        assert total.cost == pytest.approx(0.05)
        assert isinstance(total, Usage)
        assert not isinstance(total, TokenUsage)

    def test_metadata_key_skipped(self):
        """The 'metadata' key in usage dicts is not treated as a component."""
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "metadata": {"timestamp": "2025-01-01", "total_components": 1},
                    "models": {
                        "m": {
                            "cost": 0.10,
                            "input_tokens": 50,
                            "output_tokens": 25,
                            "total_tokens": 75,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": None,
                            "category": "models",
                            "component_name": "m",
                            "kind": "llm",
                        }
                    },
                },
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        total = reporter.total()

        # Only the model's cost, metadata should not contribute
        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.10)
        assert total.input_tokens == 50

    def test_skips_component_with_error(self):
        """Components with error dicts are skipped, others still counted."""
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "models": {
                        "good_model": {
                            "cost": 0.10,
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": None,
                            "category": "models",
                            "component_name": "good_model",
                            "kind": "llm",
                        },
                        "bad_model": {
                            "error": "Failed to gather usage",
                            "error_type": "RuntimeError",
                        },
                    }
                },
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        total = reporter.total()

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.10)
        assert total.input_tokens == 100

    def test_environment_direct_usage(self):
        """Environment/user usage (direct dicts with 'cost') are parsed."""
        reports = [
            {
                "task_id": "task_1",
                "repeat_idx": 0,
                "usage": {
                    "environment": {
                        "cost": 0.05,
                        "units": {"steps": 10},
                        "provider": None,
                        "category": "environment",
                        "component_name": "env",
                        "kind": "env",
                    },
                    "models": {
                        "m": {
                            "cost": 0.10,
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                            "cached_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "units": {},
                            "provider": None,
                            "category": "models",
                            "component_name": "m",
                            "kind": "llm",
                        }
                    },
                },
            },
        ]
        reporter = UsageReporter.from_reports(reports)
        total = reporter.total()

        assert total.cost == pytest.approx(0.15)


# =============================================================================
# StaticPricingCalculator — Utility Methods
# =============================================================================


class TestStaticPricingCalculatorUtilities:
    """Tests for add_model, models property, and gather_config."""

    def test_add_model(self):
        calc = StaticPricingCalculator({})
        calc.add_model("new-model", {"input": 0.01, "output": 0.02})

        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        cost = calc.calculate_cost(usage, "new-model")
        assert cost == pytest.approx(2.00)

    def test_models_property(self):
        calc = StaticPricingCalculator(
            {
                "model-a": {"input": 0.01, "output": 0.02},
                "model-b": {"input": 0.001, "output": 0.002},
            }
        )
        assert sorted(calc.models) == ["model-a", "model-b"]

    def test_gather_config(self):
        pricing = {"model-a": {"input": 0.01, "output": 0.02}}
        calc = StaticPricingCalculator(pricing)
        config = calc.gather_config()

        assert config["type"] == "StaticPricingCalculator"
        assert config["pricing"] == pricing
