"""Tests for ModelScorer abstract base class.

These tests verify that the ModelScorer ABC correctly delegates to
subclass implementations, handles logging/tracing, and provides
the expected batch and MCQ convenience methods.
"""

import pytest
from typing import Dict, List, Optional, Tuple

from maseval.core.scorer import ModelScorer


class StubScorer(ModelScorer):
    """Minimal concrete scorer for testing the ABC contract."""

    def __init__(self, scores: Dict[Tuple[str, str], float], seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._scores = scores
        self._call_log: List[Tuple[str, str]] = []

    @property
    def model_id(self) -> str:
        return "stub-model"

    def _loglikelihood_impl(self, context: str, continuation: str) -> float:
        self._call_log.append((context, continuation))
        return self._scores[(context, continuation)]


class FailingScorer(ModelScorer):
    """Scorer that raises on every call, for error-path testing."""

    @property
    def model_id(self) -> str:
        return "failing-model"

    def _loglikelihood_impl(self, context: str, continuation: str) -> float:
        raise ValueError("model exploded")


pytestmark = pytest.mark.core


class TestModelScorerLoglikelihood:
    """Tests for single-pair loglikelihood."""

    def test_delegates_to_impl(self):
        """loglikelihood() should delegate to _loglikelihood_impl()."""
        scorer = StubScorer({("ctx", " cont"): -1.5})
        result = scorer.loglikelihood("ctx", " cont")

        assert result == -1.5
        assert scorer._call_log == [("ctx", " cont")]

    def test_logs_success(self):
        """Successful call should be logged."""
        scorer = StubScorer({("a", "b"): -2.0})
        scorer.loglikelihood("a", "b")

        assert len(scorer.logs) == 1
        assert scorer.logs[0]["status"] == "success"
        assert scorer.logs[0]["type"] == "loglikelihood"
        assert scorer.logs[0]["duration_seconds"] >= 0

    def test_logs_error_and_reraises(self):
        """Failed call should be logged and the exception re-raised."""
        scorer = FailingScorer()

        with pytest.raises(ValueError, match="model exploded"):
            scorer.loglikelihood("a", "b")

        assert len(scorer.logs) == 1
        assert scorer.logs[0]["status"] == "error"
        assert scorer.logs[0]["error_type"] == "ValueError"


class TestModelScorerBatch:
    """Tests for batch loglikelihood."""

    def test_default_batch_loops_over_impl(self):
        """Default _loglikelihood_batch_impl loops over _loglikelihood_impl."""
        scores = {("q", " A"): -1.0, ("q", " B"): -2.0, ("q", " C"): -0.5}
        scorer = StubScorer(scores)

        results = scorer.loglikelihood_batch([("q", " A"), ("q", " B"), ("q", " C")])

        assert results == [-1.0, -2.0, -0.5]
        assert len(scorer._call_log) == 3

    def test_batch_logs_single_entry(self):
        """Batch call should produce one log entry (not per-pair)."""
        scores = {("q", " A"): -1.0, ("q", " B"): -2.0}
        scorer = StubScorer(scores)

        scorer.loglikelihood_batch([("q", " A"), ("q", " B")])

        assert len(scorer.logs) == 1
        assert scorer.logs[0]["type"] == "loglikelihood_batch"
        assert scorer.logs[0]["batch_size"] == 2

    def test_empty_batch(self):
        """Empty batch should return empty list."""
        scorer = StubScorer({})
        assert scorer.loglikelihood_batch([]) == []


class TestModelScorerChoices:
    """Tests for MCQ loglikelihood_choices."""

    def test_prepends_delimiter(self):
        """Choices should be prepended with the delimiter before scoring."""
        scores = {("Q?", " A"): -1.0, ("Q?", " B"): -0.5, ("Q?", " C"): -2.0}
        scorer = StubScorer(scores)

        results = scorer.loglikelihood_choices("Q?", ["A", "B", "C"])

        assert results == [-1.0, -0.5, -2.0]
        assert scorer._call_log == [("Q?", " A"), ("Q?", " B"), ("Q?", " C")]

    def test_custom_delimiter(self):
        """Custom delimiter should be used instead of default space."""
        scores = {("Q?", "\nA"): -1.0, ("Q?", "\nB"): -0.5}
        scorer = StubScorer(scores)

        results = scorer.loglikelihood_choices("Q?", ["A", "B"], delimiter="\n")

        assert results == [-1.0, -0.5]
        assert scorer._call_log == [("Q?", "\nA"), ("Q?", "\nB")]


class TestModelScorerTracing:
    """Tests for gather_traces and gather_config."""

    def test_gather_traces_includes_call_stats(self):
        """Traces should contain call counts and timing."""
        scores = {("a", "b"): -1.0, ("c", "d"): -2.0}
        scorer = StubScorer(scores)
        scorer.loglikelihood("a", "b")
        scorer.loglikelihood("c", "d")

        traces = scorer.gather_traces()

        assert traces["model_id"] == "stub-model"
        assert traces["total_calls"] == 2
        assert traces["successful_calls"] == 2
        assert traces["failed_calls"] == 0
        assert traces["total_duration_seconds"] >= 0
        assert len(traces["logs"]) == 2

    def test_gather_traces_counts_failures(self):
        """Traces should correctly count failed calls."""
        scorer = FailingScorer()
        with pytest.raises(ValueError):
            scorer.loglikelihood("a", "b")

        traces = scorer.gather_traces()

        assert traces["total_calls"] == 1
        assert traces["successful_calls"] == 0
        assert traces["failed_calls"] == 1

    def test_gather_config(self):
        """Config should include model_id, scorer_type, and seed."""
        scorer = StubScorer({}, seed=42)

        config = scorer.gather_config()

        assert config["model_id"] == "stub-model"
        assert config["scorer_type"] == "StubScorer"
        assert config["seed"] == 42

    def test_gather_config_seed_none(self):
        """Config should report None seed when unseeded."""
        scorer = StubScorer({})

        config = scorer.gather_config()

        assert config["seed"] is None


class TestModelScorerSeed:
    """Tests for seed property."""

    def test_seed_stored(self):
        scorer = StubScorer({}, seed=123)
        assert scorer.seed == 123

    def test_seed_default_none(self):
        scorer = StubScorer({})
        assert scorer.seed is None
