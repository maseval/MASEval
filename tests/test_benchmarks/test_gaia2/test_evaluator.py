"""Tests for Gaia2Evaluator and compute_gaia2_metrics.

Tests the evaluation layer that integrates with ARE's GraphPerEventJudge.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Test Gaia2Evaluator Initialization
# =============================================================================


@pytest.mark.benchmark
class TestGaia2EvaluatorInit:
    """Tests for Gaia2Evaluator initialization."""

    def test_stores_task(self, sample_gaia2_task):
        """Test evaluator stores task."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        assert evaluator.task == sample_gaia2_task

    def test_stores_environment(self, sample_gaia2_task):
        """Test evaluator stores environment."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        assert evaluator.environment == mock_env

    def test_extracts_oracle_events_from_task(self, sample_gaia2_task):
        """Test evaluator extracts oracle events from task."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        assert evaluator.oracle_events == sample_gaia2_task.evaluation_data.get("oracle_events", [])

    def test_default_use_llm_judge_is_false(self, sample_gaia2_task):
        """Test use_llm_judge defaults to False."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        assert evaluator.use_llm_judge is False


@pytest.mark.benchmark
class TestGaia2EvaluatorFilterTraces:
    """Tests for Gaia2Evaluator.filter_traces()."""

    def test_extracts_tool_invocations(self, sample_gaia2_task, sample_execution_traces):
        """Test filter_traces extracts tool invocations."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        filtered = evaluator.filter_traces(sample_execution_traces)

        assert "tool_invocations" in filtered

    def test_extracts_simulation_time(self, sample_gaia2_task, sample_execution_traces):
        """Test filter_traces extracts simulation time."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        filtered = evaluator.filter_traces(sample_execution_traces)

        assert "simulation_time" in filtered


@pytest.mark.benchmark
class TestGaia2EvaluatorCall:
    """Tests for Gaia2Evaluator.__call__()."""

    def test_returns_gsr_from_judge(self, sample_gaia2_task):
        """Test evaluator returns GSR from ARE judge."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_are_env.get_completed_events.return_value = []
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        # Mock ARE imports via sys.modules
        mock_are = MagicMock()
        mock_judge = MagicMock()
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.partial_score = 1.0
        mock_result.event_results = []
        mock_judge.evaluate.return_value = mock_result
        mock_are.simulation.validation.JudgeFactory.create.return_value = mock_judge

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.validation": mock_are.simulation.validation,
                "are.simulation.validation.config": mock_are.simulation.validation.config,
            },
        ):
            result = evaluator({}, None)

            assert result["gsr"] == 1.0
            assert result["passed"] is True

    def test_returns_zero_gsr_on_failure(self, sample_gaia2_task):
        """Test evaluator returns 0.0 GSR when judge fails."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_are_env.get_completed_events.return_value = []
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        # Mock ARE imports via sys.modules
        mock_are = MagicMock()
        mock_judge = MagicMock()
        mock_result = MagicMock()
        mock_result.passed = False
        mock_result.partial_score = 0.3
        mock_judge.evaluate.return_value = mock_result
        mock_are.simulation.validation.JudgeFactory.create.return_value = mock_judge

        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.validation": mock_are.simulation.validation,
                "are.simulation.validation.config": mock_are.simulation.validation.config,
            },
        ):
            result = evaluator({}, None)

            assert result["gsr"] == 0.0
            assert result["partial_gsr"] == 0.3
            assert result["passed"] is False

    def test_handles_missing_are_environment(self, sample_gaia2_task):
        """Test evaluator handles missing ARE environment."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()
        mock_env.get_are_environment.return_value = None

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        # Mock ARE imports via sys.modules
        mock_are = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.validation": mock_are.simulation.validation,
                "are.simulation.validation.config": mock_are.simulation.validation.config,
            },
        ):
            result = evaluator({}, None)

            assert result["gsr"] == 0.0
            assert result["passed"] is False
            assert "error" in result


# =============================================================================
# Test compute_gaia2_metrics
# =============================================================================


@pytest.mark.benchmark
class TestComputeGaia2Metrics:
    """Tests for compute_gaia2_metrics function."""

    def test_computes_overall_gsr(self):
        """Test computes overall GSR from results."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 0.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 1.0, "capability": "search"}], "status": "success"},
        ]

        metrics = compute_gaia2_metrics(results)

        # 2 out of 3 passed
        assert metrics["gsr"] == pytest.approx(2 / 3)

    def test_computes_partial_gsr(self):
        """Test computes partial GSR from results."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0, "partial_gsr": 1.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 0.0, "partial_gsr": 0.5, "capability": "execution"}], "status": "success"},
        ]

        metrics = compute_gaia2_metrics(results)

        assert metrics["partial_gsr"] == pytest.approx(0.75)

    def test_groups_by_capability(self):
        """Test groups metrics by capability."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 0.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 1.0, "capability": "search"}], "status": "success"},
            {"eval": [{"gsr": 1.0, "capability": "time"}], "status": "success"},
        ]

        metrics = compute_gaia2_metrics(results)

        assert "by_capability" in metrics
        assert metrics["by_capability"]["execution"]["gsr"] == 0.5
        assert metrics["by_capability"]["search"]["gsr"] == 1.0
        assert metrics["by_capability"]["time"]["gsr"] == 1.0

    def test_counts_task_statuses(self):
        """Test counts task statuses."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0, "capability": "execution"}], "status": "success"},
            {"eval": [{"gsr": 0.0, "capability": "execution"}], "status": "success"},
            {"eval": [], "status": "agent_error"},
        ]

        metrics = compute_gaia2_metrics(results)

        assert "status_counts" in metrics
        assert metrics["status_counts"]["success"] == 2
        assert metrics["status_counts"]["agent_error"] == 1

    def test_handles_empty_results(self):
        """Test handles empty results list."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        metrics = compute_gaia2_metrics([])

        assert metrics["total_tasks"] == 0
        assert metrics["gsr"] == 0.0

    def test_excludes_non_scoreable_statuses(self):
        """Test excludes non-scoreable statuses from GSR calculation."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0, "capability": "execution"}], "status": "success"},
            {"eval": [], "status": "environment_error"},  # Not scoreable
        ]

        metrics = compute_gaia2_metrics(results)

        # Only 1 scoreable task
        assert metrics["scored_tasks"] == 1
        assert metrics["gsr"] == 1.0

    def test_handles_missing_capability(self):
        """Test handles results without capability."""
        from maseval.benchmark.gaia2.evaluator import compute_gaia2_metrics

        results = [
            {"eval": [{"gsr": 1.0}], "status": "success"},  # No capability
        ]

        metrics = compute_gaia2_metrics(results)

        assert metrics["gsr"] == 1.0
        # Should have "unknown" capability
        assert "unknown" in metrics["by_capability"]


@pytest.mark.benchmark
class TestScoreableStatuses:
    """Tests for SCOREABLE_STATUSES constant."""

    def test_includes_success(self):
        """Test SCOREABLE_STATUSES includes success."""
        from maseval.benchmark.gaia2.evaluator import SCOREABLE_STATUSES

        assert "success" in SCOREABLE_STATUSES

    def test_includes_agent_error(self):
        """Test SCOREABLE_STATUSES includes agent_error."""
        from maseval.benchmark.gaia2.evaluator import SCOREABLE_STATUSES

        assert "agent_error" in SCOREABLE_STATUSES

    def test_includes_task_timeout(self):
        """Test SCOREABLE_STATUSES includes task_timeout."""
        from maseval.benchmark.gaia2.evaluator import SCOREABLE_STATUSES

        assert "task_timeout" in SCOREABLE_STATUSES

    def test_excludes_environment_error(self):
        """Test SCOREABLE_STATUSES excludes environment_error."""
        from maseval.benchmark.gaia2.evaluator import SCOREABLE_STATUSES

        assert "environment_error" not in SCOREABLE_STATUSES
