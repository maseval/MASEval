"""Tests for Gaia2Evaluator and compute_gaia2_metrics.

Tests the evaluation layer that integrates with ARE's GraphPerEventJudge.
"""

import pytest
from unittest.mock import MagicMock


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

    def test_extracts_judge_type_from_task(self, sample_gaia2_task):
        """Test evaluator extracts judge_type from task evaluation_data."""
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        assert evaluator.judge_type == sample_gaia2_task.evaluation_data.get("judge_type", "graph_per_event")

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
        """Test evaluator returns GSR from ARE judge on single-turn scenario."""
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        # Create scenario mock with an explicit judge (single-turn: nb_turns=1)
        mock_scenario = MagicMock()
        mock_judge = MagicMock()
        mock_judge.state = SimpleNamespace(nb_turns=1, turn_idx=-1)
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.rationale = None
        mock_judge.validate.return_value = mock_result
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        result = evaluator({}, None)

        assert result["gsr"] == 1.0
        assert result["passed"] is True
        # Single-turn: no intermediate judge calls, only validate
        mock_judge.assert_not_called()
        mock_judge.validate.assert_called_once_with(mock_are_env)

    def test_returns_zero_gsr_on_failure(self, sample_gaia2_task):
        """Test evaluator returns 0.0 GSR when judge fails."""
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        # Create scenario mock with an explicit judge (single-turn)
        mock_scenario = MagicMock()
        mock_judge = MagicMock()
        mock_judge.state = SimpleNamespace(nb_turns=1, turn_idx=-1)
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.rationale = "Failed"
        mock_judge.validate.return_value = mock_result
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        result = evaluator({}, None)

        assert result["gsr"] == 0.0
        assert result["passed"] is False

    def test_multi_turn_calls_judge_for_intermediate_turns(self, sample_gaia2_task):
        """Test evaluator calls judge(env) for intermediate turns before validate().

        ARE's intended flow for nb_turns=N: call judge(env) for turns 0..N-2,
        then judge.validate(env) for the final turn. This advances turn_idx so
        the is_last_turn check in validate() passes.
        ARE simulation/validation/base.py:104
        """
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        # Simulate judge state: turn_idx starts at -1 (no trigger conditions fired)
        state = SimpleNamespace(nb_turns=3, turn_idx=-1)
        intermediate_judgment = MagicMock()
        intermediate_judgment.success = True

        mock_judge = MagicMock()
        mock_judge.state = state

        def judge_call(env):
            state.turn_idx += 1
            return intermediate_judgment

        mock_judge.side_effect = judge_call

        mock_validate_result = MagicMock()
        mock_validate_result.success = True
        mock_validate_result.rationale = None
        mock_judge.validate.return_value = mock_validate_result

        mock_scenario = MagicMock()
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(task=sample_gaia2_task, environment=mock_env)
        result = evaluator({}, None)

        # Should call judge(env) twice for intermediate turns (0, 1)
        assert mock_judge.call_count == 2
        mock_judge.assert_any_call(mock_are_env)
        # Then validate for the final turn
        mock_judge.validate.assert_called_once_with(mock_are_env)
        assert result["gsr"] == 1.0
        assert result["passed"] is True

    def test_multi_turn_intermediate_failure_short_circuits(self, sample_gaia2_task):
        """Test that a failed intermediate turn stops further judge calls.

        When an intermediate turn fails, the evaluator breaks early. The subsequent
        validate() call returns failure via the last_turn_success check.
        ARE simulation/validation/base.py:96-100
        """
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        state = SimpleNamespace(nb_turns=3, turn_idx=-1)
        failed_judgment = MagicMock()
        failed_judgment.success = False
        failed_judgment.failure = "Turn 0 events did not match"

        mock_judge = MagicMock()
        mock_judge.state = state

        def judge_call(env):
            state.turn_idx += 1
            return failed_judgment

        mock_judge.side_effect = judge_call

        mock_validate_result = MagicMock()
        mock_validate_result.success = False
        mock_validate_result.rationale = "Last turn was already rejected"
        mock_judge.validate.return_value = mock_validate_result

        mock_scenario = MagicMock()
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(task=sample_gaia2_task, environment=mock_env)
        result = evaluator({}, None)

        # Should only call judge(env) once (broke after first failure)
        assert mock_judge.call_count == 1
        # validate() still called (returns failure via last_turn_success check)
        mock_judge.validate.assert_called_once_with(mock_are_env)
        assert result["gsr"] == 0.0
        assert result["passed"] is False

    def test_two_turn_scenario_calls_judge_once(self, sample_gaia2_task):
        """Test 2-turn scenario calls judge(env) once then validate(env).

        This is the most common multi-turn case (adaptability scenarios).
        nb_turns=2: one intermediate judge(env) call, then validate(env).
        """
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        state = SimpleNamespace(nb_turns=2, turn_idx=-1)
        intermediate_judgment = MagicMock()
        intermediate_judgment.success = True

        mock_judge = MagicMock()
        mock_judge.state = state

        def judge_call(env):
            state.turn_idx += 1
            return intermediate_judgment

        mock_judge.side_effect = judge_call

        mock_validate_result = MagicMock()
        mock_validate_result.success = True
        mock_validate_result.rationale = None
        mock_judge.validate.return_value = mock_validate_result

        mock_scenario = MagicMock()
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(task=sample_gaia2_task, environment=mock_env)
        result = evaluator({}, None)

        # One intermediate call + one validate
        assert mock_judge.call_count == 1
        mock_judge.validate.assert_called_once_with(mock_are_env)
        assert result["gsr"] == 1.0

    def test_skips_intermediate_turns_if_already_judged(self, sample_gaia2_task):
        """Test evaluator skips judge(env) calls if trigger conditions already fired.

        In online mode (default), ARE's ConditionCheckEvent trigger conditions
        call judge(env) during the simulation, advancing turn_idx. The evaluator
        checks turn_idx before calling judge(env) to avoid double-counting.
        """
        from types import SimpleNamespace

        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        # turn_idx already at nb_turns-2 (trigger conditions fired for all intermediate turns)
        state = SimpleNamespace(nb_turns=3, turn_idx=1)

        mock_judge = MagicMock()
        mock_judge.state = state

        mock_validate_result = MagicMock()
        mock_validate_result.success = True
        mock_validate_result.rationale = None
        mock_judge.validate.return_value = mock_validate_result

        mock_scenario = MagicMock()
        mock_scenario.judge = mock_judge

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        evaluator = Gaia2Evaluator(task=sample_gaia2_task, environment=mock_env)
        result = evaluator({}, None)

        # No intermediate judge calls needed (already advanced)
        mock_judge.assert_not_called()
        mock_judge.validate.assert_called_once_with(mock_are_env)
        assert result["gsr"] == 1.0

    def test_handles_missing_are_environment(self, sample_gaia2_task):
        """Test evaluator handles missing ARE environment.

        When ARE environment is not available, score is None (excluded from
        scoring), matching ARE's behavior for no_validation results.
        ARE benchmark/hf_upload_utils.py:47-48
        """
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        mock_env = MagicMock()
        mock_env.get_are_environment.return_value = None

        evaluator = Gaia2Evaluator(
            task=sample_gaia2_task,
            environment=mock_env,
        )

        result = evaluator({}, None)

        assert result["gsr"] is None
        assert result["passed"] is False
        assert "error" in result
        assert result["status"] == "no_validation"

    def test_fallback_judge_respects_judge_engine_config(self):
        """Test evaluator fallback judge creation respects judge_engine_config."""
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from maseval import Task
        from maseval.benchmark.gaia2.data_loader import Gaia2JudgeEngineConfig
        from maseval.benchmark.gaia2.evaluator import Gaia2Evaluator

        judge_engine_config = Gaia2JudgeEngineConfig(
            model_name="openai/gpt-4o",
            provider="openrouter",
        )

        task = Task(
            id="test_fallback",
            query="",
            environment_data={
                "scenario": MagicMock(scenario_id="test"),
                "capability": "execution",
            },
            evaluation_data={
                "judge_type": "graph_per_event",
                "judge_engine_config": judge_engine_config,
            },
        )

        # Scenario without a judge (triggers fallback)
        mock_scenario = MagicMock(spec=[])
        del mock_scenario.judge  # Ensure getattr returns None

        mock_env = MagicMock()
        mock_are_env = MagicMock()
        mock_env.get_are_environment.return_value = mock_are_env
        mock_env.get_scenario.return_value = mock_scenario

        # Mock ARE imports
        mock_llm_config_cls = MagicMock()
        mock_create_engine = MagicMock()
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_judge_config_cls = MagicMock()
        mock_judge = MagicMock()
        mock_judge.state = SimpleNamespace(nb_turns=1, turn_idx=-1)
        mock_validate_result = MagicMock()
        mock_validate_result.success = True
        mock_validate_result.rationale = None
        mock_judge.validate.return_value = mock_validate_result

        mock_factory = MagicMock()
        mock_factory.return_value.return_value = mock_judge

        mock_validation = MagicMock()
        mock_validation.GraphPerEventJudgeConfig = mock_judge_config_cls
        mock_validation.JudgeFactory = mock_factory

        mock_are = MagicMock()
        modules = {
            "are": mock_are,
            "are.simulation": mock_are.simulation,
            "are.simulation.validation": mock_validation,
            "are.simulation.validation.configs": MagicMock(create_judge_engine=mock_create_engine),
            "are.simulation.agents": mock_are.simulation.agents,
            "are.simulation.agents.are_simulation_agent_config": MagicMock(LLMEngineConfig=mock_llm_config_cls),
        }

        evaluator = Gaia2Evaluator(task=task, environment=mock_env)

        with patch.dict(sys.modules, modules):
            evaluator({}, None)

        # Verify LLMEngineConfig was created with custom values
        mock_llm_config_cls.assert_called_once_with(
            model_name="openai/gpt-4o",
            provider="openrouter",
            endpoint=None,
        )
        # Verify create_judge_engine was called
        mock_create_engine.assert_called_once()
        # Verify GraphPerEventJudgeConfig was created with the custom engine
        mock_judge_config_cls.assert_called_once_with(engine=mock_engine)


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
