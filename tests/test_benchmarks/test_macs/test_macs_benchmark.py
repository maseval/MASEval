"""Unit tests for MACSBenchmark and compute_benchmark_metrics."""

import pytest
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest.mock import MagicMock

from maseval import AgentAdapter, Task, User, MessageHistory
from maseval.benchmark.macs import (
    MACSBenchmark,
    MACSEnvironment,
    MACSEvaluator,
    MACSUser,
    compute_benchmark_metrics,
)

from .conftest import MACSAgentAdapter, ConcreteMACSBenchmark
from conftest import DummyModelAdapter


# =============================================================================
# Unit Tests: Initialization and Setup
# =============================================================================


@pytest.mark.benchmark
class TestMACSBenchmarkSetup:
    """Tests for MACSBenchmark initialization and setup methods."""

    def test_init_configures_benchmark(self, macs_model, sample_agent_data):
        """Benchmark initializes with optional params."""
        callbacks = [MagicMock()]
        benchmark = ConcreteMACSBenchmark(macs_model, callbacks=callbacks, n_task_repeats=3)

        assert benchmark.callbacks == callbacks
        assert benchmark.n_task_repeats == 3

    def test_macs_default_max_invocations_is_five(self, macs_model, sample_agent_data):
        """MACS benchmark defaults to max_invocations=5 per MACS paper.

        This is a MACS-specific default that differs from the base class default of 1.
        The MACS paper specifies up to 5 agent-user interaction rounds.
        """
        benchmark = ConcreteMACSBenchmark(macs_model)

        assert benchmark.max_invocations == 5

    def test_setup_environment_creates_macs_environment(self, macs_model, sample_agent_data, sample_task):
        """setup_environment returns MACSEnvironment with tools."""
        benchmark = ConcreteMACSBenchmark(macs_model)

        env = benchmark.setup_environment(sample_agent_data, sample_task)

        assert isinstance(env, MACSEnvironment)
        assert "search_flights" in env.tools

    def test_setup_user_creates_macs_user(self, macs_model, sample_agent_data, sample_task):
        """setup_user returns MACSUser with scenario from task."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)

        user = benchmark.setup_user(sample_agent_data, env, sample_task)

        assert isinstance(user, MACSUser)
        assert user.scenario == "Business trip to NYC"

    def test_setup_user_handles_no_scenario(self, macs_model, sample_agent_data, sample_task_no_scenario):
        """Handles missing scenario gracefully."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task_no_scenario)

        user = benchmark.setup_user(sample_agent_data, env, sample_task_no_scenario)

        assert user.scenario == ""

    def test_setup_evaluators_creates_user_and_system(self, macs_model, sample_agent_data, sample_task):
        """Creates both user and system evaluators."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        agents = [MACSAgentAdapter()]

        evaluators = benchmark.setup_evaluators(env, sample_task, agents, None)

        assert len(evaluators) == 2
        assert isinstance(evaluators[0], MACSEvaluator)
        assert isinstance(evaluators[1], MACSEvaluator)
        assert evaluators[0].gsr_type == "user"
        assert evaluators[1].gsr_type == "system"

    def test_setup_agents_is_abstract(self, macs_model, sample_agent_data):
        """setup_agents must be overridden in subclass."""
        import inspect

        assert inspect.isabstract(MACSBenchmark)

        # Verify IncompleteMACSBenchmark would fail (checked at class definition time)
        with pytest.raises(TypeError, match="abstract"):

            class IncompleteMACSBenchmark(MACSBenchmark):
                pass

            # This line won't be reached due to TypeError at class definition
            IncompleteMACSBenchmark(sample_agent_data, macs_model)  # type: ignore


# =============================================================================
# Unit Tests: Run Agents
# =============================================================================


@pytest.mark.benchmark
class TestRunAgents:
    """Tests for run_agents method."""

    def test_run_agents_executes_agents_with_query(self, macs_model, sample_agent_data, sample_task):
        """Agents are executed with the query parameter."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)

        agents_list, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)

        # Pass explicit query parameter
        benchmark.run_agents(agents_list, sample_task, env, query=sample_task.query)

        # Cast to MACSAgentAdapter to access run_calls
        mock_agent = agents_list[0]
        assert isinstance(mock_agent, MACSAgentAdapter)
        assert len(mock_agent.run_calls) == 1
        assert mock_agent.run_calls[0] == sample_task.query

    def test_run_agents_uses_query_parameter_not_task_query(self, macs_model, sample_agent_data, sample_task):
        """run_agents uses the query parameter, not task.query directly.

        This is critical for multi-turn interaction where the query changes
        between invocations (e.g., user's response becomes the next query).
        """
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        agents_list, _ = benchmark.setup_agents(sample_agent_data, env, sample_task, None)

        # Pass a different query than task.query
        custom_query = "This is a user response, not the task query"
        benchmark.run_agents(agents_list, sample_task, env, query=custom_query)

        mock_agent = agents_list[0]
        assert isinstance(mock_agent, MACSAgentAdapter)
        # Agent should receive the custom query, not task.query
        assert mock_agent.run_calls[0] == custom_query
        assert mock_agent.run_calls[0] != sample_task.query

    def test_run_agents_returns_answer(self, macs_model, sample_agent_data, sample_task):
        """Returns final answer(s) as MessageHistory."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        agents_list, _ = benchmark.setup_agents(sample_agent_data, env, sample_task, None)

        result = benchmark.run_agents(agents_list, sample_task, env, query=sample_task.query)

        # run_agents returns MessageHistory from the agent run
        assert isinstance(result, MessageHistory)
        assert len(result) > 0
        # Check that response content contains expected text
        assert "Response to:" in result[-1]["content"]

    def test_run_agents_single_agent(self, macs_model, sample_agent_data, sample_task):
        """Single agent returns MessageHistory."""
        benchmark = ConcreteMACSBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        agents_list, _ = benchmark.setup_agents(sample_agent_data, env, sample_task, None)

        result = benchmark.run_agents(agents_list, sample_task, env, query=sample_task.query)

        assert isinstance(result, MessageHistory)

    def test_run_agents_multiple_agents(self, macs_model, sample_agent_data, sample_task):
        """Multiple agents return list of answers."""

        class MultiAgentBenchmark(MACSBenchmark):
            def __init__(self, model_factory, **kwargs):
                self._model_factory = model_factory if callable(model_factory) else lambda _: model_factory
                super().__init__(**kwargs)

            def get_model_adapter(self, model_id: str, **kwargs):
                return self._model_factory(model_id)

            def setup_agents(  # type: ignore[invalid-method-override]
                self,
                agent_data: Dict[str, Any],
                environment: MACSEnvironment,
                task: Task,
                user: Optional[User],
                seed_generator=None,
            ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
                agent1: AgentAdapter = MACSAgentAdapter("agent1")
                agent2: AgentAdapter = MACSAgentAdapter("agent2")
                return [agent1, agent2], {"agent1": agent1, "agent2": agent2}

        benchmark = MultiAgentBenchmark(macs_model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        agents_list, _ = benchmark.setup_agents(sample_agent_data, env, sample_task, None)

        result = benchmark.run_agents(agents_list, sample_task, env, query=sample_task.query)

        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# Unit Tests: Evaluation
# =============================================================================


@pytest.mark.benchmark
class TestEvaluation:
    """Tests for evaluate method."""

    def test_evaluate_calls_both_evaluators(self, sample_agent_data, sample_task):
        """Both user and system evaluators are called."""
        # Model returns valid JSON for evaluation
        responses = [
            '[{"assertion": "User assertion", "answer": "TRUE", "evidence": "OK"}]',
            '[{"assertion": "System assertion", "answer": "TRUE", "evidence": "OK"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        _, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)
        evaluators = benchmark.setup_evaluators(env, sample_task, list(agents_dict.values()), None)

        traces = {
            "agents": {
                "test_agent": {
                    "messages": [
                        {"role": "user", "content": "Book flight"},
                        {"role": "assistant", "content": "Done"},
                    ]
                }
            },
            "tools": {},
        }

        results = benchmark.evaluate(evaluators, agents_dict, "final answer", traces)

        assert len(results) == 1  # Combined into one result dict
        assert "user_gsr" in results[0]
        assert "system_gsr" in results[0]

    def test_evaluate_returns_aggregated_metrics(self, sample_agent_data, sample_task):
        """Returns combined GSR metrics."""
        responses = [
            '[{"assertion": "A", "answer": "TRUE", "evidence": "OK"}]',
            '[{"assertion": "B", "answer": "TRUE", "evidence": "OK"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        _, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)
        evaluators = benchmark.setup_evaluators(env, sample_task, list(agents_dict.values()), None)

        traces = {
            "agents": {"test_agent": {"messages": [{"role": "user", "content": "Q"}]}},
            "tools": {},
        }

        results = benchmark.evaluate(evaluators, agents_dict, "answer", traces)

        result = results[0]
        assert "user_gsr" in result
        assert "user_partial_gsr" in result
        assert "system_gsr" in result
        assert "system_partial_gsr" in result
        assert "overall_gsr" in result
        assert "overall_partial_gsr" in result
        assert "supervisor_gsr" in result
        assert "report" in result

    def test_evaluate_overall_gsr(self, sample_agent_data, sample_task):
        """overall_gsr = 1.0 only if both user AND system pass."""
        # User passes, system fails
        responses = [
            '[{"assertion": "A", "answer": "TRUE", "evidence": "OK"}]',
            '[{"assertion": "B", "answer": "FALSE", "evidence": "Fail"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        _, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)
        evaluators = benchmark.setup_evaluators(env, sample_task, list(agents_dict.values()), None)

        traces = {
            "agents": {"test_agent": {"messages": [{"role": "user", "content": "Q"}]}},
            "tools": {},
        }

        results = benchmark.evaluate(evaluators, agents_dict, "answer", traces)

        assert results[0]["user_gsr"] == 1.0
        assert results[0]["system_gsr"] == 0.0
        assert results[0]["overall_gsr"] == 0.0  # Not all passed

    def test_evaluate_supervisor_gsr(self, sample_agent_data, sample_task):
        """supervisor_gsr = 1.0 if overall OR user passes."""
        # User passes, system fails
        responses = [
            '[{"assertion": "A", "answer": "TRUE", "evidence": "OK"}]',
            '[{"assertion": "B", "answer": "FALSE", "evidence": "Fail"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        _, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)
        evaluators = benchmark.setup_evaluators(env, sample_task, list(agents_dict.values()), None)

        traces = {
            "agents": {"test_agent": {"messages": [{"role": "user", "content": "Q"}]}},
            "tools": {},
        }

        results = benchmark.evaluate(evaluators, agents_dict, "answer", traces)

        # User passed, so supervisor_gsr should be 1.0
        assert results[0]["supervisor_gsr"] == 1.0

    def test_evaluate_combined_report(self, sample_agent_data, sample_task):
        """Report combines both evaluator reports."""
        responses = [
            '[{"assertion": "User A", "answer": "TRUE", "evidence": "OK"}]',
            '[{"assertion": "System B", "answer": "TRUE", "evidence": "OK"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        _, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, None)
        evaluators = benchmark.setup_evaluators(env, sample_task, list(agents_dict.values()), None)

        traces = {
            "agents": {"test_agent": {"messages": [{"role": "user", "content": "Q"}]}},
            "tools": {},
        }

        results = benchmark.evaluate(evaluators, agents_dict, "answer", traces)

        report = results[0]["report"]
        assert len(report) == 2
        # Check assertion types are added
        assertion_types = [item.get("assertion_type") for item in report]
        assert "user" in assertion_types
        assert "system" in assertion_types


# =============================================================================
# Unit Tests: compute_benchmark_metrics
# =============================================================================


@pytest.mark.benchmark
class TestComputeBenchmarkMetrics:
    """Tests for compute_benchmark_metrics utility."""

    def test_empty_results(self):
        """Empty results returns zeros."""
        result = compute_benchmark_metrics([])

        assert result["total_tasks"] == 0
        assert result["scored_tasks"] == 0
        assert result["successful_tasks"] == 0
        assert result["success_rate"] == 0.0
        assert result["mean_metrics"] == {}
        assert result["excluded"] == {}
        assert result["status_counts"] == {}

    def test_single_successful_result(self):
        """Single successful result counted."""
        results = [{"status": "success", "eval": [{"overall_gsr": 1.0, "user_gsr": 1.0, "system_gsr": 1.0}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["scored_tasks"] == 1
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0

    def test_single_failed_result(self):
        """Single failed result counted."""
        results = [{"status": "success", "eval": [{"overall_gsr": 0.0, "user_gsr": 0.0, "system_gsr": 0.0}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["scored_tasks"] == 1
        assert metrics["successful_tasks"] == 0
        assert metrics["success_rate"] == 0.0

    def test_multiple_results(self):
        """Multiple results aggregated correctly."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},  # Success
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},  # Fail
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},  # Success
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["scored_tasks"] == 3
        assert metrics["successful_tasks"] == 2
        assert metrics["success_rate"] == pytest.approx(2 / 3)

    def test_success_rate_calculation(self):
        """success_rate = successful/scored (not total)."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["success_rate"] == 0.5

    def test_mean_metrics_calculation(self):
        """Mean of numeric metrics computed."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0, "partial_gsr": 0.8}]},
            {"status": "success", "eval": [{"overall_gsr": 0.0, "partial_gsr": 0.4}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["mean_metrics"]["overall_gsr"] == pytest.approx(0.5)
        assert metrics["mean_metrics"]["partial_gsr"] == pytest.approx(0.6)

    def test_handles_missing_eval(self):
        """Handles results with no eval key."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "success", "no_eval_key": True},  # Missing eval
            {"status": "success", "eval": None},  # None eval
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["scored_tasks"] == 3
        assert metrics["successful_tasks"] == 1

    def test_handles_non_numeric_values(self):
        """Non-numeric values in eval are ignored for mean."""
        results = [
            {
                "status": "success",
                "eval": [
                    {
                        "overall_gsr": 1.0,
                        "report": [{"assertion": "A"}],  # Non-numeric
                        "status": "success",  # String
                    }
                ],
            }
        ]

        metrics = compute_benchmark_metrics(results)

        # Should only have numeric metrics
        assert "overall_gsr" in metrics["mean_metrics"]
        assert "report" not in metrics["mean_metrics"]
        assert "status" not in metrics["mean_metrics"]

    def test_excludes_environment_errors_from_scoring(self):
        """Environment errors are excluded from scoring."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "environment_error", "eval": None},  # Should be excluded
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["scored_tasks"] == 2  # Only success tasks
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 0.5  # 1/2, not 1/3
        assert metrics["excluded"]["environment_error"] == 1

    def test_excludes_user_errors_from_scoring(self):
        """User simulator errors are excluded from scoring."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "user_error", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 2
        assert metrics["scored_tasks"] == 1
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0  # Only the success one
        assert metrics["excluded"]["user_error"] == 1

    def test_excludes_unknown_errors_from_scoring(self):
        """Unknown execution errors are excluded from scoring."""
        results = [
            {"status": "unknown_execution_error", "eval": None},
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 2
        assert metrics["scored_tasks"] == 1
        assert metrics["success_rate"] == 0.0
        assert metrics["excluded"]["unknown_execution_error"] == 1

    def test_excludes_setup_failed_from_scoring(self):
        """Setup failures are excluded from scoring."""
        results = [
            {"status": "setup_failed", "eval": None},
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 2
        assert metrics["scored_tasks"] == 1
        assert metrics["excluded"]["setup_failed"] == 1

    def test_excludes_evaluation_failed_from_scoring(self):
        """Evaluation failures are excluded from scoring."""
        results = [
            {"status": "evaluation_failed", "eval": None},
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 2
        assert metrics["scored_tasks"] == 1
        assert metrics["success_rate"] == 1.0  # Only the success one
        assert metrics["excluded"]["evaluation_failed"] == 1

    def test_includes_agent_errors_in_scoring(self):
        """Agent errors ARE included in scoring (agent's fault)."""
        results = [
            {"status": "agent_error", "eval": [{"overall_gsr": 0.0}]},
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 2
        assert metrics["scored_tasks"] == 2  # Agent errors count!
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 0.5

    def test_status_counts_tracked(self):
        """Status counts are tracked for all tasks."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0}]},
            {"status": "success", "eval": [{"overall_gsr": 0.0}]},
            {"status": "agent_error", "eval": None},
            {"status": "environment_error", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["status_counts"]["success"] == 2
        assert metrics["status_counts"]["agent_error"] == 1
        assert metrics["status_counts"]["environment_error"] == 1

    def test_mixed_errors_comprehensive(self):
        """Comprehensive test with various error types."""
        results = [
            {"status": "success", "eval": [{"overall_gsr": 1.0, "accuracy": 0.9}]},
            {"status": "success", "eval": [{"overall_gsr": 0.0, "accuracy": 0.3}]},
            {"status": "agent_error", "eval": [{"overall_gsr": 0.0, "accuracy": 0.0}]},
            {"status": "environment_error", "eval": None},  # Excluded
            {"status": "user_error", "eval": None},  # Excluded
            {"status": "evaluation_failed", "eval": None},  # Excluded
            {"status": "setup_failed", "eval": None},  # Excluded
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 7
        assert metrics["scored_tasks"] == 3  # success(2) + agent_error(1)
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == pytest.approx(1 / 3)
        assert metrics["mean_metrics"]["accuracy"] == pytest.approx((0.9 + 0.3 + 0.0) / 3)
        assert sum(metrics["excluded"].values()) == 4


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestMACSBenchmarkIntegration:
    """Integration tests for MACSBenchmark."""

    def test_full_task_execution(self, sample_agent_data, sample_task):
        """Test complete task execution flow."""
        # Evaluator responses - user then system
        responses = [
            '[{"assertion": "Booking confirmed", "answer": "TRUE", "evidence": "Done"}]',
            '[{"assertion": "Database updated", "answer": "TRUE", "evidence": "Updated"}]',
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)

        # Setup phase
        env = benchmark.setup_environment(sample_agent_data, sample_task)
        user = benchmark.setup_user(sample_agent_data, env, sample_task)
        agents_list, agents_dict = benchmark.setup_agents(sample_agent_data, env, sample_task, user)
        evaluators = benchmark.setup_evaluators(env, sample_task, agents_list, user)

        # Run phase
        final_answer = benchmark.run_agents(agents_list, sample_task, env, query=sample_task.query)

        # Evaluate phase
        traces = {
            "agents": {
                "test_agent": {
                    "messages": [
                        {"role": "user", "content": sample_task.query},
                        {"role": "assistant", "content": final_answer},
                    ]
                }
            },
            "tools": {},
        }
        results = benchmark.evaluate(evaluators, agents_dict, final_answer, traces)

        # Verify results
        assert len(results) == 1
        assert results[0]["user_gsr"] == 1.0
        assert results[0]["system_gsr"] == 1.0
        assert results[0]["overall_gsr"] == 1.0

    def test_benchmark_with_real_environment(self, sample_agent_data, sample_task):
        """Test with real MACSEnvironment tool creation."""
        model = DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])
        benchmark = ConcreteMACSBenchmark(model)

        env = benchmark.setup_environment(sample_agent_data, sample_task)

        # Environment should have tools
        assert "search_flights" in env.tools
        assert env.tools["search_flights"].name == "search_flights"
