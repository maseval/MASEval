"""Unit tests for Tau2Evaluator."""

import json

import pytest
from unittest.mock import MagicMock, patch

from maseval import Task
from maseval.benchmark.tau2.evaluator import Tau2Evaluator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.domain = "retail"
    # Default hash values
    env.get_db_hash.return_value = "hash123"
    env.get_user_db_hash.return_value = None
    env.toolkit.has_tool.return_value = True
    env.toolkit.use_tool.return_value = True
    env.run_env_assertion.return_value = True
    return env


@pytest.fixture
def sample_task():
    task = MagicMock(spec=Task)
    task.environment_data = {"domain": "retail"}
    task.evaluation_data = {
        "reward_basis": ["DB", "ACTION", "COMMUNICATE"],
        "actions": [{"name": "check_order", "arguments": {"order_id": "123"}}],
        "communicate_info": ["refund processed"],
        "env_assertions": [{"func_name": "check_status", "arguments": {"id": "1"}, "assert_value": True}],
    }
    return task


@pytest.fixture
def evaluator(sample_task, mock_environment):
    return Tau2Evaluator(sample_task, mock_environment)


# =============================================================================
# Trace Filtering Tests
# =============================================================================


@pytest.mark.benchmark
def test_filter_traces(evaluator):
    """Test extraction of relevant traces into full_trajectory."""
    raw_traces = {
        "agents": {"agent1": {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}},
        "users": {},
        "environment": {"final_db_hash": "hash123"},
        "termination_reason": "agent_stop",
    }

    filtered = evaluator.filter_traces(raw_traces)

    assert "full_trajectory" in filtered
    assert len(filtered["full_trajectory"]) == 2
    assert filtered["full_trajectory"][0]["content"] == "Hi"
    assert filtered["full_trajectory"][1]["content"] == "Hello"

    assert filtered["environment"]["final_db_hash"] == "hash123"
    assert filtered["termination_reason"] == "agent_stop"


# =============================================================================
# Environment Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_environment_success(evaluator, mock_environment):
    """Test environment evaluation with matching hashes."""
    full_trajectory = []  # Empty trajectory — no tool calls to replay

    # Mock gold environment creation (must mock both agent and user DB hashes)
    with patch("maseval.benchmark.tau2.evaluator.get_environment_constructor") as mock_get_const:
        mock_env = MagicMock()
        mock_env.get_db_hash.return_value = "hash_gold"
        mock_env.get_user_db_hash.return_value = "user_hash_gold"
        mock_env.run_env_assertion.return_value = True
        mock_get_const.return_value.return_value = mock_env

        result = evaluator._evaluate_environment(full_trajectory)

        assert result["db_match"] is True
        assert result["db_reward"] == 1.0
        assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_environment_failure(evaluator, mock_environment):
    """Test environment evaluation with mismatching hashes."""
    full_trajectory = []

    with patch("maseval.benchmark.tau2.evaluator.get_environment_constructor") as mock_get_const:
        # Predicted env returns different hash than gold env
        call_count = [0]

        def make_env():
            call_count[0] += 1
            env = MagicMock()
            if call_count[0] == 1:
                # predicted env
                env.get_db_hash.return_value = "hash_actual"
                env.get_user_db_hash.return_value = None
            else:
                # gold env
                env.get_db_hash.return_value = "hash_expected"
                env.get_user_db_hash.return_value = None
            env.run_env_assertion.return_value = True
            return env

        mock_get_const.return_value.side_effect = make_env

        result = evaluator._evaluate_environment(full_trajectory)

        assert result["db_match"] is False
        assert result["db_reward"] == 0.0


# =============================================================================
# Action Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_actions_match(evaluator):
    """Test action evaluation with matching tool calls in trajectory."""
    full_trajectory = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "check_order", "arguments": {"order_id": "123"}}]},
        {"role": "tool", "content": "result"},
    ]

    result = evaluator._evaluate_actions(full_trajectory)

    assert result["all_matched"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_actions_mismatch(evaluator):
    """Test action evaluation with mismatching tool calls."""
    full_trajectory = [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "check_order", "arguments": {"order_id": "999"}}]},
        {"role": "tool", "content": "result"},
    ]

    result = evaluator._evaluate_actions(full_trajectory)

    assert result["all_matched"] is False
    assert result["reward"] == 0.0


@pytest.mark.benchmark
def test_evaluate_actions_missing(evaluator):
    """Test action evaluation with no tool calls in trajectory."""
    full_trajectory = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = evaluator._evaluate_actions(full_trajectory)

    assert result["all_matched"] is False
    assert result["reward"] == 0.0


# =============================================================================
# Communication Evaluation Tests
# =============================================================================


@pytest.mark.benchmark
def test_evaluate_communication_success(evaluator):
    """Test communication evaluation finding required info in trajectory."""
    full_trajectory = [{"role": "assistant", "content": "Your refund processed successfully."}]

    result = evaluator._evaluate_communication(full_trajectory)

    assert result["all_found"] is True
    assert result["reward"] == 1.0


@pytest.mark.benchmark
def test_evaluate_communication_failure(evaluator):
    """Test communication evaluation failing to find info in trajectory."""
    full_trajectory = [{"role": "assistant", "content": "I cannot help you."}]

    result = evaluator._evaluate_communication(full_trajectory)

    assert result["all_found"] is False
    assert result["reward"] == 0.0


# =============================================================================
# Score Aggregation Tests
# =============================================================================


@pytest.mark.benchmark
def test_score_aggregation_all_pass(evaluator):
    """Test overall score when all components pass."""
    # Mock individual methods to return success
    evaluator._evaluate_environment = MagicMock(return_value={"reward": 1.0, "breakdown": {"DB": 1.0}})
    evaluator._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {"ACTION": 1.0}})
    evaluator._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {"COMMUNICATE": 1.0}})
    evaluator._evaluate_nl_assertions = MagicMock(return_value={"reward": 1.0})

    result = evaluator({"termination_reason": "agent_stop", "full_trajectory": []})

    assert result["reward"] == 1.0
    assert result["passed"] is True
    assert result["reward_breakdown"] == {"DB": 1.0, "ACTION": 1.0, "COMMUNICATE": 1.0}


@pytest.mark.benchmark
def test_score_aggregation_mixed(evaluator):
    """Test overall score when some components fail."""
    # Environment fails, others pass
    evaluator._evaluate_environment = MagicMock(return_value={"reward": 0.0, "breakdown": {"DB": 0.0}})
    evaluator._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {"ACTION": 1.0}})
    evaluator._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {"COMMUNICATE": 1.0}})
    evaluator._evaluate_nl_assertions = MagicMock(return_value={"reward": 1.0})

    result = evaluator({"termination_reason": "agent_stop", "full_trajectory": []})

    assert result["reward"] == 0.0  # Multiplicative
    assert result["passed"] is False


@pytest.mark.benchmark
def test_premature_termination(evaluator):
    """Test evaluation aborts on error termination."""
    traces = {"termination_reason": "too_many_errors"}

    result = evaluator(traces)

    assert result["reward"] == 0.0
    assert result["passed"] is False
    assert "prematurely" in result["note"]


@pytest.mark.benchmark
def test_max_steps_termination(evaluator):
    """Test evaluation aborts on max_steps termination."""
    traces = {"termination_reason": "max_steps"}

    result = evaluator(traces)

    assert result["reward"] == 0.0
    assert result["passed"] is False
    assert "prematurely" in result["note"]


# =============================================================================
# Metrics Tests
# =============================================================================


@pytest.mark.benchmark
class TestComputeBenchmarkMetrics:
    """Tests for compute_benchmark_metrics function."""

    def test_empty_results(self):
        """Empty results returns zeros."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        result = compute_benchmark_metrics([])

        assert result["total_tasks"] == 0
        assert result["successful_tasks"] == 0
        assert result["success_rate"] == 0.0
        assert result["mean_reward"] == 0.0

    def test_single_success(self):
        """Single successful result counted."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [{"status": "success", "eval": [{"reward": 1.0, "passed": True}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == 1.0
        assert metrics["mean_reward"] == 1.0

    def test_single_failure(self):
        """Single failed result counted. H9: ALL simulations count."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [{"status": "agent_error", "eval": [{"reward": 0.0, "passed": False}]}]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 1
        assert metrics["successful_tasks"] == 0
        assert metrics["success_rate"] == 0.0

    def test_mixed_results(self):
        """Mixed success/failure results aggregated. H9: ALL simulations count."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "success", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "agent_error", "eval": [{"reward": 0.0, "passed": False}]},
            {"status": "task_timeout", "eval": [{"reward": 0.5, "passed": False}]},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 3
        assert metrics["successful_tasks"] == 1
        assert metrics["success_rate"] == pytest.approx(1 / 3)
        assert metrics["mean_reward"] == pytest.approx(0.5)

    def test_all_simulations_count(self):
        """H9: ALL simulations count in denominator (terminated ones get reward=0.0)."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "success", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "environment_error", "eval": None},
            {"status": "user_error", "eval": None},
            {"status": "setup_failed", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["total_tasks"] == 4
        assert metrics["successful_tasks"] == 1
        # H9: success_rate denominator is total_tasks, not scored_tasks
        assert metrics["success_rate"] == pytest.approx(1 / 4)

    def test_status_counts(self):
        """Status counts tracked correctly."""
        from maseval.benchmark.tau2.evaluator import compute_benchmark_metrics

        results = [
            {"status": "success", "eval": [{"reward": 1.0, "passed": True}]},
            {"status": "agent_error", "eval": [{"reward": 0.0, "passed": False}]},
            {"status": "environment_error", "eval": None},
        ]

        metrics = compute_benchmark_metrics(results)

        assert metrics["status_counts"]["success"] == 1
        assert metrics["status_counts"]["agent_error"] == 1
        assert metrics["status_counts"]["environment_error"] == 1


@pytest.mark.benchmark
class TestComputePassAtK:
    """Tests for compute_pass_at_k function."""

    def test_all_pass(self):
        """All attempts pass gives 1.0 for all k."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 1.0
        assert pass_k["pass@2"] == 1.0
        assert pass_k["pass@3"] == 1.0

    def test_all_fail(self):
        """All attempts fail gives 0.0 for all k."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 0.0
        assert pass_k["pass@2"] == 0.0
        assert pass_k["pass@3"] == 0.0

    def test_mixed_results(self):
        """Mixed results with pass on second attempt."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 0.0  # First attempt failed
        assert pass_k["pass@2"] == 1.0  # Second attempt passed
        assert pass_k["pass@3"] == 1.0  # At least one passed

    def test_insufficient_attempts(self):
        """Insufficient attempts for k returns 0.0."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2, 3])

        assert pass_k["pass@1"] == 1.0
        assert pass_k["pass@2"] == 0.0  # Not enough attempts
        assert pass_k["pass@3"] == 0.0

    def test_multiple_tasks(self):
        """Multiple tasks with different outcomes."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k

        results = [
            # Task 1: passes on first try
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            # Task 2: fails all
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_k = compute_pass_at_k(results, k_values=[1, 2])

        assert pass_k["pass@1"] == 0.5  # 1/2 tasks pass@1
        assert pass_k["pass@2"] == 0.5  # 1/2 tasks pass@2


# =============================================================================
# Pass^k Tests (Combinatorial Metric)
# =============================================================================


@pytest.mark.benchmark
class TestPassHatK:
    """Tests for pass^k (combinatorial) metric."""

    def test_pass_hat_k_basic(self):
        """Basic pass^k calculation."""
        from maseval.benchmark.tau2.evaluator import pass_hat_k

        # 4 trials, 2 successes, k=1: C(2,1)/C(4,1) = 2/4 = 0.5
        assert pass_hat_k(4, 2, 1) == 0.5

        # 4 trials, 2 successes, k=2: C(2,2)/C(4,2) = 1/6 ≈ 0.167
        assert abs(pass_hat_k(4, 2, 2) - 1 / 6) < 0.001

        # 4 trials, 4 successes, k=4: C(4,4)/C(4,4) = 1
        assert pass_hat_k(4, 4, 4) == 1.0

        # 4 trials, 0 successes, k=1: C(0,1)/C(4,1) = 0
        assert pass_hat_k(4, 0, 1) == 0.0

    def test_pass_hat_k_insufficient_successes(self):
        """pass^k returns 0 when success_count < k."""
        from maseval.benchmark.tau2.evaluator import pass_hat_k

        # 4 trials, 1 success, k=2: can't get 2 successes from 1
        assert pass_hat_k(4, 1, 2) == 0.0

    def test_pass_hat_k_invalid_k(self):
        """pass^k raises error when k > num_trials."""
        from maseval.benchmark.tau2.evaluator import pass_hat_k

        with pytest.raises(ValueError):
            pass_hat_k(2, 1, 3)  # k=3 > num_trials=2

    def test_compute_pass_hat_k_single_task(self):
        """compute_pass_hat_k with single task."""
        from maseval.benchmark.tau2.evaluator import compute_pass_hat_k

        # 4 attempts, 2 successes
        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_hat = compute_pass_hat_k(results, k_values=[1, 2])

        # k=1: C(2,1)/C(4,1) = 2/4 = 0.5
        assert pass_hat["pass^1"] == 0.5
        # k=2: C(2,2)/C(4,2) = 1/6 ≈ 0.167
        assert abs(pass_hat["pass^2"] - 1 / 6) < 0.001

    def test_compute_pass_hat_k_multiple_tasks(self):
        """compute_pass_hat_k averages across tasks."""
        from maseval.benchmark.tau2.evaluator import compute_pass_hat_k

        results = [
            # Task 1: 2/2 successes
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            # Task 2: 0/2 successes
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task2", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_hat = compute_pass_hat_k(results, k_values=[1, 2])

        # Task 1: pass^1 = C(2,1)/C(2,1) = 1.0
        # Task 2: pass^1 = C(0,1)/C(2,1) = 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert pass_hat["pass^1"] == 0.5

        # Task 1: pass^2 = C(2,2)/C(2,2) = 1.0
        # Task 2: pass^2 = C(0,2)/C(2,2) = 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert pass_hat["pass^2"] == 0.5

    def test_compute_pass_hat_k_auto_k_values(self):
        """compute_pass_hat_k auto-determines k values if not provided."""
        from maseval.benchmark.tau2.evaluator import compute_pass_hat_k

        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
        ]

        pass_hat = compute_pass_hat_k(results)

        # Should have pass^1, pass^2, pass^3
        assert "pass^1" in pass_hat
        assert "pass^2" in pass_hat
        assert "pass^3" in pass_hat
        assert pass_hat["pass^1"] == 1.0
        assert pass_hat["pass^2"] == 1.0
        assert pass_hat["pass^3"] == 1.0

    def test_compute_pass_hat_k_empty_results(self):
        """compute_pass_hat_k with empty results."""
        from maseval.benchmark.tau2.evaluator import compute_pass_hat_k

        pass_hat = compute_pass_hat_k([])
        assert pass_hat == {}

    def test_pass_at_k_vs_pass_hat_k_difference(self):
        """Demonstrate the difference between pass@k and pass^k."""
        from maseval.benchmark.tau2.evaluator import compute_pass_at_k, compute_pass_hat_k

        # 4 attempts: [True, False, False, False] (1 success out of 4)
        results = [
            {"task_id": "task1", "status": "success", "eval": [{"passed": True}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
            {"task_id": "task1", "status": "success", "eval": [{"passed": False}]},
        ]

        pass_at = compute_pass_at_k(results, k_values=[1, 2, 4])
        pass_hat = compute_pass_hat_k(results, k_values=[1, 2, 4])

        # pass@1: First attempt succeeded → 1.0
        assert pass_at["pass@1"] == 1.0
        # pass^1: C(1,1)/C(4,1) = 1/4 = 0.25
        assert pass_hat["pass^1"] == 0.25

        # pass@2: At least one of first 2 succeeded → 1.0
        assert pass_at["pass@2"] == 1.0
        # pass^2: C(1,2)/C(4,2) = 0 (can't pick 2 from 1 success)
        assert pass_hat["pass^2"] == 0.0

        # pass@4: At least one of all 4 succeeded → 1.0
        assert pass_at["pass@4"] == 1.0
        # pass^4: C(1,4)/C(4,4) = 0 (can't pick 4 from 1 success)
        assert pass_hat["pass^4"] == 0.0


# =============================================================================
# _build_full_trajectory Tests — Lines 146-189
# =============================================================================


@pytest.mark.benchmark
class TestBuildFullTrajectory:
    """Tests for Tau2Evaluator._build_full_trajectory() message merging."""

    def test_empty_agents_returns_empty(self):
        """No agent messages → empty trajectory."""
        traces = {"agents": {}, "users": {"u1": {"messages": [{"role": "user", "content": "hi"}]}}}
        assert Tau2Evaluator._build_full_trajectory(traces) == []

    def test_agent_only_no_user(self):
        """Agent messages only, no user traces → returns agent messages."""
        agent_msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        traces = {"agents": {"a1": {"messages": agent_msgs}}, "users": {}}
        result = Tau2Evaluator._build_full_trajectory(traces)
        assert result == agent_msgs

    def test_greeting_first_in_trajectory(self):
        """Greeting from user trace appears first in trajectory."""
        agent_msgs = [{"role": "user", "content": "I need help"}, {"role": "assistant", "content": "Sure"}]
        user_msgs = [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "I need help"},
        ]
        traces = {"agents": {"a1": {"messages": agent_msgs}}, "users": {"u1": {"messages": user_msgs}}}
        result = Tau2Evaluator._build_full_trajectory(traces)
        assert result[0]["content"] == "Hi! How can I help you today?"

    def test_user_tool_calls_inserted(self):
        """User tool call sequences inserted after matching agent text."""
        agent_msgs = [
            {"role": "user", "content": "I need help"},
            {"role": "assistant", "content": "Here's the result"},
        ]
        user_msgs = [
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "I need help"},
            {"role": "assistant", "content": "Here's the result"},
            {"role": "user", "tool_calls": [{"name": "pay_bill", "id": "tc1"}], "content": ""},
            {"role": "tool", "content": "paid"},
        ]
        traces = {"agents": {"a1": {"messages": agent_msgs}}, "users": {"u1": {"messages": user_msgs}}}
        result = Tau2Evaluator._build_full_trajectory(traces)

        # User tool calls should appear somewhere in the trajectory
        has_user_tc = any(m.get("tool_calls") and m.get("role") == "user" for m in result)
        has_tool_response = any(m.get("role") == "tool" for m in result)
        assert has_user_tc
        assert has_tool_response

    def test_agent_tool_calls_preserved(self):
        """Agent tool calls from agent msgs remain in trajectory."""
        agent_msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"name": "lookup"}], "content": ""},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Found it"},
        ]
        traces = {"agents": {"a1": {"messages": agent_msgs}}, "users": {}}
        result = Tau2Evaluator._build_full_trajectory(traces)
        assert any(m.get("tool_calls") for m in result)

    def test_empty_user_messages(self):
        """Empty user message list → same as no user traces."""
        agent_msgs = [{"role": "user", "content": "hi"}]
        traces = {"agents": {"a1": {"messages": agent_msgs}}, "users": {"u1": {"messages": []}}}
        result = Tau2Evaluator._build_full_trajectory(traces)
        assert result == agent_msgs


# =============================================================================
# _evaluate_nl_assertions Tests — Lines 485-558
# =============================================================================


@pytest.mark.benchmark
class TestEvaluateNLAssertions:
    """Tests for _evaluate_nl_assertions() NL evaluation."""

    def test_no_assertions_returns_1(self, evaluator):
        """No NL assertions → reward=1.0 (skipped)."""
        evaluator.nl_assertions = None
        result = evaluator._evaluate_nl_assertions([])
        assert result["reward"] == 1.0

    def test_no_model_skips(self):
        """NL assertions without model → reward=1.0 with skip note."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": ["The agent should greet the user"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
        }
        env = MagicMock()
        ev = Tau2Evaluator(task, env, nl_model=None)
        result = ev._evaluate_nl_assertions([{"role": "assistant", "content": "Hello"}])
        assert result["reward"] == 1.0
        assert "skipped" in result.get("note", "")

    def test_all_met(self):
        """All NL assertions met → reward=1.0."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(
            content=json.dumps({"results": [{"expectedOutcome": "greet", "metExpectation": True, "reasoning": "ok"}]})
        )
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": ["greet"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock(), nl_model=mock_model)
        result = ev._evaluate_nl_assertions([{"role": "assistant", "content": "Hello!"}])
        assert result["reward"] == 1.0
        assert len(result["nl_checks"]) == 1
        assert result["nl_checks"][0]["met"] is True

    def test_some_not_met(self):
        """Some NL assertions not met → reward=0.0."""
        mock_model = MagicMock()
        mock_model.chat.return_value = MagicMock(
            content=json.dumps(
                {
                    "results": [
                        {"expectedOutcome": "greet", "metExpectation": True, "reasoning": "ok"},
                        {"expectedOutcome": "apologize", "metExpectation": False, "reasoning": "no"},
                    ]
                }
            )
        )
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": ["greet", "apologize"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock(), nl_model=mock_model)
        result = ev._evaluate_nl_assertions([{"role": "assistant", "content": "Hi!"}])
        assert result["reward"] == 0.0

    def test_model_error_returns_0(self):
        """Model exception → reward=0.0 (graceful degradation)."""
        mock_model = MagicMock()
        mock_model.chat.side_effect = Exception("API error")
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": ["greet"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock(), nl_model=mock_model)
        result = ev._evaluate_nl_assertions([{"role": "assistant", "content": "Hi"}])
        assert result["reward"] == 0.0


# =============================================================================
# __call__ Branch Tests — Lines 246-247
# =============================================================================


@pytest.mark.benchmark
class TestEvaluatorCallBranches:
    """Tests for Tau2Evaluator.__call__() reward_basis branches."""

    def test_nl_assertion_in_reward_basis(self):
        """NL_ASSERTION in reward_basis contributes to final reward."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["NL_ASSERTION"],
            "nl_assertions": ["something"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        ev._evaluate_environment = MagicMock(return_value={"reward": 1.0, "breakdown": {}})  # type: ignore[assignment]
        ev._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {}})  # type: ignore[assignment]
        ev._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {}})  # type: ignore[assignment]
        ev._evaluate_nl_assertions = MagicMock(return_value={"reward": 0.5, "breakdown": {"NL_ASSERTION": 0.5}})  # type: ignore[assignment]

        result = ev({"termination_reason": "agent_stop", "full_trajectory": []})
        assert result["reward"] == 0.5
        assert "NL_ASSERTION" in result["reward_breakdown"]

    def test_env_assertion_in_reward_basis(self):
        """ENV_ASSERTION in reward_basis contributes to final reward."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["ENV_ASSERTION"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": [{"func_name": "f", "assert_value": True}],
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        ev._evaluate_environment = MagicMock(return_value={"reward": 0.0, "breakdown": {"ENV_ASSERTION": 0.0}})  # type: ignore[assignment]
        ev._evaluate_actions = MagicMock(return_value={"reward": 1.0, "breakdown": {}})  # type: ignore[assignment]
        ev._evaluate_communication = MagicMock(return_value={"reward": 1.0, "breakdown": {}})  # type: ignore[assignment]
        ev._evaluate_nl_assertions = MagicMock(return_value={"reward": 1.0})  # type: ignore[assignment]

        result = ev({"termination_reason": "agent_stop", "full_trajectory": []})
        assert result["reward"] == 0.0
        assert "ENV_ASSERTION" in result["reward_breakdown"]

    def test_no_environment_criteria(self):
        """No actions/assertions → environment returns reward=1.0."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["COMMUNICATE"],
            "actions": None,
            "communicate_info": ["hello"],
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        result = ev._evaluate_environment([])
        assert result["reward"] == 1.0

    def test_no_action_criteria(self):
        """No actions → actions returns reward=1.0."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["COMMUNICATE"],
            "actions": None,
            "communicate_info": ["hello"],
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        result = ev._evaluate_actions([])
        assert result["reward"] == 1.0

    def test_no_communicate_criteria(self):
        """No communicate_info → communication returns reward=1.0."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["DB"],
            "actions": None,
            "communicate_info": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        result = ev._evaluate_communication([])
        assert result["reward"] == 1.0


# =============================================================================
# _evaluate_actions Branch Tests — Lines 380-389
# =============================================================================


@pytest.mark.benchmark
class TestEvaluateActionsBranches:
    """Tests for _evaluate_actions() edge cases."""

    def test_function_format_in_trajectory(self):
        """Tool calls in OpenAI 'function' format are parsed correctly."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["ACTION"],
            "actions": [{"name": "check_order", "arguments": {"order_id": "123"}}],
            "communicate_info": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        trajectory = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "check_order", "arguments": '{"order_id": "123"}'}, "id": "tc1"}],
            },
            {"role": "tool", "content": "result"},
        ]
        result = ev._evaluate_actions(trajectory)
        assert result["all_matched"] is True
        assert result["reward"] == 1.0

    def test_string_arguments_parsed(self):
        """String arguments in tool calls are JSON-parsed."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["ACTION"],
            "actions": [{"name": "get_user", "arguments": {"user_id": "u1"}}],
            "communicate_info": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        trajectory = [
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_user", "arguments": '{"user_id": "u1"}', "id": "tc1"}]},
            {"role": "tool", "content": "result"},
        ]
        result = ev._evaluate_actions(trajectory)
        assert result["all_matched"] is True

    def test_user_tool_calls_included(self):
        """User tool calls in trajectory also checked."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["ACTION"],
            "actions": [{"name": "pay_bill", "arguments": {"bill_id": "b1"}}],
            "communicate_info": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        trajectory = [
            {"role": "user", "content": "", "tool_calls": [{"name": "pay_bill", "arguments": {"bill_id": "b1"}}]},
            {"role": "tool", "content": "paid"},
        ]
        result = ev._evaluate_actions(trajectory)
        assert result["all_matched"] is True


# =============================================================================
# _evaluate_communication Branch Tests — Lines 447-452
# =============================================================================


@pytest.mark.benchmark
class TestEvaluateCommunicationBranches:
    """Tests for _evaluate_communication() edge cases."""

    def test_list_content_handled(self):
        """List content (multi-part messages) is joined for matching."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["COMMUNICATE"],
            "communicate_info": ["refund processed"],
            "actions": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        trajectory = [{"role": "assistant", "content": [{"text": "Your refund processed OK"}]}]
        result = ev._evaluate_communication(trajectory)
        assert result["all_found"] is True

    def test_empty_content_skipped(self):
        """Empty assistant content is skipped."""
        task = MagicMock(spec=Task)
        task.environment_data = {"domain": "retail"}
        task.evaluation_data = {
            "reward_basis": ["COMMUNICATE"],
            "communicate_info": ["hello"],
            "actions": None,
            "env_assertions": None,
            "nl_assertions": None,
        }
        ev = Tau2Evaluator(task, MagicMock())
        trajectory = [{"role": "assistant", "content": ""}, {"role": "assistant", "content": "hello world"}]
        result = ev._evaluate_communication(trajectory)
        assert result["all_found"] is True
