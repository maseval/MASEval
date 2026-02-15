"""Integration tests for MultiAgentBench using real MARBLE data.

These tests validate that MultiAgentBench components work correctly with
actual MARBLE data, not synthetic fixtures.  They are marked ``live`` +
``slow`` + ``benchmark`` because they require MARBLE to be cloned and
exercise the full pipeline.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_multiagentbench/test_integration_real_data.py -v
"""

import pytest

from conftest import DummyModelAdapter
from maseval.benchmark.multiagentbench.data_loader import (
    VALID_DOMAINS,
    configure_model_ids,
    load_tasks,
)
from maseval.benchmark.multiagentbench.environment import (
    INFRASTRUCTURE_DOMAINS,
    MultiAgentBenchEnvironment,
)
from maseval.benchmark.multiagentbench.evaluator import MultiAgentBenchEvaluator

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]


@pytest.fixture(autouse=True)
def _mock_marble_environment():
    """Override: integration tests use real marble."""
    yield


# Domains that can be tested without external infrastructure (Docker, Minecraft Server)
NON_INFRA_DOMAINS = sorted(VALID_DOMAINS - INFRASTRUCTURE_DOMAINS - {"minecraft", "werewolf"})

# All domains except minecraft (untested upstream, requires game server)
EVALUATABLE_DOMAINS = sorted(VALID_DOMAINS - {"minecraft"})


# =============================================================================
# Fixture: ensure MARBLE is available
# =============================================================================


@pytest.fixture(scope="module")
def marble_data_dir():
    """Ensure MARBLE is available and return the multiagentbench data dir."""
    from maseval.benchmark.multiagentbench.data_loader import ensure_marble_exists

    marble_dir = ensure_marble_exists(auto_download=True)
    data_dir = marble_dir / "multiagentbench"
    assert data_dir.exists(), f"MARBLE data dir not found: {data_dir}"
    return data_dir


# =============================================================================
# Data Loading with Real Data
# =============================================================================


class TestMultiAgentBenchRealDataLoading:
    """Test that load_tasks works with real MARBLE data for all domains."""

    @pytest.mark.parametrize("domain", sorted(VALID_DOMAINS - {"werewolf", "minecraft"}))
    def test_load_tasks_returns_tasks(self, domain, marble_data_dir):
        """load_tasks(domain) returns a non-empty list of Tasks."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=5)
        assert len(tasks) > 0, f"No tasks loaded for domain '{domain}'. Check test_data_integrity tests first."

    @pytest.mark.parametrize("domain", sorted(VALID_DOMAINS - {"werewolf", "minecraft"}))
    def test_tasks_have_agents(self, domain, marble_data_dir):
        """Every loaded task has at least one agent."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=3)
        for task in tasks:
            agents = task.environment_data.get("agents", [])
            assert len(agents) >= 1, f"Task {task.id} in domain '{domain}' has {len(agents)} agents, expected at least 1."

    @pytest.mark.parametrize("domain", sorted(VALID_DOMAINS - {"werewolf", "minecraft"}))
    def test_configure_model_ids_modifies_tasks(self, domain, marble_data_dir):
        """configure_model_ids() sets llm and evaluator model_id."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=2)
        configure_model_ids(tasks, agent_model_id="test-model")

        for task in tasks:
            assert task.environment_data.get("llm") == "test-model", (
                f"Task {task.id}: environment_data['llm'] not set after configure_model_ids()"
            )
            assert task.evaluation_data.get("model_id") == "test-model", (
                f"Task {task.id}: evaluation_data['model_id'] not set after configure_model_ids()"
            )

    def test_load_werewolf_tasks(self, marble_data_dir):
        """Werewolf tasks load from config files."""
        tasks = load_tasks("werewolf", data_dir=marble_data_dir)
        assert len(tasks) > 0, "No werewolf tasks loaded"


# =============================================================================
# Environment Setup with Real Data
# =============================================================================


class TestMultiAgentBenchRealEnvironment:
    """Test MultiAgentBenchEnvironment with real MARBLE task data."""

    @pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)
    def test_environment_creates(self, domain, marble_data_dir):
        """MultiAgentBenchEnvironment can be created from a real task."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=1)
        assert len(tasks) > 0

        env = MultiAgentBenchEnvironment(task_data=tasks[0].environment_data)
        assert env is not None
        assert env.domain == domain

    @pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)
    def test_environment_setup_state(self, domain, marble_data_dir):
        """setup_state() extracts domain and max_iterations from real data."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=1)
        env = MultiAgentBenchEnvironment(task_data=tasks[0].environment_data)
        state = env.setup_state(tasks[0].environment_data)

        assert isinstance(state, dict)
        assert state.get("domain") == domain
        assert "max_iterations" in state
        assert state["max_iterations"] > 0

    @pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)
    def test_environment_gather_traces(self, domain, marble_data_dir):
        """gather_traces() returns dict with expected keys."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=1)
        env = MultiAgentBenchEnvironment(task_data=tasks[0].environment_data)
        env.setup_state(tasks[0].environment_data)
        traces = env.gather_traces()

        assert isinstance(traces, dict)
        assert "domain" in traces
        assert traces["domain"] == domain

    @pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)
    def test_environment_gather_config(self, domain, marble_data_dir):
        """gather_config() returns dict with domain and tool info."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=1)
        env = MultiAgentBenchEnvironment(task_data=tasks[0].environment_data)
        env.setup_state(tasks[0].environment_data)
        config = env.gather_config()

        assert isinstance(config, dict)
        assert config.get("domain") == domain


# =============================================================================
# Evaluator with Real Data
# =============================================================================


class TestMultiAgentBenchRealEvaluation:
    """Test evaluator creation with real task data."""

    @pytest.mark.parametrize("domain", EVALUATABLE_DOMAINS)
    def test_evaluator_creates_from_real_domain(self, domain, marble_data_dir):
        """MultiAgentBenchEvaluator can be created for each domain."""
        model = DummyModelAdapter(
            model_id="test-eval-model",
            responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )
        evaluator = MultiAgentBenchEvaluator(
            domain=domain,
            model_adapter=model,
        )
        assert evaluator is not None

    @pytest.mark.parametrize("domain", EVALUATABLE_DOMAINS)
    def test_evaluator_filter_traces(self, domain, marble_data_dir):
        """filter_traces() processes a synthetic trace structure."""
        model = DummyModelAdapter(
            model_id="test-eval-model",
            responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
        )
        evaluator = MultiAgentBenchEvaluator(
            domain=domain,
            model_adapter=model,
        )

        # Minimal synthetic traces
        traces = {
            "agents": {
                "agent_1": {
                    "action_log": [],
                    "communication_log": [],
                    "token_usage": 100,
                }
            },
            "environment": {
                "domain": domain,
            },
        }

        filtered = evaluator.filter_traces(traces)
        assert isinstance(filtered, dict)


# =============================================================================
# Pipeline Smoke Test
# =============================================================================


class TestMultiAgentBenchPipelineSmoke:
    """Smoke test for the full pipeline with real data."""

    @pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)
    def test_full_pipeline_single_task(self, domain, marble_data_dir, concrete_multiagentbench_benchmark):
        """benchmark.run() on one real task produces results."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=1)
        assert len(tasks) > 0, f"No tasks for domain '{domain}'. Check test_data_integrity tests first."

        # Configure model IDs (required for evaluation)
        configure_model_ids(tasks, agent_model_id="test-model")

        benchmark = concrete_multiagentbench_benchmark(
            progress_bar=False,
            max_invocations=1,
        )

        results = benchmark.run(tasks, agent_data={})

        assert len(results) == 1, f"Expected 1 result for domain '{domain}', got {len(results)}."

        result = results[0]
        assert "status" in result, "Result missing 'status' key"
        # All possible TaskExecutionStatus values
        known_statuses = {
            "success",
            "agent_error",
            "environment_error",
            "user_error",
            "task_timeout",
            "unknown_execution_error",
            "evaluation_failed",
            "setup_failed",
        }
        assert result["status"] in known_statuses, f"Unexpected status '{result['status']}' for domain '{domain}'. Known: {known_statuses}"
