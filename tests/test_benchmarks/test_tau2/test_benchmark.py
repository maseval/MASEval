"""Unit tests for Tau2Benchmark."""

import pytest
from unittest.mock import MagicMock, patch
from maseval import Task
from maseval.benchmark.tau2 import Tau2Benchmark, Tau2User


class DummyTau2Benchmark(Tau2Benchmark):
    """Subclass for testing abstract base class."""

    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        return [], {}

    def get_model_adapter(self, model_id, **kwargs):
        return MagicMock()


@pytest.fixture
def benchmark():
    return DummyTau2Benchmark()


@pytest.fixture
def task():
    t = MagicMock(spec=Task)
    t.environment_data = {"domain": "retail"}
    t.user_data = {"model_id": "gpt-4o", "instructions": "Call about order."}
    t.evaluation_data = {"actions": None, "reward_basis": ["DB"]}
    t.query = "Hello"
    return t


# =============================================================================
# Class Structure Tests
# =============================================================================


@pytest.mark.benchmark
class TestTau2BenchmarkClassStructure:
    """Tests for Tau2Benchmark class structure."""

    def test_inherits_from_benchmark(self):
        """Tau2Benchmark inherits from Benchmark base class."""
        from maseval.core.benchmark import Benchmark

        assert issubclass(Tau2Benchmark, Benchmark)

    def test_has_abstract_methods(self):
        """Tau2Benchmark requires setup_agents and get_model_adapter."""

        # These should be abstract in Tau2Benchmark or its parent
        assert hasattr(Tau2Benchmark, "setup_agents")
        assert hasattr(Tau2Benchmark, "get_model_adapter")

    def test_default_max_invocations(self):
        """MAX_INVOCATIONS is 50 (matching tau2-bench max_steps/4)."""
        assert Tau2Benchmark.MAX_INVOCATIONS == 50

    def test_setup_evaluators_returns_tau2_evaluator(self, benchmark, task, seed_gen):
        """setup_evaluators returns Tau2Evaluator."""
        from maseval.benchmark.tau2.evaluator import Tau2Evaluator

        mock_env = MagicMock()
        evaluators = benchmark.setup_evaluators(mock_env, task, [], None, seed_gen)

        assert len(evaluators) == 1
        assert isinstance(evaluators[0], Tau2Evaluator)

    def test_run_agents_returns_results(self, benchmark, task, seed_gen):
        """run_agents executes agents and returns results."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Done"
        mock_env = MagicMock()

        result = benchmark.run_agents([mock_agent], task, mock_env, "query")

        assert result == "Done"
        mock_agent.run.assert_called_once_with("query")


# =============================================================================
# DefaultAgentTau2Benchmark Tests
# =============================================================================


@pytest.mark.benchmark
class TestDefaultAgentTau2BenchmarkSetupAgents:
    """Tests for DefaultAgentTau2Benchmark.setup_agents."""

    def test_raises_without_model_id(self, seed_gen):
        """Raises ValueError when model_id missing from agent_data."""
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark

        class Concrete(DefaultAgentTau2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return MagicMock()

        benchmark = Concrete()
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Policy"

        with pytest.raises(ValueError, match="model_id not configured"):
            benchmark.setup_agents({}, mock_env, MagicMock(spec=Task), None, seed_gen)

    def test_creates_agent_with_model_id(self, seed_gen):
        """Creates DefaultTau2Agent with model_id from agent_data."""
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark, DefaultTau2AgentAdapter

        class Concrete(DefaultAgentTau2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return MagicMock()

        benchmark = Concrete()
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {"tool1": lambda: None}
        mock_env.policy = "Policy"

        agents_list, agents_dict = benchmark.setup_agents({"model_id": "test-model"}, mock_env, MagicMock(spec=Task), None, seed_gen)

        assert len(agents_list) == 1
        assert isinstance(agents_list[0], DefaultTau2AgentAdapter)
        assert "default_agent" in agents_dict


# =============================================================================
# Setup Method Tests
# =============================================================================


@pytest.mark.benchmark
def test_setup_environment(benchmark, task):
    """Test environment setup."""
    from maseval.core.seeding import DefaultSeedGenerator

    seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
    with patch("maseval.benchmark.tau2.tau2.Tau2Environment") as mock_env_cls:
        benchmark.setup_environment({}, task, seed_gen)

        mock_env_cls.assert_called_once_with(task_data=task.environment_data)


@pytest.mark.benchmark
def test_setup_user(benchmark, task):
    """Test user setup."""
    from maseval.core.seeding import DefaultSeedGenerator

    mock_env = MagicMock()
    mock_env.create_user_tools.return_value = {}
    seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

    user = benchmark.setup_user({}, mock_env, task, seed_gen)

    assert isinstance(user, Tau2User)
    assert user.scenario == "Call about order."
    # Check that model adapter was requested with correct ID
    # Since we use DummyTau2Benchmark which returns a mock, we assume it worked if user is created.


@pytest.mark.benchmark
def test_setup_user_missing_model_id(benchmark, task):
    """Test that missing model_id raises ValueError."""
    from maseval.core.seeding import DefaultSeedGenerator

    task.user_data = {}  # Remove model_id
    seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

    with pytest.raises(ValueError, match="not configured"):
        benchmark.setup_user({}, MagicMock(), task, seed_gen)


# =============================================================================
# Seeding Tests
# =============================================================================


@pytest.mark.benchmark
class TestTau2BenchmarkSeeding:
    """Tests for Tau2Benchmark seeding behavior."""

    def test_setup_user_passes_seed_to_get_model_adapter(self, task):
        """Test that setup_user derives and passes seed to get_model_adapter."""
        from maseval.core.seeding import DefaultSeedGenerator

        captured_kwargs = []

        class CapturingBenchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return MagicMock()

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()
        mock_env.create_user_tools.return_value = {}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        benchmark.setup_user({}, mock_env, task, seed_generator=seed_gen)

        assert len(captured_kwargs) == 1
        assert "seed" in captured_kwargs[0]
        assert captured_kwargs[0]["seed"] is not None
        assert isinstance(captured_kwargs[0]["seed"], int)

    def test_setup_user_passes_none_seed_when_seeding_disabled(self, task):
        """Test that setup_user passes None seed when global_seed is None."""
        from maseval.core.seeding import DefaultSeedGenerator

        captured_kwargs = []

        class CapturingBenchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return MagicMock()

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()
        mock_env.create_user_tools.return_value = {}

        # Use a seed generator with global_seed=None to test disabled seeding
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
        benchmark.setup_user({}, mock_env, task, seed_generator=seed_gen)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("seed") is None

    def test_setup_user_uses_correct_seed_path(self, task):
        """Test that setup_user uses the documented seed path."""
        from maseval.core.seeding import DefaultSeedGenerator

        class CapturingBenchmark(Tau2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return MagicMock()

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()
        mock_env.create_user_tools.return_value = {}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        benchmark.setup_user({}, mock_env, task, seed_generator=seed_gen)

        assert "simulators/user" in seed_gen.seed_log


@pytest.mark.benchmark
class TestDefaultAgentTau2BenchmarkSeeding:
    """Tests for DefaultAgentTau2Benchmark seeding behavior."""

    def test_setup_agents_passes_seed_to_get_model_adapter(self):
        """Test that setup_agents derives and passes seed to get_model_adapter."""
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark
        from maseval.core.seeding import DefaultSeedGenerator

        captured_kwargs = []

        class CapturingBenchmark(DefaultAgentTau2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return MagicMock()

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Test policy"

        mock_task = MagicMock(spec=Task)
        mock_task.environment_data = {"domain": "retail"}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        # agent_data is passed to setup_agents, not the constructor
        benchmark.setup_agents({"model_id": "test-model"}, mock_env, mock_task, None, seed_generator=seed_gen)

        assert len(captured_kwargs) == 1
        assert "seed" in captured_kwargs[0]
        assert captured_kwargs[0]["seed"] is not None

    def test_setup_agents_uses_correct_seed_path(self):
        """Test that setup_agents uses the documented seed path."""
        from maseval.benchmark.tau2 import DefaultAgentTau2Benchmark
        from maseval.core.seeding import DefaultSeedGenerator

        class CapturingBenchmark(DefaultAgentTau2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return MagicMock()

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Test policy"

        mock_task = MagicMock(spec=Task)
        mock_task.environment_data = {"domain": "retail"}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        # agent_data is passed to setup_agents, not the constructor
        benchmark.setup_agents({"model_id": "test-model"}, mock_env, mock_task, None, seed_generator=seed_gen)

        assert "agents/default_agent" in seed_gen.seed_log
