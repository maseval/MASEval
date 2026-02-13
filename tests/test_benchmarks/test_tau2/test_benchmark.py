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
    t.query = "Hello"
    return t


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
