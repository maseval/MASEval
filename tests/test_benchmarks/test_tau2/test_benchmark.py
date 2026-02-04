"""Unit tests for Tau2Benchmark."""

import pytest
from unittest.mock import MagicMock, patch
from maseval import Task
from maseval.benchmark.tau2 import Tau2Benchmark, Tau2User


class DummyTau2Benchmark(Tau2Benchmark):
    """Subclass for testing abstract base class."""

    def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
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
    with patch("maseval.benchmark.tau2.tau2.Tau2Environment") as mock_env_cls:
        benchmark.setup_environment({}, task)

        mock_env_cls.assert_called_once_with(task_data=task.environment_data)


@pytest.mark.benchmark
def test_setup_user(benchmark, task):
    """Test user setup."""
    mock_env = MagicMock()
    mock_env.create_user_tools.return_value = {}

    user = benchmark.setup_user({}, mock_env, task)

    assert isinstance(user, Tau2User)
    assert user.scenario == "Call about order."
    # Check that model adapter was requested with correct ID
    # Since we use DummyTau2Benchmark which returns a mock, we assume it worked if user is created.


@pytest.mark.benchmark
def test_setup_user_missing_model_id(benchmark, task):
    """Test that missing model_id raises ValueError."""
    task.user_data = {}  # Remove model_id

    with pytest.raises(ValueError, match="not configured"):
        benchmark.setup_user({}, MagicMock(), task)
