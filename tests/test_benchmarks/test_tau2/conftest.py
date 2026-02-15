"""Shared fixtures for Tau 2 benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, etc.)
- tests/test_benchmarks/test_tau2/conftest.py: Tau2-specific fixtures (this file)

Tau2 tests can use fixtures from both levels - pytest handles this automatically.
"""

import pytest
from typing import Any, Dict, Optional, Sequence, Tuple
from unittest.mock import MagicMock

from conftest import DummyModelAdapter
from maseval import AgentAdapter, Task, ModelAdapter


# =============================================================================
# Session-Scoped Setup
# =============================================================================


@pytest.fixture(scope="session")
def ensure_tau2_data():
    """Download Tau2 domain data to the package's default data directory.

    Downloads data files (db.json, tasks.json, policy.md) if not already present.
    Uses ensure_data_exists() which caches: skips download when files exist.

    Tests that need real data should depend on this fixture and be marked @pytest.mark.live.
    Tests that don't need data (structural, mock-based) should NOT depend on this fixture.
    """
    from maseval.benchmark.tau2.data_loader import ensure_data_exists, DEFAULT_DATA_DIR

    for domain in ["retail", "airline", "telecom"]:
        ensure_data_exists(domain=domain, verbose=0)

    return DEFAULT_DATA_DIR


# =============================================================================
# Domain Database Fixtures
# =============================================================================


@pytest.fixture
def retail_db(ensure_tau2_data):
    """Load the retail domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config
    from maseval.benchmark.tau2.domains.retail import RetailDB

    config = load_domain_config("retail")
    return RetailDB.load(config["db_path"])


@pytest.fixture
def airline_db(ensure_tau2_data):
    """Load the airline domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config
    from maseval.benchmark.tau2.domains.airline import AirlineDB

    config = load_domain_config("airline")
    return AirlineDB.load(config["db_path"])


@pytest.fixture
def telecom_db(ensure_tau2_data):
    """Load the telecom domain database."""
    from maseval.benchmark.tau2.data_loader import load_domain_config
    from maseval.benchmark.tau2.domains.telecom import TelecomDB

    config = load_domain_config("telecom")
    return TelecomDB.load(config["db_path"])


# =============================================================================
# Toolkit Fixtures
# =============================================================================


@pytest.fixture
def retail_toolkit(retail_db):
    """Create a retail toolkit with database."""
    from maseval.benchmark.tau2.domains.retail import RetailTools

    return RetailTools(retail_db)


@pytest.fixture
def airline_toolkit(airline_db):
    """Create an airline toolkit with database."""
    from maseval.benchmark.tau2.domains.airline import AirlineTools

    return AirlineTools(airline_db)


@pytest.fixture
def telecom_toolkit(telecom_db):
    """Create a telecom toolkit with database."""
    from maseval.benchmark.tau2.domains.telecom import TelecomTools

    return TelecomTools(telecom_db)


@pytest.fixture
def telecom_user_toolkit(telecom_db):
    """Create a telecom user toolkit with database."""
    from maseval.benchmark.tau2.domains.telecom import TelecomUserTools

    # Ensure user_db is initialized (handled by toolkit init, but good to be explicit)
    return TelecomUserTools(telecom_db)


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def retail_environment(ensure_tau2_data):
    """Create a retail environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "retail"})


@pytest.fixture
def airline_environment(ensure_tau2_data):
    """Create an airline environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "airline"})


@pytest.fixture
def telecom_environment(ensure_tau2_data):
    """Create a telecom environment."""
    from maseval.benchmark.tau2 import Tau2Environment

    return Tau2Environment({"domain": "telecom"})


# =============================================================================
# Task Fixtures
# =============================================================================


@pytest.fixture
def sample_retail_task():
    """Sample retail domain task."""
    return Task(
        query="I want to cancel my order",
        environment_data={"domain": "retail"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Cancel order #12345"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "retail"},
    )


@pytest.fixture
def sample_airline_task():
    """Sample airline domain task."""
    return Task(
        query="I need to change my flight",
        environment_data={"domain": "airline"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Change flight reservation"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "airline"},
    )


@pytest.fixture
def sample_telecom_task():
    """Sample telecom domain task."""
    return Task(
        query="My internet is not working",
        environment_data={"domain": "telecom"},
        user_data={
            "model_id": "test-model",
            "instructions": {"reason_for_call": "Internet connectivity issue"},
        },
        evaluation_data={"model_id": "test-model"},
        metadata={"domain": "telecom"},
    )


# =============================================================================
# Concrete Benchmark Implementation
# =============================================================================


@pytest.fixture
def concrete_tau2_benchmark():
    """Create a concrete Tau2Benchmark class for testing."""
    from maseval.benchmark.tau2 import Tau2Benchmark, Tau2Environment

    class ConcreteTau2Benchmark(Tau2Benchmark):
        """Concrete implementation for testing."""

        def __init__(self, model_factory: Optional[Any] = None, **kwargs: Any):
            if model_factory is None:
                self._model_factory = lambda name: DummyModelAdapter(
                    model_id=f"test-{name}",
                    responses=["I have completed the task. ###STOP###"],
                )
            elif callable(model_factory):
                self._model_factory = model_factory
            else:
                self._model_factory = lambda name: model_factory
            super().__init__(**kwargs)

        def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
            factory_key = kwargs.get("register_name", model_id)
            adapter = self._model_factory(factory_key)
            register_name = kwargs.get("register_name")
            if register_name:
                try:
                    self.register("models", register_name, adapter)
                except ValueError:
                    pass
            return adapter

        def setup_agents(  # type: ignore[override]
            self,
            agent_data: Dict[str, Any],
            environment: Tau2Environment,
            task: Task,
            user: Optional[Any],
            seed_generator,
        ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
            agent = MagicMock(spec=AgentAdapter)
            agent.name = "test_agent"
            agent.run.return_value = "Task completed"
            agent.gather_traces.return_value = {"type": "MockAgent", "messages": []}
            agent.callbacks = []
            return [agent], {"test_agent": agent}

    return ConcreteTau2Benchmark


@pytest.fixture
def seed_gen():
    """Seed generator fixture for direct setup method calls.

    When seeding is disabled (global_seed=None), derive_seed() returns None.
    This fixture is for tests that call setup methods directly outside of run().
    """
    from maseval.core.seeding import DefaultSeedGenerator

    return DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
