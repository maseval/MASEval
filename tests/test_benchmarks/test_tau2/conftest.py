"""Shared fixtures for Tau 2 benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, etc.)
- tests/test_benchmarks/test_tau2/conftest.py: Tau2-specific fixtures (this file)

Tau2 tests can use fixtures from both levels - pytest handles this automatically.
"""

import pytest

from maseval import Task


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
