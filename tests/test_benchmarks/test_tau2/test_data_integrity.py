"""Data integrity tests for Tau2 benchmark.

These tests download real data from the tau2-bench GitHub repository and
validate that the downloaded files are structurally sound.  They are marked
``live`` (network required) and ``slow`` (download takes time) so that they
are excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_tau2/test_data_integrity.py -v
"""

import json
from pathlib import Path

import pytest

from maseval.benchmark.tau2.data_loader import (
    BASE_SPLIT_COUNTS,
    VALID_DOMAINS,
    download_domain_data,
    load_domain_config,
    load_tasks,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]


# =============================================================================
# File Existence & Format
# =============================================================================


class TestTau2FileIntegrity:
    """Validate that downloaded files exist and are parseable."""

    _data_dir: Path

    @pytest.fixture(scope="class", autouse=True)
    def _download_data(self, tmp_path_factory):
        """Download Tau2 data into a temporary directory once for this class."""
        self.__class__._data_dir = tmp_path_factory.mktemp("tau2_data")
        download_domain_data(data_dir=self._data_dir, verbose=0)

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_tasks_json_exists_and_parses(self, domain):
        """tasks.json exists and is valid JSON."""
        path = self._data_dir / domain / "tasks.json"
        assert path.exists(), f"Missing {path}"
        data = json.loads(path.read_text())
        assert isinstance(data, list)

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_policy_md_exists(self, domain):
        """policy.md exists and is non-empty."""
        path = self._data_dir / domain / "policy.md"
        assert path.exists(), f"Missing {path}"
        assert path.stat().st_size > 0

    @pytest.mark.parametrize(
        "domain,ext",
        [("retail", ".json"), ("airline", ".json"), ("telecom", ".toml")],
    )
    def test_db_file_exists_and_nonempty(self, domain, ext):
        """Database file exists and is non-empty."""
        path = self._data_dir / domain / f"db{ext}"
        assert path.exists(), f"Missing {path}"
        assert path.stat().st_size > 100, f"db file suspiciously small: {path.stat().st_size} bytes"


# =============================================================================
# Task Counts
# =============================================================================


class TestTau2TaskCounts:
    """Validate that each domain has at least the expected number of tasks."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_base_split_count(self, domain, ensure_tau2_data):
        """Base split has at least the expected number of tasks."""
        tasks = load_tasks(domain, split="base", data_dir=ensure_tau2_data)
        expected = BASE_SPLIT_COUNTS[domain]
        assert len(tasks) >= expected, f"{domain} base split: expected >= {expected} tasks, got {len(tasks)}"


# =============================================================================
# Task Schema
# =============================================================================


class TestTau2TaskSchema:
    """Validate that task objects have the expected structure."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_task_fields_populated(self, domain, ensure_tau2_data):
        """Every task has a non-empty query and correctly shaped data dicts."""
        tasks = load_tasks(domain, split="base", limit=10, data_dir=ensure_tau2_data)

        for task in tasks:
            assert task.query, f"Task {task.id} has empty query"
            assert isinstance(task.environment_data, dict)
            assert task.environment_data.get("domain") == domain
            assert task.environment_data.get("policy"), f"Task {task.id} missing policy"
            assert task.environment_data.get("db_path"), f"Task {task.id} missing db_path"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_evaluation_criteria_present(self, domain, ensure_tau2_data):
        """Every task has evaluation criteria."""
        tasks = load_tasks(domain, split="base", limit=10, data_dir=ensure_tau2_data)

        for task in tasks:
            eval_data = task.evaluation_data
            assert isinstance(eval_data, dict)
            assert "reward_basis" in eval_data, f"Task {task.id} missing reward_basis"


# =============================================================================
# Database Content
# =============================================================================


class TestTau2DatabaseContent:
    """Validate that domain databases contain minimum expected data.

    These checks correspond to the entities that Tau2 domain tool tests
    depend on.  If these fail, the conditional ``pytest.skip`` calls in the
    domain tool tests would fire, silently reducing coverage.
    """

    def test_retail_db_has_entities(self, ensure_tau2_data):
        """Retail DB has users, orders, and products."""
        from maseval.benchmark.tau2.domains.retail import RetailDB

        config = load_domain_config("retail", ensure_tau2_data)
        db = RetailDB.load(config["db_path"])

        assert len(db.users) > 0, "retail DB has no users"
        assert len(db.orders) > 0, "retail DB has no orders"
        assert len(db.products) > 0, "retail DB has no products"

    def test_retail_db_users_have_payment_methods(self, ensure_tau2_data):
        """At least one retail user has multiple payment methods."""
        from maseval.benchmark.tau2.domains.retail import RetailDB

        config = load_domain_config("retail", ensure_tau2_data)
        db = RetailDB.load(config["db_path"])

        max_methods = max(len(u.payment_methods) for u in db.users.values())
        assert max_methods >= 2, f"No retail user has >= 2 payment methods (max={max_methods}). This will cause test_retail_tools skips."

    def test_airline_db_has_entities(self, ensure_tau2_data):
        """Airline DB has users, reservations, and flights."""
        from maseval.benchmark.tau2.domains.airline import AirlineDB

        config = load_domain_config("airline", ensure_tau2_data)
        db = AirlineDB.load(config["db_path"])

        assert len(db.users) > 0, "airline DB has no users"
        assert len(db.reservations) > 0, "airline DB has no reservations"
        assert len(db.flights) > 0, "airline DB has no flights"

    @pytest.mark.xfail(reason="v0.2.0 upstream data has no reservations with nonfree baggages")
    def test_airline_db_has_nonfree_baggages(self, ensure_tau2_data):
        """At least one airline reservation has nonfree baggages."""
        from maseval.benchmark.tau2.domains.airline import AirlineDB

        config = load_domain_config("airline", ensure_tau2_data)
        db = AirlineDB.load(config["db_path"])

        has_nonfree = any(r.nonfree_baggages for r in db.reservations.values())
        assert has_nonfree, "No airline reservation has nonfree_baggages. This will cause test_airline_tools skips."

    def test_telecom_db_has_entities(self, ensure_tau2_data):
        """Telecom DB has customers, lines, bills, and plans."""
        from maseval.benchmark.tau2.domains.telecom import TelecomDB

        config = load_domain_config("telecom", ensure_tau2_data)
        db = TelecomDB.load(config["db_path"])

        assert len(db.customers) > 0, "telecom DB has no customers"
        assert len(db.lines) > 0, "telecom DB has no lines"
        assert len(db.bills) > 0, "telecom DB has no bills"
        assert len(db.plans) > 0, "telecom DB has no plans"
