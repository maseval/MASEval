"""Data integrity tests for the CONVERSE benchmark.

These tests download real data from the upstream CONVERSE GitHub repository and
validate that the downloaded files are structurally sound.  They are marked
``live`` (network required) and ``slow`` (download takes time) so that they
are excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_converse/test_data_integrity.py -v
"""

import json

import pytest

from maseval.benchmark.converse.data_loader import (
    DOMAIN_MAP,
    PERSONAS,
    ensure_data_exists,
    load_tasks,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]

VALID_DOMAINS = tuple(DOMAIN_MAP.keys())

# Minimum number of attack tasks we expect per domain (across all personas).
# CONVERSE ships several items per persona; these are conservative lower bounds.
MIN_PRIVACY_TASKS = 4
MIN_SECURITY_TASKS = 4


# =============================================================================
# Fixture: download data once for the whole module
# =============================================================================


@pytest.fixture(scope="module")
def converse_data_dir(tmp_path_factory):
    """Download CONVERSE data into a temporary directory for all domains."""
    data_dir = tmp_path_factory.mktemp("converse_data")
    for domain in VALID_DOMAINS:
        ensure_data_exists(domain=domain, data_dir=data_dir)
    return data_dir


# =============================================================================
# File Existence & Format
# =============================================================================


class TestConverseFileIntegrity:
    """Validate that downloaded files exist and are parseable."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_options_txt_exists_and_nonempty(self, converse_data_dir, domain):
        """options.txt exists and is non-empty."""
        path = converse_data_dir / domain / "options.txt"
        assert path.exists(), f"Missing {path}"
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    @pytest.mark.parametrize("persona_id", PERSONAS)
    def test_persona_env_file_exists(self, converse_data_dir, domain, persona_id):
        """env_persona<N>.txt exists and is non-empty."""
        path = converse_data_dir / domain / f"env_persona{persona_id}.txt"
        assert path.exists(), f"Missing {path}"
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    @pytest.mark.parametrize("persona_id", PERSONAS)
    def test_privacy_json_exists_and_parses(self, converse_data_dir, domain, persona_id):
        """Privacy attack JSON exists and is valid JSON."""
        path = converse_data_dir / domain / "privacy" / f"attacks_p{persona_id}.json"
        assert path.exists(), f"Missing {path}"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    @pytest.mark.parametrize("persona_id", PERSONAS)
    def test_security_json_exists_and_parses(self, converse_data_dir, domain, persona_id):
        """Security attack JSON exists and is valid JSON."""
        path = converse_data_dir / domain / "security" / f"attacks_p{persona_id}.json"
        assert path.exists(), f"Missing {path}"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)


# =============================================================================
# Privacy Attack Schema
# =============================================================================


class TestConversePrivacySchema:
    """Validate privacy attack JSON structure."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    @pytest.mark.parametrize("persona_id", PERSONAS)
    def test_privacy_json_has_categories(self, converse_data_dir, domain, persona_id):
        """Privacy JSON contains a non-empty 'categories' key."""
        path = converse_data_dir / domain / "privacy" / f"attacks_p{persona_id}.json"
        data = json.loads(path.read_text())
        assert "categories" in data, f"Missing 'categories' key in {path}"
        assert isinstance(data["categories"], dict)

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_privacy_items_have_required_fields(self, converse_data_dir, domain):
        """Each privacy attack item has 'data_item' and 'attack_action'.

        Note: 'user_task' is optional — the ``unrelated_to_*`` categories in
        the upstream data omit it, and the parser falls back to a default.
        """
        for persona_id in PERSONAS:
            path = converse_data_dir / domain / "privacy" / f"attacks_p{persona_id}.json"
            data = json.loads(path.read_text())
            for cat_name, cat_data in data.get("categories", {}).items():
                for item in cat_data.get("items", []):
                    assert "data_item" in item, f"Missing 'data_item' in {path} / {cat_name}"
                    assert "attack_action" in item, f"Missing 'attack_action' in {path} / {cat_name}"


# =============================================================================
# Security Attack Schema
# =============================================================================


class TestConverseSecuritySchema:
    """Validate security attack JSON structure."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_security_items_have_required_fields(self, converse_data_dir, domain):
        """Each security attack item has 'attack_action' and 'user_task'."""
        for persona_id in PERSONAS:
            path = converse_data_dir / domain / "security" / f"attacks_p{persona_id}.json"
            data = json.loads(path.read_text())

            # Collect attacks from either schema format
            items = []
            if "security_attacks" in data:
                for _cat_name, cat_attacks in data["security_attacks"].items():
                    if isinstance(cat_attacks, dict):
                        for _key, attack_data in cat_attacks.items():
                            if isinstance(attack_data, dict):
                                items.append(attack_data)
            else:
                for section in ("toolkit_attacks", "final_package_attacks"):
                    for _key, attack_data in data.get(section, {}).items():
                        if isinstance(attack_data, dict):
                            items.append(attack_data)

            for item in items:
                assert "attack_action" in item, f"Missing 'attack_action' in {path}"
                assert "user_task" in item, f"Missing 'user_task' in {path}"


# =============================================================================
# Task Loading & Counts
# =============================================================================


class TestConverseTaskCounts:
    """Validate that each domain produces a reasonable number of tasks."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_privacy_task_count(self, converse_data_dir, domain):
        """Privacy split produces at least the expected number of tasks."""
        tasks = load_tasks(domain, split="privacy", data_dir=converse_data_dir)
        assert len(tasks) >= MIN_PRIVACY_TASKS, f"{domain} privacy: expected >= {MIN_PRIVACY_TASKS} tasks, got {len(tasks)}"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_security_task_count(self, converse_data_dir, domain):
        """Security split produces at least the expected number of tasks."""
        tasks = load_tasks(domain, split="security", data_dir=converse_data_dir)
        assert len(tasks) >= MIN_SECURITY_TASKS, f"{domain} security: expected >= {MIN_SECURITY_TASKS} tasks, got {len(tasks)}"


# =============================================================================
# Task Schema
# =============================================================================


class TestConverseTaskSchema:
    """Validate that parsed Task objects have the expected structure."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_privacy_task_fields(self, converse_data_dir, domain):
        """Privacy tasks have correct query, user_data, and evaluation_data."""
        tasks = load_tasks(domain, split="privacy", limit=5, data_dir=converse_data_dir)

        for task in tasks:
            assert task.query, f"Task {task.id} has empty query"
            assert isinstance(task.environment_data, dict)
            assert task.environment_data.get("domain") == domain
            assert task.environment_data.get("persona_text"), f"Task {task.id} missing persona_text"

            assert task.user_data.get("attack_type") == "privacy"
            assert task.user_data.get("attack_goal"), f"Task {task.id} missing attack_goal"

            assert task.evaluation_data.get("type") == "privacy"
            assert "target_info" in task.evaluation_data, f"Task {task.id} missing target_info"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_security_task_fields(self, converse_data_dir, domain):
        """Security tasks have correct query, user_data, and evaluation_data."""
        tasks = load_tasks(domain, split="security", limit=5, data_dir=converse_data_dir)

        for task in tasks:
            assert task.query, f"Task {task.id} has empty query"
            assert isinstance(task.environment_data, dict)
            assert task.environment_data.get("domain") == domain

            assert task.user_data.get("attack_type") == "security"

            assert task.evaluation_data.get("type") == "security"
            assert "forbidden_tools" in task.evaluation_data, f"Task {task.id} missing forbidden_tools"
            assert isinstance(task.evaluation_data["forbidden_tools"], list)
