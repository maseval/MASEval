"""Data integrity tests for MultiAgentBench benchmark.

These tests validate that MARBLE data (either locally cloned or freshly
downloaded) is structurally sound.  They are marked ``live`` (network may
be required for initial clone) and ``slow`` (git clone takes time) so
that they are excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_multiagentbench/test_data_integrity.py -v
"""

import json

import pytest

from maseval.benchmark.multiagentbench.data_loader import (
    VALID_DOMAINS,
    load_tasks,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]

# JSONL domains that can be loaded with load_tasks().
# Werewolf uses config-based loading (no JSONL data).
# Minecraft has JSONL data but entries lack 'scenario' and 'task_id' fields
# required by _parse_task_entry() — a pre-existing data loader limitation.
JSONL_DOMAINS = sorted(VALID_DOMAINS - {"werewolf", "minecraft"})

# Expected minimum task count per domain.  MARBLE has 100 tasks per JSONL domain.
MIN_TASKS_PER_DOMAIN = 50


# =============================================================================
# Fixture: ensure MARBLE is available
# =============================================================================


@pytest.fixture(scope="module")
def marble_dir():
    """Ensure MARBLE data is available.

    Uses ensure_marble_exists() which reuses an existing clone if present,
    or downloads from GitHub if not.
    """
    from maseval.benchmark.multiagentbench.data_loader import ensure_marble_exists

    return ensure_marble_exists(auto_download=True)


@pytest.fixture(scope="module")
def marble_data_dir(marble_dir):
    """Resolve the MARBLE multiagentbench data directory."""
    data_dir = marble_dir / "multiagentbench"
    assert data_dir.exists(), f"MARBLE multiagentbench directory not found at {data_dir}. The MARBLE clone may be incomplete."
    return data_dir


# =============================================================================
# MARBLE Data Presence
# =============================================================================


class TestMarbleDataPresence:
    """Validate that MARBLE data files exist for all domains."""

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_domain_directory_exists(self, domain, marble_data_dir):
        """Domain directory exists in MARBLE multiagentbench/."""
        domain_dir = marble_data_dir / domain
        assert domain_dir.exists(), (
            f"Domain directory missing: {domain_dir}. MARBLE clone may be incomplete or the domain was renamed upstream."
        )

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_domain_has_jsonl(self, domain, marble_data_dir):
        """Domain has its {domain}_main.jsonl file."""
        jsonl_path = marble_data_dir / domain / f"{domain}_main.jsonl"
        assert jsonl_path.exists(), f"JSONL file missing: {jsonl_path}. MARBLE data structure may have changed upstream."

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_jsonl_is_non_empty(self, domain, marble_data_dir):
        """JSONL file has content."""
        jsonl_path = marble_data_dir / domain / f"{domain}_main.jsonl"
        assert jsonl_path.stat().st_size > 100, f"JSONL file suspiciously small ({jsonl_path.stat().st_size} bytes): {jsonl_path}"


# =============================================================================
# Task Structure
# =============================================================================


class TestMarbleTaskStructure:
    """Validate loaded tasks have expected structure."""

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_minimum_task_count(self, domain, marble_data_dir):
        """Each JSONL domain has at least the expected number of tasks."""
        tasks = load_tasks(domain, data_dir=marble_data_dir)
        assert len(tasks) >= MIN_TASKS_PER_DOMAIN, (
            f"Domain '{domain}' has {len(tasks)} tasks, expected >= {MIN_TASKS_PER_DOMAIN}. This may indicate upstream data loss."
        )

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_required_fields_in_environment_data(self, domain, marble_data_dir):
        """Every task has required fields in environment_data."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=10)
        for task in tasks:
            assert "scenario" in task.environment_data, f"Task {task.id} missing 'scenario' in environment_data"
            assert "agents" in task.environment_data, f"Task {task.id} missing 'agents' in environment_data"
            assert "relationships" in task.environment_data, f"Task {task.id} missing 'relationships' in environment_data"

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_agents_have_ids(self, domain, marble_data_dir):
        """Every agent in every task has an agent_id."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=10)
        for task in tasks:
            agents = task.environment_data.get("agents", [])
            assert len(agents) > 0, f"Task {task.id} in domain '{domain}' has no agents. MultiAgentBench tasks require at least one agent."
            for agent in agents:
                assert "agent_id" in agent, f"Agent in task {task.id} missing 'agent_id': {agent}"

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_tasks_have_queries(self, domain, marble_data_dir):
        """Every task has a non-empty query."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=10)
        for task in tasks:
            assert task.query, f"Task {task.id} in domain '{domain}' has empty query"

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_tasks_have_metadata_domain(self, domain, marble_data_dir):
        """Every task records its domain in metadata."""
        tasks = load_tasks(domain, data_dir=marble_data_dir, limit=5)
        for task in tasks:
            assert task.metadata.get("domain") == domain, (
                f"Task {task.id} metadata domain is '{task.metadata.get('domain')}', expected '{domain}'"
            )


# =============================================================================
# JSONL Raw Data Validation
# =============================================================================


class TestMarbleRawJsonl:
    """Validate raw JSONL files parse correctly and have required schema."""

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_jsonl_entries_parse(self, domain, marble_data_dir):
        """Every line in the JSONL file is valid JSON."""
        jsonl_path = marble_data_dir / domain / f"{domain}_main.jsonl"
        with jsonl_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {idx} in {jsonl_path.name} is invalid JSON: {e}")
                assert isinstance(entry, dict), f"Line {idx} in {jsonl_path.name} is not a JSON object"

    @pytest.mark.parametrize("domain", JSONL_DOMAINS)
    def test_jsonl_required_fields(self, domain, marble_data_dir):
        """Raw JSONL entries have the fields required by _parse_task_entry()."""
        jsonl_path = marble_data_dir / domain / f"{domain}_main.jsonl"
        required = {"scenario", "task_id", "task", "agents", "relationships"}

        with jsonl_path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                missing = required - set(entry.keys())
                assert not missing, f"Line {idx} in {domain}_main.jsonl missing fields: {missing}"
                if idx >= 9:  # Check first 10 entries
                    break


# =============================================================================
# Werewolf Config Files
# =============================================================================


class TestMarbleWerewolfConfigs:
    """Validate werewolf YAML config files."""

    def test_werewolf_configs_directory_exists(self, marble_dir):
        """MARBLE configs directory exists."""
        configs_dir = marble_dir / "marble" / "configs"
        assert configs_dir.exists(), f"MARBLE configs directory not found at {configs_dir}. The MARBLE clone structure may have changed."

    def test_werewolf_config_files_exist(self, marble_dir):
        """At least one werewolf config YAML exists."""
        configs_dir = marble_dir / "marble" / "configs"
        werewolf_configs = list(configs_dir.glob("**/werewolf_config*.yaml"))
        assert len(werewolf_configs) > 0, f"No werewolf config files found in {configs_dir}. Expected files matching: **/werewolf_config*.yaml"

    def test_werewolf_tasks_load(self, marble_data_dir):
        """Werewolf tasks can be loaded from config files."""
        tasks = load_tasks("werewolf", data_dir=marble_data_dir)
        assert len(tasks) > 0, "No werewolf tasks loaded. Check that werewolf config YAML files exist in marble/configs/."

    def test_werewolf_tasks_have_agents(self, marble_data_dir):
        """Werewolf tasks have agent specifications with roles."""
        tasks = load_tasks("werewolf", data_dir=marble_data_dir)
        for task in tasks:
            agents = task.environment_data.get("agents", [])
            assert len(agents) > 0, f"Werewolf task {task.id} has no agents"
            for agent in agents:
                assert "agent_id" in agent
                assert "role" in agent, f"Werewolf agent {agent.get('agent_id')} missing 'role'"
