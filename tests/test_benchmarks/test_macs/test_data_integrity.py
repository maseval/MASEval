"""Data integrity tests for MACS benchmark.

These tests download real data from the AWS GitHub repository and validate
that the downloaded and restructured files are structurally sound.  They are
marked ``live`` (network required) and ``slow`` (download + restructure takes
time) so that they are excluded from the default fast test suite.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_macs/test_data_integrity.py -v
"""

import json

import pytest

from maseval.benchmark.macs.data_loader import (
    VALID_DOMAINS,
    download_original_data,
    download_prompt_templates,
    restructure_data,
    load_tasks,
    load_agent_config,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]

# Expected minimum scenario count per domain (AWS benchmark has 30 each).
MIN_SCENARIOS_PER_DOMAIN = 25


# =============================================================================
# Fixture: download + restructure once for the whole module
# =============================================================================


@pytest.fixture(scope="module")
def macs_data_dir(tmp_path_factory):
    """Download and restructure MACS data into a temporary directory."""
    data_dir = tmp_path_factory.mktemp("macs_data")
    download_original_data(data_dir=data_dir, verbose=0)
    download_prompt_templates(data_dir=data_dir, verbose=0)
    restructure_data(data_dir=data_dir, verbose=0)
    return data_dir


# =============================================================================
# Original Data Integrity
# =============================================================================


class TestMACSOriginalData:
    """Validate that raw downloaded files are present and valid JSON."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_agents_json_exists(self, macs_data_dir, domain):
        """agents.json exists and contains agents."""
        path = macs_data_dir / "original" / domain / "agents.json"
        assert path.exists(), f"Missing {path}"
        data = json.loads(path.read_text())
        # AWS format: dict with "agents" key or a list
        if isinstance(data, dict):
            assert "agents" in data, f"agents.json for {domain} missing 'agents' key"
            assert len(data["agents"]) > 0
        else:
            assert len(data) > 0

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_scenarios_json_exists(self, macs_data_dir, domain):
        """scenarios.json exists and contains scenarios."""
        path = macs_data_dir / "original" / domain / "scenarios.json"
        assert path.exists(), f"Missing {path}"
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            scenarios = data.get("scenarios", data)
        else:
            scenarios = data
        assert len(scenarios) >= MIN_SCENARIOS_PER_DOMAIN, f"{domain}: expected >= {MIN_SCENARIOS_PER_DOMAIN} scenarios, got {len(scenarios)}"


# =============================================================================
# Restructured Data Integrity
# =============================================================================


class TestMACSRestructuredData:
    """Validate that restructured data is correct."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_restructured_tasks_json(self, macs_data_dir, domain):
        """restructured tasks.json exists and has tasks with expected shape."""
        path = macs_data_dir / "restructured" / domain / "tasks.json"
        assert path.exists(), f"Missing {path}"

        tasks = json.loads(path.read_text())
        assert isinstance(tasks, list)
        assert len(tasks) >= MIN_SCENARIOS_PER_DOMAIN

        for task in tasks:
            assert "id" in task, "Task missing 'id'"
            assert "query" in task, "Task missing 'query'"
            assert task["query"], f"Task {task['id']} has empty query"
            assert "environment_data" in task
            assert "evaluation_data" in task

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_restructured_agents_json(self, macs_data_dir, domain):
        """restructured agents.json exists and has agent hierarchy."""
        path = macs_data_dir / "restructured" / domain / "agents.json"
        assert path.exists(), f"Missing {path}"

        data = json.loads(path.read_text())
        assert "agents" in data
        assert len(data["agents"]) > 0, f"{domain} has no agents"
        assert "primary_agent_id" in data, f"{domain} missing primary_agent_id"

        # Each agent should have tool names (strings), not full tool dicts
        for agent in data["agents"]:
            assert "agent_id" in agent
            assert "tools" in agent
            for tool_ref in agent["tools"]:
                assert isinstance(tool_ref, str), f"Agent {agent['agent_id']} tool ref should be a name string, got {type(tool_ref).__name__}"


# =============================================================================
# Prompt Templates
# =============================================================================


class TestMACSPromptTemplates:
    """Validate that prompt templates were downloaded and parsed."""

    @pytest.mark.parametrize("name", ["user.txt", "system.txt", "issues.txt"])
    def test_template_exists_and_nonempty(self, macs_data_dir, name):
        """Prompt template file exists and is non-empty."""
        # Templates are stored alongside the data_loader module, not in data/
        # but download_prompt_templates() puts them in data_dir/../prompt_templates/
        path = macs_data_dir.parent / "prompt_templates" / name
        if not path.exists():
            # Fallback: templates may be in the module's own prompt_templates dir
            from maseval.benchmark.macs.data_loader import DEFAULT_DATA_DIR

            path = DEFAULT_DATA_DIR.parent / "prompt_templates" / name
        assert path.exists(), f"Missing prompt template: {name}"
        assert path.stat().st_size > 0, f"Prompt template {name} is empty"


# =============================================================================
# Load Functions with Real Data
# =============================================================================


class TestMACSLoadFunctions:
    """Validate load_tasks and load_agent_config with real restructured data."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_load_tasks_returns_task_objects(self, macs_data_dir, domain):
        """load_tasks returns Task objects with correct fields."""
        tasks = load_tasks(domain, data_dir=macs_data_dir)

        assert len(tasks) >= MIN_SCENARIOS_PER_DOMAIN
        for task in tasks:
            assert task.id is not None
            assert task.query
            assert isinstance(task.environment_data, dict)
            assert isinstance(task.evaluation_data, dict)

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_load_agent_config_returns_hierarchy(self, macs_data_dir, domain):
        """load_agent_config returns a valid agent hierarchy."""
        config = load_agent_config(domain, data_dir=macs_data_dir)

        assert "agents" in config
        assert len(config["agents"]) > 0
        assert "primary_agent_id" in config

        # Primary agent should exist in the agents list
        agent_ids = {a["agent_id"] for a in config["agents"]}
        assert config["primary_agent_id"] in agent_ids, f"primary_agent_id '{config['primary_agent_id']}' not in agents: {agent_ids}"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_tasks_have_assertions(self, macs_data_dir, domain):
        """Every task has at least one evaluation assertion."""
        tasks = load_tasks(domain, data_dir=macs_data_dir)

        for task in tasks:
            assertions = task.evaluation_data.get("assertions", [])
            assert len(assertions) > 0, f"Task {task.id} in {domain} has no assertions"
