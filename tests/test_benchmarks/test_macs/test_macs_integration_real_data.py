"""Integration tests for MACS benchmark using real downloaded data.

These tests validate that MACS components work correctly with actual AWS
benchmark data, not synthetic fixtures. They are marked `live` + `slow` +
`benchmark` because they download real data from GitHub.

Run with::

    pytest -m "live and slow" tests/test_benchmarks/test_macs/test_macs_integration_real_data.py -v
"""

import pytest

from maseval.benchmark.macs import MACSEnvironment, MACSEvaluator
from maseval.benchmark.macs.data_loader import (
    VALID_DOMAINS,
    download_original_data,
    download_prompt_templates,
    restructure_data,
    load_tasks,
    load_agent_config,
)

pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]


# =============================================================================
# Fixture: Download Real MACS Data
# =============================================================================


@pytest.fixture(scope="module")
def real_macs_data(tmp_path_factory):
    """Download and restructure real MACS data once for all tests in this module.

    This fixture mirrors the approach in test_data_integrity.py but makes the
    downloaded data available for integration testing.
    """
    data_dir = tmp_path_factory.mktemp("macs_integration_data")
    download_original_data(data_dir=data_dir, verbose=0)
    download_prompt_templates(data_dir=data_dir, verbose=0)
    restructure_data(data_dir=data_dir, verbose=0)
    return data_dir


# =============================================================================
# Integration Tests with Real Data
# =============================================================================


class TestMACSRealDataWithEnvironment:
    """Test that real downloaded MACS tasks work with MACSEnvironment."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_tasks_create_valid_environments(self, domain, real_macs_data, macs_model_factory):
        """Real downloaded tasks work with MACSEnvironment."""
        # Load real tasks from downloaded data
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=5)
        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        # Test that each task creates a valid environment
        for task in tasks:
            env = MACSEnvironment(task.environment_data, macs_model_factory)

            # Validate environment was created successfully
            assert env.tools is not None
            assert isinstance(env.tools, dict)
            assert len(env.tools) > 0, f"Task {task.id} has no tools"

            # Validate tools are callable
            for tool_name, tool in env.tools.items():
                assert hasattr(tool, "__call__"), f"Tool {tool_name} is not callable"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_agent_config_assigns_tools_correctly(self, domain, real_macs_data, macs_model_factory):
        """Real agent configurations work with tool assignment."""
        # Load real data
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=1)
        agent_config = load_agent_config(domain, data_dir=real_macs_data)

        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"
        task = tasks[0]

        # Create environment
        env = MACSEnvironment(task.environment_data, macs_model_factory)

        # Validate that agent config works with environment
        assert "agents" in agent_config
        assert len(agent_config["agents"]) > 0

        # Test tool assignment for each agent
        for agent_spec in agent_config["agents"]:
            agent_tools = env.get_tools_for_agent(agent_spec)  # type: ignore[arg-type]

            # Validate each agent has tools assigned
            assert len(agent_tools) > 0, (
                f"Agent {agent_spec.get('agent_id', 'unknown')} has no tools. "
                f"Agent tool refs: {agent_spec.get('tools', [])}. "
                f"Available environment tools: {list(env.tools.keys())}"
            )

            # Validate assigned tools are callable
            for tool_name in agent_tools:
                assert tool_name in env.tools, f"Tool {tool_name} not in environment"


class TestMACSRealDataWithEvaluators:
    """Test that real downloaded MACS tasks work with MACSEvaluator."""

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_tasks_create_valid_evaluators(self, domain, real_macs_data, macs_model_evaluator):
        """Real downloaded tasks work with MACSEvaluator."""
        # Load real tasks
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=5)
        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        # Test both evaluator types with each task
        for task in tasks:
            # Test user evaluator
            user_eval = MACSEvaluator(macs_model_evaluator, task, gsr_type="user")
            assert user_eval.gsr_type == "user"
            assert user_eval.task == task
            assert "{{scenario}}" in user_eval.template
            assert "{{history}}" in user_eval.template
            assert "{{assertions}}" in user_eval.template

            # Test system evaluator
            system_eval = MACSEvaluator(macs_model_evaluator, task, gsr_type="system")
            assert system_eval.gsr_type == "system"
            assert "{{invocations}}" in system_eval.template

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_assertions_are_parseable(self, domain, real_macs_data, macs_model_evaluator):
        """Real task assertions can be parsed by evaluators."""
        # Load real tasks
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=5)
        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        for task in tasks:
            # Validate assertions exist
            assertions = task.evaluation_data.get("assertions", [])
            assert len(assertions) > 0, f"Task {task.id} has no assertions"

            # Create evaluators
            user_eval = MACSEvaluator(macs_model_evaluator, task, gsr_type="user")
            system_eval = MACSEvaluator(macs_model_evaluator, task, gsr_type="system")

            # Parse assertions for both types
            user_assertions = user_eval._parse_assertions(assertions)
            system_assertions = system_eval._parse_assertions(assertions)

            # Validate parsing worked
            # Note: At least one type should have assertions (could be all user, all system, or mixed)
            total_parsed = len(user_assertions) + len(system_assertions)
            assert total_parsed > 0, f"Task {task.id} assertions failed to parse for both evaluator types. Raw assertions: {assertions}"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_task_metadata_is_valid(self, domain, real_macs_data):
        """Real tasks have valid metadata structure."""
        # Load real tasks
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=10)
        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        for task in tasks:
            # Validate required fields
            assert task.id is not None, "Task has no ID"
            assert task.query, "Task has empty query"
            assert isinstance(task.environment_data, dict), "environment_data is not a dict"
            assert isinstance(task.evaluation_data, dict), "evaluation_data is not a dict"

            # Validate environment data has tools
            assert "tools" in task.environment_data or len(task.environment_data) > 0, f"Task {task.id} environment_data has no tools"

            # Validate evaluation data has assertions
            assertions = task.evaluation_data.get("assertions", [])
            assert len(assertions) > 0, f"Task {task.id} has no assertions"
