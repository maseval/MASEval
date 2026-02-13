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
    configure_model_ids,
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
            env = MACSEnvironment({"environment_data": task.environment_data}, macs_model_factory)

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
        env = MACSEnvironment({"environment_data": task.environment_data}, macs_model_factory)

        # Validate that agent config works with environment
        assert "agents" in agent_config
        assert len(agent_config["agents"]) > 0

        # Test tool assignment for each agent
        for agent_spec in agent_config["agents"]:
            agent_tools = env.get_tools_for_agent(agent_spec)  # type: ignore[arg-type]
            agent_tool_refs = agent_spec.get("tools", [])

            # Agents may legitimately have no tools (e.g., coordinator agents).
            # Only assert tools resolved for agents that reference them.
            if agent_tool_refs:
                assert len(agent_tools) > 0, (
                    f"Agent {agent_spec.get('agent_id', 'unknown')} references tools "
                    f"{agent_tool_refs} but none were resolved. "
                    f"Available environment tools: {list(env.tools.keys())}"
                )

            # Validate assigned tools exist in environment
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


# =============================================================================
# End-to-End Integration Tests with Real Data
# =============================================================================


class TestMACSBenchmarkWithRealData:
    """End-to-end tests running the full MACS benchmark with real data.

    These tests validate that the complete benchmark pipeline works correctly
    with actual downloaded MACS tasks, not synthetic fixtures. This is the
    real data equivalent of TestEndToEndPipeline in test_macs_integration.py.
    """

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_task_complete_lifecycle(self, domain, real_macs_data, macs_model_factory):
        """Test complete task lifecycle with real MACS data: setup → run → evaluate."""
        from conftest import DummyModelAdapter
        from .conftest import ConcreteMACSBenchmark
        from maseval.core.seeding import DefaultSeedGenerator

        # Load real data
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=1)
        agent_config = load_agent_config(domain, data_dir=real_macs_data)
        configure_model_ids(
            tasks,
            tool_model_id="test-model",
            user_model_id="test-model",
            evaluator_model_id="test-model",
        )

        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"
        task = tasks[0]

        # Create model with responses for this test
        # Note: Using DummyModelAdapter to avoid actual LLM calls in tests
        responses = [
            '{"text": "I can help with that.", "details": {}}',  # Tool simulation
            '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',  # User evaluation
            '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',  # System evaluation
        ]
        model = DummyModelAdapter(responses=responses)
        benchmark = ConcreteMACSBenchmark(model)

        # Setup phase - using real task data
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task(str(task.id)).for_repetition(0)

        from maseval.benchmark.macs import MACSEnvironment, MACSUser

        env = benchmark.setup_environment(agent_config, task, seed_gen)
        user = benchmark.setup_user(agent_config, env, task, seed_gen)
        agents_list, agents_dict = benchmark.setup_agents(agent_config, env, task, user, seed_gen)
        evaluators = benchmark.setup_evaluators(env, task, agents_list, user, seed_gen)

        # Verify setup with real data
        assert isinstance(env, MACSEnvironment)
        assert isinstance(user, MACSUser)
        assert len(agents_list) > 0, f"No agents created for {domain}"
        assert len(evaluators) == 2, "Should have user and system evaluators"

        # Verify environment has real tools from task
        assert len(env.tools) > 0, f"No tools in environment for {domain} task {task.id}"

        # Run phase
        final_answer = benchmark.run_agents(agents_list, task, env)
        assert final_answer is not None

        # Evaluate phase with real assertions
        traces = {
            "agents": {
                "test_agent": {
                    "messages": [
                        {"role": "user", "content": task.query},
                        {"role": "assistant", "content": final_answer},
                    ]
                }
            },
            "tools": {},
        }
        results = benchmark.evaluate(evaluators, agents_dict, final_answer, traces)

        # Verify evaluation results
        assert len(results) == 1
        assert "user_gsr" in results[0]
        assert "system_gsr" in results[0]
        assert "overall_gsr" in results[0]

    @pytest.mark.xfail(reason="DummyModelAdapter cycling responses don't align with domain-specific call sequences")
    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_task_full_benchmark_run(self, domain, real_macs_data):
        """Full end-to-end test: real task through benchmark.run()."""
        from conftest import DummyModelAdapter
        from .conftest import ConcreteMACSBenchmark

        # Load real data
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=1)
        agent_config = load_agent_config(domain, data_dir=real_macs_data)
        configure_model_ids(
            tasks,
            tool_model_id="test-model",
            user_model_id="test-model",
            evaluator_model_id="test-model",
        )

        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        # Create model with appropriate responses
        model = DummyModelAdapter(
            responses=[
                '{"text": "Response to user query", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(model)
        reports = benchmark.run(tasks, agent_data=agent_config)

        # Verify complete report structure with real data
        assert len(reports) == 1, f"Expected 1 report for {domain}, got {len(reports)}"
        report = reports[0]

        assert report["task_id"] == str(tasks[0].id)
        assert report["repeat_idx"] == 0
        assert report["status"] == "success", f"Task failed: {report.get('error', 'unknown error')}"
        assert "traces" in report
        assert "config" in report
        assert "eval" in report

        # Verify evaluation contains real assertions
        eval_data = report["eval"]
        assert "user_gsr" in eval_data or "system_gsr" in eval_data, "No evaluation results"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_multiple_tasks_from_domain(self, domain, real_macs_data):
        """Run benchmark with multiple real tasks from the same domain."""
        from conftest import DummyModelAdapter
        from .conftest import ConcreteMACSBenchmark

        # Load multiple tasks
        tasks = load_tasks(domain, data_dir=real_macs_data, limit=3)
        agent_config = load_agent_config(domain, data_dir=real_macs_data)
        configure_model_ids(
            tasks,
            tool_model_id="test-model",
            user_model_id="test-model",
            evaluator_model_id="test-model",
        )

        assert len(tasks) > 0, f"No tasks loaded for domain {domain}"

        model = DummyModelAdapter(
            responses=[
                '{"text": "Response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(model)
        reports = benchmark.run(tasks, agent_data=agent_config)

        # Verify all tasks completed
        assert len(reports) == len(tasks), f"Expected {len(tasks)} reports, got {len(reports)}"

        for i, report in enumerate(reports):
            assert report["status"] == "success", f"Task {i} failed: {report.get('error', 'unknown')}"
            assert report["task_id"] == str(tasks[i].id)
            assert "eval" in report

    def test_real_tasks_across_all_domains(self, real_macs_data):
        """Verify benchmark works with real tasks from all MACS domains."""
        from conftest import DummyModelAdapter
        from .conftest import ConcreteMACSBenchmark

        # Collect one task from each domain
        all_tasks = []
        domain_configs = {}

        for domain in VALID_DOMAINS:
            tasks = load_tasks(domain, data_dir=real_macs_data, limit=1)
            if tasks:
                configure_model_ids(
                    tasks,
                    tool_model_id="test-model",
                    user_model_id="test-model",
                    evaluator_model_id="test-model",
                )
                all_tasks.extend(tasks)
                # Store first domain's config (they should all work with same agent structure)
                if not domain_configs:
                    domain_configs = load_agent_config(domain, data_dir=real_macs_data)

        assert len(all_tasks) > 0, "No tasks loaded from any domain"
        assert domain_configs, "No agent config loaded"

        model = DummyModelAdapter(
            responses=[
                '{"text": "Response", "details": {}}',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
                '[{"assertion": "test", "answer": "TRUE", "evidence": "ok"}]',
            ]
        )

        benchmark = ConcreteMACSBenchmark(model)
        reports = benchmark.run(all_tasks, agent_data=domain_configs)

        # Verify cross-domain execution
        assert len(reports) == len(all_tasks)
        for report in reports:
            assert report["status"] == "success", f"Cross-domain task failed: {report.get('error', 'unknown')}"


# =============================================================================
# Real Data Validation Tests
# =============================================================================


class TestMACSRealDataIntegrity:
    """Validate that real MACS data has expected structure and quality.

    These tests fail loudly if real data is missing or malformed, similar to
    tau2's approach of using assertions instead of skips.
    """

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_data_has_sufficient_tasks(self, domain, real_macs_data):
        """Real MACS data should have a reasonable number of tasks per domain."""
        tasks = load_tasks(domain, data_dir=real_macs_data)

        assert len(tasks) > 0, (
            f"Real MACS {domain} domain should have tasks. This indicates download issue or upstream data problem. Found {len(tasks)} tasks."
        )

        # MACS domains should have at least a few tasks
        assert len(tasks) >= 3, (
            f"Real MACS {domain} domain should have at least 3 tasks. Found {len(tasks)} tasks. This may indicate incomplete download."
        )

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_data_has_agent_config(self, domain, real_macs_data):
        """Real MACS data should have agent configurations."""
        agent_config = load_agent_config(domain, data_dir=real_macs_data)

        assert agent_config is not None, (
            f"Real MACS {domain} domain should have agent config. This indicates download issue or data structure problem."
        )

        assert "agents" in agent_config, f"Agent config for {domain} should have 'agents' key. Found keys: {list(agent_config.keys())}"

        assert len(agent_config["agents"]) > 0, (
            f"Agent config for {domain} should have at least one agent. Found {len(agent_config['agents'])} agents."
        )

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_tasks_have_valid_tools(self, domain, real_macs_data):
        """All real tasks should have valid tool specifications."""
        tasks = load_tasks(domain, data_dir=real_macs_data)

        assert len(tasks) > 0, f"No tasks loaded for {domain}"

        for task in tasks:
            tools = task.environment_data.get("tools", [])

            assert len(tools) > 0, f"Task {task.id} in {domain} should have tools. This indicates data quality issue. Found {len(tools)} tools."

            # Validate tool structure
            for tool_group in tools:
                assert "tool_name" in tool_group, f"Tool group in task {task.id} missing 'tool_name'"
                assert "actions" in tool_group, f"Tool group in task {task.id} missing 'actions'"
                assert len(tool_group["actions"]) > 0, f"Tool group {tool_group['tool_name']} has no actions"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_tasks_have_valid_assertions(self, domain, real_macs_data):
        """All real tasks should have evaluation assertions."""
        tasks = load_tasks(domain, data_dir=real_macs_data)

        assert len(tasks) > 0, f"No tasks loaded for {domain}"

        for task in tasks:
            assertions = task.evaluation_data.get("assertions", [])

            assert len(assertions) > 0, (
                f"Task {task.id} in {domain} should have assertions. This indicates evaluation data issue. Found {len(assertions)} assertions."
            )

            # Validate assertion format (should be strings with prefixes)
            for assertion in assertions:
                assert isinstance(assertion, str), f"Assertion should be string, got {type(assertion)}"
                assert len(assertion) > 0, "Assertion should not be empty"

    @pytest.mark.parametrize("domain", VALID_DOMAINS)
    def test_real_tasks_have_scenarios(self, domain, real_macs_data):
        """Real tasks should have scenario metadata for user simulation."""
        tasks = load_tasks(domain, data_dir=real_macs_data)

        assert len(tasks) > 0, f"No tasks loaded for {domain}"

        # Check a sample of tasks (not all may have scenarios, but most should)
        tasks_with_scenarios = 0
        for task in tasks[: min(10, len(tasks))]:
            if task.metadata and "scenario" in task.metadata:
                scenario = task.metadata["scenario"]
                if scenario and len(scenario.strip()) > 0:
                    tasks_with_scenarios += 1

        # At least some tasks should have scenarios
        assert tasks_with_scenarios > 0, (
            f"Real MACS {domain} tasks should have scenario metadata. "
            f"Checked {min(10, len(tasks))} tasks, found {tasks_with_scenarios} with scenarios. "
            "This may indicate data quality issue."
        )
