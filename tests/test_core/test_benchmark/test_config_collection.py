"""Test configuration collection from all components.

These tests verify that the configuration collection system gathers
reproducibility data from all registered components correctly.
"""

import pytest
from maseval import TaskQueue


@pytest.mark.core
class TestConfigCollection:
    """Tests for comprehensive configuration collection."""

    def test_config_collected_from_all_components(self):
        """Test that configs are collected from all registered components."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Verify config has the expected structure
        assert "metadata" in config
        assert "agents" in config
        assert "environment" in config
        assert "benchmark" in config

        # Verify metadata
        assert config["metadata"]["total_components"] > 0
        assert "timestamp" in config["metadata"]

        # Verify agent configs
        assert "test_agent" in config["agents"]
        agent_config = config["agents"]["test_agent"]
        assert "type" in agent_config
        assert "gathered_at" in agent_config

        # Verify environment config
        assert config["environment"] is not None
        env_config = config["environment"]
        assert "type" in env_config
        assert "gathered_at" in env_config

    def test_config_includes_benchmark_level_info(self):
        """Test that benchmark-level configuration is included."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Verify benchmark-level config exists
        assert "benchmark" in config
        bench_config = config["benchmark"]

        # Should include system info, python version, etc.
        assert "python" in bench_config
        assert "system" in bench_config
        assert "timestamp" in bench_config

    def test_config_includes_system_info(self):
        """Test that system information is captured."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        system_info = config["benchmark"]["system"]
        assert "system" in system_info  # OS name (Darwin, Linux, Windows)
        assert "hostname" in system_info

    def test_config_includes_git_info(self):
        """Test that git information is captured when available."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Git info may not be available in all environments
        assert "git" in config["benchmark"]
        git_info = config["benchmark"]["git"]

        # Either has git info or has an error message
        assert ("branch" in git_info and "commit_hash" in git_info) or ("error" in git_info)

    def test_config_includes_package_versions(self):
        """Test that installed package versions are captured."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Should capture package information
        if "packages" in config["benchmark"]:
            packages = config["benchmark"]["packages"]
            # At minimum, should have pytest since we're running tests
            assert isinstance(packages, dict)

    def test_config_structure_matches_spec(self):
        """Test that config structure matches expected specification."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Top-level keys
        expected_keys = [
            "metadata",
            "agents",
            "models",
            "tools",
            "simulators",
            "callbacks",
            "environment",
            "user",
            "other",
            "benchmark",
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

        # Agents should be a dict
        assert isinstance(config["agents"], dict)

        # Environment should be a dict (not nested)
        assert isinstance(config["environment"], dict)

        # User may be None
        assert config["user"] is None or isinstance(config["user"], dict)

    def test_config_handles_component_errors_gracefully(self):
        """Test that config collection continues even if a component fails."""
        from conftest import DummyBenchmark
        from maseval import AgentAdapter

        class FailingConfigAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return "success"

            def gather_config(self):
                # Simulate an error during config gathering
                raise RuntimeError("Config gathering failed")

        class TestBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                from conftest import DummyAgent

                agent = DummyAgent()
                agent_adapter = FailingConfigAdapter(agent, "failing_agent")
                return [agent_adapter], {"failing_agent": agent_adapter}  # type: ignore[return-value]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        # Should complete without raising, with error info in config
        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Verify config collection handled the error
        assert "agents" in config
        agent_config = config["agents"]["failing_agent"]
        assert "error" in agent_config
        assert "Config gathering failed" in agent_config["error"]

    def test_config_json_serializable(self):
        """Test that all configs are JSON serializable."""
        import json
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {"key": "value"}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Should be able to serialize to JSON
        try:
            json_str = json.dumps(config)
            assert len(json_str) > 0

            # Should be able to deserialize back
            config_restored = json.loads(json_str)
            assert config_restored["metadata"]["total_components"] == config["metadata"]["total_components"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Config is not JSON serializable: {e}")

    def test_config_contains_timestamps(self):
        """Test that all config components include timestamps."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Check metadata timestamp
        assert "timestamp" in config["metadata"]

        # Check agent timestamp
        agent_config = config["agents"]["test_agent"]
        assert "gathered_at" in agent_config

        # Check environment timestamp
        env_config = config["environment"]
        assert "gathered_at" in env_config

        # Check benchmark timestamp
        assert "timestamp" in config["benchmark"]

    def test_config_includes_component_types(self):
        """Test that all configs include component type information."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        config = reports[0]["config"]

        # Check agent type
        agent_config = config["agents"]["test_agent"]
        assert "type" in agent_config
        assert agent_config["type"] == "DummyAgentAdapter"

        # Check environment type
        env_config = config["environment"]
        assert "type" in env_config
        assert env_config["type"] == "DummyEnvironment"

    def test_config_different_per_repetition(self):
        """Test that each repetition has its own config snapshot."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark(n_task_repeats=3)

        reports = benchmark.run(tasks, agent_data={"model": "test"})

        # Should have 3 reports
        assert len(reports) == 3

        # Each should have its own config
        for i, report in enumerate(reports):
            assert "config" in report
            assert report["repeat_idx"] == i
            # Config should be present and complete
            assert "benchmark" in report["config"]
            assert "agents" in report["config"]
