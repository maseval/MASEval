"""Tests for benchmark seeding integration.

These tests verify that the seeding system integrates correctly with
the benchmark execution lifecycle, including seed_generator propagation
to setup methods and seeding config in reports.
"""

import pytest
from maseval import TaskQueue
from maseval.core.seeding import DefaultSeedGenerator, SeedGenerator


pytestmark = pytest.mark.core


# =============================================================================
# Benchmark Seeding Initialization Tests
# =============================================================================


class TestBenchmarkSeedingInitialization:
    """Tests for Benchmark seed/seed_generator initialization."""

    def test_benchmark_seed_parameter_creates_generator(self):
        """seed parameter creates a DefaultSeedGenerator."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark(seed=42)

        assert benchmark.seed_generator is not None
        assert isinstance(benchmark.seed_generator, DefaultSeedGenerator)
        assert benchmark.seed_generator.global_seed == 42

    def test_benchmark_seed_generator_parameter(self):
        """seed_generator parameter is stored directly."""
        from conftest import DummyBenchmark

        custom_gen = DefaultSeedGenerator(global_seed=123)
        benchmark = DummyBenchmark(seed_generator=custom_gen)

        assert benchmark.seed_generator is custom_gen
        assert benchmark.seed_generator.global_seed == 123

    def test_benchmark_no_seed_no_generator(self):
        """No seed or seed_generator results in None."""
        from conftest import DummyBenchmark

        benchmark = DummyBenchmark()

        assert benchmark.seed_generator is None

    def test_benchmark_seed_and_generator_raises_value_error(self):
        """Providing both seed and seed_generator raises ValueError."""
        from conftest import DummyBenchmark

        with pytest.raises(ValueError, match="Cannot provide both"):
            DummyBenchmark(seed=42, seed_generator=DefaultSeedGenerator(42))


# =============================================================================
# Seed Generator Propagation Tests
# =============================================================================


class TestSeedGeneratorPropagation:
    """Tests verifying seed_generator is passed to all setup methods."""

    def test_seed_generator_passed_to_setup_environment(self):
        """setup_environment receives seed_generator."""
        from conftest import DummyBenchmark, DummyEnvironment

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_environment(self, agent_data, task, seed_generator=None):
                captured["seed_generator"] = seed_generator
                return DummyEnvironment(task.environment_data)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured["seed_generator"] is not None
        assert isinstance(captured["seed_generator"], SeedGenerator)

    def test_seed_generator_passed_to_setup_user(self):
        """setup_user receives seed_generator."""
        from conftest import DummyBenchmark

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_user(self, agent_data, environment, task, seed_generator=None):
                captured["seed_generator"] = seed_generator
                return None

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured["seed_generator"] is not None

    def test_seed_generator_passed_to_setup_agents(self):
        """setup_agents receives seed_generator."""
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                captured["seed_generator"] = seed_generator
                agent = DummyAgent()
                adapter = DummyAgentAdapter(agent, "test_agent")
                return [adapter], {"test_agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured["seed_generator"] is not None

    def test_seed_generator_passed_to_setup_evaluators(self):
        """setup_evaluators receives seed_generator."""
        from conftest import DummyBenchmark, DummyEvaluator

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_evaluators(self, environment, task, agents, user, seed_generator=None):
                captured["seed_generator"] = seed_generator
                return [DummyEvaluator(task, environment, user)]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        assert captured["seed_generator"] is not None

    def test_seed_generator_none_when_no_seed(self):
        """seed_generator is None in setup methods when seeding disabled."""
        from conftest import DummyBenchmark

        captured = {}

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                captured["seed_generator"] = seed_generator
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark()  # No seed
        benchmark.run(tasks, agent_data={})

        assert captured["seed_generator"] is None


# =============================================================================
# Seeding Config in Reports Tests
# =============================================================================


class TestSeedingConfigInReports:
    """Tests verifying seeding config appears in benchmark reports."""

    def test_seeding_config_appears_in_report(self):
        """Seeding configuration appears in report config."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark(seed=42)

        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        config = reports[0]["config"]
        assert "seeding" in config
        assert "seed_generator" in config["seeding"]
        assert config["seeding"]["seed_generator"]["global_seed"] == 42

    def test_seeding_config_includes_seed_log(self):
        """Seeding config includes all derived seeds."""
        from conftest import DummyBenchmark, DummyAgent, DummyAgentAdapter

        class SeedUsingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    agent_gen = seed_generator.child("agents")
                    agent_gen.derive_seed("test_agent")
                agent = DummyAgent()
                adapter = DummyAgentAdapter(agent, "test_agent")
                return [adapter], {"test_agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = SeedUsingBenchmark(seed=42)

        reports = benchmark.run(tasks, agent_data={})

        config = reports[0]["config"]
        seeds = config["seeding"]["seed_generator"]["seeds"]
        assert "agents/test_agent" in seeds

    def test_no_seeding_config_when_disabled(self):
        """No seeding config when seeding is disabled."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()  # No seed

        reports = benchmark.run(tasks, agent_data={})

        config = reports[0]["config"]
        # seeding key may exist but be empty, or not exist at all
        if "seeding" in config:
            assert config["seeding"] == {} or config["seeding"].get("seed_generator") is None


# =============================================================================
# Seeding Across Repetitions Tests
# =============================================================================


class TestSeedingAcrossRepetitions:
    """Tests verifying seeding behavior across task repetitions."""

    def test_different_seeds_per_repetition(self):
        """Different repetitions get different seeds (per_repetition=True)."""
        from conftest import DummyBenchmark

        captured_seeds = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("agent", per_repetition=True)
                    captured_seeds.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42, n_task_repeats=3)
        benchmark.run(tasks, agent_data={})

        assert len(captured_seeds) == 3
        assert len(set(captured_seeds)) == 3  # All different

    def test_same_seed_across_repetitions_when_per_rep_false(self):
        """Same seed across repetitions when per_repetition=False."""
        from conftest import DummyBenchmark

        captured_seeds = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("baseline", per_repetition=False)
                    captured_seeds.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42, n_task_repeats=3)
        benchmark.run(tasks, agent_data={})

        assert len(captured_seeds) == 3
        assert len(set(captured_seeds)) == 1  # All same


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Tests verifying reproducible benchmark runs with seeding."""

    def test_same_seed_produces_same_derived_seeds(self):
        """Same global seed produces identical derived seeds across runs."""
        from conftest import DummyBenchmark

        class CapturingBenchmark(DummyBenchmark):
            def __init__(self, capture_list, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._capture_list = capture_list

            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("agent")
                    self._capture_list.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])

        # Run 1
        seeds_run1 = []
        benchmark1 = CapturingBenchmark(seeds_run1, seed=42)
        benchmark1.run(tasks, agent_data={})

        # Run 2
        seeds_run2 = []
        benchmark2 = CapturingBenchmark(seeds_run2, seed=42)
        benchmark2.run(tasks, agent_data={})

        assert seeds_run1 == seeds_run2

    def test_different_seeds_produce_different_derived_seeds(self):
        """Different global seeds produce different derived seeds."""
        from conftest import DummyBenchmark

        class CapturingBenchmark(DummyBenchmark):
            def __init__(self, capture_list, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._capture_list = capture_list

            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    seed = seed_generator.derive_seed("agent")
                    self._capture_list.append(seed)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])

        # Run with seed 42
        seeds_42 = []
        benchmark1 = CapturingBenchmark(seeds_42, seed=42)
        benchmark1.run(tasks, agent_data={})

        # Run with seed 123
        seeds_123 = []
        benchmark2 = CapturingBenchmark(seeds_123, seed=123)
        benchmark2.run(tasks, agent_data={})

        assert seeds_42 != seeds_123


# =============================================================================
# Seed Generator Scoping Tests
# =============================================================================


class TestSeedGeneratorScoping:
    """Tests verifying seed_generator is properly scoped to task and repetition."""

    def test_seed_generator_scoped_to_task(self):
        """seed_generator in setup methods is scoped to current task."""
        from conftest import DummyBenchmark

        task_ids_seen = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    # Access internal state to verify task scoping
                    task_ids_seen.append(seed_generator._task_id)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
            ]
        )
        benchmark = CapturingBenchmark(seed=42)
        benchmark.run(tasks, agent_data={})

        # Should have seen 2 different task IDs
        assert len(task_ids_seen) == 2
        assert len(set(task_ids_seen)) == 2  # All unique

    def test_seed_generator_scoped_to_repetition(self):
        """seed_generator in setup methods is scoped to current repetition."""
        from conftest import DummyBenchmark

        rep_indices_seen = []

        class CapturingBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    # Access internal state to verify rep scoping
                    rep_indices_seen.append(seed_generator._rep_index)
                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = CapturingBenchmark(seed=42, n_task_repeats=3)
        benchmark.run(tasks, agent_data={})

        # Should have seen rep indices 0, 1, 2
        assert rep_indices_seen == [0, 1, 2]

    def test_child_generators_share_seed_log(self):
        """Child generators created with child() share the same seed log."""
        from conftest import DummyBenchmark

        class SeedUsingBenchmark(DummyBenchmark):
            captured_config = None

            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                if seed_generator is not None:
                    # Create multiple child generators
                    agent_gen = seed_generator.child("agents")
                    env_gen = seed_generator.child("environment")

                    # Derive seeds from each
                    agent_gen.derive_seed("orchestrator")
                    agent_gen.derive_seed("worker")
                    env_gen.derive_seed("tool_weather")

                return super().setup_agents(agent_data, environment, task, user, seed_generator)

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = SeedUsingBenchmark(seed=42)
        reports = benchmark.run(tasks, agent_data={})

        # All derived seeds should be in the seed log
        seeds = reports[0]["config"]["seeding"]["seed_generator"]["seeds"]
        assert "agents/orchestrator" in seeds
        assert "agents/worker" in seeds
        assert "environment/tool_weather" in seeds
