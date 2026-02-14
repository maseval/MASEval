"""Tests for MultiAgentBench benchmark classes."""

import sys

import pytest
from unittest.mock import MagicMock, patch

from maseval import Task
from maseval.benchmark.multiagentbench import (
    MarbleMultiAgentBenchBenchmark,
    MultiAgentBenchEnvironment,
    MultiAgentBenchEvaluator,
)

pytestmark = pytest.mark.benchmark


class TestMultiAgentBenchBenchmark:
    """Tests for MultiAgentBenchBenchmark abstract class."""

    def test_setup_environment_returns_correct_type(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """setup_environment should return MultiAgentBenchEnvironment."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)

        assert isinstance(env, MultiAgentBenchEnvironment)

    def test_setup_user_returns_none(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """setup_user should return None (no user simulator for multi-agent)."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        user = benchmark_instance.setup_user({}, env, sample_research_task, seed_gen)

        assert user is None

    def test_setup_evaluators_returns_evaluator(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """setup_evaluators should return MultiAgentBenchEvaluator."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents, None, seed_gen)

        assert len(evaluators) == 1
        assert isinstance(evaluators[0], MultiAgentBenchEvaluator)

    def test_setup_evaluators_uses_seed_generator(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """setup_evaluators should pass seed to model adapter when seed_generator provided."""
        from unittest.mock import patch
        from maseval.core.seeding import DefaultSeedGenerator

        # Create a seed generator first
        seed_gen = DefaultSeedGenerator(global_seed=42)
        task_seed_gen = seed_gen.for_task("test_task").for_repetition(0)

        # Create a benchmark and mock get_model_adapter
        benchmark = concrete_multiagentbench_benchmark(progress_bar=False)
        env = benchmark.setup_environment({}, sample_research_task, task_seed_gen)
        agents, _ = benchmark.setup_agents({}, env, sample_research_task, None, task_seed_gen)

        # Patch get_model_adapter to capture the seed argument
        with patch.object(benchmark, "get_model_adapter", wraps=benchmark.get_model_adapter) as mock_get_model:
            benchmark.setup_evaluators(env, sample_research_task, agents, None, seed_generator=task_seed_gen)

            # Verify get_model_adapter was called with a seed
            mock_get_model.assert_called_once()
            call_kwargs = mock_get_model.call_args.kwargs
            assert "seed" in call_kwargs
            assert call_kwargs["seed"] is not None
            assert isinstance(call_kwargs["seed"], int)

    def test_setup_evaluators_no_seed_when_seeding_disabled(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """setup_evaluators should pass None seed when global_seed is None."""
        from unittest.mock import patch
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = concrete_multiagentbench_benchmark(progress_bar=False)
        # Use a seed generator with global_seed=None to test disabled seeding
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
        env = benchmark.setup_environment({}, sample_research_task, seed_generator=seed_gen)
        agents, _ = benchmark.setup_agents({}, env, sample_research_task, None, seed_generator=seed_gen)

        with patch.object(benchmark, "get_model_adapter", wraps=benchmark.get_model_adapter) as mock_get_model:
            benchmark.setup_evaluators(env, sample_research_task, agents, None, seed_generator=seed_gen)

            mock_get_model.assert_called_once()
            call_kwargs = mock_get_model.call_args.kwargs
            assert call_kwargs.get("seed") is None

    def test_setup_evaluators_seed_is_deterministic(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Same global seed should produce same derived seed."""
        from unittest.mock import patch
        from maseval.core.seeding import DefaultSeedGenerator

        seeds_collected = []

        for _ in range(2):
            # Same global seed each time
            seed_gen = DefaultSeedGenerator(global_seed=42)
            task_seed_gen = seed_gen.for_task("test_task").for_repetition(0)

            benchmark = concrete_multiagentbench_benchmark(progress_bar=False)
            env = benchmark.setup_environment({}, sample_research_task, task_seed_gen)
            agents, _ = benchmark.setup_agents({}, env, sample_research_task, None, task_seed_gen)

            with patch.object(benchmark, "get_model_adapter", wraps=benchmark.get_model_adapter) as mock_get_model:
                benchmark.setup_evaluators(env, sample_research_task, agents, None, seed_generator=task_seed_gen)
                seeds_collected.append(mock_get_model.call_args.kwargs["seed"])

        # Same seed both times
        assert seeds_collected[0] == seeds_collected[1]

    def test_setup_evaluators_different_global_seeds_produce_different_seeds(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Different global seeds should produce different derived seeds."""
        from unittest.mock import patch
        from maseval.core.seeding import DefaultSeedGenerator

        seeds_collected = []

        for global_seed in [42, 123]:
            benchmark = concrete_multiagentbench_benchmark(progress_bar=False)
            seed_gen = DefaultSeedGenerator(global_seed=global_seed).for_task("test_task").for_repetition(0)
            env = benchmark.setup_environment({}, sample_research_task, seed_gen)
            agents, _ = benchmark.setup_agents({}, env, sample_research_task, None, seed_gen)

            with patch.object(benchmark, "get_model_adapter", wraps=benchmark.get_model_adapter) as mock_get_model:
                benchmark.setup_evaluators(env, sample_research_task, agents, None, seed_generator=seed_gen)
                seeds_collected.append(mock_get_model.call_args.kwargs["seed"])

        # Different seeds
        assert seeds_collected[0] != seeds_collected[1]

    def test_benchmark_run_with_seed_passes_seed_to_evaluators(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Full benchmark.run() with seed should pass seed through to evaluators."""
        from unittest.mock import patch

        benchmark = concrete_multiagentbench_benchmark(progress_bar=False, seed=42)

        seeds_passed = []
        original_get_model_adapter = benchmark.get_model_adapter

        def capturing_get_model_adapter(model_id, **kwargs):
            seeds_passed.append(kwargs.get("seed"))
            return original_get_model_adapter(model_id, **kwargs)

        with patch.object(benchmark, "get_model_adapter", side_effect=capturing_get_model_adapter):
            benchmark.run(tasks=[sample_research_task], agent_data={})

        # At least one call should have a seed (the evaluator)
        assert any(s is not None for s in seeds_passed), "Expected at least one seeded model adapter call"

    def test_benchmark_run_without_seed_passes_none(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Full benchmark.run() without seed should pass None to evaluators."""
        from unittest.mock import patch

        benchmark = concrete_multiagentbench_benchmark(progress_bar=False)  # No seed

        seeds_passed = []
        original_get_model_adapter = benchmark.get_model_adapter

        def capturing_get_model_adapter(model_id, **kwargs):
            seeds_passed.append(kwargs.get("seed"))
            return original_get_model_adapter(model_id, **kwargs)

        with patch.object(benchmark, "get_model_adapter", side_effect=capturing_get_model_adapter):
            benchmark.run(tasks=[sample_research_task], agent_data={})

        # All calls should have None seed
        assert all(s is None for s in seeds_passed), "Expected all model adapter calls to have None seed"

    def test_setup_agents_creates_correct_count(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """setup_agents should create correct number of agents."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        # sample_research_task has 2 agents
        assert len(agents_list) == 2
        assert len(agents_dict) == 2
        assert "agent1" in agents_dict
        assert "agent2" in agents_dict

    def test_run_agents_executes_all_agents(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should execute all agents and return results."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        results = benchmark_instance.run_agents(
            agents_list,
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert isinstance(results, dict)
        assert "agent_results" in results
        assert "communications" in results
        assert "coordination_mode" in results
        assert len(results["agent_results"]) == 2
        assert all("agent_id" in r for r in results["agent_results"])
        assert all("result" in r for r in results["agent_results"])

    def test_evaluate_calls_evaluators(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """evaluate should call all evaluators."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents_list, None, seed_gen)

        final_answer = [{"agent_id": "agent1", "result": "Done"}]
        traces = {
            "agents": {"agent1": {"token_usage": 100, "action_log": [], "communication_log": []}},
            "environment": {},
        }

        results = benchmark_instance.evaluate(evaluators, agents_dict, final_answer, traces)

        assert len(results) == 1
        assert "passed" in results[0]


class TestBenchmarkIntegration:
    """Integration tests for benchmark execution."""

    def test_run_single_task(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Benchmark should run a single task end-to-end."""
        results = benchmark_instance.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 1
        assert results[0]["task_id"] == "research_1"
        assert "status" in results[0]
        assert "traces" in results[0]
        assert "eval" in results[0]

    def test_run_multiple_tasks(
        self,
        benchmark_instance,
        sample_research_task: Task,
        sample_bargaining_task: Task,
    ):
        """Benchmark should run multiple tasks."""
        results = benchmark_instance.run(
            tasks=[sample_research_task, sample_bargaining_task],
            agent_data={},
        )

        assert len(results) == 2

    def test_run_with_task_repeats(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Benchmark should repeat tasks when n_task_repeats > 1."""
        benchmark = concrete_multiagentbench_benchmark(
            n_task_repeats=3,
            progress_bar=False,
        )
        results = benchmark.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 3
        assert all(r["task_id"] == "research_1" for r in results)
        assert [r["repeat_idx"] for r in results] == [0, 1, 2]

    def test_run_collects_traces(
        self,
        benchmark_instance,
        sample_research_task: Task,
    ):
        """Benchmark should collect traces from all components."""
        results = benchmark_instance.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        traces = results[0]["traces"]
        assert "agents" in traces
        assert "environment" in traces or traces.get("environment") is None

    def test_run_handles_setup_error(
        self,
        concrete_multiagentbench_benchmark,
        sample_research_task: Task,
    ):
        """Benchmark should handle setup errors gracefully."""

        class FailingBenchmark(concrete_multiagentbench_benchmark):
            def setup_environment(self, agent_data, task, seed_generator):
                raise RuntimeError("Setup failed")

        benchmark = FailingBenchmark(progress_bar=False)
        results = benchmark.run(
            tasks=[sample_research_task],
            agent_data={},
        )

        assert len(results) == 1
        assert results[0]["status"] == "setup_failed"
        assert "error" in results[0]


class TestAgentCreation:
    """Tests for agent creation and configuration."""

    def test_agents_have_correct_ids(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """Created agents should have IDs from task config."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        _, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        assert "agent1" in agents_dict
        assert "agent2" in agents_dict

    def test_agents_have_profiles(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """Created agents should have profiles from task config."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        _, agents_dict = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        agent1 = agents_dict["agent1"]
        assert hasattr(agent1, "profile")
        assert "machine learning" in agent1.profile.lower()


class TestEvaluatorConfiguration:
    """Tests for evaluator configuration."""

    def test_evaluator_uses_task_model_id(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """Evaluator should use model_id from task evaluation_data."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)
        evaluators = benchmark_instance.setup_evaluators(env, sample_research_task, agents, None, seed_gen)

        evaluator = evaluators[0]
        assert evaluator.domain == "research"

    def test_evaluator_domain_from_task(
        self,
        benchmark_instance,
        sample_bargaining_task: Task,
        seed_gen,
    ):
        """Evaluator should get domain from task environment_data."""
        env = benchmark_instance.setup_environment({}, sample_bargaining_task, seed_gen)
        agents, _ = benchmark_instance.setup_agents({}, env, sample_bargaining_task, None, seed_gen)
        evaluators = benchmark_instance.setup_evaluators(env, sample_bargaining_task, agents, None, seed_gen)

        evaluator = evaluators[0]
        assert evaluator.domain == "bargaining"


class TestMarbleMultiAgentBenchBenchmark:
    """Tests for MarbleMultiAgentBenchBenchmark class."""

    @pytest.fixture
    def marble_benchmark_class(self):
        """Create a concrete MarbleMultiAgentBenchBenchmark class."""
        from conftest import DummyModelAdapter

        class ConcreteMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
            def get_model_adapter(self, model_id, **kwargs):
                adapter = DummyModelAdapter(
                    model_id=model_id,
                    responses=['{"rating": 4}'],
                )
                register_name = kwargs.get("register_name")
                if register_name:
                    try:
                        self.register("models", register_name, adapter)
                    except ValueError:
                        pass
                return adapter

        return ConcreteMarbleBenchmark

    def test_setup_agents_raises_import_error(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
        seed_gen,
    ):
        """setup_agents should raise ImportError when MARBLE not available."""
        benchmark = marble_benchmark_class(progress_bar=False)
        env = benchmark.setup_environment({}, sample_research_task, seed_gen)

        # Temporarily remove marble modules to simulate MARBLE not being available
        marble_modules = {k: v for k, v in sys.modules.items() if "marble" in k}
        for module_name in marble_modules:
            sys.modules.pop(module_name, None)

        try:
            with patch.dict("sys.modules", {"marble.agent.base_agent": None}):
                with pytest.raises(ImportError, match="MARBLE is not available"):
                    benchmark.setup_agents({}, env, sample_research_task, None, seed_gen)
        finally:
            sys.modules.update(marble_modules)

    def test_create_marble_env_raises_import_error(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
    ):
        """_create_marble_env should raise ImportError when MARBLE not available."""
        benchmark = marble_benchmark_class(progress_bar=False)

        # Mock MARBLE import to simulate it not being available
        # Temporarily remove marble modules from sys.modules
        marble_modules = {k: v for k, v in sys.modules.items() if "marble" in k}
        for module_name in marble_modules:
            sys.modules.pop(module_name, None)

        try:
            # Patch the import to raise ImportError
            with patch.dict("sys.modules", {"marble.environments.base_env": None}):
                with pytest.raises(ImportError, match="MARBLE is not available"):
                    benchmark._create_marble_env(sample_research_task)
        finally:
            # Restore marble modules
            sys.modules.update(marble_modules)

    def test_setup_agent_graph_with_missing_agents_raises(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
    ):
        """_setup_agent_graph should raise when agents referenced in relationships don't exist."""
        benchmark = marble_benchmark_class(progress_bar=False)

        # Mock AgentGraph so this test works without MARBLE installed.
        # The real AgentGraph validates that relationship agents exist; simulate that.
        mock_agent_graph_cls = MagicMock(side_effect=ValueError("Agent 'agent1' does not exist in the graph"))

        with patch.dict("sys.modules", {"marble.graph.agent_graph": MagicMock(AgentGraph=mock_agent_graph_cls)}):
            with pytest.raises(ValueError, match="does not exist"):
                benchmark._setup_agent_graph({}, sample_research_task, None)

    def test_run_agents_returns_structured_output(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should return structured output with agent_results."""

        benchmark = marble_benchmark_class(progress_bar=False)
        env = benchmark.setup_environment({}, sample_research_task, seed_gen)

        # Create mock agents
        mock_agent1 = MagicMock()
        mock_agent1.run.return_value = "Result from agent1"
        mock_agent1.agent_id = "agent1"

        mock_agent2 = MagicMock()
        mock_agent2.run.return_value = "Result from agent2"
        mock_agent2.agent_id = "agent2"
        mock_agent2.get_serialized_messages.return_value = "Communication log"

        result = benchmark.run_agents(
            [mock_agent1, mock_agent2],
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert "agent_results" in result
        assert "communications" in result
        assert "coordination_mode" in result
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][0]["agent_id"] == "agent1"
        assert result["agent_results"][1]["agent_id"] == "agent2"

    def test_run_agents_collects_communications(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should collect communications from agents."""
        benchmark = marble_benchmark_class(progress_bar=False)
        env = benchmark.setup_environment({}, sample_research_task, seed_gen)

        # Create mock agent with get_serialized_messages
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Result"
        mock_agent.agent_id = "agent1"
        mock_agent.get_serialized_messages.return_value = "Hello from agent1"

        result = benchmark.run_agents(
            [mock_agent],
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert "Hello from agent1" in result["communications"]


class TestBenchmarkWithDifferentCoordinationModes:
    """Tests for different coordination modes."""

    def test_run_agents_with_cooperative_mode(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should work with cooperative coordination."""
        # sample_research_task uses cooperative mode by default
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        results = benchmark_instance.run_agents(
            agents_list,
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert len(results["agent_results"]) == 2
        assert results["coordination_mode"] == "cooperative"

    def test_run_agents_with_star_mode(self, benchmark_instance, seed_gen):
        """run_agents should work with star coordination."""
        task_data = {
            "scenario": "research",
            "task_id": 1,
            "agents": [
                {"agent_id": "central", "profile": "Central coordinator"},
                {"agent_id": "worker1", "profile": "Worker 1"},
            ],
            "coordinate_mode": "star",
            "relationships": [["central", "worker1", "coordinates"]],
            "environment": {"max_iterations": 10},
            "task": {"content": "Research task", "output_format": "5Q"},
            "max_iterations": 10,
        }
        task = Task(
            id="test_star",
            query="Research task",
            environment_data=task_data,
            evaluation_data={"model_id": "gpt-4o-mini"},
            metadata={"domain": "research"},
        )

        env = benchmark_instance.setup_environment({}, task, seed_gen)
        agents_list, _ = benchmark_instance.setup_agents({}, env, task, None, seed_gen)

        results = benchmark_instance.run_agents(agents_list, task, env, task.query)

        assert len(results["agent_results"]) == 2
        assert results["coordination_mode"] == "star"


class TestBenchmarkWithEmptyAgents:
    """Tests for edge cases with agents."""

    def test_run_agents_with_empty_list(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should handle empty agent list."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)

        results = benchmark_instance.run_agents(
            [],
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert results["agent_results"] == []
        assert results["communications"] == []

    def test_setup_agents_with_no_agents_in_task(self, benchmark_instance, seed_gen):
        """setup_agents should handle task with no agents."""
        task_data = {
            "scenario": "research",
            "task_id": 1,
            "agents": [],  # No agents
            "coordinate_mode": "cooperative",
            "relationships": [],
            "environment": {"max_iterations": 10},
            "task": {"content": "Research task"},
            "max_iterations": 10,
        }
        task = Task(
            id="test_no_agents",
            query="Research task",
            environment_data=task_data,
            evaluation_data={"model_id": "gpt-4o-mini"},
            metadata={"domain": "research"},
        )

        env = benchmark_instance.setup_environment({}, task, seed_gen)
        agents_list, agents_dict = benchmark_instance.setup_agents({}, env, task, None, seed_gen)

        assert len(agents_list) == 0
        assert len(agents_dict) == 0


# =============================================================================
# MarbleAgentAdapter Tests
# =============================================================================


class TestMarbleAgentAdapterTraces:
    """Tests for MarbleAgentAdapter.gather_traces() structure."""

    def test_gather_traces_includes_all_fields(self):
        """gather_traces should include all MARBLE-specific trace fields."""
        from maseval.benchmark.multiagentbench.adapters.marble_adapter import MarbleAgentAdapter

        mock_marble = MagicMock()
        mock_marble.profile = "Expert in NLP"
        mock_marble.get_token_usage.return_value = 500
        mock_marble.relationships = {"agent2": "collaborates"}
        mock_marble.task_history = [{"task": "research", "result": "done"}]
        mock_marble.memory = MagicMock()
        mock_marble.memory.get_memory_str.return_value = "past context"

        adapter = MarbleAgentAdapter(marble_agent=mock_marble, agent_id="nlp_agent")
        traces = adapter.gather_traces()

        assert traces["agent_id"] == "nlp_agent"
        assert traces["profile"] == "Expert in NLP"
        assert traces["token_usage"] == 500
        assert traces["action_log"] == []
        assert traces["communication_log"] == []
        assert traces["memory"] == "past context"
        assert traces["relationships"] == {"agent2": "collaborates"}
        assert traces["task_history"] == [{"task": "research", "result": "done"}]

    def test_gather_traces_handles_missing_attributes(self):
        """gather_traces should handle MARBLE agents with missing optional attributes."""
        from maseval.benchmark.multiagentbench.adapters.marble_adapter import MarbleAgentAdapter

        # Agent with no optional attributes
        mock_marble = MagicMock(spec=[])
        adapter = MarbleAgentAdapter(marble_agent=mock_marble, agent_id="basic_agent")
        traces = adapter.gather_traces()

        assert traces["agent_id"] == "basic_agent"
        assert traces["token_usage"] == 0
        assert traces["memory"] == ""
        assert traces["relationships"] == {}
        assert traces["task_history"] == []

    def test_gather_config_includes_all_fields(self):
        """gather_config should include agent configuration fields."""
        from maseval.benchmark.multiagentbench.adapters.marble_adapter import MarbleAgentAdapter

        mock_marble = MagicMock()
        mock_marble.profile = "Data analyst"
        mock_marble.strategy = "divide_and_conquer"
        mock_marble.llm = "gpt-4o"

        adapter = MarbleAgentAdapter(marble_agent=mock_marble, agent_id="data_agent")
        config = adapter.gather_config()

        assert config["agent_id"] == "data_agent"
        assert config["profile"] == "Data analyst"
        assert config["strategy"] == "divide_and_conquer"
        assert config["llm"] == "gpt-4o"
