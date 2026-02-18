"""Tests for MultiAgentBench benchmark classes."""

import sys
from typing import Any

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

    def test_run_agents_skips_empty_communications(
        self,
        marble_benchmark_class,
        sample_research_task: Task,
        seed_gen,
    ):
        """run_agents should not include empty communications."""
        benchmark = marble_benchmark_class(progress_bar=False)
        env = benchmark.setup_environment({}, sample_research_task, seed_gen)

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Result"
        mock_agent.agent_id = "agent1"
        mock_agent.get_serialized_messages.return_value = ""  # Empty communication

        result = benchmark.run_agents(
            [mock_agent],
            sample_research_task,
            env,
            sample_research_task.query,
        )

        assert result["communications"] == []


class TestExecutionLoop:
    """Tests for the execution_loop overrides."""

    def test_base_execution_loop_calls_run_agents_once(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """MultiAgentBenchBenchmark.execution_loop should call run_agents exactly once."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        with patch.object(benchmark_instance, "run_agents", wraps=benchmark_instance.run_agents) as mock_run:
            result = benchmark_instance.execution_loop(agents_list, sample_research_task, env, None)
            mock_run.assert_called_once_with(agents_list, sample_research_task, env, sample_research_task.query)

        assert isinstance(result, dict)
        assert "agent_results" in result

    def test_base_execution_loop_uses_task_query(
        self,
        benchmark_instance,
        sample_research_task: Task,
        seed_gen,
    ):
        """execution_loop should pass task.query to run_agents."""
        env = benchmark_instance.setup_environment({}, sample_research_task, seed_gen)
        agents_list, _ = benchmark_instance.setup_agents({}, env, sample_research_task, None, seed_gen)

        benchmark_instance.execution_loop(agents_list, sample_research_task, env, None)

        # Verify agents received the task query
        for agent in agents_list:
            assert len(agent.run_calls) == 1
            assert agent.run_calls[0] == sample_research_task.query


# =============================================================================
# Helpers for coordination mode tests
# =============================================================================


def _make_mock_adapter(agent_id: str, run_return: str = "result", communication: str = "") -> MagicMock:
    """Create a mock MarbleAgentAdapter with standard attributes.

    Args:
        agent_id: Agent identifier
        run_return: Default return value for adapter.run()
        communication: If non-empty, adapter.run() will append to _communication_log
    """
    adapter = MagicMock()
    adapter.agent_id = agent_id
    adapter._communication_log = []
    adapter.marble_agent = MagicMock()
    adapter.marble_agent.plan_task.return_value = f"planned task for {agent_id}"

    if communication:
        # Simulate MARBLE adapter behavior: run() appends to _communication_log
        def _run_with_comm(query):
            adapter._communication_log.append({"communication": communication})
            return run_return

        adapter.run.side_effect = _run_with_comm
    else:
        adapter.run.return_value = run_return

    return adapter


def _make_mock_benchmark(
    coordinate_mode: str = "graph",
    domain: str = "research",
    max_iterations: int = 10,
    planner_decide_sequence: Any = None,
    agents_dict: Any = None,
) -> MagicMock:
    """Create a mock MarbleMultiAgentBenchBenchmark with standard planner wiring."""
    from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

    mock_planner = MagicMock()
    mock_planner.summarize_output.return_value = MagicMock(content="summary")
    if planner_decide_sequence is not None:
        mock_planner.decide_next_step.side_effect = planner_decide_sequence
    else:
        mock_planner.decide_next_step.return_value = False

    benchmark = MagicMock(spec=MarbleMultiAgentBenchBenchmark)
    benchmark._marble_planner = mock_planner
    benchmark._task_content = "Do the task"
    benchmark._output_format = "output format"
    benchmark._coordinate_mode = coordinate_mode
    benchmark._domain = domain
    benchmark._marble_max_iterations = max_iterations
    benchmark._summarize_results_marble = MarbleMultiAgentBenchBenchmark._summarize_results_marble

    # Mock MARBLE evaluator with proper metrics structure matching evaluator.py:31-40
    mock_evaluator = MagicMock()
    mock_evaluator.metrics = {
        "task_completion": [],
        "token_consumption": [],
        "planning_score": [],
        "communication_score": [],
        "task_evaluation": {},
        "total_milestones": 0,
        "agent_kpis": {},
        "code_quality": {},
    }
    benchmark._marble_evaluator = mock_evaluator
    benchmark._task_data = {}

    if agents_dict is not None:
        benchmark._agents_dict = agents_dict
    return benchmark


class TestMarbleExecutionLoopDispatch:
    """Tests for MarbleMultiAgentBenchBenchmark.execution_loop mode dispatch."""

    def test_dispatches_to_correct_mode(self):
        """execution_loop should dispatch to the handler matching coordinate_mode."""
        from maseval.benchmark.multiagentbench import MarbleMultiAgentBenchBenchmark

        for mode in ("graph", "star", "chain", "tree"):
            benchmark = MagicMock(spec=MarbleMultiAgentBenchBenchmark)
            benchmark._coordinate_mode = mode
            handler = MagicMock(return_value={"agent_results": []})
            setattr(benchmark, f"_{mode}_coordinate", handler)

            MarbleMultiAgentBenchBenchmark.execution_loop(benchmark, [], MagicMock(), MagicMock(), None)
            handler.assert_called_once()

    def test_raises_on_unknown_mode(self):
        """execution_loop should raise ValueError for unsupported modes."""
        from maseval.benchmark.multiagentbench import MarbleMultiAgentBenchBenchmark

        benchmark = MagicMock(spec=MarbleMultiAgentBenchBenchmark)
        benchmark._coordinate_mode = "unknown_mode"

        with pytest.raises(ValueError, match="Unsupported coordinate mode"):
            MarbleMultiAgentBenchBenchmark.execution_loop(benchmark, [], MagicMock(), MagicMock(), None)


class TestGraphCoordinate:
    """Tests for _graph_coordinate — the most common MARBLE coordination mode."""

    def test_initial_assignment_runs_all_agents(self):
        """All agents receive the task content in the initial assignment."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1", "r1")
        adapter2 = _make_mock_adapter("agent2", "r2")
        benchmark = _make_mock_benchmark(planner_decide_sequence=[False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter1, adapter2])

        adapter1.run.assert_called_once_with("Do the task")
        adapter2.run.assert_called_once_with("Do the task")
        assert len(result["agent_results"]) == 2
        assert result["agent_results"][0] == {"agent_id": "agent1", "result": "r1"}

    def test_planner_stop_after_initial_returns_initial_results(self):
        """When planner stops after initial assignment, initial results are returned."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "initial_result")
        benchmark = _make_mock_benchmark(planner_decide_sequence=[False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # Only one call (no iterations)
        assert adapter.run.call_count == 1
        assert adapter.marble_agent.plan_task.call_count == 0
        # update_progress NOT called when planner stops after initial (engine.py:283-288)
        benchmark._marble_planner.update_progress.assert_not_called()
        assert result["agent_results"][0]["result"] == "initial_result"

    def test_iteration_uses_plan_task(self):
        """In iterations, agents call plan_task() and execute the planned task."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        adapter.marble_agent.plan_task.return_value = "my planned task"
        # Continue after initial, then stop after 1 iteration
        benchmark = _make_mock_benchmark(planner_decide_sequence=[True, False])

        MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        assert adapter.run.call_count == 2
        assert adapter.run.call_args_list[0].args[0] == "Do the task"  # initial
        assert adapter.run.call_args_list[1].args[0] == "my planned task"  # iteration
        adapter.marble_agent.plan_task.assert_called_once()

    def test_update_progress_receives_planner_return(self):
        """After initial assignment, update_progress receives summarize_output's return."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        planner_return = MagicMock(content="planner summary")
        benchmark = _make_mock_benchmark(planner_decide_sequence=[True, False])
        benchmark._marble_planner.summarize_output.return_value = planner_return

        MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # update_progress should receive the planner's return object, not raw string
        benchmark._marble_planner.update_progress.assert_called_once_with(planner_return)

    def test_respects_max_iterations(self):
        """Loop should not exceed per-task max_iterations even if planner always continues."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        # Planner always says continue
        benchmark = _make_mock_benchmark(max_iterations=3, planner_decide_sequence=None)
        benchmark._marble_planner.decide_next_step.return_value = True

        MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # initial(iter0→1) + iterations at 1→2, 2→3 = 3 total agent.run calls
        assert adapter.run.call_count == 3

    def test_returns_latest_iteration_results_on_stop(self):
        """When planner stops mid-loop, results from that iteration are returned."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        adapter.run.side_effect = ["initial_r", "iter1_r"]
        # Continue after initial, stop after first iteration
        benchmark = _make_mock_benchmark(planner_decide_sequence=[True, False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # Should return the iteration results, not initial
        assert result["agent_results"][0]["result"] == "iter1_r"

    def test_communication_captured_during_initial(self):
        """Communication produced during initial assignment should be captured."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1", communication="hello from agent1")
        benchmark = _make_mock_benchmark(planner_decide_sequence=[False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        assert result["communications"] == ["hello from agent1"]

    def test_communication_captured_during_iteration(self):
        """Communication produced during iteration should be captured."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1", communication="iter comm")
        adapter.marble_agent.plan_task.return_value = "planned"
        # Continue after initial, stop after iteration
        benchmark = _make_mock_benchmark(planner_decide_sequence=[True, False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # Communication from both initial and iteration
        assert len(result["communications"]) >= 1
        assert "iter comm" in result["communications"]

    def test_minecraft_domain_uses_file_check(self):
        """Graph mode with minecraft domain should check block_hit_rate file."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1")
        benchmark = _make_mock_benchmark(domain="minecraft", planner_decide_sequence=None)
        # Bind the real _minecraft_should_continue (patched to return False → stop)
        benchmark._minecraft_should_continue = MagicMock(return_value=False)

        MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # minecraft check was called, and it stopped iteration
        benchmark._minecraft_should_continue.assert_called()
        assert adapter.run.call_count == 1

    def test_agent_error_during_initial_does_not_crash(self):
        """If one agent fails during initial assignment, others still execute."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1")
        adapter1.run.side_effect = RuntimeError("LLM timeout")
        adapter2 = _make_mock_adapter("agent2", "ok")
        benchmark = _make_mock_benchmark(planner_decide_sequence=[False])

        result = MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter1, adapter2])

        # agent2 still ran and produced results
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent_id"] == "agent2"

    def test_agent_error_during_iteration_does_not_crash(self):
        """If agent.plan_task() or run() fails during iteration, loop continues."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        adapter.marble_agent.plan_task.side_effect = RuntimeError("plan failed")
        # Continue after initial, stop after iteration
        benchmark = _make_mock_benchmark(planner_decide_sequence=[True, False])

        MarbleMultiAgentBenchBenchmark._graph_coordinate(benchmark, [adapter])

        # Initial ran ok, iteration failed — planner still consulted
        assert adapter.run.call_count == 1  # only initial succeeded
        assert benchmark._marble_planner.decide_next_step.call_count == 2


class TestStarCoordinate:
    """Tests for _star_coordinate — centralized task assignment by planner."""

    def test_planner_assigns_tasks_to_agents(self):
        """Planner.assign_tasks() distributes work; agents execute assigned tasks."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1")
        adapter2 = _make_mock_adapter("agent2")
        agents_dict = {"agent1": adapter1, "agent2": adapter2}

        benchmark = _make_mock_benchmark(
            coordinate_mode="star",
            agents_dict=agents_dict,
            planner_decide_sequence=[False],
        )
        benchmark._planning_method = "naive"
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"agent1": "search papers", "agent2": "analyze data"}}

        result = MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter1, adapter2])

        adapter1.run.assert_called_once_with("search papers")
        adapter2.run.assert_called_once_with("analyze data")
        assert len(result["agent_results"]) == 2

    def test_star_update_progress_uses_raw_summary(self):
        """Star mode passes the raw summary string to update_progress (not planner return)."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="star", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._planning_method = "naive"
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"agent1": "task"}}

        MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter])

        # update_progress should receive raw string, not planner object
        call_arg = benchmark._marble_planner.update_progress.call_args[0][0]
        assert isinstance(call_arg, str)
        assert call_arg.startswith("Agents' Results Summary:")

    def test_star_communication_captured(self):
        """Star mode should capture communication from agents."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1", communication="star comm")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="star", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._planning_method = "naive"
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"agent1": "task"}}

        result = MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter])

        assert result["communications"] == ["star comm"]

    def test_star_skips_unknown_agent(self):
        """Star mode should skip (not crash) when planner assigns to unknown agent."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="star", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._planning_method = "naive"
        # Planner assigns to "unknown_agent" which doesn't exist in agents_dict
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"unknown_agent": "task", "agent1": "real task"}}

        result = MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter])

        # Only agent1 ran
        adapter.run.assert_called_once_with("real task")
        assert len(result["agent_results"]) == 1

    def test_star_agent_error_logged_not_raised(self):
        """Star mode should log agent errors and continue (matching graph behavior)."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1")
        adapter1.run.side_effect = RuntimeError("LLM error")
        adapter2 = _make_mock_adapter("agent2", "ok")
        agents_dict = {"agent1": adapter1, "agent2": adapter2}
        benchmark = _make_mock_benchmark(coordinate_mode="star", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._planning_method = "naive"
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"agent1": "t1", "agent2": "t2"}}

        result = MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter1, adapter2])

        # agent2 still produced results despite agent1 failure
        assert len(result["agent_results"]) == 1
        assert result["agent_results"][0]["agent_id"] == "agent2"

    def test_star_respects_max_iterations(self):
        """Star loop stops at max_iterations."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="star", max_iterations=2, agents_dict=agents_dict)
        benchmark._planning_method = "naive"
        benchmark._marble_planner.decide_next_step.return_value = True
        benchmark._marble_planner.assign_tasks.return_value = {"tasks": {"agent1": "t"}}

        MarbleMultiAgentBenchBenchmark._star_coordinate(benchmark, [adapter])

        assert adapter.run.call_count == 2


class TestChainCoordinate:
    """Tests for _chain_coordinate — sequential agent handoff."""

    def test_starts_with_agent1(self):
        """Chain should start with agent1 as per MARBLE's _select_initial_agent."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1", "r1")
        adapter2 = _make_mock_adapter("agent2", "r2")
        agents_dict = {"agent1": adapter1, "agent2": adapter2}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        adapter1.marble_agent.plan_next_agent.return_value = ("agent2", "next task")

        MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter1, adapter2])

        # agent1 should have been called first
        assert adapter1.run.call_args_list[0].args[0] == "Do the task"

    def test_task_always_updates_to_plan(self):
        """Chain should always update task to the plan, even if plan is empty string."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1")
        agents_dict = {"agent1": adapter1}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        # Agent returns empty plan — chain should still use it
        adapter1.marble_agent.plan_next_agent.return_value = ("agent1", "")
        # Stop after first link
        benchmark._marble_planner.decide_next_step.side_effect = [False]

        MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter1])

        # Only one call (planner stopped it)
        adapter1.run.assert_called_once_with("Do the task")

    def test_chain_handoff_to_next_agent(self):
        """Chain should hand off to the agent selected by plan_next_agent."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter1 = _make_mock_adapter("agent1", "r1")
        adapter2 = _make_mock_adapter("agent2", "r2")
        agents_dict = {"agent1": adapter1, "agent2": adapter2}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        # agent1 picks agent2, agent2 picks agent1 (but planner stops)
        adapter1.marble_agent.plan_next_agent.return_value = ("agent2", "task for agent2")
        adapter2.marble_agent.plan_next_agent.return_value = ("agent1", "back to agent1")
        benchmark._marble_planner.decide_next_step.side_effect = [True, False]

        result = MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter1, adapter2])

        # agent1 ran first, agent2 ran second
        adapter1.run.assert_called_once_with("Do the task")
        adapter2.run.assert_called_once_with("task for agent2")
        assert len(result["agent_results"]) == 2

    def test_chain_max_length(self):
        """Chain stops at max_chain_length = max_iterations * num_agents."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        agents_dict = {"agent1": adapter}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", max_iterations=2, agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        adapter.marble_agent.plan_next_agent.return_value = ("agent1", "keep going")
        benchmark._marble_planner.decide_next_step.return_value = True

        MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

        # max_chain_length = 2 * 1 = 2
        assert adapter.run.call_count == 2

    def test_chain_aborts_when_agent1_not_found(self):
        """Chain should abort (not fallback) when 'agent1' is not in agents_dict."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("researcher")
        agents_dict = {"researcher": adapter}  # No "agent1" key

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()

        result = MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

        # Should return empty — no silent fallback to first agent
        assert result["agent_results"] == []
        adapter.run.assert_not_called()

    def test_chain_error_propagates(self):
        """Chain should NOT catch agent errors — they must propagate (matching MARBLE)."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1")
        adapter.run.side_effect = RuntimeError("LLM failure")
        agents_dict = {"agent1": adapter}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"

        with pytest.raises(RuntimeError, match="LLM failure"):
            MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

    def test_chain_communication_captured(self):
        """Chain mode should capture communication from agents."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1", communication="chain comm")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        adapter.marble_agent.plan_next_agent.return_value = ("agent1", "next")

        result = MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

        assert result["communications"] == ["chain comm"]

    def test_chain_fallback_to_current_when_next_not_found(self):
        """Chain should fall back to current agent when next_agent_id is not in agents_dict."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "r1")
        agents_dict = {"agent1": adapter}
        benchmark = _make_mock_benchmark(coordinate_mode="chain", max_iterations=1, agents_dict=agents_dict)
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        # Agent picks "nonexistent" — falls back to self (engine.py:709-717)
        adapter.marble_agent.plan_next_agent.return_value = ("nonexistent", "task2")
        benchmark._marble_planner.decide_next_step.return_value = True

        MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

        # Fallback to current agent, so agent1 ran at least once
        assert adapter.run.call_count >= 1

    def test_chain_post_loop_update_progress(self):
        """Chain should call update_progress with accumulated summary after loop ends."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        adapter = _make_mock_adapter("agent1", "result_text")
        agents_dict = {"agent1": adapter}

        benchmark = _make_mock_benchmark(coordinate_mode="chain", agents_dict=agents_dict, planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_agent_profiles_linked.return_value = "profiles"
        adapter.marble_agent.plan_next_agent.return_value = ("agent1", "next")

        MarbleMultiAgentBenchBenchmark._chain_coordinate(benchmark, [adapter])

        # update_progress called twice: once in-loop (engine.py:719) + once post-loop (engine.py:768)
        assert benchmark._marble_planner.update_progress.call_count == 2
        # Post-loop call should be the accumulated summary string
        post_loop_arg = benchmark._marble_planner.update_progress.call_args_list[-1][0][0]
        assert isinstance(post_loop_arg, str)
        assert post_loop_arg.startswith("Agents' Results Summary:")


class TestTreeCoordinate:
    """Tests for _tree_coordinate — recursive hierarchical execution."""

    def test_recursive_execution_from_root(self):
        """Tree mode should execute from root, delegating to children recursively."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        # Build a tree: root → child
        child_agent = MagicMock()
        child_agent.agent_id = "child1"
        child_agent.children = []
        child_agent.act.return_value = ("child result", None)

        root_agent = MagicMock()
        root_agent.agent_id = "root"
        root_agent.children = [child_agent]
        root_agent.plan_tasks_for_children.return_value = {"child1": "sub-task"}
        root_agent.act.return_value = ("root result", "root comm")

        child_adapter = _make_mock_adapter("child1", "child traced result")
        root_adapter = _make_mock_adapter("root", "root traced result")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = root_agent
        benchmark._agents_dict = {"root": root_adapter, "child1": child_adapter}
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [root_adapter, child_adapter])

        # Child was executed
        child_adapter.run.assert_called_once_with("sub-task")
        # Root was executed with children's results appended
        assert root_adapter.run.call_count == 1
        root_call_arg = root_adapter.run.call_args[0][0]
        assert "results of the children" in root_call_arg

    def test_tree_passes_results_directly_to_planner(self):
        """Tree mode passes recursive results directly (not reformatted) to summarizer and planner."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        leaf_agent = MagicMock()
        leaf_agent.agent_id = "leaf"
        leaf_agent.children = []
        leaf_agent.act.return_value = ("leaf result", None)

        leaf_adapter = _make_mock_adapter("leaf", "leaf result")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = leaf_agent
        benchmark._agents_dict = {"leaf": leaf_adapter}
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [leaf_adapter])

        # decide_next_step should receive results in recursive format: [{"agent_id": ..., "result": ...}]
        decide_call_arg = benchmark._marble_planner.decide_next_step.call_args[0][0]
        assert isinstance(decide_call_arg, list)
        assert "agent_id" in decide_call_arg[0]

    def test_tree_leaf_communication_captured(self):
        """Leaf agent's communication from act() should be captured."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        leaf_agent = MagicMock()
        leaf_agent.agent_id = "leaf"
        leaf_agent.children = []
        leaf_agent.act.return_value = ("result", "leaf comm")

        # No adapter — uses direct agent call
        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = leaf_agent
        benchmark._agents_dict = {}  # No adapters
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        result = MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [])

        assert result["communications"] == ["leaf comm"]

    def test_tree_no_root_agent_returns_empty(self):
        """Tree mode should return empty results when no root agent found."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = None

        result = MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [])

        assert result["agent_results"] == []
        assert result["communications"] == []

    def test_tree_leaf_communication_via_adapter(self):
        """Leaf agent's communication should be captured via adapter when available."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        leaf_agent = MagicMock()
        leaf_agent.agent_id = "leaf"
        leaf_agent.children = []

        leaf_adapter = _make_mock_adapter("leaf", "leaf result", communication="adapter leaf comm")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = leaf_agent
        benchmark._agents_dict = {"leaf": leaf_adapter}
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        result = MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [leaf_adapter])

        assert "adapter leaf comm" in result["communications"]

    def test_tree_parent_no_adapter_uses_direct_act(self):
        """Parent agent without adapter should fall back to marble_agent.act()."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        child_agent = MagicMock()
        child_agent.agent_id = "child"
        child_agent.children = []
        child_agent.act.return_value = ("child result", None)

        root_agent = MagicMock()
        root_agent.agent_id = "root"
        root_agent.children = [child_agent]
        root_agent.plan_tasks_for_children.return_value = {"child": "sub-task"}
        root_agent.act.return_value = ("root result", "root comm")

        # Only child has adapter; root does not
        child_adapter = _make_mock_adapter("child", "child traced")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = root_agent
        benchmark._agents_dict = {"child": child_adapter}  # No "root" adapter
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        result = MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [child_adapter])

        # root.act() was called directly (not via adapter)
        root_agent.act.assert_called_once()
        assert "root comm" in result["communications"]

    def test_tree_empty_child_task_skipped(self):
        """Children with empty task string should be skipped."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        child_agent = MagicMock()
        child_agent.agent_id = "child"
        child_agent.children = []

        root_agent = MagicMock()
        root_agent.agent_id = "root"
        root_agent.children = [child_agent]
        # plan_tasks_for_children returns empty string for child
        root_agent.plan_tasks_for_children.return_value = {"child": ""}
        root_agent.act.return_value = ("root result", None)

        root_adapter = _make_mock_adapter("root", "root traced")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", planner_decide_sequence=[False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = root_agent
        benchmark._agents_dict = {"root": root_adapter, "child": _make_mock_adapter("child")}
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [root_adapter])

        # child.act() should NOT have been called (empty task)
        child_agent.act.assert_not_called()

    def test_tree_multiple_iterations(self):
        """Tree mode should iterate when planner continues."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        leaf_agent = MagicMock()
        leaf_agent.agent_id = "leaf"
        leaf_agent.children = []
        leaf_agent.act.return_value = ("result", None)

        leaf_adapter = _make_mock_adapter("leaf", "traced result")

        benchmark = _make_mock_benchmark(coordinate_mode="tree", max_iterations=3, planner_decide_sequence=[True, True, False])
        benchmark._marble_graph = MagicMock()
        benchmark._marble_graph.get_root_agent.return_value = leaf_agent
        benchmark._agents_dict = {"leaf": leaf_adapter}
        benchmark._execute_agent_task_recursive = MarbleMultiAgentBenchBenchmark._execute_agent_task_recursive.__get__(benchmark)

        MarbleMultiAgentBenchBenchmark._tree_coordinate(benchmark, [leaf_adapter])

        # 3 iterations: planner continues twice, stops on third
        assert leaf_adapter.run.call_count == 3


class TestMinecraftTermination:
    """Tests for _minecraft_should_continue helper."""

    def test_returns_false_when_block_hit_rate_is_1(self):
        """Should return False (stop) when block_hit_rate is 1."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # _minecraft_should_continue reads Path(_MARBLE_ROOT).parent / "data" / "score.json"
            # Set up the directory structure so the path resolves correctly.
            marble_root = os.path.join(tmpdir, "marble")
            os.makedirs(marble_root, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            score_path = os.path.join(data_dir, "score.json")
            with open(score_path, "w") as f:
                json.dump([{"block_hit_rate": 1}], f)

            with patch("maseval.benchmark.multiagentbench._constants._MARBLE_ROOT", marble_root):
                assert MarbleMultiAgentBenchBenchmark._minecraft_should_continue() is False

    def test_returns_true_when_block_hit_rate_below_1(self):
        """Should return True (continue) when block_hit_rate is below 1."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            marble_root = os.path.join(tmpdir, "marble")
            os.makedirs(marble_root, exist_ok=True)
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir, exist_ok=True)
            score_path = os.path.join(data_dir, "score.json")
            with open(score_path, "w") as f:
                json.dump([{"block_hit_rate": 0.5}], f)

            with patch("maseval.benchmark.multiagentbench._constants._MARBLE_ROOT", marble_root):
                assert MarbleMultiAgentBenchBenchmark._minecraft_should_continue() is True

    def test_returns_true_when_file_missing(self):
        """Should return True (continue) when score.json doesn't exist."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Point _MARBLE_ROOT to a dir where score.json doesn't exist
            with patch("maseval.benchmark.multiagentbench._constants._MARBLE_ROOT", tmpdir):
                assert MarbleMultiAgentBenchBenchmark._minecraft_should_continue() is True


class TestSummarizeResults:
    """Tests for _summarize_results_marble helper."""

    def test_format_matches_marble(self):
        """Output format should match MARBLE's Engine._summarize_results."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        agents_results = [
            {"agent1": "Research finding"},
            {"agent2": "NLP results"},
        ]
        summary = MarbleMultiAgentBenchBenchmark._summarize_results_marble(agents_results)

        assert summary.startswith("Agents' Results Summary:\n")
        assert "agent1" in summary
        assert "agent2" in summary

    def test_truncates_at_1000_chars(self):
        """Each result line should be truncated to 1000 chars."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        long_result = "x" * 2000
        summary = MarbleMultiAgentBenchBenchmark._summarize_results_marble([{"agent1": long_result}])

        lines = summary.strip().split("\n")
        for line in lines[1:]:  # Skip header
            assert len(line) <= 1000

    def test_handles_recursive_result_format(self):
        """Should also work with tree's [{"agent_id": ..., "result": ...}] format."""
        from maseval.benchmark.multiagentbench.multiagentbench import MarbleMultiAgentBenchBenchmark

        results = [{"agent_id": "root", "result": "done"}]
        summary = MarbleMultiAgentBenchBenchmark._summarize_results_marble(results)

        assert "agent_id" in summary
        assert "root" in summary


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
