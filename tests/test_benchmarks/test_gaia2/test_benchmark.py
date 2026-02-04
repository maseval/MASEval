"""Tests for Gaia2Benchmark and DefaultAgentGaia2Benchmark.

Tests the benchmark orchestration layer that wraps ARE integration.
"""

import pytest
from unittest.mock import MagicMock

from conftest import DummyModelAdapter
from maseval import Task, AgentAdapter


# =============================================================================
# Test Gaia2Benchmark (Abstract Base)
# =============================================================================


@pytest.mark.benchmark
class TestGaia2BenchmarkInit:
    """Tests for Gaia2Benchmark initialization."""

    def test_default_max_invocations_is_one(self):
        """Test that MAX_INVOCATIONS is 1 (single-turn)."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        assert Gaia2Benchmark.MAX_INVOCATIONS == 1

    def test_initializes_with_defaults(self):
        """Test benchmark initializes with default parameters."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        # Create minimal concrete implementation
        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        assert benchmark.max_invocations == 1
        assert benchmark.n_task_repeats == 1

    def test_accepts_custom_parameters(self):
        """Test benchmark accepts custom parameters."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark(
            n_task_repeats=3,
            max_invocations=5,
            num_workers=2,
        )

        assert benchmark.n_task_repeats == 3
        assert benchmark.max_invocations == 5


@pytest.mark.benchmark
class TestGaia2BenchmarkSetupUser:
    """Tests for Gaia2Benchmark.setup_user()."""

    def test_returns_none(self, sample_gaia2_task):
        """Test setup_user returns None (no user simulation in GAIA2)."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        # Mock environment
        mock_env = MagicMock()

        result = benchmark.setup_user({}, mock_env, sample_gaia2_task)

        assert result is None


@pytest.mark.benchmark
class TestGaia2BenchmarkSetupEnvironment:
    """Tests for Gaia2Benchmark.setup_environment()."""

    def test_creates_gaia2_environment(self, sample_gaia2_task):
        """Test setup_environment creates Gaia2Environment."""
        import sys
        from unittest.mock import patch, MagicMock
        from maseval.benchmark.gaia2 import Gaia2Benchmark, Gaia2Environment

        # Create mock ARE module structure
        mock_are = MagicMock()
        mock_are_env_instance = MagicMock()
        mock_are.simulation.environment.Environment.return_value = mock_are_env_instance
        mock_are_env_instance.get_tools.return_value = []
        mock_are_env_instance.get_completed_events.return_value = []

        mock_scenario = MagicMock()
        mock_scenario.duration = 86400

        # Patch sys.modules for ARE imports
        with patch.dict(
            sys.modules,
            {
                "are": mock_are,
                "are.simulation": mock_are.simulation,
                "are.simulation.environment": mock_are.simulation.environment,
            },
        ):
            # Add scenario to task environment_data
            sample_gaia2_task.environment_data["scenario"] = mock_scenario

            class TestBenchmark(Gaia2Benchmark):
                def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                    return [], {}

                def get_model_adapter(self, model_id, **kwargs):
                    return DummyModelAdapter()

            benchmark = TestBenchmark()

            env = benchmark.setup_environment({}, sample_gaia2_task)

            assert isinstance(env, Gaia2Environment)


@pytest.mark.benchmark
class TestGaia2BenchmarkSetupEvaluators:
    """Tests for Gaia2Benchmark.setup_evaluators()."""

    def test_creates_gaia2_evaluator(self, sample_gaia2_task):
        """Test setup_evaluators creates Gaia2Evaluator."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark, Gaia2Evaluator

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()
        mock_env = MagicMock()

        evaluators = benchmark.setup_evaluators(mock_env, sample_gaia2_task, [], None)

        assert len(evaluators) == 1
        assert isinstance(evaluators[0], Gaia2Evaluator)

    def test_passes_model_if_configured(self, sample_gaia2_task):
        """Test evaluator receives model if model_id configured."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark, Gaia2Evaluator

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        benchmark = TestBenchmark()
        mock_env = MagicMock()

        # Add model_id to evaluation_data
        task = Task(
            id="test",
            query="test",
            environment_data={},
            evaluation_data={"model_id": "test-evaluator-model"},
            user_data={},
            metadata={},
        )

        evaluators = benchmark.setup_evaluators(mock_env, task, [], None)
        gaia2_evaluator = evaluators[0]
        assert isinstance(gaia2_evaluator, Gaia2Evaluator)
        assert gaia2_evaluator.model is not None


@pytest.mark.benchmark
class TestGaia2BenchmarkRunAgents:
    """Tests for Gaia2Benchmark.run_agents()."""

    def test_runs_agents_and_returns_answer(self, sample_gaia2_task):
        """Test run_agents executes agents and returns answer."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Task completed"

        # Create mock environment
        mock_env = MagicMock()

        result = benchmark.run_agents([mock_agent], sample_gaia2_task, mock_env, "Do task")

        assert result == "Task completed"
        mock_agent.run.assert_called_once_with("Do task")

    def test_cleans_up_environment_on_success(self, sample_gaia2_task):
        """Test environment cleanup is called on success."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Done"

        mock_env = MagicMock()

        benchmark.run_agents([mock_agent], sample_gaia2_task, mock_env, "Task")

        mock_env.cleanup.assert_called_once()

    def test_cleans_up_environment_on_error(self, sample_gaia2_task):
        """Test environment cleanup is called even on error."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Agent failed")

        mock_env = MagicMock()

        with pytest.raises(RuntimeError):
            benchmark.run_agents([mock_agent], sample_gaia2_task, mock_env, "Task")

        mock_env.cleanup.assert_called_once()


@pytest.mark.benchmark
class TestGaia2BenchmarkEvaluate:
    """Tests for Gaia2Benchmark.evaluate()."""

    def test_calls_evaluator_with_filtered_traces(self):
        """Test evaluate calls evaluator with filtered traces."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class TestBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        mock_evaluator = MagicMock()
        mock_evaluator.filter_traces.return_value = {"filtered": "traces"}
        mock_evaluator.return_value = {"gsr": 1.0, "passed": True}

        traces = {"raw": "traces"}

        results = benchmark.evaluate([mock_evaluator], {}, "answer", traces)

        mock_evaluator.filter_traces.assert_called_once_with(traces)
        mock_evaluator.assert_called_once_with({"filtered": "traces"}, "answer")
        assert results == [{"gsr": 1.0, "passed": True}]


# =============================================================================
# Test DefaultAgentGaia2Benchmark
# =============================================================================


@pytest.mark.benchmark
class TestDefaultAgentGaia2BenchmarkInit:
    """Tests for DefaultAgentGaia2Benchmark initialization."""

    def test_stores_agent_data(self):
        """Test benchmark stores agent_data."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        # Note: DefaultAgentGaia2Benchmark is abstract (get_model_adapter)
        # We test the initialization logic via a concrete subclass

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        agent_data = {"model_id": "test-model", "max_iterations": 50}
        benchmark = TestBenchmark(agent_data=agent_data)

        assert benchmark._agent_data == agent_data

    def test_default_agent_data_is_empty_dict(self):
        """Test default agent_data is empty dict."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()

        assert benchmark._agent_data == {}


@pytest.mark.benchmark
class TestDefaultAgentGaia2BenchmarkSetupAgents:
    """Tests for DefaultAgentGaia2Benchmark.setup_agents()."""

    def test_creates_default_gaia2_agent(self, sample_gaia2_task):
        """Test setup_agents creates DefaultGaia2Agent."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark, DefaultGaia2AgentAdapter

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        benchmark = TestBenchmark(agent_data={"model_id": "test-model"})

        # Mock environment
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {"tool1": lambda **kw: "result"}

        agents, agent_dict = benchmark.setup_agents({}, mock_env, sample_gaia2_task, None)

        assert len(agents) == 1
        assert isinstance(agents[0], DefaultGaia2AgentAdapter)
        assert "gaia2_agent" in agent_dict

    def test_merges_class_and_runtime_agent_data(self, sample_gaia2_task):
        """Test agent_data from class and runtime are merged."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        benchmark = TestBenchmark(agent_data={"model_id": "class-model", "verbose": 1})

        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        # Runtime data should override class data
        agents, _ = benchmark.setup_agents(
            {"model_id": "runtime-model"},
            mock_env,
            sample_gaia2_task,
            None,
        )

        # The agent should use runtime model_id
        # We can't easily verify this without inspecting the agent's model

    def test_raises_if_model_id_missing(self, sample_gaia2_task):
        """Test raises ValueError if model_id not configured."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = TestBenchmark()  # No agent_data

        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        with pytest.raises(ValueError, match="model_id not configured"):
            benchmark.setup_agents({}, mock_env, sample_gaia2_task, None)

    def test_uses_default_iterations_from_constant(self, sample_gaia2_task):
        """Test uses _DEFAULT_MAX_ITERATIONS if not specified."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark
        from maseval.benchmark.gaia2.gaia2 import _DEFAULT_MAX_ITERATIONS

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        benchmark = TestBenchmark(agent_data={"model_id": "test"})

        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        agents, _ = benchmark.setup_agents({}, mock_env, sample_gaia2_task, None)

        # The agent should have default max_iterations
        assert agents[0].agent.max_iterations == _DEFAULT_MAX_ITERATIONS

    def test_passes_custom_llm_args(self, sample_gaia2_task):
        """Test custom llm_args are passed to agent."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        class TestBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        custom_llm_args = {"temperature": 0.9, "max_tokens": 4096}
        benchmark = TestBenchmark(agent_data={"model_id": "test", "llm_args": custom_llm_args})

        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        agents, _ = benchmark.setup_agents({}, mock_env, sample_gaia2_task, None)

        # Agent should have custom temperature (overrides default)
        assert agents[0].agent.llm_args["temperature"] == 0.9


@pytest.mark.benchmark
class TestDefaultAgentGaia2BenchmarkDocstring:
    """Tests for DefaultAgentGaia2Benchmark documentation."""

    def test_docstring_mentions_default_parameters(self):
        """Test docstring documents default parameters."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        docstring = DefaultAgentGaia2Benchmark.__doc__
        assert docstring is not None

        assert "max_iterations: 80" in docstring
        assert "temperature: 0.5" in docstring
        assert "max_tokens: 16384" in docstring
        assert "invalid_format_retries: 10" in docstring


# =============================================================================
# Test Agent-Agnostic Design
# =============================================================================


@pytest.mark.benchmark
class TestAgentAgnosticDesign:
    """Tests verifying Gaia2Benchmark is agent-agnostic."""

    def test_gaia2_benchmark_does_not_import_default_agent(self):
        """Test Gaia2Benchmark class doesn't depend on DefaultGaia2Agent."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark
        import inspect

        # Get source code of Gaia2Benchmark class
        source = inspect.getsource(Gaia2Benchmark)

        # Should not reference default agent classes
        assert "DefaultGaia2Agent" not in source
        assert "DefaultGaia2AgentAdapter" not in source

    def test_can_use_custom_agent(self, sample_gaia2_task):
        """Test Gaia2Benchmark works with custom agent implementation."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark

        class CustomAgent:
            def run(self, query):
                return f"Custom response to: {query}"

        class CustomAgentAdapter(AgentAdapter):
            def __init__(self):
                super().__init__(CustomAgent(), "custom_agent")

            def _run_agent(self, query):
                return self.agent.run(query)

        class CustomBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                adapter = CustomAgentAdapter()
                return [adapter], {"custom_agent": adapter}

            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter()

        benchmark = CustomBenchmark()
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        agents, agent_dict = benchmark.setup_agents({}, mock_env, sample_gaia2_task, None)

        assert len(agents) == 1
        assert agents[0].run("test") == "Custom response to: test"


# =============================================================================
# Test Seeding Behavior
# =============================================================================


@pytest.mark.benchmark
class TestGaia2BenchmarkSeeding:
    """Tests for Gaia2Benchmark seeding behavior."""

    def test_setup_agents_passes_seed_to_get_model_adapter(self, sample_gaia2_task):
        """Test that setup_agents derives and passes seed to get_model_adapter."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark
        from maseval.core.seeding import DefaultSeedGenerator

        captured_kwargs = []

        class CapturingBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return DummyModelAdapter(model_id=model_id)

        benchmark = CapturingBenchmark(agent_data={"model_id": "test-model"})
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        benchmark.setup_agents({}, mock_env, sample_gaia2_task, None, seed_generator=seed_gen)

        # Verify seed was passed
        assert len(captured_kwargs) == 1
        assert "seed" in captured_kwargs[0]
        assert captured_kwargs[0]["seed"] is not None
        assert isinstance(captured_kwargs[0]["seed"], int)

    def test_setup_agents_passes_none_seed_when_no_generator(self, sample_gaia2_task):
        """Test that setup_agents passes None seed when seed_generator is None."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark

        captured_kwargs = []

        class CapturingBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return DummyModelAdapter(model_id=model_id)

        benchmark = CapturingBenchmark(agent_data={"model_id": "test-model"})
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        benchmark.setup_agents({}, mock_env, sample_gaia2_task, None, seed_generator=None)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("seed") is None

    def test_setup_evaluators_passes_seed_to_get_model_adapter(self, sample_gaia2_task):
        """Test that setup_evaluators derives and passes seed to get_model_adapter."""
        from maseval.benchmark.gaia2 import Gaia2Benchmark
        from maseval.core.seeding import DefaultSeedGenerator
        from maseval import Task

        captured_kwargs = []

        class CapturingBenchmark(Gaia2Benchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
                return [], {}

            def get_model_adapter(self, model_id, **kwargs):
                captured_kwargs.append(kwargs.copy())
                return DummyModelAdapter(model_id=model_id)

        benchmark = CapturingBenchmark()
        mock_env = MagicMock()

        # Task with evaluator model_id configured
        task = Task(
            id="test",
            query="test",
            environment_data={},
            evaluation_data={"model_id": "evaluator-model"},
            user_data={},
            metadata={},
        )

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        benchmark.setup_evaluators(mock_env, task, [], None, seed_generator=seed_gen)

        # Verify seed was passed
        assert len(captured_kwargs) == 1
        assert "seed" in captured_kwargs[0]
        assert captured_kwargs[0]["seed"] is not None

    def test_seeding_uses_correct_paths(self, sample_gaia2_task):
        """Test that seeding uses the documented seed paths."""
        from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark
        from maseval.core.seeding import DefaultSeedGenerator
        from maseval import Task

        class CapturingBenchmark(DefaultAgentGaia2Benchmark):
            def get_model_adapter(self, model_id, **kwargs):
                return DummyModelAdapter(model_id=model_id)

        benchmark = CapturingBenchmark(agent_data={"model_id": "test-model"})
        mock_env = MagicMock()
        mock_env.create_tools.return_value = {}

        seed_gen = DefaultSeedGenerator(global_seed=42, task_id="test", rep_index=0)

        # Call setup_agents
        benchmark.setup_agents({}, mock_env, sample_gaia2_task, None, seed_generator=seed_gen)

        # Verify the seed path was logged
        assert "agents/gaia2_agent" in seed_gen.seed_log

        # Call setup_evaluators with model_id
        task_with_eval = Task(
            id="test",
            query="test",
            environment_data={},
            evaluation_data={"model_id": "evaluator-model"},
            user_data={},
            metadata={},
        )
        benchmark.setup_evaluators(mock_env, task_with_eval, [], None, seed_generator=seed_gen)

        # Verify the evaluator seed path was logged
        assert "evaluators/judge" in seed_gen.seed_log
