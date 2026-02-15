"""Shared fixtures for MultiAgentBench tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, etc.)
  These are automatically available via pytest's conftest inheritance.
- tests/test_benchmarks/test_multiagentbench/conftest.py: MultiAgentBench-specific fixtures

MultiAgentBench-Specific Components
-----------------------------------
- MultiAgentBenchAgentAdapter: Test adapter for multi-agent scenarios
- ConcreteMultiAgentBenchBenchmark: Concrete implementation for testing
- Sample task fixtures for different domains
"""

import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

from conftest import DummyModelAdapter
from maseval import AgentAdapter, Task, MessageHistory
from maseval.benchmark.multiagentbench.environment import MultiAgentBenchEnvironment


# =============================================================================
# Session-Scoped Setup
# =============================================================================


@pytest.fixture(scope="session")
def ensure_marble_data():
    """Ensure MARBLE data is available, downloading if necessary.

    Uses ensure_marble_exists() which caches: skips clone when marble/ already exists.

    Tests that need real MARBLE data should depend on this fixture and be marked
    @pytest.mark.live.
    Tests that don't need data (structural, mock-based) should NOT depend on this fixture.

    Returns:
        Path to the MARBLE directory
    """
    from maseval.benchmark.multiagentbench.data_loader import ensure_marble_exists

    return ensure_marble_exists(auto_download=True)


@pytest.fixture(autouse=True)
def _mock_marble_environment():
    """Mock MARBLE environment creation for all Tier 1 (offline) tests.

    MultiAgentBenchEnvironment.setup_state() calls _create_marble_environment()
    which imports marble — a vendored dependency not available in CI. This fixture
    replaces that method with a MagicMock so structural tests can exercise the full
    benchmark pipeline without marble installed.

    Tests that need the real _create_marble_environment can override this fixture
    with a no-op at the class or file level (standard pytest fixture scoping).
    """
    mock_env = MagicMock()
    mock_env.is_done.return_value = False
    mock_env.is_task_completed.return_value = False
    mock_env.get_state.return_value = {}
    mock_env._action_handlers = {}
    mock_env.action_handler_descriptions = {}

    with patch.object(
        MultiAgentBenchEnvironment,
        "_create_marble_environment",
        return_value=mock_env,
    ):
        yield


# =============================================================================
# Sample Task Data
# =============================================================================


@pytest.fixture
def sample_research_task_data() -> Dict[str, Any]:
    """Sample task data for research domain."""
    return {
        "scenario": "research",
        "task_id": 1,
        "agents": [
            {
                "agent_id": "agent1",
                "profile": "I am a researcher focused on machine learning.",
                "type": "BaseAgent",
            },
            {
                "agent_id": "agent2",
                "profile": "I am a researcher focused on NLP.",
                "type": "BaseAgent",
            },
        ],
        "coordinate_mode": "cooperative",
        "relationships": [
            ["agent1", "agent2", "collaborate with"],
        ],
        "environment": {
            "type": "Research",
            "max_iterations": 10,
        },
        "task": {
            "content": "Collaborate to generate a research idea about federated learning.",
            "output_format": "Present the idea in 5Q format.",
        },
        "llm": "gpt-4o-mini",
        "memory": {"type": "SharedMemory"},
        "metrics": {
            "diversity_of_perspectives": True,
            "engagement_level": True,
            "relevance": True,
        },
        "max_iterations": 10,
        "engine_planner": {"initial_progress": "Starting research collaboration."},
        "output": {"format": "jsonl"},
        "raw_marble_config": {},
    }


@pytest.fixture
def sample_bargaining_task_data() -> Dict[str, Any]:
    """Sample task data for bargaining domain."""
    return {
        "scenario": "bargaining",
        "task_id": 1,
        "agents": [
            {
                "agent_id": "buyer",
                "profile": "I am a buyer looking for the best price.",
                "type": "BaseAgent",
            },
            {
                "agent_id": "seller",
                "profile": "I am a seller trying to maximize profit.",
                "type": "BaseAgent",
            },
        ],
        "coordinate_mode": "cooperative",
        "relationships": [
            ["buyer", "seller", "negotiate with"],
        ],
        "environment": {
            "type": "WorldSimulation",
            "max_iterations": 10,
        },
        "task": {
            "content": "Negotiate the price of a used laptop.",
            "output_format": "Final agreed price or disagreement.",
        },
        "llm": "gpt-4o-mini",
        "memory": {"type": "SharedMemory"},
        "metrics": {},
        "max_iterations": 10,
        "engine_planner": {},
        "output": {},
        "raw_marble_config": {},
    }


@pytest.fixture
def sample_research_task(sample_research_task_data: Dict[str, Any]) -> Task:
    """Create a sample research Task."""
    return Task(
        id="research_1",
        query=sample_research_task_data["task"]["content"],
        environment_data=sample_research_task_data,
        evaluation_data={
            "metrics": sample_research_task_data.get("metrics", {}),
            "output_format": sample_research_task_data["task"].get("output_format", ""),
            "model_id": "gpt-4o-mini",
        },
        metadata={
            "domain": "research",
            "task_id": 1,
        },
    )


@pytest.fixture
def sample_bargaining_task(sample_bargaining_task_data: Dict[str, Any]) -> Task:
    """Create a sample bargaining Task."""
    return Task(
        id="bargaining_1",
        query=sample_bargaining_task_data["task"]["content"],
        environment_data=sample_bargaining_task_data,
        evaluation_data={
            "metrics": sample_bargaining_task_data.get("metrics", {}),
            "output_format": sample_bargaining_task_data["task"].get("output_format", ""),
            "model_id": "gpt-4o-mini",
        },
        metadata={
            "domain": "bargaining",
            "task_id": 1,
        },
    )


# =============================================================================
# Mock Components
# =============================================================================


class MultiAgentBenchAgentAdapter(AgentAdapter):
    """Test agent adapter for MultiAgentBench tests.

    Provides controllable responses without needing a real agent implementation.
    """

    def __init__(
        self,
        agent_id: str = "test_agent",
        profile: str = "Test agent profile",
    ):
        super().__init__(agent_instance=MagicMock(), name=agent_id)
        self._agent_id = agent_id
        self._profile = profile
        self._responses: List[str] = []
        self._call_count = 0
        self.run_calls: List[str] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def profile(self) -> str:
        return self._profile

    def set_responses(self, responses: List[str]) -> None:
        """Set canned responses for the agent."""
        self._responses = responses

    def _run_agent(self, query: str) -> MessageHistory:
        self.run_calls.append(query)
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
        else:
            response = f"Agent {self._agent_id} response to: {query[:50]}..."
        return MessageHistory([{"role": "assistant", "content": response}])

    def get_token_usage(self) -> int:
        return self._call_count * 100

    def gather_traces(self) -> Dict[str, Any]:
        traces = super().gather_traces()
        traces["agent_id"] = self._agent_id
        traces["profile"] = self._profile
        traces["token_usage"] = self.get_token_usage()
        traces["action_log"] = [{"task": q, "result": f"Response {i}", "has_communication": False} for i, q in enumerate(self.run_calls)]
        traces["communication_log"] = []
        return traces


# =============================================================================
# Concrete Benchmark Implementation
# =============================================================================


@pytest.fixture
def concrete_multiagentbench_benchmark():
    """Create a concrete MultiAgentBenchBenchmark for testing."""
    from maseval.benchmark.multiagentbench import MultiAgentBenchBenchmark

    class ConcreteMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
        """Concrete implementation for testing."""

        def __init__(
            self,
            model_factory: Optional[Any] = None,
            **kwargs: Any,
        ):
            if model_factory is None:
                self._model_factory = lambda model_name: DummyModelAdapter(
                    model_id=f"test-model-{model_name}",
                    responses=['{"innovation": 4, "safety": 4, "feasibility": 4}'],
                )
            elif callable(model_factory):
                self._model_factory = model_factory
            else:
                self._model_factory = lambda model_name: model_factory
            super().__init__(**kwargs)

        def get_model_adapter(self, model_id: str, **kwargs):
            factory_key = kwargs.get("register_name", model_id)
            adapter = self._model_factory(factory_key)
            register_name = kwargs.get("register_name")
            if register_name:
                try:
                    self.register("models", register_name, adapter)
                except ValueError:
                    pass
            return adapter

        def setup_agents(
            self,
            agent_data: Dict[str, Any],
            environment: Any,
            task: Task,
            user: Optional[Any],
            seed_generator,
        ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
            agent_configs = task.environment_data.get("agents", [])
            agents_list: List[AgentAdapter] = []
            agents_dict: Dict[str, AgentAdapter] = {}

            for config in agent_configs:
                agent_id = config.get("agent_id", f"agent_{len(agents_list)}")
                profile = config.get("profile", "")
                adapter = MultiAgentBenchAgentAdapter(
                    agent_id=agent_id,
                    profile=profile,
                )
                agents_list.append(adapter)
                agents_dict[agent_id] = adapter

            return agents_list, agents_dict

    return ConcreteMultiAgentBenchBenchmark


@pytest.fixture
def benchmark_instance(concrete_multiagentbench_benchmark):
    """Create a benchmark instance with default settings."""
    return concrete_multiagentbench_benchmark(
        progress_bar=False,
        max_invocations=1,
    )


@pytest.fixture
def seed_gen():
    """Seed generator fixture for direct setup method calls.

    When seeding is disabled (global_seed=None), derive_seed() returns None.
    This fixture is for tests that call setup methods directly outside of run().
    """
    from maseval.core.seeding import DefaultSeedGenerator

    return DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
