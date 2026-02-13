"""MultiAgentBench benchmark implementations.

This module provides benchmark classes for the MARBLE MultiAgentBench suite:
- MultiAgentBenchBenchmark: Abstract base for framework-agnostic evaluation
- MarbleMultiAgentBenchBenchmark: Exact MARBLE reproduction mode
"""

import logging
from abc import abstractmethod
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from maseval import (
    AgentAdapter,
    Benchmark,
    Environment,
    Evaluator,
    ModelAdapter,
    Task,
    User,
)
from maseval.core.callback import BenchmarkCallback
from maseval.core.seeding import SeedGenerator

from maseval.benchmark.multiagentbench._constants import MARBLE_IMPORT_ERROR
from maseval.benchmark.multiagentbench.environment import MultiAgentBenchEnvironment
from maseval.benchmark.multiagentbench.evaluator import (
    MultiAgentBenchEvaluator,
)

if TYPE_CHECKING:
    from maseval.benchmark.multiagentbench.adapters.marble_adapter import MarbleAgentAdapter

logger = logging.getLogger(__name__)


class MultiAgentBenchBenchmark(Benchmark):
    """Abstract base class for framework-agnostic MultiAgentBench evaluation.

    This benchmark provides the infrastructure for evaluating multi-agent systems
    on MARBLE's MultiAgentBench tasks. Subclasses implement `setup_agents()` with
    their specific agent framework.

    The benchmark supports:
    - Multiple coordination modes (star, cooperative, tree, hierarchical)
    - Multiple domains (research, bargaining, coding, database, etc.)
    - LLM-based evaluation matching MARBLE's metrics
    - Comprehensive tracing of agent interactions

    Example:
        ```python
        class MyMultiAgentBenchmark(MultiAgentBenchBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                # Derive seeds for agents (returns None if seeding disabled)
                agents_gen = seed_generator.child("agents")
                agent_seeds = {}
                for config in task.environment_data.get("agents", []):
                    agent_id = config.get("agent_id")
                    agent_seeds[agent_id] = agents_gen.derive_seed(agent_id)

                # Create agents using your framework with seeds
                agents_list = []
                agents_dict = {}
                for config in task.environment_data.get("agents", []):
                    agent_id = config.get("agent_id")
                    model = self.get_model_adapter(
                        agent_data.get("model_id", "gpt-4o"),
                        register_name=f"agent_{agent_id}",
                        seed=agent_seeds.get(agent_id),
                    )
                    # Create your agent with the seeded model...
                    ...
                return agents_list, agents_dict

            def get_model_adapter(self, model_id, **kwargs):
                seed = kwargs.pop("seed", None)
                adapter = MyModelAdapter(model_id, seed=seed)
                if "register_name" in kwargs:
                    self.register("models", kwargs["register_name"], adapter)
                return adapter

        benchmark = MyMultiAgentBenchmark(seed=42)  # Enable seeding
        results = benchmark.run(tasks, agent_data={"model_id": "gpt-4o"})
        ```
    """

    def __init__(
        self,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 10,
        num_workers: int = 1,
        fail_on_setup_error: bool = False,
        fail_on_task_error: bool = False,
        fail_on_evaluation_error: bool = False,
        progress_bar: bool | str = True,
        seed: Optional[int] = None,
        seed_generator=None,
    ):
        """Initialize the benchmark.

        Args:
            callbacks: Optional list of callbacks
            n_task_repeats: Number of times to repeat each task
            max_invocations: Maximum agent invocations per task
            num_workers: Number of parallel workers
            fail_on_setup_error: Raise on setup errors
            fail_on_task_error: Raise on task errors
            fail_on_evaluation_error: Raise on evaluation errors
            progress_bar: Progress bar configuration
            seed: Global seed for reproducible benchmark runs
            seed_generator: Custom seed generator (takes precedence over seed)
        """
        super().__init__(
            callbacks=callbacks,
            n_task_repeats=n_task_repeats,
            max_invocations=max_invocations,
            num_workers=num_workers,
            fail_on_setup_error=fail_on_setup_error,
            fail_on_task_error=fail_on_task_error,
            fail_on_evaluation_error=fail_on_evaluation_error,
            progress_bar=progress_bar,
            seed=seed,
            seed_generator=seed_generator,
        )

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Environment:
        """Create the MultiAgentBench environment.

        Args:
            agent_data: Agent configuration
            task: The task to set up
            seed_generator: Seed generator for reproducibility

        Returns:
            MultiAgentBenchEnvironment instance
        """
        return MultiAgentBenchEnvironment(
            task_data=task.environment_data,
        )

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Optional[User]:
        """MultiAgentBench tasks don't use user simulators.

        The multi-agent coordination replaces user interaction.

        Args:
            agent_data: Agent configuration
            environment: The environment instance
            task: The task
            seed_generator: Seed generator (unused)

        Returns:
            None
        """
        return None

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for the task (implement in subclass).

        Subclasses should:
        1. Read agent specifications from task.environment_data["agents"]
        2. Derive seeds from seed_generator for each agent's model
        3. Create agents using their framework with seeded models
        4. Wrap them in AgentAdapter
        5. Set up relationships from task.environment_data["relationships"]

        Args:
            agent_data: Agent configuration (model IDs, etc.)
            environment: The environment instance
            task: The task containing agent specs
            user: User simulator (None for MultiAgentBench)
            seed_generator: Seed generator for deriving deterministic seeds.
                Use `seed_generator.child("agents")` to create a namespace, then
                `derive_seed(agent_id)` for each agent's model. Returns None if
                seeding is disabled.

        Returns:
            Tuple of (agents_to_run, agents_dict)

        Example:
            ```python
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agents_gen = seed_generator.child("agents")

                for config in task.environment_data.get("agents", []):
                    agent_id = config.get("agent_id")
                    seed = agents_gen.derive_seed(agent_id)  # Returns None if seeding disabled
                    model = self.get_model_adapter(model_id, seed=seed)
                    # Create agent with seeded model...
            ```
        """
        pass

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        """Create evaluators for the task.

        Args:
            environment: The environment
            task: The task with evaluation data
            agents: The agents
            user: User simulator (None for MultiAgentBench)
            seed_generator: Seed generator for reproducibility

        Returns:
            List of evaluators
        """
        # Get evaluation model ID from task or default
        eval_model_id = task.evaluation_data.get("model_id", "gpt-4o-mini")

        # Derive seed for evaluator model (returns None if seeding disabled)
        evaluator_seed = seed_generator.derive_seed("evaluators/multiagentbench_evaluator")

        # Create model adapter for evaluation
        model_adapter = self.get_model_adapter(
            eval_model_id,
            register_name="evaluator_model",
            seed=evaluator_seed,
        )

        # Get domain-specific evaluation configuration
        domain = task.environment_data.get("scenario", "")
        metrics_config = task.evaluation_data.get("metrics", {})
        output_format = task.evaluation_data.get("output_format", "")

        return [
            MultiAgentBenchEvaluator(
                domain=domain,
                model_adapter=model_adapter,
                metrics_config=metrics_config,
                output_format=output_format,
            )
        ]

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Provide a model adapter (implement in subclass).

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments including register_name

        Returns:
            ModelAdapter instance
        """
        pass

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Dict[str, Any]:
        """Execute the multi-agent system.

        For MultiAgentBench, this runs all agents on the task and
        collects their outputs.

        Args:
            agents: Agents to run
            task: The task
            environment: The environment
            query: The query/task content

        Returns:
            Dict with agent_results, communications, and coordination_mode
        """
        results: List[Dict[str, Any]] = []
        communications: List[str] = []

        coordination_mode = task.environment_data.get("coordinate_mode", "cooperative")

        for agent in agents:
            result = agent.run(query)
            agent_id = getattr(agent, "agent_id", str(agent))

            results.append(
                {
                    "agent_id": agent_id,
                    "result": result,
                }
            )

            # Collect communication logs if available
            if hasattr(agent, "get_serialized_messages"):
                comm = agent.get_serialized_messages()  # type: ignore[operator]
                if comm:
                    communications.append(comm)

        return {
            "agent_results": results,
            "communications": communications,
            "coordination_mode": coordination_mode,
        }

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute evaluators on the results.

        Args:
            evaluators: The evaluators
            agents: Dict of all agents
            final_answer: The combined agent outputs
            traces: Execution traces

        Returns:
            List of evaluation results
        """
        results = []

        for evaluator in evaluators:
            # MultiAgentBenchEvaluator expects traces in a specific format
            result = evaluator(traces, final_answer)
            results.append(result)

        return results


class MarbleMultiAgentBenchBenchmark(MultiAgentBenchBenchmark):
    """MARBLE reproduction mode for MultiAgentBench.

    This benchmark uses MARBLE's native agents and engine for exact
    reproduction of published results. It wraps MARBLE components
    in MASEval adapters for unified tracing.

    Example:
        ```python
        from maseval.benchmark.multiagentbench import (
            MarbleMultiAgentBenchBenchmark,
            load_tasks,
            configure_model_ids,
        )

        class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
            def get_model_adapter(self, model_id, **kwargs):
                from maseval.interface.openai import OpenAIModelAdapter
                adapter = OpenAIModelAdapter(model_id)
                if "register_name" in kwargs:
                    self.register("models", kwargs["register_name"], adapter)
                return adapter

        tasks = load_tasks("research", limit=5)
        configure_model_ids(tasks, agent_model_id="gpt-4o")

        benchmark = MyMarbleBenchmark()
        results = benchmark.run(tasks, agent_data={})
        ```
    """

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create MARBLE agents wrapped in MASEval adapters.

        Note:
            MARBLE agents use their own internal LLM handling with a model ID string,
            not MASEval's ModelAdapter. This means seed_generator cannot be applied
            to agent LLM calls in this implementation. For reproducible agent behavior,
            use `MultiAgentBenchBenchmark` with a custom `setup_agents` that creates
            agents using seeded MASEval ModelAdapters.

        Args:
            agent_data: Agent configuration
            environment: The environment
            task: The task with agent specifications
            user: User simulator (None)
            seed_generator: Seed generator (not used for MARBLE agents,
                but seeding is applied to evaluators)

        Returns:
            Tuple of (agents_to_run, agents_dict)
        """
        from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
            create_marble_agents,
        )

        # Get agent configurations from task
        agent_configs = task.environment_data.get("agents", [])
        model_id = task.environment_data.get("llm", "gpt-4o-mini")

        # Get MARBLE environment from our wrapper
        marble_env = None
        if isinstance(environment, MultiAgentBenchEnvironment):
            marble_env = environment._marble_env

        # Create MARBLE environment if not available
        if marble_env is None:
            marble_env = self._create_marble_env(task)

        # Create agents using factory function
        agents_list, agents_dict = create_marble_agents(
            agent_configs=agent_configs,
            marble_env=marble_env,
            model=model_id,
        )

        # Set up agent graph for inter-agent communication
        self._setup_agent_graph(agents_dict, task, marble_env)

        # Register agents for tracing
        for agent_id, adapter in agents_dict.items():
            self.register("agents", agent_id, adapter)

        return agents_list, agents_dict  # type: ignore[return-value]

    def _create_marble_env(self, task: Task) -> Any:
        """Create a MARBLE environment for the task.

        Args:
            task: The task with environment configuration

        Returns:
            MARBLE environment instance
        """
        try:
            from .marble.marble.environments.base_env import BaseEnvironment  # type: ignore[unresolved-import]
        except ImportError as e:
            raise ImportError(MARBLE_IMPORT_ERROR.format(error=e)) from e

        env_config = task.environment_data.get("environment", {})
        task_config = task.environment_data.get("task", {})

        config = {
            "description": f"{task.environment_data.get('scenario', '')} environment",
            "task_description": task_config.get("content", "") if isinstance(task_config, dict) else str(task_config),
            "max_iterations": env_config.get("max_iterations") or task.environment_data.get("max_iterations", 10),
        }

        return BaseEnvironment(name=config["description"], config=config)

    def _setup_agent_graph(
        self,
        agents_dict: Dict[str, "MarbleAgentAdapter"],
        task: Task,
        marble_env: Any,
    ) -> None:
        """Set up MARBLE's AgentGraph for inter-agent communication.

        Args:
            agents_dict: Dict of MARBLE agent adapters
            task: Task with relationship data
            marble_env: MARBLE environment
        """
        try:
            from .marble.marble.graph.agent_graph import AgentGraph  # type: ignore[unresolved-import]
        except ImportError:
            # MARBLE not available, skip graph setup
            return

        # Extract MARBLE agents from adapters
        marble_agents = [adapter.marble_agent for adapter in agents_dict.values()]

        # Build config for AgentGraph (MARBLE expects an object with these attributes)
        relationships = task.environment_data.get("relationships", [])
        coordination_mode = task.environment_data.get("coordinate_mode", "cooperative")
        config = SimpleNamespace(coordination_mode=coordination_mode, relationships=relationships)

        try:
            # Create agent graph
            graph = AgentGraph(marble_agents, config)  # type: ignore

            # Set graph on all agents
            for agent in marble_agents:
                agent.set_agent_graph(graph)

        except (ValueError, KeyError, AttributeError) as e:
            # Graph creation failed, agents will work without inter-agent communication
            logger.warning("AgentGraph setup failed, agents will run without inter-agent communication: %s", e)

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        """Provide a model adapter (implement in subclass).

        Args:
            model_id: Model identifier
            **kwargs: Additional arguments

        Returns:
            ModelAdapter instance
        """
        pass
