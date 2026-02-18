"""MultiAgentBench benchmark implementations.

This module provides benchmark classes for the MARBLE MultiAgentBench suite:
- MultiAgentBenchBenchmark: Abstract base for framework-agnostic evaluation
- MarbleMultiAgentBenchBenchmark: Exact MARBLE reproduction mode
"""

import json
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

from maseval.benchmark.multiagentbench._constants import MARBLE_IMPORT_ERROR, ensure_marble_on_path
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
        seed_generator: Optional[SeedGenerator] = None,
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

    def execution_loop(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        user: Optional[User],
    ) -> Any:
        """Execute agents in a single pass.

        MultiAgentBench uses multi-agent coordination instead of user interaction.
        The base class ``execution_loop`` breaks after one call when ``user is None``,
        so this override makes the single-pass behavior explicit.

        Subclasses (e.g. ``MarbleMultiAgentBenchBenchmark``) override this with
        multi-iteration coordination loops matching their framework's orchestration.
        """
        return self.run_agents(agents, task, environment, task.query)

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

        Also creates MARBLE's orchestration components (``EnginePlanner``,
        ``SharedMemory``, ``AgentGraph``) needed by ``execution_loop`` to
        replicate MARBLE's multi-iteration coordination.

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

        # Set up agent graph for inter-agent communication (stores self._marble_graph)
        self._setup_agent_graph(agents_dict, task, marble_env)

        # Create MARBLE orchestration components for execution_loop
        self._setup_orchestration(task, model_id)

        # Store agents dict for ID-based lookup in coordination modes
        self._agents_dict: Dict[str, "MarbleAgentAdapter"] = agents_dict  # type: ignore[assignment]
        self._marble_env_instance = marble_env

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
        ensure_marble_on_path()
        try:
            from marble.environments.base_env import BaseEnvironment  # type: ignore[import-untyped]
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

        Stores the graph as ``self._marble_graph`` for use by coordination modes.

        Args:
            agents_dict: Dict of MARBLE agent adapters
            task: Task with relationship data
            marble_env: MARBLE environment
        """
        ensure_marble_on_path()
        try:
            from marble.graph.agent_graph import AgentGraph  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(MARBLE_IMPORT_ERROR.format(error=e)) from e

        # Extract MARBLE agents from adapters
        marble_agents = [adapter.marble_agent for adapter in agents_dict.values()]

        # Build config for AgentGraph (MARBLE expects an object with these attributes)
        relationships = task.environment_data.get("relationships", [])
        coordination_mode = task.environment_data.get("coordinate_mode", "cooperative")
        config = SimpleNamespace(coordination_mode=coordination_mode, relationships=relationships)

        # Create agent graph
        graph = AgentGraph(marble_agents, config)  # type: ignore

        # Set graph on all agents
        for agent in marble_agents:
            agent.set_agent_graph(graph)

        self._marble_graph = graph

    def _setup_orchestration(self, task: Task, model_id: str) -> None:
        """Create MARBLE's EnginePlanner and SharedMemory for the execution loop.

        These components are imported directly from the vendored MARBLE package.
        See ``marble/engine/engine.py:61-98`` for the original Engine.__init__.

        Args:
            task: The task with orchestration configuration
            model_id: LLM model identifier for the planner
        """
        ensure_marble_on_path()
        try:
            from marble.engine.engine_planner import EnginePlanner  # type: ignore[import-untyped]
            from marble.memory.base_memory import BaseMemory  # type: ignore[import-untyped]
            from marble.memory.shared_memory import SharedMemory  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(MARBLE_IMPORT_ERROR.format(error=e)) from e

        # Initialize memory matching marble/engine/engine.py:179-198
        memory_config = task.environment_data.get("memory", {})
        memory_type = memory_config.get("type", "SharedMemory")
        if memory_type == "SharedMemory":
            self._marble_memory: Any = SharedMemory()
        else:
            self._marble_memory: Any = BaseMemory()

        # Extract task metadata for the planner
        engine_planner_config = task.environment_data.get("engine_planner", {})
        task_content = task.query
        task_data = task.environment_data.get("task", {})
        output_format = (
            task_data.get("output_format", "You are free to define your own output format to answer the task properly.")
            if isinstance(task_data, dict)
            else "You are free to define your own output format to answer the task properly."
        )

        # Create engine planner (marble/engine/engine.py:89-96)
        self._marble_planner: Any = EnginePlanner(
            agent_graph=self._marble_graph,
            memory=self._marble_memory,
            config=engine_planner_config,
            task=task_content,
            model=model_id,
        )

        # Store task metadata for coordination methods
        self._task_content = task_content
        self._output_format = output_format
        self._coordinate_mode = task.environment_data.get("coordinate_mode", "graph")
        self._planning_method = engine_planner_config.get("planning_method", "naive")
        self._domain = task.environment_data.get("scenario", "")

        # Per-task iteration limit matching marble/engine/engine.py:97
        # MARBLE reads config.environment["max_iterations"] with default 10.
        # task.environment_data["environment"] is always populated by data_loader.
        env_config = task.environment_data["environment"]
        self._marble_max_iterations: int = env_config.get("max_iterations", 10)

    def execution_loop(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        user: Optional[User],
    ) -> Any:
        """Execute MARBLE's multi-iteration coordination loop.

        Dispatches to the appropriate coordination handler based on the task's
        ``coordinate_mode``. Replicates ``Engine.start()`` from
        ``marble/engine/engine.py:1034-1055``.

        Args:
            agents: MARBLE agents wrapped in MarbleAgentAdapter
            task: The task being solved
            environment: The environment
            user: Always None for MultiAgentBench

        Returns:
            Dict with agent_results, communications, and coordination_mode

        Raises:
            ValueError: If coordinate_mode is not supported
        """
        mode = self._coordinate_mode
        if mode == "star":
            return self._star_coordinate(agents)
        elif mode == "chain":
            return self._chain_coordinate(agents)
        elif mode == "tree":
            return self._tree_coordinate(agents)
        elif mode == "graph":
            return self._graph_coordinate(agents)
        else:
            raise ValueError(f"Unsupported coordinate mode: {mode}")

    # -- Coordination mode implementations --
    # Each replicates its counterpart in marble/engine/engine.py.

    def _graph_coordinate(self, agents: Sequence[AgentAdapter]) -> Dict[str, Any]:
        """Graph-based coordination. Replicates ``Engine.graph_coordinate()``
        (marble/engine/engine.py:200-492).

        Flow:
        1. Initial assignment — all agents receive the task
        2. Summarize → decide → update progress
        3. Iteration loop — each agent plans its own next task via ``plan_task()``
        """
        planner = self._marble_planner
        task_content = self._task_content
        output_format = self._output_format
        current_iteration = 0

        # -- Initial assignment (engine.py:211-260) --
        agents_results: List[Dict[str, Any]] = []
        latest_results: List[Dict[str, Any]] = []
        communications: List[str] = []

        for adapter in agents:
            try:
                comm_count_before = len(adapter._communication_log)  # type: ignore[union-attr]
                result = adapter.run(task_content)
                agents_results.append({adapter.agent_id: result})  # type: ignore[union-attr]
                latest_results.append({"agent_id": adapter.agent_id, "result": result})  # type: ignore[union-attr]
                # Only capture NEW communication (avoid stale re-capture)
                if len(adapter._communication_log) > comm_count_before:  # type: ignore[union-attr]
                    comm_text = adapter._communication_log[-1].get("communication", "")  # type: ignore[union-attr]
                    if comm_text:
                        communications.append(comm_text)
            except Exception as e:
                logger.error(f"Error executing initial task for agent '{getattr(adapter, 'agent_id', adapter)}': {e}")

        # Summarize (engine.py:262-267)
        # engine.py:264 overwrites summary with planner return
        summary = self._summarize_results_marble(agents_results)
        summary = planner.summarize_output(summary, task_content, output_format)

        # Decide whether to continue (engine.py:270-289)
        if self._domain.lower() == "minecraft":
            continue_simulation = self._minecraft_should_continue()
        else:
            continue_simulation = planner.decide_next_step(agents_results)

        if continue_simulation:
            # engine.py:288 passes the planner return object (not raw string)
            planner.update_progress(summary)
            current_iteration += 1

        # -- Iteration loop (engine.py:323-433) --
        # Use end_on_iter_0 flag matching engine.py:319-322
        end_on_iter_0 = not continue_simulation

        while current_iteration < self._marble_max_iterations and not end_on_iter_0:
            agents_results = []
            iteration_results: List[Dict[str, Any]] = []
            communications = []

            for adapter in agents:
                try:
                    # Each agent plans its own task (engine.py:344)
                    planned_task = adapter.marble_agent.plan_task()  # type: ignore[union-attr]
                    # Agent acts on planned task (engine.py:356)
                    comm_count_before = len(adapter._communication_log)  # type: ignore[union-attr]
                    result = adapter.run(planned_task)
                    agents_results.append({adapter.agent_id: result})  # type: ignore[union-attr]
                    iteration_results.append({"agent_id": adapter.agent_id, "result": result})  # type: ignore[union-attr]
                    # Only capture NEW communication (avoid stale re-capture)
                    if len(adapter._communication_log) > comm_count_before:  # type: ignore[union-attr]
                        comm_text = adapter._communication_log[-1].get("communication", "")  # type: ignore[union-attr]
                        if comm_text:
                            communications.append(comm_text)
                except Exception as e:
                    logger.error(f"Error in agent '{getattr(adapter, 'agent_id', adapter)}' during planning or action: {e}")

            # Summarize (engine.py:379-387)
            summary = self._summarize_results_marble(agents_results)
            current_iteration += 1
            planner.summarize_output(summary, task_content, output_format)

            # Update latest results from this iteration
            latest_results = iteration_results

            # Decide whether to continue (engine.py:414-433)
            if self._domain.lower() == "minecraft":
                continue_simulation = self._minecraft_should_continue()
            else:
                continue_simulation = planner.decide_next_step(agents_results)

            if not continue_simulation:
                break

        return {
            "agent_results": latest_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
        }

    def _star_coordinate(self, agents: Sequence[AgentAdapter]) -> Dict[str, Any]:
        """Centralized coordination. Replicates ``Engine.star_coordinate()``
        (marble/engine/engine.py:494-653).

        Flow: planner assigns tasks → agents execute → summarize → decide.
        """
        planner = self._marble_planner
        task_content = self._task_content
        output_format = self._output_format
        current_iteration = 0

        latest_results: List[Dict[str, Any]] = []
        iteration_results: List[Dict[str, Any]] = []
        communications: List[str] = []

        while current_iteration < self._marble_max_iterations:
            # Planner assigns tasks (engine.py:519-523)
            assignment = planner.assign_tasks(planning_method=self._planning_method)
            tasks = assignment.get("tasks", {})

            agents_results: List[Dict[str, Any]] = []
            iteration_results = []
            communications = []

            for agent_id, agent_task in tasks.items():
                adapter = self._agents_dict.get(agent_id)
                if adapter is None:
                    logger.error(f"Agent '{agent_id}' not found in agents dict.")
                    continue
                try:
                    comm_count_before = len(adapter._communication_log)
                    result = adapter.run(agent_task)
                    agents_results.append({agent_id: result})
                    iteration_results.append({"agent_id": agent_id, "result": result})
                    # Only capture NEW communication (avoid stale re-capture)
                    if len(adapter._communication_log) > comm_count_before:
                        comm_text = adapter._communication_log[-1].get("communication", "")
                        if comm_text:
                            communications.append(comm_text)
                except Exception as e:
                    logger.error(f"Error executing task for agent '{agent_id}': {e}")

            # Summarize and update progress (engine.py:550-557)
            # star mode passes raw summary string to update_progress (engine.py:556)
            summary = self._summarize_results_marble(agents_results)
            planner.summarize_output(summary, task_content, output_format)
            planner.update_progress(summary)
            current_iteration += 1

            # Update latest results from this iteration
            latest_results = iteration_results

            # Decide whether to continue (engine.py:584-595)
            continue_simulation = planner.decide_next_step(agents_results)
            if not continue_simulation:
                break

        return {
            "agent_results": latest_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
        }

    def _chain_coordinate(self, agents: Sequence[AgentAdapter]) -> Dict[str, Any]:
        """Chain-based coordination. Replicates ``Engine.chain_coordinate()``
        (marble/engine/engine.py:655-813).

        Flow: sequential agent chain, each agent picks the next.
        """
        planner = self._marble_planner
        task_content = self._task_content
        output_format = self._output_format
        graph = self._marble_graph

        # Select initial agent (engine.py:1015-1032: starts with "agent1")
        # MARBLE aborts chain if starting agent not found (engine.py:668-670)
        starting_agent_id = "agent1"
        current_adapter = self._agents_dict.get(starting_agent_id)
        if current_adapter is None:
            logger.error(f"Starting agent '{starting_agent_id}' not found for chain coordination.")
            return {"agent_results": [], "communications": [], "coordination_mode": self._coordinate_mode}

        max_chain_length = self._marble_max_iterations * len(self._agents_dict)
        chain_length = 0
        current_task = task_content

        all_results: List[Dict[str, Any]] = []
        agents_results_accumulated: List[Dict[str, Any]] = []
        communications: List[str] = []

        # No try/except around loop body — matching MARBLE (engine.py:680-764).
        # Errors propagate to MASEval's task error handling.
        while current_adapter and chain_length < max_chain_length:
            # Agent acts (engine.py:691)
            comm_count_before = len(current_adapter._communication_log)
            result = current_adapter.run(current_task)
            agents_results_accumulated.append({current_adapter.agent_id: result})
            all_results.append({"agent_id": current_adapter.agent_id, "result": result})

            # Only capture NEW communication (engine.py:720)
            if len(current_adapter._communication_log) > comm_count_before:
                comm_text = current_adapter._communication_log[-1].get("communication", "")
                if comm_text:
                    communications.append(comm_text)

            # Current agent chooses next agent (engine.py:702-717)
            agent_profiles = graph.get_agent_profiles_linked(current_adapter.agent_id)
            next_agent_id, plan = current_adapter.marble_agent.plan_next_agent(result, agent_profiles)

            # engine.py:709-717: fall back to current agent if next not found
            prev_adapter = current_adapter
            try:
                next_adapter = self._agents_dict.get(next_agent_id) if next_agent_id else None
            except Exception:
                next_adapter = None
            current_adapter = next_adapter if next_adapter is not None else prev_adapter
            # engine.py:717: always update task to the plan
            current_task = plan

            chain_length += 1
            planner.update_progress(result)

            # Summarize (engine.py:734-738)
            summary = self._summarize_results_marble(agents_results_accumulated)
            planner.summarize_output(summary, task_content, output_format)

            # Decide whether to continue (engine.py:755-764)
            continue_simulation = planner.decide_next_step([{"root_agent": result}])
            if not continue_simulation:
                break

        # Post-loop update_progress (engine.py:766-768)
        summary = self._summarize_results_marble(agents_results_accumulated)
        planner.update_progress(summary)

        return {
            "agent_results": all_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
        }

    def _tree_coordinate(self, agents: Sequence[AgentAdapter]) -> Dict[str, Any]:
        """Tree-based coordination. Replicates ``Engine.tree_coordinate()``
        (marble/engine/engine.py:815-949).

        Flow: recursive execution from root agent through children.
        """
        planner = self._marble_planner
        task_content = self._task_content
        output_format = self._output_format
        graph = self._marble_graph
        current_iteration = 0

        root_marble_agent = graph.get_root_agent()
        if not root_marble_agent:
            logger.error("No root agent found in the tree.")
            return {"agent_results": [], "communications": [], "coordination_mode": self._coordinate_mode}

        all_results: List[Dict[str, Any]] = []
        communications: List[str] = []

        while current_iteration < self._marble_max_iterations:
            current_iteration += 1

            # Recursive execution from root (engine.py:843-845)
            results, communication, _tasks = self._execute_agent_task_recursive(root_marble_agent, task_content)

            all_results = results
            if communication:
                communications = [communication] if isinstance(communication, str) else communication

            # Summarize (engine.py:847-855)
            # tree_coordinate passes recursive results directly (not reformatted)
            summary = self._summarize_results_marble(results)
            summary = planner.summarize_output(summary, task_content, output_format)
            planner.update_progress(summary)

            # Decide whether to continue (engine.py:884)
            # tree_coordinate passes results directly to decide_next_step
            continue_simulation = planner.decide_next_step(results)
            if not continue_simulation:
                break

        return {
            "agent_results": all_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
        }

    def _execute_agent_task_recursive(self, marble_agent: Any, task: str) -> Tuple[List[Dict[str, Any]], Optional[str], List[Any]]:
        """Recursively execute tasks in a tree structure.

        Replicates ``Engine._execute_agent_task_recursive()``
        (marble/engine/engine.py:951-1013).

        Args:
            marble_agent: MARBLE BaseAgent (not adapter)
            task: Task string to execute

        Returns:
            Tuple of (results_list, communications_string, tasks_list)
        """
        tasks_collected: List[Any] = []
        agent_id = marble_agent.agent_id

        # Find the MASEval adapter for this MARBLE agent
        adapter = self._agents_dict.get(agent_id)

        if hasattr(marble_agent, "children") and len(marble_agent.children) > 0:
            # Agent assigns tasks to children (engine.py:968)
            tasks_for_children = marble_agent.plan_tasks_for_children(task)
            tasks_collected.append(tasks_for_children)

            children_results: List[Dict[str, Any]] = []
            communications_list: List[str] = []

            for child in marble_agent.children:
                child_task = tasks_for_children.get(child.agent_id, "")
                if child_task:
                    child_results, child_comm, child_tasks = self._execute_agent_task_recursive(child, child_task)
                    tasks_collected += child_tasks
                    if child_comm:
                        communications_list.append(child_comm)
                    children_results += child_results

            # Parent acts with children's results (engine.py:985-995)
            results_str = "\n".join(json.dumps(r)[:500] for r in children_results)
            task_for_parent = (
                task
                + "\nHere are the results of the children: "
                + results_str
                + "\nPlease don't repeat the same task and continue to work on the original task. "
                "You may also need to communicate with other agents or summarize the results "
                "or just continue to work on the original task."
            )

            # engine.py:995-998: agent acts and communication is captured
            if adapter is not None:
                comm_count_before = len(adapter._communication_log)
                own_result = adapter.run(task_for_parent)
                # Only capture NEW communication (avoid stale re-capture)
                if len(adapter._communication_log) > comm_count_before:
                    last_comm = adapter._communication_log[-1].get("communication", "")
                    if last_comm:
                        communications_list.append(last_comm)
            else:
                own_result, communication = marble_agent.act(task_for_parent)
                if communication:
                    communications_list.append(communication)

            communications_str = "\n".join(communications_list) if communications_list else None

            results = [{"agent_id": agent_id, "result": own_result}] + children_results
            return results, communications_str, tasks_collected
        else:
            # Leaf agent acts directly (engine.py:1007-1013)
            if adapter is not None:
                comm_count_before = len(adapter._communication_log)
                result = adapter.run(task)
                # Only capture NEW communication (avoid stale re-capture)
                communication = None
                if len(adapter._communication_log) > comm_count_before:
                    communication = adapter._communication_log[-1].get("communication", "")
            else:
                result, communication = marble_agent.act(task)

            return [{"agent_id": agent_id, "result": result}], communication, tasks_collected

    # -- Helpers --

    @staticmethod
    def _summarize_results_marble(agents_results: List[Dict[str, Any]]) -> str:
        """Format agent results for the MARBLE planner.

        Replicates ``Engine._summarize_results()``
        (marble/engine/engine.py:1069-1088).
        """
        summary = "Agents' Results Summary:\n"
        for result in agents_results:
            shorten_result = f"- {result}"
            shorten_result = shorten_result[:1000]
            summary += f"{shorten_result}\n"
        return summary

    @staticmethod
    def _minecraft_should_continue() -> bool:
        """Check Minecraft block_hit_rate for termination.

        Replicates the Minecraft-specific termination logic from
        ``Engine.graph_coordinate()`` (marble/engine/engine.py:270-279).
        """
        try:
            with open("../data/score.json", "r") as f:
                block_hit_rate = json.load(f)[-1]["block_hit_rate"]
        except Exception:
            block_hit_rate = 0.0
        return int(block_hit_rate) != 1

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
