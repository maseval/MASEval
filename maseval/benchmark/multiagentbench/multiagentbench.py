"""MultiAgentBench benchmark implementations.

This module provides benchmark classes for the MARBLE MultiAgentBench suite:
- MultiAgentBenchBenchmark: Abstract base for framework-agnostic evaluation
- MarbleMultiAgentBenchBenchmark: Exact MARBLE reproduction mode

Original Repository: https://github.com/ulab-uiuc/MARBLE
Fork Used: https://github.com/cemde/MARBLE (contains bug fixes for MASEval integration)
Code License: MIT

Citation:
    Zhu, et al. (2025). MultiAgentBench: Evaluating the Collaboration and Competition
    of LLM agents. arXiv:2503.01935.
"""

import json
import logging
from abc import abstractmethod
from pathlib import Path
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
    MarbleReproductionEvaluator,
    MultiAgentBenchEvaluator,
)

if TYPE_CHECKING:
    from maseval.benchmark.multiagentbench.adapters.marble_adapter import MarbleAgentAdapter

logger = logging.getLogger(__name__)


def _create_patched_marble_evaluator(metrics_config: Dict[str, Any]) -> Any:
    """Create a MARBLE Evaluator with corrected ``evaluator_prompts.json`` path.

    The original ``Evaluator.__init__`` (evaluator.py:41) uses a hardcoded
    relative path ``'evaluator/evaluator_prompts.json'``. This function creates
    a subclass that fixes the path while inheriting all ~600 lines of evaluation
    methods (``evaluate_communication``, ``evaluate_planning``, ``evaluate_kpi``,
    ``evaluate_task_research``, ``evaluate_task_world``, ``evaluate_task_db``,
    ``evaluate_code_quality``, and all ``parse_*`` methods) unchanged.

    Args:
        metrics_config: MARBLE metrics configuration dict. The ``evaluate_llm``
            key controls which model is used for LLM-based evaluation calls.
            Structure: ``{"evaluate_llm": {"model": "gpt-4o"}}``

    Returns:
        Instance of patched MARBLE Evaluator with all methods inherited.
    """
    ensure_marble_on_path()
    from marble.evaluator.evaluator import Evaluator as MarbleEvaluator  # type: ignore[import-untyped]
    from marble.utils.logger import get_logger  # type: ignore[import-untyped]

    class _Patched(MarbleEvaluator):
        def __init__(self, mc: Dict[str, Any]) -> None:
            # Replicate evaluator.py:22-45 with corrected prompts path.
            # We duplicate only these ~15 init lines; all ~600 lines of
            # evaluation methods are inherited from MarbleEvaluator.
            self.logger = get_logger(self.__class__.__name__)
            self.metrics_config = mc
            # Metrics dict structure from evaluator.py:31-40
            self.metrics: Dict[str, Any] = {
                "task_completion": [],
                "token_consumption": [],
                "planning_score": [],
                "communication_score": [],
                "task_evaluation": {},
                "total_milestones": 0,
                "agent_kpis": {},
                "code_quality": {},
            }
            # Fix: absolute path to vendored evaluator_prompts.json
            # (original uses relative 'evaluator/evaluator_prompts.json')
            prompts_path = Path(__file__).parent / "marble" / "marble" / "evaluator" / "evaluator_prompts.json"
            with open(prompts_path, "r", encoding="utf-8") as f:
                self.evaluation_prompts = json.load(f)
            # LLM model selection from evaluator.py:44-45
            evaluate_llm_config = self.metrics_config.get("evaluate_llm", {})
            # Default 'gpt-3.5-turbo' from evaluator.py:45
            self.llm = evaluate_llm_config.get("model", "gpt-3.5-turbo") if isinstance(evaluate_llm_config, dict) else evaluate_llm_config

        def evaluate_communication(self, task: str, communications: str) -> None:
            """Override evaluator.py:66-93 to fix prompt formatting bug.

            MARBLE's evaluator_prompts.json has a bug in the Graph.Communication
            prompt: ``{"rating": X}`` uses single braces, unlike the research
            and bargaining prompts which correctly use ``{{`` double braces.
            This causes ``str.format()`` to raise ``KeyError: '"rating"'``.

            We use ``.replace()`` instead of ``.format()`` to avoid this.
            All other logic is identical to evaluator.py:66-93.
            """
            from marble.llms.model_prompting import model_prompting  # type: ignore[import-untyped]

            communication_prompt_template = self.evaluation_prompts["Graph"]["Communication"]["prompt"]
            # Fix: .replace() instead of .format() — see docstring
            prompt = communication_prompt_template.replace("{task}", task).replace("{communications}", communications)
            result = model_prompting(
                llm_model=self.llm,
                messages=[{"role": "user", "content": prompt}],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]
            assert isinstance(result.content, str)
            score = self.parse_score(result.content)
            self.metrics["communication_score"].append(score)

    return _Patched(metrics_config)


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

    Warning:
        ``communication_score`` is only computed when agents use
        ``MarbleAgentAdapter``, which populates the ``communication_log`` trace
        key from ``BaseAgent.act()``. Custom ``setup_agents()`` implementations
        using other adapters must explicitly populate ``communication_log`` in
        each adapter's ``gather_traces()`` output for communication evaluation
        to work. See ``MultiAgentBenchEvaluator._extract_communications()`` for
        the expected format.

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
        # Get evaluation model ID from task or default.
        # Default matches MARBLE evaluator.py:45 (gpt-3.5-turbo).
        eval_model_id = task.evaluation_data.get("model_id", "gpt-3.5-turbo")

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

        coordination_mode = task.environment_data.get("coordinate_mode", "graph")

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
        # MARBLE Config.py:26 defaults llm to "" (empty string).
        # Require explicit model via configure_model_ids() rather than
        # silently substituting a default. Empty string will cause MARBLE's
        # model_prompting() to fail with a clear API error.
        model_id = task.environment_data.get("llm", "")

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

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        """Create a thin evaluator for MARBLE reproduction mode.

        All LLM-based evaluation happens inside the coordination loop via
        MARBLE's Evaluator (imported directly). The ``MarbleReproductionEvaluator``
        only reformats pre-computed metrics into MASEval's result format.

        No ``ModelAdapter`` is needed — evaluation LLM calls are handled by
        MARBLE's ``model_prompting()`` in the coordination loop.

        Args:
            environment: The environment
            task: The task with evaluation data
            agents: The agents
            user: User simulator (None for MultiAgentBench)
            seed_generator: Seed generator for reproducibility

        Returns:
            List containing a single MarbleReproductionEvaluator
        """
        domain = task.environment_data.get("scenario", "")
        return [MarbleReproductionEvaluator(domain=domain)]

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

        task_config = task.environment_data.get("task", {})

        config = {
            "description": f"{task.environment_data.get('scenario', '')} environment",
            "task_description": task_config.get("content", "") if isinstance(task_config, dict) else str(task_config),
            # Use the already-resolved max_iterations from data_loader (which
            # applies correct per-domain defaults from MARBLE YAML configs).
            "max_iterations": task.environment_data.get("max_iterations", 10),
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
        coordination_mode = task.environment_data.get("coordinate_mode", "graph")
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

        # Per-task iteration limit matching marble/engine/engine.py:97.
        # Converted from domain-specific MARBLE YAML configs by data_loader.
        self._marble_max_iterations: int = task.environment_data["max_iterations"]

        # Store task data for domain evaluation (DB needs labels, root_causes;
        # Coding needs workspace_dir). Matches engine.py:83 (self.config.task).
        self._task_data = task_data

        # Create MARBLE evaluator for in-loop evaluation (engine.py:82).
        # MARBLE: Evaluator(metrics_config=config.metrics)
        # metrics_config.evaluate_llm.model controls the LLM (evaluator.py:44-45).
        marble_metrics_config: Dict[str, Any] = dict(task.evaluation_data.get("metrics", {}))
        eval_model_id = task.evaluation_data.get("model_id")
        if eval_model_id:
            marble_metrics_config["evaluate_llm"] = {"model": eval_model_id}
        self._marble_evaluator: Any = _create_patched_marble_evaluator(marble_metrics_config)

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
        # Matches engine.py:223 iteration_data["summary"] = "" initialization
        iteration_summary = ""

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
        # engine.py:267: iteration_data["summary"] = summary.content
        iteration_summary = summary.content

        # Decide whether to continue (engine.py:270-289)
        if self._domain.lower() == "minecraft":
            continue_simulation = self._minecraft_should_continue()
        else:
            continue_simulation = planner.decide_next_step(agents_results)

        if continue_simulation:
            # engine.py:288 passes the planner return object (not raw string)
            planner.update_progress(summary)
            current_iteration += 1

        # Per-iteration evaluation: graph mode always -1 (engine.py:293-317)
        # Communication eval is COMMENTED OUT in MARBLE's graph_coordinate
        # (engine.py:297-299), always stores -1. Same for planning (engine.py:317).
        self._marble_evaluator.metrics["communication_score"].append(-1)
        self._marble_evaluator.metrics["planning_score"].append(-1)

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
            summary_from_planner = planner.summarize_output(summary, task_content, output_format)
            # engine.py:387: iteration_data["summary"] = summary_from_planner.content
            iteration_summary = summary_from_planner.content

            # Per-iteration evaluation: graph mode always -1 (engine.py:389-413)
            self._marble_evaluator.metrics["communication_score"].append(-1)
            self._marble_evaluator.metrics["planning_score"].append(-1)

            # Update latest results from this iteration
            latest_results = iteration_results

            # Decide whether to continue (engine.py:414-433)
            if self._domain.lower() == "minecraft":
                continue_simulation = self._minecraft_should_continue()
            else:
                continue_simulation = planner.decide_next_step(agents_results)

            if not continue_simulation:
                break

        # Domain-specific final evaluation (engine.py:451-484)
        # Uses last iteration's planner summary, matching MARBLE exactly.
        self._run_domain_evaluation(iteration_summary, mode="graph")

        return {
            "agent_results": latest_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
            "marble_evaluation": self._marble_evaluator.metrics,
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
        # Matches engine.py:511 iteration_data["summary"] = "" initialization
        iteration_summary = ""

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
            summary_from_planner = planner.summarize_output(summary, task_content, output_format)
            # engine.py:554: iteration_data["summary"] = summary_from_planner.content
            iteration_summary = summary_from_planner.content
            planner.update_progress(summary)
            current_iteration += 1

            # Per-iteration evaluation (engine.py:559-581)
            # Communication (engine.py:560-567)
            if communications:
                communications_str = self._format_communications(communications)
                self._marble_evaluator.evaluate_communication(task_content, communications_str)
            else:
                self._marble_evaluator.metrics["communication_score"].append(-1)
            # Planning (engine.py:570-580)
            agent_profiles = self._get_agent_profiles()
            agent_tasks_str = self._format_agent_tasks(tasks)
            results_str = self._format_results(agents_results)
            self._marble_evaluator.evaluate_planning(iteration_summary, agent_profiles, agent_tasks_str, results_str)
            # KPI (engine.py:581)
            self._marble_evaluator.evaluate_kpi(task_content, results_str)

            # Update latest results from this iteration
            latest_results = iteration_results

            # Decide whether to continue (engine.py:584-595)
            continue_simulation = planner.decide_next_step(agents_results)
            if not continue_simulation:
                break

        # Domain-specific final evaluation (engine.py:606-644)
        self._run_domain_evaluation(iteration_summary, mode="star")

        return {
            "agent_results": latest_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
            "marble_evaluation": self._marble_evaluator.metrics,
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
        # Matches engine.py:684 iteration_data["summary"] implicit "" initialization
        iteration_summary = ""

        all_results: List[Dict[str, Any]] = []
        agents_results_accumulated: List[Dict[str, Any]] = []
        communications: List[str] = []

        # No try/except around loop body — matching MARBLE (engine.py:680-764).
        # Errors propagate to MASEval's task error handling.
        while current_adapter and chain_length < max_chain_length:
            # Agent acts (engine.py:691)
            comm_count_before = len(current_adapter._communication_log)
            result = current_adapter.run(current_task)
            # engine.py:692: formatted result string for KPI evaluation
            result_str = f"AgentID: '{current_adapter.agent_id}' completed task with result: {result}"
            # engine.py:695: per-step task assignment for planning eval
            step_task_assignments = {current_adapter.agent_id: current_task}
            agents_results_accumulated.append({current_adapter.agent_id: result})
            all_results.append({"agent_id": current_adapter.agent_id, "result": result})

            # Capture per-step communication (engine.py:720)
            # Known MARBLE bug: engine.py:720 stores raw `communication` (None
            # or dict) into iteration_data["communications"], then asserts it's
            # a list at L725 → crashes if the agent actually communicated.
            # MASEval uses adapter._communication_log (always a list), which
            # avoids this crash. See PROVENANCE.md for details.
            step_communication: List[Any] = []
            if len(current_adapter._communication_log) > comm_count_before:
                comm_text = current_adapter._communication_log[-1].get("communication", "")
                if comm_text:
                    communications.append(comm_text)
                    step_communication.append(comm_text)

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

            # Per-step communication evaluation (engine.py:722-732)
            # Note: in chain mode, communication eval comes BEFORE summarize
            if step_communication:
                communications_str = self._format_communications(step_communication)
                self._marble_evaluator.evaluate_communication(task_content, communications_str)
            else:
                self._marble_evaluator.metrics["communication_score"].append(-1)

            # Summarize (engine.py:734-738)
            summary = self._summarize_results_marble(agents_results_accumulated)
            summary_from_planner = planner.summarize_output(summary, task_content, output_format)
            # engine.py:738: iteration_data["summary"] = summary_from_planner.content
            iteration_summary = summary_from_planner.content

            # Planning evaluation (engine.py:740-752)
            # Chain mode uses _get_agent_profiles() (all agents), per-step task_assignments,
            # and passes raw `result` (not formatted results_str) to evaluate_planning
            agent_profiles_all = self._get_agent_profiles()
            agent_tasks_str = self._format_agent_tasks(step_task_assignments)
            self._marble_evaluator.evaluate_planning(iteration_summary, agent_profiles_all, agent_tasks_str, result)
            # engine.py:752: KPI uses result_str (formatted with AgentID prefix)
            self._marble_evaluator.evaluate_kpi(task_content, result_str)

            # Decide whether to continue (engine.py:755-764)
            continue_simulation = planner.decide_next_step([{"root_agent": result}])
            if not continue_simulation:
                break

        # Post-loop update_progress (engine.py:766-768)
        summary = self._summarize_results_marble(agents_results_accumulated)
        planner.update_progress(summary)

        # Domain-specific final evaluation (engine.py:780-803)
        self._run_domain_evaluation(iteration_summary, mode="chain")

        return {
            "agent_results": all_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
            "marble_evaluation": self._marble_evaluator.metrics,
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
        # Matches engine.py:838 implicit "" initialization
        iteration_summary = ""

        root_marble_agent = graph.get_root_agent()
        if not root_marble_agent:
            logger.error("No root agent found in the tree.")
            return {"agent_results": [], "communications": [], "coordination_mode": self._coordinate_mode}

        all_results: List[Dict[str, Any]] = []
        communications: List[str] = []

        while current_iteration < self._marble_max_iterations:
            current_iteration += 1

            # Recursive execution from root (engine.py:843-845)
            results, communication, tasks_collected = self._execute_agent_task_recursive(root_marble_agent, task_content)

            all_results = results
            if communication:
                communications = [communication] if isinstance(communication, str) else communication

            # Summarize (engine.py:847-855)
            # tree_coordinate passes recursive results directly (not reformatted)
            summary = self._summarize_results_marble(results)
            summary = planner.summarize_output(summary, task_content, output_format)
            # engine.py:851: iteration_data["summary"] = summary.content
            iteration_summary = summary.content
            planner.update_progress(summary)

            # Per-iteration evaluation (engine.py:859-881)
            # Communication (engine.py:860-867)
            # Note: communication from _execute_agent_task_recursive is a string
            # or None. MARBLE passes it directly to _format_communications which
            # iterates over it (engine.py:861-863). We replicate this exactly.
            if communication:
                communications_str = self._format_communications(communication)
                self._marble_evaluator.evaluate_communication(task_content, communications_str)
            else:
                self._marble_evaluator.metrics["communication_score"].append(-1)
            # Planning (engine.py:870-880)
            # tasks_collected is a list from recursive exec; _format_agent_tasks
            # handles this via its except branch (engine.py:1147-1148).
            agent_profiles = self._get_agent_profiles()
            agent_tasks_str = self._format_agent_tasks(tasks_collected)
            results_str = self._format_results(results)
            self._marble_evaluator.evaluate_planning(iteration_summary, agent_profiles, agent_tasks_str, results_str)
            # KPI (engine.py:881)
            self._marble_evaluator.evaluate_kpi(task_content, results_str)

            # Decide whether to continue (engine.py:884)
            # tree_coordinate passes results directly to decide_next_step
            continue_simulation = planner.decide_next_step(results)
            if not continue_simulation:
                break

        # Domain-specific final evaluation (engine.py:902-940)
        self._run_domain_evaluation(iteration_summary, mode="tree")

        return {
            "agent_results": all_results,
            "communications": communications,
            "coordination_mode": self._coordinate_mode,
            "marble_evaluation": self._marble_evaluator.metrics,
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
        MARBLE reads ``../data/score.json`` relative to cwd. We resolve
        relative to vendored MARBLE root for consistent behavior.
        """
        from maseval.benchmark.multiagentbench._constants import _MARBLE_ROOT

        score_path = Path(_MARBLE_ROOT).parent / "data" / "score.json"
        try:
            with open(score_path, "r") as f:
                block_hit_rate = json.load(f)[-1]["block_hit_rate"]
        except Exception:
            block_hit_rate = 0.0
        return int(block_hit_rate) != 1

    # -- MARBLE Engine helper methods for evaluation --
    # Copied from marble/engine/engine.py:1119-1163. These format data
    # for MARBLE's evaluator methods (evaluate_communication, evaluate_planning,
    # evaluate_kpi) which are called per-iteration inside coordination loops.

    def _format_communications(self, communications: Any) -> str:
        """Format communications into a string for the evaluator.

        Copied from engine.py:1119-1124. Accepts any iterable — in star/chain
        mode this is a list of strings; in tree mode MARBLE passes the raw
        communication string from ``_execute_agent_task_recursive``
        (engine.py:856, 861).
        """
        # Assuming each communication is a string or can be converted to string
        return "\n".join(str(c) for c in communications)

    def _get_agent_profiles(self) -> str:
        """Retrieve and format agent profiles from the agent graph.

        Copied from engine.py:1126-1136.
        """
        agent_profiles = []
        for agent in self._marble_graph.get_all_agents():
            agent_profiles.append(f"Agent ID: {agent.agent_id}, Profile: {agent.profile}")
        return "\n".join(agent_profiles)

    def _format_agent_tasks(self, agent_tasks: Any) -> str:
        """Format agent task assignments into a string.

        Copied from engine.py:1138-1148. Accepts a dict (star mode) or a list
        (tree mode where ``_execute_agent_task_recursive`` returns a list of
        collected task dicts). The ``except`` branch handles the list case.
        """
        try:
            return "\n".join(f"Agent {agent_id}: Task: {task}" for agent_id, task in agent_tasks.items())
        except Exception:
            return "\n".join(json.dumps(item) for item in agent_tasks)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format task results into a string.

        Copied from engine.py:1150-1163.
        """
        results_str = []
        for result in results:
            if "agent_id" in result and "result" in result:
                agent_id = result["agent_id"]
                res_content = result["result"]
                results_str.append(f"AgentID: {agent_id}: Result: {res_content}")
            else:
                for agent_id, res_content in result.items():
                    results_str.append(f"Agent {agent_id}: Result: {res_content}")
        return "\n".join(results_str)

    def _read_code_from_file(self) -> str:
        """Read solution code from workspace for coding evaluation.

        Replicates engine.py:49-59. MARBLE reads from a hardcoded path
        ``MARBLE/marble/workspace/solution.py`` (engine.py:615, 911).
        Resolved relative to vendored MARBLE root so it works regardless
        of the current working directory.
        """
        from maseval.benchmark.multiagentbench._constants import _MARBLE_ROOT

        file_path = Path(_MARBLE_ROOT) / "marble" / "workspace" / "solution.py"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except IOError as e:
            logger.error(f"Failed to read code from {file_path}: {e}")
            return ""

    # Domain evaluation coverage per coordination mode, from engine.py.
    # Each mode only evaluates specific domains; others are skipped and their
    # evaluator.metrics fields stay at __init__ defaults (e.g. code_quality={}).
    _DOMAIN_EVAL_COVERAGE: Dict[str, frozenset] = {
        "graph": frozenset({"research", "bargaining", "database", "minecraft"}),  # engine.py:451-484
        "star": frozenset({"research", "coding", "bargaining", "database"}),  # engine.py:606-644
        "chain": frozenset({"research", "bargaining", "database"}),  # engine.py:780-803
        "tree": frozenset({"research", "coding", "bargaining", "database"}),  # engine.py:902-940
    }

    def _run_domain_evaluation(self, summary: str, mode: str) -> None:
        """Run domain-specific final evaluation via MARBLE evaluator.

        Replicates the domain dispatch at the end of each coordination method:
        engine.py:451-484 (graph), engine.py:606-644 (star),
        engine.py:780-803 (chain), engine.py:902-940 (tree).

        Each coordination mode only evaluates certain domains. For example,
        graph mode has no coding evaluation (engine.py:451-484 has no Coding
        branch), so ``code_quality`` stays ``{}``. This method skips domains
        not evaluated in the given mode, matching MARBLE exactly.

        Args:
            summary: The planner's summary string (``iteration_data["summary"]``
                in MARBLE, which is ``planner.summarize_output().content``).
            mode: Coordination mode (``"graph"``, ``"star"``, ``"chain"``,
                ``"tree"``).
        """
        domain = self._domain.lower()
        evaluator = self._marble_evaluator

        # Skip domains not evaluated in this coordination mode
        allowed = self._DOMAIN_EVAL_COVERAGE.get(mode, frozenset())
        if domain not in allowed:
            return

        if domain == "research":
            # engine.py:454, 607, 781, 903
            evaluator.evaluate_task_research(self._task_content, summary)
        elif domain == "bargaining":
            # engine.py:460, 628, 787, 924
            # MARBLE checks environment.name == "World Simulation Environment"
            evaluator.evaluate_task_world(self._task_content, summary)
        elif domain == "database":
            # engine.py:473-479, 634-640, 793-799, 930-936
            # MARBLE reads labels/root_causes from self.config.task
            task_data = self._task_data
            evaluator.evaluate_task_db(
                self._task_content,
                summary,
                task_data["labels"],
                task_data["number_of_labels_pred"],
                task_data["root_causes"],
            )
        elif domain == "coding":
            # engine.py:614-626, 910-921
            # MARBLE reads from workspace/solution.py
            code = self._read_code_from_file()
            if code:
                evaluator.evaluate_code_quality(task=self._task_content, code_result=code)
        elif domain == "minecraft":
            # engine.py:465-471
            # MARBLE reads ../data/score.json relative to cwd. We resolve
            # relative to vendored MARBLE root for consistent behavior.
            from maseval.benchmark.multiagentbench._constants import _MARBLE_ROOT

            score_path = Path(_MARBLE_ROOT).parent / "data" / "score.json"
            try:
                with open(score_path, "r") as f:
                    block_hit_rate = json.load(f)[-1]["block_hit_rate"]
            except Exception:
                block_hit_rate = 0.0
            evaluator.metrics["task_evaluation"] = block_hit_rate * 5

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
