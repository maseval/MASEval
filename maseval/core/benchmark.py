from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable, Optional, Sequence, Tuple, Union, cast
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import warnings
import traceback
import logging

from .evaluator import Evaluator
from .task import Task, BaseTaskQueue, SequentialTaskQueue
from .environment import Environment
from .agent import AgentAdapter
from .model import ModelAdapter
from .callback_handler import CallbackHandler
from .callback import BenchmarkCallback
from .user import User
from .tracing import TraceableMixin
from .registry import ComponentRegistry, RegisterableComponent
from .context import TaskContext
from .utils.system_info import gather_benchmark_config
from .callbacks.progress_bar import (
    ProgressBarCallback,
    TqdmProgressBarCallback,
    RichProgressBarCallback,
)
from .exceptions import AgentError, EnvironmentError, UserError, TaskTimeoutError
from .seeding import SeedGenerator, DefaultSeedGenerator


class TaskExecutionStatus(Enum):
    """Status of task execution and evaluation.

    This enum tracks the execution state of a task through the benchmark lifecycle,
    enabling graceful failure handling and comprehensive result reporting.

    The status distinguishes between errors caused by the agent (agent's fault) and
    errors caused by the evaluation infrastructure (environment, user simulator).
    This enables fair scoring by excluding infrastructure failures.

    Attributes:
        SUCCESS: Task executed and evaluated successfully.
        AGENT_ERROR: Agent violated contract at a boundary (agent's fault, counts against score).
        ENVIRONMENT_ERROR: Environment/tool infrastructure failed (not agent's fault, exclude from scoring).
        USER_ERROR: User simulator failed (not agent's fault, exclude from scoring).
        TASK_TIMEOUT: Task execution exceeded configured timeout (resource constraint).
        UNKNOWN_EXECUTION_ERROR: Unclassified execution error (e.g., agent framework internal failure).
        EVALUATION_FAILED: Task executed but evaluation raised an exception.
        SETUP_FAILED: Setup phase (environment, agents, evaluators) raised an exception.

    Scoring Guidance:
        - Include in agent score: SUCCESS, AGENT_ERROR
        - Exclude from agent score: ENVIRONMENT_ERROR, USER_ERROR, TASK_TIMEOUT, UNKNOWN_EXECUTION_ERROR
        - Handle separately: EVALUATION_FAILED, SETUP_FAILED
    """

    SUCCESS = "success"
    AGENT_ERROR = "agent_error"
    ENVIRONMENT_ERROR = "environment_error"
    USER_ERROR = "user_error"
    TASK_TIMEOUT = "task_timeout"
    UNKNOWN_EXECUTION_ERROR = "unknown_execution_error"
    EVALUATION_FAILED = "evaluation_failed"
    SETUP_FAILED = "setup_failed"

    # Deprecated: kept for backward compatibility, use specific error types instead
    TASK_EXECUTION_FAILED = "task_execution_failed"


class Benchmark(ABC):
    """Abstract base class for orchestrating multi-agent system execution and evaluation.

    Benchmark provides a structured framework for running reproducible agent experiments across
    a collection of tasks. It manages the complete execution lifecycle: environment initialization,
    agent instantiation, task execution, and performance evaluation. The class enforces a
    three-stage pattern (setup, run, evaluate) while handling task iteration, callback orchestration,
    message history collection, and result aggregation.

    How to use:
        1. **Subclass Benchmark** and implement the abstract methods to define your benchmark logic
        2. **Implement setup methods** to specify how environments, agents, and evaluators are created
        3. **Implement run_agents** to define how your multi-agent system solves a single task
        4. **Instantiate your benchmark** with agent configuration
        5. **Call benchmark.run(tasks)** to execute the complete benchmark

        Example workflow:
            ```python
            class MyBenchmark(Benchmark):
                def setup_environment(self, agent_data, task):
                    return MyEnvironment(task.environment_data)

                def setup_agents(self, agent_data, environment, task, user):
                    agent = MyAgent(model=agent_data["model"])
                    agent_adapter = AgentAdapter(agent, "agent")
                    return [agent_adapter], {"agent": agent_adapter}

                def run_agents(self, agents, task, environment, query):
                    return agents[0].run(query)

                # ... implement other abstract methods

            # Run the benchmark
            config = {"model": "gpt-4", "temperature": 0.7}
            benchmark = MyBenchmark()
            reports = benchmark.run(tasks=my_tasks, agent_data=config)

            # Retry failed tasks elegantly (graceful task failure handling by default)
            failed_tasks = benchmark.get_failed_tasks()
            if len(failed_tasks) > 0:
                retry_reports = benchmark.run(tasks=failed_tasks, agent_data=config)

            # Parallel execution for I/O-bound workloads
            benchmark = MyBenchmark(num_workers=4)
            reports = benchmark.run(tasks=my_tasks, agent_data=config)

            # Or use strict mode for debugging (fail fast)
            benchmark = MyBenchmark(
                fail_on_task_error=True,
                fail_on_evaluation_error=True,
                fail_on_setup_error=True
            )
            reports = benchmark.run(tasks=my_tasks, agent_data=config)
            ```

        The framework handles task iteration, repetitions for statistical robustness, callback
        notifications, and result collection. You focus on defining agent behavior and evaluation
        criteria for your specific domain.

    Note:
        Task timeout (via ``TaskProtocol.timeout_seconds``) is checked at phase boundaries
        (after setup, before execution, before evaluation). Timeout detection during agent
        execution is not yet supported.
    """

    def __init__(
        self,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 1,
        num_workers: int = 1,
        fail_on_setup_error: bool = False,
        fail_on_task_error: bool = False,
        fail_on_evaluation_error: bool = False,
        progress_bar: bool | str = True,
        seed: Optional[int] = None,
        seed_generator: Optional[SeedGenerator] = None,
    ):
        """Initialize a benchmark with execution configuration.

        Args:
            callbacks: Optional list of callback handlers for monitoring execution, tracing messages,
                or collecting custom metrics during the benchmark run.
            n_task_repeats: Number of times to repeat each task. Useful for measuring variance in
                stochastic agent behaviors. Must be at least 1.
            max_invocations: Maximum number of agent invocations per task in the execution loop.
                For simple benchmarks, the default (1) means agents run once per task. For interactive
                benchmarks with user feedback loops, set higher (e.g., 5 for MACS) to allow multiple
                agent-user interaction rounds.
            num_workers: Number of parallel task executions. Default 1 (sequential).
                Set higher for I/O-bound workloads (e.g., LLM API calls). This controls the
                ThreadPoolExecutor worker count for concurrent task processing.
            fail_on_setup_error: If True, raise exceptions when setup fails (environment, agents, evaluators).
                If False (default), catch exceptions during setup and record them in the report with status
                SETUP_FAILED. This allows the benchmark to continue running remaining tasks even if setup fails.
            fail_on_task_error: If True, raise exceptions when task execution fails. If False (default),
                catch exceptions during task execution and record them in the report with status
                TASK_EXECUTION_FAILED. This allows the benchmark to continue running remaining tasks.
            fail_on_evaluation_error: If True, raise exceptions when evaluation fails. If False (default),
                catch exceptions during evaluation and record them in the report with status
                EVALUATION_FAILED. This allows the benchmark to continue even if evaluation logic has errors.
            progress_bar: Controls progress bar during benchmark runs. When enabled, one of the
                ProgressBarCallback will be added. Options:
                - True (default): Automatically adds a TqdmProgressBarCallback if no ProgressBarCallback
                  is already present in the `callbacks` argument.
                - False: Does not automatically add a progress bar callback. Manually provided
                  ProgressBarCallback instances in the callbacks list will still work normally.
                - "tqdm": Automatically adds a TqdmProgressBarCallback (same as True)
                - "rich": Automatically adds a RichProgressBarCallback (uses the rich library)
            seed: Global seed for reproducible benchmark runs. When provided, a `DefaultSeedGenerator`
                is created and passed to setup methods, enabling deterministic seed derivation for
                all components (agents, tools, user simulators, etc.). Seeds cascade: changing the
                global seed changes all derived seeds.
            seed_generator: Custom seed generator for advanced use cases. If provided, takes
                precedence over the `seed` parameter. Use this to provide a custom `SeedGenerator`
                implementation (e.g., database-backed seeds, different hash algorithms).

        Raises:
            ValueError: If n_task_repeats is less than 1.
            ValueError: If both `seed` and `seed_generator` are provided.

        How to use:
            Configure execution settings at initialization:

            ```python
            # Sequential execution (default)
            benchmark = MyBenchmark()

            # Parallel execution for faster I/O-bound workloads
            benchmark = MyBenchmark(num_workers=4)

            # Strict mode - fail fast on any error (useful for debugging)
            benchmark = MyBenchmark(
                fail_on_task_error=True,
                fail_on_evaluation_error=True,
                fail_on_setup_error=True
            )

            # Progress bar configuration
            benchmark = MyBenchmark()  # Default: adds TqdmProgressBarCallback
            benchmark = MyBenchmark(progress_bar=True)  # Explicit: TqdmProgressBarCallback
            benchmark = MyBenchmark(progress_bar="rich")  # Uses RichProgressBarCallback
            benchmark = MyBenchmark(progress_bar=False)  # No automatic callback

            # Custom callbacks
            benchmark = MyBenchmark(callbacks=[MyCustomProgressBarCallback()])
            ```
        """
        # Initialize tasks to empty queue (will be set in run())
        self.tasks: BaseTaskQueue = SequentialTaskQueue([])

        self.callback_handler = CallbackHandler()
        self.callbacks = callbacks or []

        # Attach default progress bar if enabled and no progress bar in callbacks
        if progress_bar is not False:
            # Check if user already provided a progress bar callback
            has_progress_bar = any(isinstance(cb, ProgressBarCallback) for cb in self.callbacks)

            if not has_progress_bar:
                # Attach default progress bar based on type
                if progress_bar is True or progress_bar == "tqdm":
                    self.callbacks.append(TqdmProgressBarCallback())
                elif progress_bar == "rich":
                    self.callbacks.append(RichProgressBarCallback())
                else:
                    raise ValueError(f"Invalid progress_bar value: {progress_bar!r}. Must be True, False, 'tqdm', or 'rich'.")

        self.n_task_repeats = n_task_repeats
        if self.n_task_repeats < 1:
            raise ValueError("n_task_repeats must be at least 1")

        # Execution configuration
        self.max_invocations = max_invocations
        self.num_workers = num_workers

        # Failure handling configuration
        self.fail_on_task_error = fail_on_task_error
        self.fail_on_evaluation_error = fail_on_evaluation_error
        self.fail_on_setup_error = fail_on_setup_error

        # Seeding configuration - always create a generator (with global_seed=None if seeding disabled)
        if seed is not None and seed_generator is not None:
            raise ValueError("Cannot provide both `seed` and `seed_generator`. Use one or the other.")
        if seed_generator is not None:
            self._seed_generator: SeedGenerator = seed_generator
        else:
            # Create generator with seed (can be None to disable seeding)
            self._seed_generator = DefaultSeedGenerator(global_seed=seed)

        # Gather benchmark-level configuration (system, git, packages, etc.)
        self.benchmark_level_config = gather_benchmark_config()

        # Thread-safe component registry (replaces inline registry dicts)
        self._registry = ComponentRegistry(benchmark_config=self.benchmark_level_config)

        # Thread safety locks for parallel execution
        self._reports_lock = threading.Lock()
        self._callback_lock = threading.Lock()

        # Persistent benchmark-level reports (stored across all task repetitions)
        # Each report contains both traces and configs for a single task repetition
        self.reports: List[Dict[str, Any]] = []

    @property
    def seed_generator(self) -> SeedGenerator:
        """The seed generator for this benchmark.

        The seed generator is configured at benchmark initialization via the `seed`
        or `seed_generator` parameters. When `seed=None` (the default), the generator's
        `derive_seed()` method returns `None`, effectively disabling seeding while
        maintaining a uniform interface.

        Returns:
            The root `SeedGenerator` instance.
        """
        return self._seed_generator

    def register(self, category: str, name: str, component: RegisterableComponent) -> RegisterableComponent:
        """Register a component for comprehensive trace and configuration collection.

        All core MASEval components (AgentAdapter, ModelAdapter, Environment,
        User, LLMSimulator, BenchmarkCallback) inherit from TraceableMixin and/or
        ConfigurableMixin, and are automatically registered for both trace and
        configuration collection before evaluation.

        **Note:** Most components are **automatically registered** when returned from
        setup methods (`setup_environment`, `setup_user`, `setup_agents`). You only
        need to manually register additional components like models, simulators, or
        tools that aren't automatically captured.

        Args:
            category: Component category (e.g., "agents", "models", "tools", "simulators",
                     "callbacks", "user", "environment", "seeding"). Use plural form to match
                     the structure in collect_all_traces() and collect_all_configs().
            name: Unique identifier for this component within its category
            component: Any object inheriting from TraceableMixin and/or ConfigurableMixin

        Returns:
            The component (for chaining convenience)

        Raises:
            ValueError: If the component is already registered under a different name

        How to use:
            Most components are auto-registered. Manual registration is only needed
            for additional components:

            ```python
            def setup_agents(self, agent_data, environment, task, user):
                # Create model (needs manual registration)
                model = MyModelAdapter(...)
                self.register("models", "main_model", model)

                # Create agent (auto-registered when returned)
                agent = MyAgent(model=model)
                agent_adapter = AgentAdapter(agent, "agent1")

                # Environment and user are also auto-registered
                return [agent_adapter], {"agent1": agent_adapter}
            ```

            Traces and configs are automatically collected before evaluation via
            `collect_all_traces()` and `collect_all_configs()` which are called
            internally by the `run()` method.
        """
        return self._registry.register(category, name, component)

    def clear_registry(self) -> None:
        """Clear the component registry after a task repetition completes.

        This method is called automatically by `run()` after each task repetition
        to ensure components are not carried over between repetitions. The
        reports list persists across all repetitions for aggregated analysis.
        """
        self._registry.clear()

    def collect_all_traces(self) -> Dict[str, Any]:
        """Collect execution traces from all registered components for the current task repetition.

        This method is called automatically by `run()` after each task repetition completes
        and before evaluation begins. It gathers comprehensive traces from all registered
        components (agents, models, tools, simulators, callbacks, etc.) for that specific
        repetition. After collection, the registry is cleared for the next repetition.

        The collected traces are stored in `benchmark.reports` list along with configs
        for persistent access across all task repetitions.

        Output fields:

        - `metadata` - Collection timestamp and thread info
        - `agents` - Dict mapping agent names to their traces (messages, execution data)
        - `models` - Dict mapping model names to their traces (API calls, timing, errors)
        - `tools` - Dict mapping tool names to their traces (invocations, parameters)
        - `simulators` - Dict mapping simulator names to their traces (attempts, outcomes)
        - `callbacks` - Dict mapping callback names to their traces (custom data)
        - `environment` - Direct traces from the environment (not nested), or `None` if not present
        - `user` - Direct traces from the user simulator (not nested), or `None` if not present
        - `other` - Dict for any other registered components

        Returns:
            Structured dictionary containing execution traces from all registered components.

        How to use:
            This method is called automatically by `run()` after each task repetition:

            ```python
            # Automatic collection (recommended)
            results = benchmark.run()

            # Access all collected reports (traces + configs) across repetitions
            for report in benchmark.reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}")
                # Agents is a dict: agent_name -> traces
                print(f"Agent messages: {report['traces']['agents']['my_agent']}")
                # Environment and user are direct (not nested)
                print(f"Environment state: {report['traces']['environment']}")
                print(f"User interactions: {report['traces']['user']}")
            ```

            The collected traces are passed to the evaluator's `evaluate()` method
            and stored in `benchmark.reports` for later analysis.
        """
        return self._registry.collect_traces()

    def collect_all_configs(self) -> Dict[str, Any]:
        """Collect configuration from all registered components for the current task repetition.

        This method is called automatically by `run()` after each task repetition completes
        and before evaluation begins. It gathers comprehensive configuration from all registered
        components (agents, models, tools, simulators, callbacks, etc.) for that specific
        repetition. After collection, the registry is cleared for the next repetition.

        The collected configs are stored in `benchmark.reports` list along with traces
        for persistent access across all task repetitions.

        Output fields:

        - `metadata` - Collection timestamp and thread info
        - `agents` - Dict mapping agent names to their config (settings, parameters)
        - `models` - Dict mapping model names to their config (model IDs, parameters)
        - `tools` - Dict mapping tool names to their config (specifications, settings)
        - `simulators` - Dict mapping simulator names to their config (parameters, templates)
        - `callbacks` - Dict mapping callback names to their config (settings)
        - `environment` - Direct config from the environment (not nested), or `None` if not present
        - `user` - Direct config from the user simulator (not nested), or `None` if not present
        - `other` - Dict for any other registered components
        - `benchmark` - Benchmark-level configuration (git, system, packages)

        Returns:
            Structured dictionary containing configuration from all registered components.

        How to use:
            This method is called automatically by `run()` after each task repetition:

            ```python
            # Automatic collection (recommended)
            results = benchmark.run()

            # Access all collected reports (traces + configs) across repetitions
            for report in benchmark.reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}")
                # Agents is a dict: agent_name -> config
                print(f"Agent config: {report['config']['agents']['my_agent']}")
                # Environment and user are direct (not nested)
                print(f"Environment config: {report['config']['environment']}")
                print(f"User config: {report['config']['user']}")
                # Benchmark-level config
                print(f"Git commit: {report['config']['benchmark']['git']['commit_hash']}")
            ```

            The collected configs are available in the results for reproducibility analysis.
        """
        return self._registry.collect_configs()

    def _invoke_callbacks(self, method_name: str, *args, suppress_errors: bool = True, **kwargs) -> List[Exception]:
        """Invoke a callback method on all registered callbacks (thread-safe).

        This method serializes all callback invocations using an internal lock,
        so users don't need to implement thread-safe callbacks.

        Callback errors are caught and logged by default to prevent one failing
        callback from disrupting the entire benchmark run. This is especially
        important in parallel execution where callback failures could otherwise
        cause difficult-to-debug issues.

        Args:
            method_name: Name of the callback method to invoke (e.g., "on_task_start").
            *args: Positional arguments to pass to the callback method.
            suppress_errors: If True (default), catch and log callback errors instead
                of propagating them. If False, first callback error will be raised.
            **kwargs: Keyword arguments to pass to the callback method.

        Returns:
            List of exceptions that occurred during callback invocation (empty if none).

        Raises:
            Exception: First callback exception if suppress_errors=False.
        """
        errors: List[Exception] = []
        logger = logging.getLogger(__name__)

        with self._callback_lock:
            for cb in self.callbacks:
                method = getattr(cb, method_name, None)
                if method is not None:
                    try:
                        method(*args, **kwargs)
                    except Exception as e:
                        if not suppress_errors:
                            raise

                        # Log error with full context
                        logger.error(
                            f"Callback {cb.__class__.__name__}.{method_name}() failed: {e}",
                            exc_info=True,
                        )
                        errors.append(e)

        return errors

    def _append_report_safe(self, report: Dict[str, Any]) -> None:
        """Append a report to the reports list (thread-safe).

        Args:
            report: The report dictionary to append.
        """
        with self._reports_lock:
            self.reports.append(report)

    def add_callback(self, callback: BenchmarkCallback) -> None:
        """Register a callback handler to monitor benchmark execution.

        Args:
            callback: A BenchmarkCallback instance that will receive execution events.

        How to use:
            Callbacks receive notifications at key lifecycle points for tracing, progress tracking,
            or custom metrics collection. See [`BenchmarkCallback`][maseval.core.callback.BenchmarkCallback]
            for available hooks and their signatures.

            ```python
            from maseval.core.callbacks import MessageTracingCallback

            benchmark = MyBenchmark(tasks=tasks, agent_data=config)
            benchmark.add_callback(MessageTracingCallback(output_dir="logs"))
            results = benchmark.run()
            ```
        """
        self.callback_handler.register(callback.on_event)

    @abstractmethod
    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Environment:
        """Create and initialize the environment for a task.

        This method is called once per task execution to create a fresh environment with the
        task's initial conditions.

        **Note:** The returned environment is **automatically registered** for tracing.
        You don't need to manually call `register()` for it.

        Args:
            agent_data: Configuration dict containing agent specifications, model parameters,
                and other settings that may influence environment setup (e.g., framework type).
            task: The Task object containing environment_data, query, and metadata needed to
                construct the environment.
            seed_generator: Seed generator for deriving deterministic seeds for environment
                components (tools, simulators, etc.). Use `derive_seed()` to get seeds for
                individual components. Returns `None` if seeding is disabled (global_seed=None).

        Returns:
            An Environment instance with initialized state and tools for this specific task.

        How to use:
            The environment encapsulates task state and provides tools/APIs that agents can use.
            Your implementation should:

            - Extract environment state from task.environment_data
            - Initialize any databases, simulators, or API clients
            - Create and configure tools that agents can invoke
            - Set up domain-specific state (inventory, user profiles, etc.)
            - Optionally use agent_data for framework-specific tool initialization
            - Use seed_generator to derive seeds for tool simulators

            ```python
            def setup_environment(self, agent_data, task, seed_generator):
                env = TravelEnvironment(task.environment_data)

                # Use nested child() for logical paths like "environment/tools/weather"
                # derive_seed() returns None if seeding is disabled
                env_gen = seed_generator.child("environment")
                tools_gen = env_gen.child("tools")
                for tool in env.tools:
                    tool_seed = tools_gen.derive_seed(tool.name)  # Optional[int]
                    tool_model = self.get_model_adapter(model_id, seed=tool_seed)
                    tool.set_simulator(tool_model)

                return env
            ```

            The environment is automatically registered for tracing when returned.
        """
        pass

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        seed_generator: SeedGenerator,
    ) -> Optional[User]:
        """Create an optional user simulator for interactive tasks.

        This method is optional. Return None if your benchmark does not require user simulation.

        **Note:** The returned user is **automatically registered** for tracing.
        You don't need to manually call `register()` for it.

        Args:
            agent_data: Configuration dict containing agent specifications and settings that may
                influence user simulator setup (e.g., framework type for creating compatible tools).
            environment: The environment instance created for this task.
            task: The Task object with user profile data or scenario information.
            seed_generator: Seed generator for deriving deterministic seeds for the user simulator.
                `derive_seed()` returns `None` if seeding is disabled (global_seed=None).

        Returns:
            A User instance that can respond to agent queries, or None if not needed.

        How to use:
            User simulators enable agent-user interactions by responding to queries with preferences,
            clarifications, or feedback. Useful for benchmarks testing conversational agents or
            systems requiring user input during execution.

            ```python
            def setup_user(self, agent_data, environment, task, seed_generator):
                # Use child() to create logical namespace - results in "simulators/user"
                # derive_seed() returns None if seeding is disabled
                sim_gen = seed_generator.child("simulators")
                user_seed = sim_gen.derive_seed("user")  # Optional[int]

                user_model = self.get_model_adapter(model_id, seed=user_seed)
                return LLMUser(model=user_model, ...)

            # Or skip user simulation entirely
            def setup_user(self, agent_data, environment, task, seed_generator):
                return None
            ```

            The user is automatically registered for tracing when returned.
        """
        pass

    @abstractmethod
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Instantiate and configure the agent system for a task.

        **Note:** All agents in the returned `agents_dict` are **automatically registered**
        for tracing. You don't need to manually call `register()` for them. However, you
        should manually register models, simulators, or other components used by agents.

        Args:
            agent_data: Configuration dict containing agent specifications, model parameters, and
                tool assignments for this task.
            environment: The initialized environment providing tools to the agents.
            task: The Task object with query and metadata.
            user: Optional user simulator for agent-user interactions.
            seed_generator: Seed generator for deriving deterministic seeds for agents and their
                models. Use `per_repetition=True` for agents that should vary across repetitions,
                or `per_repetition=False` for baseline agents that should remain constant.
                `derive_seed()` returns `None` if seeding is disabled (global_seed=None).

        Returns:
            A tuple of (agents_to_run, agents_dict) where:

            - agents_to_run: Sequence of agents to invoke in run_agents() (typically 1 orchestrator)
            - agents_dict: Dictionary mapping agent names/IDs to all agent instances for monitoring

        How to use:
            This method constructs your agent architecture—single agent, multiple collaborative agents,
            or an orchestrator managing workers. Each agent is wrapped in AgentAdapter for uniform
            message history tracking.

            The dual return structure serves different purposes:

            - **agents_to_run**: Only agents directly invoked in run_agents() (typically the orchestrator)
            - **agents_dict**: All agents in the system for message history collection from workers
              called indirectly through the orchestrator

            ```python
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                # Use child() for logical paths like "agents/experimental"
                # derive_seed() returns None if seeding is disabled
                agent_gen = seed_generator.child("agents")

                # Vary experimental agent per rep, keep baseline constant
                experimental_seed = agent_gen.derive_seed("experimental", per_repetition=True)
                baseline_seed = agent_gen.derive_seed("baseline", per_repetition=False)

                # For worker agents, nest further: "agents/workers/analyst"
                worker_gen = agent_gen.child("workers")
                analyst_seed = worker_gen.derive_seed("analyst")

                # Create agents with seeds (model adapters accept Optional[int])
                model = self.get_model_adapter(model_id, seed=experimental_seed)
                # ... create agents ...

                return [orchestrator_adapter], all_agents
            ```
        """
        pass

    @abstractmethod
    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        """Create evaluators to assess agent performance on a task.

        Args:
            environment: The environment instance with final state after agent execution.
            task: The Task object with evaluation criteria in evaluation_data.
            agents: The agents that will execute the task (useful for context in evaluation).
            user: Optional user simulator, if evaluation needs user interaction data.
            seed_generator: Seed generator for deriving deterministic seeds for evaluators
                that use LLMs (e.g., LLM judges). `derive_seed()` returns `None` if seeding
                is disabled (global_seed=None).

        Returns:
            A sequence of Evaluator instances that will be called with execution traces.

        How to use:
            Evaluators judge whether agents successfully completed the task or satisfied specific
            criteria. Multiple evaluators can measure different performance aspects (accuracy,
            efficiency, conversation quality, etc.).

            ```python
            def setup_evaluators(self, environment, task, agents, user, seed_generator):
                # Use child() to create logical namespace - results in "evaluators/judge"
                # derive_seed() returns None if seeding is disabled
                eval_gen = seed_generator.child("evaluators")
                judge_seed = eval_gen.derive_seed("judge")  # Optional[int]

                return [
                    SuccessEvaluator(task.evaluation_data["gold_answer"]),
                    LLMJudgeEvaluator(model=self.get_model_adapter(model_id, seed=judge_seed))
                ]
            ```
        """
        pass

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """Provide a ModelAdapter for benchmark components that require LLM access.

        Many benchmark components beyond the agents themselves require access to language
        models. Common examples include:

        - **Tool simulators**: Simulating tool responses when real APIs aren't available
        - **User simulators**: Generating realistic user responses in multi-turn dialogues
        - **Judges/Evaluators**: Using LLMs to assess agent performance against criteria
        - **Reward models**: Computing scores for reinforcement learning

        This method centralizes model provisioning, giving you control over which models
        are used throughout the benchmark. Implement this to return a configured ModelAdapter
        for the requested model.

        Args:
            model_id: The model identifier to use (e.g., "gemini-2.5-flash",
                "openrouter/google/gemini-2.5-flash", "gpt-4o"). This is passed by the
                benchmark when setting up components that need model access.
            **kwargs: Additional arguments for adapter creation or registration. Common kwargs:
                - register_category: Category for trace registration (e.g., "models")
                - register_name: Name for trace registration (e.g., "evaluator_user_gsr")

        Returns:
            A ModelAdapter instance configured for the specified model. For proper tracing,
            return a fresh adapter for each call rather than reusing instances. You can
            still share the underlying API client for efficiency.

        How to use:
            For proper tracing, register the adapter after creation using the kwargs:

            ```python
            def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
                adapter = GoogleGenAIModelAdapter(self.client, model_id=model_id)

                # Register for tracing if registration info provided
                category = kwargs.get("register_category", "models")
                name = kwargs.get("register_name", model_id)
                self.register(category, name, adapter)

                return adapter
            ```

            The benchmark calls this method when setting up tools, user simulators,
            and evaluators. Each call creates a fresh adapter with its own trace log.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute evaluators to score agent performance on the task.

        This method calls each evaluator with the collected execution data to produce evaluation
        results. The framework automatically collects comprehensive execution traces from all
        registered components before calling this method.

        Args:
            evaluators: The evaluator instances created by setup_evaluators().
            agents: Dictionary of all agent instances, useful if evaluators need to query agent
                state beyond message history.
            final_answer: The return value from run_agents() - typically the final output, answer,
                or result from the agent system's execution.
            traces: Comprehensive execution traces from all registered components including:
                - agents: Dict mapping agent names to their traces (messages, execution metadata)
                - models: Dict mapping model names to traces (API calls, timing, token usage, errors)
                - tools: Dict mapping tool names to traces (invocations with parameters and results)
                - simulators: Dict mapping simulator names to traces (attempts and outcomes)
                - callbacks: Dict mapping callback names to traces (custom callback data)
                - environment: Direct environment traces (state, tools) - not nested in a dict
                - user: Direct user simulator traces (interactions) - not nested in a dict
                - metadata: Collection timestamp and thread info
                Plus any other registered component categories.

        Returns:
            List of evaluation result dictionaries, typically one per evaluator. Each dict should
            contain metrics, scores, or judgments about the agent's performance.

        How to use:
            Evaluators can access comprehensive execution data including final answers and traces:

            ```python
            def evaluate(self, evaluators, agents, final_answer, traces):
                results = []
                for evaluator in evaluators:
                    # Access agent message histories from traces
                    agent_traces = {name: trace_data.get('messages', MessageHistory())
                                   for name, trace_data in traces.get('agents', {}).items()}

                    # Access additional execution data
                    model_traces = traces.get("models", {})
                    tool_traces = traces.get("tools", {})
                    environment_traces = traces.get("environment")  # Direct, not nested
                    user_traces = traces.get("user")  # Direct, not nested

                    # Pass comprehensive data to evaluator
                    result = evaluator(
                        final_answer=final_answer,
                        traces=agent_traces,
                        model_calls=model_traces,
                        tool_usage=tool_traces,
                        environment_state=environment_traces,
                        user_interactions=user_traces
                    )
                    results.append(result)
                return results
            ```
        """
        pass

    @abstractmethod
    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment, query: str) -> Any:
        """Execute the agent system to solve a single task instance.

        This method is called once per task repetition by the framework's `run()` loop.

        Args:
            agents: Sequence of agents to execute (typically just the orchestrator or main agent).
            task: The Task object with the query and any metadata needed for execution.
            environment: The environment instance providing tools and state.
            query: The query string to pass to agents. For single-turn benchmarks this is
                typically task.query. For multi-turn with users, this may be an initial
                prompt or simulated user response.

        Returns:
            The final answer or result from the agent system's execution. This could be:
            - A string containing the final answer
            - A dict with structured output
            - A list of answers from multiple agents
            - Any other representation of the task solution

            Note: Message traces are captured automatically through the tracing system and
            passed to evaluate() via the traces parameter. Do NOT return message histories here.

        How to use:
            This is where you implement your multi-agent system logic for solving one task, one time.
            You define how your agents interact, communicate, and collaborate to complete the task.

            The agents parameter contains only the primary agents to run (from agents_to_run in
            setup_agents). Worker agents called indirectly through an orchestrator do not appear here.

            Note: This method handles a single task execution. The framework's `run()` method manages
            task iteration, repetitions, and the complete benchmark lifecycle.

            ```python
            def run_agents(self, agents, task, environment, query):
                # Simple single-agent execution - returns final answer string
                orchestrator = agents[0]
                final_answer = orchestrator.run(query)
                return final_answer

            # Or for multiple agents returning a list of answers:
            def run_agents(self, agents, task, environment, query):
                answers = []
                for agent in agents:
                    answer = agent.run(query)
                    answers.append(answer)
                return answers
            ```
        """
        pass

    def execution_loop(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        user: Optional[User],
    ) -> Any:
        """Execute agents with optional user interaction loop.

        This method orchestrates the agent-user interaction pattern. When a user is
        present, the user initiates the conversation using `user.get_initial_query()`.
        If no user is present, ``task.query`` is used as the initial query.

        Interaction Flow:
            By default, agents execute once (``max_invocations=1``). For multi-turn
            interaction, set ``self.max_invocations > 1`` in your benchmark's ``__init__``.
            The loop continues until ``max_invocations`` is reached or ``user.is_done()``
            returns True (e.g., max turns reached or stop token detected).

        Note:
            Override this method in your benchmark subclass to implement custom
            interaction patterns (e.g., agent-initiated conversations, different
            termination conditions, or specialized query routing).

        Args:
            agents: Agents to execute (typically the orchestrator).
            task: The task being solved.
            environment: The environment providing tools and state.
            user: Optional user simulator. If provided, the user initiates and drives
                the conversation. If None, a single agent execution with ``task.query``.

        Returns:
            Final answer from the last agent execution.

        Example:
            For interactive benchmarks, enable multi-turn interaction::

                def __init__(self, ...):
                    super().__init__(...)
                    self.max_invocations = 5  # Up to 5 agent-user exchanges
        """

        final_answer = None

        # Determine initial query text
        if user is not None:
            query_text = user.get_initial_query()
        else:
            query_text = task.query

        for _ in range(self.max_invocations):
            # Execute agents with query
            final_answer = self.run_agents(agents, task, environment, query_text)

            # No user means single execution
            if user is None:
                break

            # Simulate user response (handles message recording, stop token detection, turn counting)
            user_response = user.respond(str(final_answer) if final_answer else "")

            # Check if user is done (cheap state check - no LLM call)
            if user.is_done():
                break

            # Use user's response as next query
            query_text = user_response

        return final_answer

    def _execute_task_repetition(
        self,
        task: Task,
        agent_data: Dict[str, Any],
        repeat_idx: int,
    ) -> Dict[str, Any]:
        """Execute a single task repetition with timeout handling.

        This method encapsulates the complete execution of one task repetition,
        including setup, execution, trace collection, and evaluation. It is
        designed to be called from both sequential and parallel execution paths.

        Args:
            task: The task to execute.
            agent_data: Agent configuration for this task.
            repeat_idx: Repetition index (0 to n_task_repeats-1).

        Returns:
            Report dictionary containing execution results.
        """
        # Initialize status and error tracking
        execution_status = TaskExecutionStatus.SUCCESS
        error_info: Optional[Dict[str, Any]] = None
        final_answers: Any = None
        eval_results: Any = None
        execution_traces: Dict[str, Any] = {}
        execution_configs: Dict[str, Any] = {}
        evaluators: Sequence[Evaluator] = []
        agents_dict: Dict[str, AgentAdapter] = {}

        # Create execution context with optional timeout
        timeout = task.protocol.timeout_seconds
        context = TaskContext(deadline=timeout)

        # Create scoped seed generator for this task+repetition
        task_seed_gen = self._seed_generator.for_task(str(task.id)).for_repetition(repeat_idx)

        try:
            # 1. Setup
            environment = self.setup_environment(agent_data, task, seed_generator=task_seed_gen)
            user = self.setup_user(agent_data, environment, task, seed_generator=task_seed_gen)
            if user is None and self.max_invocations > 1:
                # Warn if multi-turn is enabled but no user to drive interaction
                warnings.warn(
                    f"max_invocations={self.max_invocations} > 1 but no user simulator provided. "
                    f"Falling back to single-turn execution for task {task.id}."
                )
            agents_to_run, agents_dict = self.setup_agents(agent_data, environment, task, user, seed_generator=task_seed_gen)
            evaluators = self.setup_evaluators(environment, task, agents_to_run, user, seed_generator=task_seed_gen)

            # Auto-register components returned from setup methods
            # Environment
            if environment is not None and isinstance(environment, TraceableMixin):
                self.register("environment", "env", environment)

            # User
            if user is not None and isinstance(user, TraceableMixin):
                self.register("user", "user", user)

            # Agents (use their names from agents_dict)
            for agent_name, agent in agents_dict.items():
                if isinstance(agent, TraceableMixin):
                    self.register("agents", agent_name, agent)

            # Register seed generator for config collection
            self.register("seeding", "seed_generator", task_seed_gen)

            # Check timeout after setup
            context.check_timeout()

        except TaskTimeoutError as e:
            # Timeout during setup
            execution_status = TaskExecutionStatus.TASK_TIMEOUT
            error_info = {
                "error_type": "TaskTimeoutError",
                "error_message": str(e),
                "elapsed": e.elapsed,
                "timeout": e.timeout,
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            # Create a minimal report for this timeout
            report = {
                "task_id": str(task.id),
                "repeat_idx": repeat_idx,
                "status": execution_status.value,
                "error": error_info,
                "traces": e.partial_traces,
                "config": {},
                "eval": None,
            }
            self.clear_registry()
            return report

        except Exception as e:
            # Setup failed - record error
            execution_status = TaskExecutionStatus.SETUP_FAILED
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            # Create a minimal report for this failed setup
            report = {
                "task_id": str(task.id),
                "repeat_idx": repeat_idx,
                "status": execution_status.value,
                "error": error_info,
                "traces": {},
                "config": {},
                "eval": None,
            }
            self.clear_registry()

            if self.fail_on_setup_error:
                raise

            return report

        # 2. Execute agent system with optional user interaction loop
        try:
            # Check timeout before execution
            context.check_timeout()
            final_answers = self.execution_loop(agents_to_run, task, environment, user)
        except TaskTimeoutError as e:
            # Task timed out during execution
            execution_status = TaskExecutionStatus.TASK_TIMEOUT
            error_info = {
                "error_type": "TaskTimeoutError",
                "error_message": str(e),
                "elapsed": e.elapsed,
                "timeout": e.timeout,
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }
            final_answers = None
        except AgentError as e:
            # Agent violated contract at boundary (agent's fault)
            execution_status = TaskExecutionStatus.AGENT_ERROR
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "component": e.component,
                "details": e.details,
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            if self.fail_on_task_error:
                self.clear_registry()
                raise

            final_answers = None
        except EnvironmentError as e:
            # Environment/tool infrastructure failed (not agent's fault)
            execution_status = TaskExecutionStatus.ENVIRONMENT_ERROR
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "component": e.component,
                "details": e.details,
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            if self.fail_on_task_error:
                self.clear_registry()
                raise

            final_answers = None
        except UserError as e:
            # User simulator failed (not agent's fault)
            execution_status = TaskExecutionStatus.USER_ERROR
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "component": e.component,
                "details": e.details,
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            if self.fail_on_task_error:
                self.clear_registry()
                raise

            final_answers = None
        except Exception as e:
            # Unclassified error (e.g., agent framework internal failure)
            execution_status = TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }

            if self.fail_on_task_error:
                self.clear_registry()
                raise

            final_answers = None

        # 3. Collect traces and configs (always attempt this)
        try:
            execution_configs = self.collect_all_configs()
            execution_traces = self.collect_all_traces()
            # Store in context for potential timeout errors
            context.set_collected_traces(execution_traces)
        except Exception as e:
            # If trace/config collection fails, record it but continue
            execution_configs = {
                "error": f"Failed to collect configs: {e}",
                "error_type": type(e).__name__,
            }
            execution_traces = {
                "error": f"Failed to collect traces: {e}",
                "error_type": type(e).__name__,
            }

        # 4. Evaluate (skip if task execution failed)
        if execution_status == TaskExecutionStatus.SUCCESS:
            try:
                # Check timeout before evaluation
                context.check_timeout()
                eval_results = self.evaluate(evaluators, agents_dict, final_answers, execution_traces)
            except TaskTimeoutError as e:
                execution_status = TaskExecutionStatus.TASK_TIMEOUT
                error_info = {
                    "error_type": "TaskTimeoutError",
                    "error_message": str(e),
                    "elapsed": e.elapsed,
                    "timeout": e.timeout,
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                }
                eval_results = None
            except Exception as e:
                execution_status = TaskExecutionStatus.EVALUATION_FAILED
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                }

                if self.fail_on_evaluation_error:
                    self.clear_registry()
                    raise

                eval_results = None
        else:
            # Task execution failed, so skip evaluation
            eval_results = None

        # 5. Build report
        report: Dict[str, Any] = {
            "task_id": str(task.id),
            "repeat_idx": repeat_idx,
            "status": execution_status.value,
            "traces": execution_traces,
            "config": execution_configs,
            "eval": eval_results,
        }

        # Add error info if present
        if error_info is not None:
            report["error"] = error_info

        # Clear registry after task repetition completes
        self.clear_registry()

        return report

    def _run_sequential(
        self,
        queue: BaseTaskQueue,
        agent_data_lookup: Dict[str, Dict[str, Any]],
    ) -> None:
        """Execute tasks sequentially with optional timeout support.

        Args:
            queue: Task queue providing task ordering.
            agent_data_lookup: Mapping from task_id to agent_data configuration.
        """
        for task in queue:
            agent_data = agent_data_lookup[str(task.id)]

            # Callbacks at the start of each task
            self._invoke_callbacks("on_task_start", self, task)

            for repeat_idx in range(self.n_task_repeats):
                self._invoke_callbacks("on_task_repeat_start", self, task, repeat_idx)

                report = self._execute_task_repetition(task, agent_data, repeat_idx)
                self._append_report_safe(report)

                self._invoke_callbacks("on_task_repeat_end", self, report)

            # Callbacks at the end of each task
            task_reports = [r for r in self.reports if r["task_id"] == str(task.id)]
            last_report = task_reports[-1] if task_reports else {}
            self._invoke_callbacks("on_task_end", self, task, last_report)

    def _run_parallel(
        self,
        queue: BaseTaskQueue,
        agent_data_lookup: Dict[str, Dict[str, Any]],
        num_workers: int,
    ) -> None:
        """Execute tasks in parallel with thread pool.

        Args:
            queue: Task queue providing task ordering.
            agent_data_lookup: Mapping from task_id to agent_data configuration.
            num_workers: Number of concurrent workers.
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures: Dict[Any, Tuple[Task, int]] = {}
            task_repeat_counts: Dict[str, int] = {}  # Track submitted repeats per task

            def submit_task_repeats(task: Task) -> None:
                """Submit all repeats for a task."""
                task_id = str(task.id)
                task_repeat_counts[task_id] = 0
                agent_data = agent_data_lookup[task_id]

                self._invoke_callbacks("on_task_start", self, task)

                for repeat_idx in range(self.n_task_repeats):
                    self._invoke_callbacks("on_task_repeat_start", self, task, repeat_idx)

                    future = executor.submit(
                        self._execute_task_repetition,
                        task,
                        agent_data,
                        repeat_idx,
                    )
                    futures[future] = (task, repeat_idx)
                    task_repeat_counts[task_id] += 1

            # Submit initial batch from queue
            submitted_tasks: List[Task] = []
            queue_iter = iter(queue)  # Create iterator once
            queue_exhausted = False

            # Submit initial batch
            try:
                while len(futures) < num_workers * 2:
                    task = next(queue_iter)
                    submit_task_repeats(task)
                    submitted_tasks.append(task)
            except StopIteration:
                queue_exhausted = True

            # Process completions
            completed_task_ids: set = set()

            while futures:
                # Wait for at least one completion
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # No futures done yet, wait a bit
                    import time

                    time.sleep(0.01)
                    continue

                for future in done_futures:
                    task, repeat_idx = futures.pop(future)
                    task_id = str(task.id)

                    try:
                        report = future.result()
                    except Exception as e:
                        # Create error report for unexpected failures
                        report = {
                            "task_id": task_id,
                            "repeat_idx": repeat_idx,
                            "status": TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value,
                            "error": {
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                            },
                            "traces": {},
                            "config": {},
                            "eval": None,
                        }

                    self._append_report_safe(report)

                    self._invoke_callbacks("on_task_repeat_end", self, report)

                    # Check if all repeats for this task are done
                    task_reports = [r for r in self.reports if r["task_id"] == task_id]
                    if len(task_reports) >= self.n_task_repeats and task_id not in completed_task_ids:
                        completed_task_ids.add(task_id)
                        last_report = task_reports[-1] if task_reports else {}
                        self._invoke_callbacks("on_task_end", self, task, last_report)

                # Submit more work if queue not exhausted
                if not queue_exhausted and len(futures) < num_workers:
                    try:
                        task = next(queue_iter)
                        submit_task_repeats(task)
                        submitted_tasks.append(task)
                    except StopIteration:
                        queue_exhausted = True

    def run(
        self,
        tasks: Union[Task, BaseTaskQueue, Iterable[Union[Task, dict]]],
        agent_data: Dict[str, Any] | Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Initialize and execute the complete benchmark loop across all tasks.

        Args:
            tasks: Task source for execution. Can be:
                - A single Task object
                - A BaseTaskQueue (SequentialTaskQueue, PriorityTaskQueue, or custom AdaptiveTaskQueue)
                - An iterable of Task objects or dicts that will be converted to Tasks

                When a BaseTaskQueue is provided, it controls the task ordering. AdaptiveTaskQueue
                subclasses are automatically registered as callbacks to receive task completion
                notifications.
            agent_data: Configuration for agents. Either a single dict applied to all tasks, or
                an iterable of dicts with one configuration per task. Agent data typically includes
                model parameters, agent architecture details, and tool specifications.

        Returns:
            List of report dictionaries, one per task repetition. Each report contains:
            - task_id: Task identifier (UUID)
            - repeat_idx: Repetition index (0 to n_task_repeats-1)
            - status: Execution status (one of TaskExecutionStatus enum values)
            - traces: Execution traces from all registered components
            - config: Configuration from all registered components and benchmark level
            - eval: Evaluation results (None if task or evaluation failed)
            - error: Error details dict (only present if status is not SUCCESS), containing:
                - error_type: Exception class name
                - error_message: Exception message
                - traceback: Full traceback string

        Raises:
            ValueError: If agent_data length doesn't match number of tasks (when agent_data is an iterable).

        How to use:
            This is the framework's main orchestration method that runs your entire benchmark. It
            iterates through all tasks, handles repetitions, and manages the three-stage lifecycle
            for each execution. You don't implement this method—instead, you call it to start the
            benchmark after implementing the setup and execution methods.

            By default, the benchmark will continue executing remaining tasks even if some fail.
            You can change this behavior by setting `fail_on_task_error=True`,
            `fail_on_evaluation_error=True`, or `fail_on_setup_error=True` when instantiating
            the benchmark. Each task execution returns a status indicating success or the specific
            failure type (see [`TaskExecutionStatus`][maseval.core.benchmark.TaskExecutionStatus]).

            For each task execution, the framework:

            1. Calls your setup methods to initialize components
            2. Calls your `run_agents()` method to execute the task
            3. Collects message histories and calls evaluators
            4. Stores results and triggers callbacks

            Pseudocode structure:
                ```
                for task in tasks:
                    for repeat in range(n_task_repeats):
                        # Setup stage
                        environment = setup_environment(agent_data, task)
                        user = setup_user(agent_data, environment, task)
                        agents_to_run, agents_dict = setup_agents(agent_data, environment, task, user)
                        evaluators = setup_evaluators(environment, task, agents_to_run, user)

                        # Run stage (execution_loop handles multi-turn if user exists)
                        agents_output = execution_loop(agents_to_run, task, environment, user)

                        # Evaluate stage
                        traces = collect_message_histories(agents_dict)
                        eval_results = evaluate(evaluators, traces, agents_dict)

                        # Store results
                        store_result(task_id, traces, eval_results)
                ```

            Callback hooks are triggered at these points:

            - on_run_start: Before processing any tasks
            - on_task_start: Before processing a task (once per task, not per repeat)
            - on_task_repeat_start: Before each repetition of a task
            - on_task_repeat_end: After each repetition completes
            - on_task_end: After all repetitions of a task complete
            - on_run_end: After all tasks complete

            ```python
            # Typical usage
            benchmark = MyBenchmark()
            reports = benchmark.run(tasks=tasks, agent_data=config)

            # Analyze results
            for report in reports:
                print(f"Task {report['task_id']}, Repeat {report['repeat_idx']}: {report['eval']}")
                print(f"Config: {report['config']}")
                print(f"Traces: {report['traces']}")

            # Parallel execution with 4 workers
            benchmark = MyBenchmark(num_workers=4)
            reports = benchmark.run(tasks=tasks, agent_data=config)

            # Single agent config for all tasks
            reports = benchmark.run(tasks=tasks, agent_data={"model": "gpt-4"})

            # Task-specific agent configs (must match task count)
            reports = benchmark.run(
                tasks=tasks,
                agent_data=[
                    {"model": "gpt-4", "difficulty": "easy"},
                    {"model": "gpt-4", "difficulty": "hard"},
                ]
            )

            # Priority-based execution
            from maseval.core.task import PriorityTaskQueue
            for task in tasks:
                task.protocol.priority = compute_priority(task)
            queue = PriorityTaskQueue(tasks)
            reports = benchmark.run(tasks=queue, agent_data=config)

            # Adaptive queue (auto-registered as callback)
            queue = MyAdaptiveTaskQueue(tasks)
            reports = benchmark.run(tasks=queue)  # queue receives on_task_complete callbacks
            ```
        """
        # Normalize tasks into a queue
        queue: BaseTaskQueue
        if isinstance(tasks, Task):
            # Single task - wrap in SequentialTaskQueue
            queue = SequentialTaskQueue([tasks])
        elif isinstance(tasks, BaseTaskQueue):
            # Already a queue - use directly
            queue = tasks
        else:
            # Iterable of tasks or dicts - wrap in SequentialTaskQueue
            queue = SequentialTaskQueue.from_list(list(tasks))

        # Store tasks reference for get_failed_tasks() compatibility
        self.tasks = queue

        # Build agent_data lookup (task_id -> agent_data)
        agent_data_lookup = self._build_agent_data_lookup(queue, agent_data)

        # Clear reports from previous run() calls to prevent accumulation
        self.reports = []

        # Auto-register queue as callback if it's a BenchmarkCallback (e.g., AdaptiveTaskQueue)
        queue_as_callback: Optional[BenchmarkCallback] = None
        if isinstance(queue, BenchmarkCallback) and queue not in self.callbacks:
            queue_as_callback = queue
            self.callbacks.append(queue_as_callback)

        try:
            # Callbacks at the start of the run
            self._invoke_callbacks("on_run_start", self)

            # Execute based on num_workers
            if self.num_workers == 1:
                self._run_sequential(queue, agent_data_lookup)
            else:
                self._run_parallel(queue, agent_data_lookup, self.num_workers)

            # Callbacks at the end of the run
            self._invoke_callbacks("on_run_end", self, self.reports)
        finally:
            # Remove queue from callbacks if we added it
            if queue_as_callback is not None:
                self.callbacks.remove(queue_as_callback)

        return self.reports

    def _build_agent_data_lookup(
        self, tasks: BaseTaskQueue, agent_data: Dict[str, Any] | Iterable[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from task_id to agent_data configuration.

        Args:
            tasks: The task queue containing all tasks.
            agent_data: Agent configuration(s) to map to tasks.

        Returns:
            Dict mapping task_id (string) to agent_data configuration.

        Raises:
            ValueError: If agent_data is a list but doesn't match the number of tasks.
        """
        if isinstance(agent_data, dict):
            # Single config - replicate for all tasks
            return {str(task.id): cast(Dict[str, Any], agent_data) for task in tasks}

        # List of configs - pair by position
        agent_data_list = list(agent_data)
        if len(agent_data_list) != len(tasks):
            raise ValueError(
                f"`agent_data` must either be a single dict or an iterable matching the number of tasks. "
                f"Got {len(agent_data_list)} agent configs for {len(tasks)} tasks."
            )

        return {str(task.id): agent_data_list[i] for i, task in enumerate(tasks)}

    def get_failed_tasks(
        self,
        status_filter: Optional[Union[TaskExecutionStatus, List[TaskExecutionStatus]]] = None,
        reports: Optional[List[Dict[str, Any]]] = None,
    ) -> SequentialTaskQueue:
        """Get tasks that failed during benchmark execution.

        This method retrieves failed tasks based on their execution status, useful for
        debugging, retry logic, or failure analysis.

        Args:
            status_filter: Filter by specific failure status(es). If None, returns all failed tasks
                (any status except SUCCESS). Can be a single TaskExecutionStatus or a list of them.
                Examples:
                - TaskExecutionStatus.TASK_EXECUTION_FAILED: Only tasks that failed during execution
                - TaskExecutionStatus.EVALUATION_FAILED: Only tasks where evaluation failed
                - [TaskExecutionStatus.TASK_EXECUTION_FAILED, TaskExecutionStatus.SETUP_FAILED]:
                  Tasks that failed during execution or setup
            reports: Optional list of reports to analyze. If None, uses the reports from the last
                run() call. This allows analyzing externally stored or modified reports.

        Returns:
            SequentialTaskQueue containing the failed tasks. Empty if no failures match the filter.

        Raises:
            RuntimeError: If reports is None and run() has not been executed yet.

        How to use:
            ```python
            # Run benchmark
            benchmark = MyBenchmark()
            reports = benchmark.run(tasks=tasks, agent_data=config)

            # Get all failed tasks (from internal state)
            failed = benchmark.get_failed_tasks()
            print(f"Failed: {len(failed)}/{len(benchmark.tasks)} tasks")

            # Or work with returned reports (safe from internal state changes)
            failed = benchmark.get_failed_tasks(reports=reports)

            # Get only tasks that failed during execution (not evaluation)
            execution_failures = benchmark.get_failed_tasks(
                TaskExecutionStatus.TASK_EXECUTION_FAILED,
                reports=reports
            )

            # Get setup and execution failures
            critical_failures = benchmark.get_failed_tasks(
                status_filter=[
                    TaskExecutionStatus.SETUP_FAILED,
                    TaskExecutionStatus.TASK_EXECUTION_FAILED
                ],
                reports=reports
            )

            # Retry failed tasks elegantly - this is the key use case!
            if len(failed) > 0:
                retry_reports = benchmark.run(tasks=failed)

            # Or more concisely
            reports = benchmark.run(tasks=tasks)
            retry_reports = benchmark.run(tasks=benchmark.get_failed_tasks())
            ```
        """
        # Use provided reports or fall back to internal state
        if reports is None:
            if not self.reports:
                raise RuntimeError("get_failed_tasks() must be called after run(). No reports found.")
            reports = self.reports

        # Normalize status_filter to a set of status values (strings)
        if status_filter is None:
            # All non-success statuses (includes all classified and unclassified failures)
            filter_values = {
                TaskExecutionStatus.AGENT_ERROR.value,
                TaskExecutionStatus.ENVIRONMENT_ERROR.value,
                TaskExecutionStatus.USER_ERROR.value,
                TaskExecutionStatus.TASK_TIMEOUT.value,
                TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value,
                TaskExecutionStatus.EVALUATION_FAILED.value,
                TaskExecutionStatus.SETUP_FAILED.value,
                # Include deprecated status for backward compatibility
                TaskExecutionStatus.TASK_EXECUTION_FAILED.value,
            }
        elif isinstance(status_filter, TaskExecutionStatus):
            filter_values = {status_filter.value}
        else:
            filter_values = {status.value for status in status_filter}

        # Collect unique failed task IDs matching the filter
        failed_task_ids = set()
        for report in reports:
            if report["status"] in filter_values:
                failed_task_ids.add(report["task_id"])

        # Build queue from original tasks that failed
        failed_tasks = [task for task in self.tasks if str(task.id) in failed_task_ids]
        return SequentialTaskQueue(failed_tasks)
