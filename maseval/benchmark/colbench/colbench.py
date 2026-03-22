"""
ColBench Benchmark — MASEval Benchmark orchestrator.

Subclasses ``maseval.core.Benchmark`` to wire together:
    - **Environment** → ``ColBenchEnvironment``  (task state)
    - **User**        → ``ColBenchUser``         (human simulator)
    - **Agent**       → ``ColBenchAgentInner``   (agent-under-test)
    - **Evaluator**   → ``ColBenchCodeEvaluator`` (unit-test scoring)

The ``execution_loop`` flow (inherited from ``Benchmark``) maps exactly to
the original sweet_rl interaction loop::

    # sweet_rl                          # MASEval execution_loop
    obs = env.reset(desc, gt)           query = user.get_initial_query()
    for step in range(max_steps):       for _ in range(max_invocations):
        response = agent(obs)               answer = run_agents(query)
        obs, _, done = env.step(resp)       reply = user.respond(answer)
        if done: break                      if user.is_done(): break
                                            query = reply
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from maseval.core.benchmark import Benchmark
from maseval.core.task import Task, SequentialTaskQueue
from maseval.core.environment import Environment
from maseval.core.agent import AgentAdapter
from maseval.core.model import ModelAdapter
from maseval.core.evaluator import Evaluator
from maseval.core.user import User
from maseval.core.callback import BenchmarkCallback
from maseval.core.seeding import SeedGenerator

from .environment import ColBenchEnvironment
from .user import ColBenchUser, DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT
from .agent import ColBenchAgentInner, ColBenchAgentAdapter, DEFAULT_AGENT_CODE_PROMPT
from .evaluator import ColBenchCodeEvaluator

logger = logging.getLogger(__name__)

ModelFactory = Callable[..., ModelAdapter]


class ColBenchBenchmark(Benchmark):
    """MASEval benchmark for ColBench collaborative agent evaluation.

    Args:
        model_factory: Callable ``(model_id, **kwargs) → ModelAdapter``.
        human_simulator_model_id: Model ID for the human-simulator LLM.
        agent_model_id: Default agent model ID.
        human_prompt: Prompt template for the simulator.
        agent_prompt: System prompt for the agent.
        max_steps: Maximum interaction rounds (default 10).
        agent_temperature: Agent sampling temperature (default 1.0).
        agent_max_tokens: Max tokens per agent response (default 1024).
        **kwargs: Forwarded to ``Benchmark.__init__()``.
    """

    def __init__(
        self,
        model_factory: Optional[ModelFactory] = None,
        human_simulator_model_id: str = "meta-llama/Llama-3.1-70B-Instruct",
        agent_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        human_prompt: Optional[str] = None,
        agent_prompt: Optional[str] = None,
        max_steps: int = 10,
        agent_temperature: float = 1.0,
        agent_max_tokens: int = 1024,
        callbacks: Optional[List[BenchmarkCallback]] = None,
        n_task_repeats: int = 1,
        num_workers: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            callbacks=callbacks,
            n_task_repeats=n_task_repeats,
            max_invocations=max_steps + 1,
            num_workers=num_workers,
            seed=seed,
            **kwargs,
        )
        self._model_factory = model_factory
        self.human_simulator_model_id = human_simulator_model_id
        self.agent_model_id = agent_model_id
        self.human_prompt = human_prompt or DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT
        self.agent_prompt = agent_prompt or DEFAULT_AGENT_CODE_PROMPT
        self.max_steps = max_steps
        self.agent_temperature = agent_temperature
        self.agent_max_tokens = agent_max_tokens

        # Per-task inner agent reference (set in setup_agents, used in run_agents)
        self._current_inner_agent: Optional[ColBenchAgentInner] = None

    # ── Abstract method implementations ──────────────────────────────────

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
        seed_generator: SeedGenerator,
    ) -> ColBenchEnvironment:
        env_data = task.environment_data or {}
        return ColBenchEnvironment(task_data={
            "problem_description": env_data.get("problem_description", task.query),
            "ground_truth": env_data.get("ground_truth", ""),
            "test_cases": env_data.get("test_cases", {}),
            "task_type": env_data.get("task_type", "code"),
        })

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        seed_generator: SeedGenerator,
    ) -> ColBenchUser:
        env: ColBenchEnvironment = environment  # type: ignore[assignment]

        sim_model_id = agent_data.get("env_model", self.human_simulator_model_id)
        simulator_model = self.get_model_adapter(
            sim_model_id,
            register_category="models",
            register_name="human_simulator",
        )

        return ColBenchUser(
            problem_description=env.problem_description,
            hidden_information=env.ground_truth,
            model=simulator_model,
            human_prompt=agent_data.get("human_prompt", self.human_prompt),
            max_steps=agent_data.get("max_steps", self.max_steps),
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        model_id = agent_data.get("model", self.agent_model_id)
        model = self.get_model_adapter(
            model_id,
            register_category="models",
            register_name="agent_model",
        )

        # Create inner agent with dialogue state
        inner_agent = ColBenchAgentInner(
            model=model,
            system_prompt=agent_data.get("agent_prompt", self.agent_prompt),
            temperature=agent_data.get("temperature", self.agent_temperature),
            max_tokens=agent_data.get("max_tokens", self.agent_max_tokens),
        )

        # Store reference for run_agents() to call directly
        self._current_inner_agent = inner_agent

        # Wrap in concrete AgentAdapter for MASEval tracing
        agent_adapter = ColBenchAgentAdapter(inner_agent)

        agents_dict = {"colbench_agent": agent_adapter}
        return [agent_adapter], agents_dict

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Sequence[Evaluator]:
        return [
            ColBenchCodeEvaluator(
                task=task,
                environment=environment,
                user=user,
            )
        ]

    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        category = kwargs.pop("register_category", "models")
        name = kwargs.pop("register_name", model_id)

        if self._model_factory is None:
            raise RuntimeError(
                "No model_factory provided. Either pass model_factory= to "
                "ColBenchBenchmark() or subclass and override get_model_adapter()."
            )

        adapter = self._model_factory(model_id, **kwargs)
        self.register(category, name, adapter)
        return adapter

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: Environment,
        query: str,
    ) -> Any:
        """Execute one agent turn via the inner agent.

        Calls ``ColBenchAgentInner.run(query)`` directly (bypassing
        ``AgentAdapter`` delegation) to ensure the correct ``chat()``
        call is made.
        """
        assert self._current_inner_agent is not None, (
            "setup_agents() must be called before run_agents()"
        )
        return self._current_inner_agent.run(query)

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        results = []
        for evaluator in evaluators:
            filtered = evaluator.filter_traces(traces)
            result = evaluator(filtered, final_answer=final_answer)
            results.append(result)
        return results

    # ── Task loading ─────────────────────────────────────────────────────

    @staticmethod
    def load_tasks(
        jsonl_path: str,
        num_tasks: Optional[int] = None,
    ) -> SequentialTaskQueue:
        """Load ColBench tasks from a JSONL file.

        Each line::

            {"problem_description": "...", "ground_truth": "...",
             "test_cases": {"t1": "foo(1)", ...}}
        """
        tasks: List[Task] = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if num_tasks is not None and i >= num_tasks:
                    break
                raw = json.loads(line)
                task = Task(
                    query=raw["problem_description"],
                    environment_data={
                        "problem_description": raw["problem_description"],
                        "ground_truth": raw["ground_truth"],
                        "test_cases": raw.get("test_cases", {}),
                        "task_type": "code",
                    },
                    evaluation_data={
                        "ground_truth": raw["ground_truth"],
                        "test_cases": raw.get("test_cases", {}),
                    },
                )
                tasks.append(task)

        logger.info("Loaded %d ColBench tasks from %s", len(tasks), jsonl_path)
        return SequentialTaskQueue(tasks)