import os
from typing import Any, Dict, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from maseval import AgentAdapter
from maseval.benchmark.gaia2 import DefaultAgentGaia2Benchmark, compute_gaia2_metrics, load_tasks
from maseval.benchmark.gaia2.environment import Gaia2Environment
from maseval.core.seeding import SeedGenerator
from maseval.core.task import Task
from maseval.core.user import User
from maseval.interface.inference import OpenAIModelAdapter

load_dotenv()


class MyGaia2(DefaultAgentGaia2Benchmark):
    """Run GAIA2 benchmark using OpenAI."""

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> OpenAIModelAdapter:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        adapter = OpenAIModelAdapter(client, model_id=model_id, seed=kwargs.get("seed"))
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_agents(self, agent_data: Dict[str, Any], environment: Gaia2Environment, task: Task, user: User, seed_generator: SeedGenerator):
        """Set up your own agents here"""
        pass

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Gaia2Environment, query: str = "") -> Any:
        """How to run the agents on the task."""
        pass


if __name__ == "__main__":
    tasks = load_tasks(capability="execution", limit=2)

    benchmark = MyGaia2(
        agent_data={"model_id": "gpt-4o", "verbose": 1},
        progress_bar=True,
    )

    results = benchmark.run(tasks=tasks)
    summary = compute_gaia2_metrics(results)
