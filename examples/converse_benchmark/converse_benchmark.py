"""Run the CONVERSE benchmark with SmolAgents or LangGraph.

Examples:
    uv run python examples/converse_benchmark/converse_benchmark.py --framework default --domain travel --split privacy --model gpt-4o-mini
    uv run python examples/converse_benchmark/converse_benchmark.py --framework smolagents --domain travel --split privacy --model gpt-4o
    uv run python examples/converse_benchmark/converse_benchmark.py --framework langgraph --domain insurance --split security --model gpt-4o
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from maseval import AgentAdapter, Environment, ModelAdapter, Task, User
from maseval.benchmark.converse import ConverseBenchmark, DefaultAgentConverseBenchmark, load_tasks
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.seeding import SeedGenerator
from maseval.interface.inference import OpenAIModelAdapter


def create_model_adapter(model_id: str, seed: Optional[int] = None) -> ModelAdapter:
    """Create an OpenAI model adapter for benchmark components."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    return OpenAIModelAdapter(client=client, model_id=model_id, seed=seed)


class SmolAgentsConverseBenchmark(ConverseBenchmark):
    """CONVERSE benchmark implementation for SmolAgents."""

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        seed = kwargs.get("seed")
        adapter = create_model_adapter(model_id, seed=seed)
        register_name = kwargs.get("register_name")
        register_category = kwargs.get("register_category", "models")
        if register_name:
            self.register(register_category, register_name, adapter)
        return adapter

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        _ = task
        from smolagents import OpenAIServerModel, Tool, ToolCallingAgent

        from maseval.interface.agents.smolagents import SmolAgentAdapter

        raw_tools = environment.get_tools()

        smol_tools = []
        for name, tool in raw_tools.items():

            class WrappedTool(Tool):
                name = name
                description = getattr(tool, "description", "Tool")
                inputs = {"payload": {"type": "string", "description": "Tool payload"}}
                output_type = "string"

                def forward(self, payload: str) -> str:
                    return tool(payload)

            smol_tools.append(WrappedTool())

        if user is not None:

            class TalkToExternalAgentTool(Tool):
                name = "talk_to_external_agent"
                description = "Send a message to the external service provider and get a response."
                inputs = {"message": {"type": "string", "description": "Message for the external agent"}}
                output_type = "string"

                def forward(self, message: str) -> str:
                    return user.respond(message)

            smol_tools.append(TalkToExternalAgentTool())

        model = OpenAIServerModel(model_id=agent_data.get("model_id", "gpt-4o"), api_key=os.getenv("OPENAI_API_KEY"))
        agent = ToolCallingAgent(tools=smol_tools, model=model, max_steps=10)
        adapter = SmolAgentAdapter(agent, "Assistant")
        return [adapter], {"Assistant": adapter}


class LangGraphConverseBenchmark(ConverseBenchmark):
    """CONVERSE benchmark implementation for LangGraph."""

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        seed = kwargs.get("seed")
        adapter = create_model_adapter(model_id, seed=seed)
        register_name = kwargs.get("register_name")
        register_category = kwargs.get("register_category", "models")
        if register_name:
            self.register(register_category, register_name, adapter)
        return adapter

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        _ = task, seed_generator
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        from maseval.interface.agents.langgraph import LangGraphAgentAdapter

        tools = [StructuredTool.from_function(tool) for tool in environment.get_tools().values()]

        if user is not None:

            def talk_to_external_agent(message: str) -> str:
                """Talk to the external service provider."""
                return user.respond(message)

            tools.append(StructuredTool.from_function(talk_to_external_agent))

        llm = ChatOpenAI(model=agent_data.get("model_id", "gpt-4o"))
        graph = create_react_agent(llm, tools)
        adapter = LangGraphAgentAdapter(graph, "Assistant")
        return [adapter], {"Assistant": adapter}


class OpenAIDefaultConverseBenchmark(DefaultAgentConverseBenchmark):
    """CONVERSE benchmark using the built-in default Converse agent."""

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        seed = kwargs.get("seed")
        adapter = create_model_adapter(model_id, seed=seed)
        register_name = kwargs.get("register_name")
        register_category = kwargs.get("register_category", "models")
        if register_name:
            self.register(register_category, register_name, adapter)
        return adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CONVERSE benchmark")
    parser.add_argument("--framework", choices=["default", "smolagents", "langgraph"], required=True)
    parser.add_argument("--domain", choices=["travel", "real_estate", "insurance"], required=True)
    parser.add_argument("--split", choices=["privacy", "security", "all"], default="privacy")
    parser.add_argument("--model", default="gpt-4o", help="Model ID for the assistant agent")
    parser.add_argument("--attacker-model", default="gpt-4o", help="Model ID for the adversarial external agent")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks")
    parser.add_argument("--output-dir", default="results", help="Output directory for result logs")
    args = parser.parse_args()

    tasks = load_tasks(domain=args.domain, split=args.split, limit=args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"converse_{args.domain}_{args.framework}_{{timestamp}}.jsonl",
    )

    config = {
        "model_id": args.model,
        "attacker_model_id": args.attacker_model,
        "max_turns": 10,
    }

    if args.framework == "default":
        benchmark_cls = OpenAIDefaultConverseBenchmark
    elif args.framework == "smolagents":
        benchmark_cls = SmolAgentsConverseBenchmark
    else:
        benchmark_cls = LangGraphConverseBenchmark
    benchmark = benchmark_cls(callbacks=[logger], fail_on_setup_error=True)
    results = benchmark.run(tasks=tasks, agent_data=config)

    privacy_total = 0
    privacy_passed = 0
    security_total = 0
    security_passed = 0

    for report in results:
        for eval_result in report.get("eval") or []:
            if "privacy_leak" in eval_result:
                privacy_total += 1
                if not eval_result["privacy_leak"]:
                    privacy_passed += 1
            if "security_violation" in eval_result:
                security_total += 1
                if not eval_result["security_violation"]:
                    security_passed += 1

    print(f"Completed {len(results)} task run(s).")
    if privacy_total > 0:
        print(f"Privacy robustness: {privacy_passed}/{privacy_total} ({(privacy_passed / privacy_total) * 100:.1f}%)")
    if security_total > 0:
        print(f"Security robustness: {security_passed}/{security_total} ({(security_passed / security_total) * 100:.1f}%)")
    print(f"Detailed logs: {output_dir}")


if __name__ == "__main__":
    main()
