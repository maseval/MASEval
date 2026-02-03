"""Tau 2 Benchmark Example.

This example demonstrates running the Tau 2 benchmark with three framework options:
- default: MASEval's built-in DefaultAgentTau2Benchmark (mirrors original tau2-bench)
- smolagents: HuggingFace smolagents framework
- langgraph: LangChain's LangGraph framework

The Tau 2 benchmark evaluates single-agent customer service tasks across three domains:
- airline: Flight reservation management (50 base tasks)
- retail: E-commerce order management (114 base tasks)
- telecom: Telecom customer service (114 base tasks)

Key Features:
- Real tool implementations that modify actual database state
- Deterministic evaluation via database state comparison
- Single-agent customer service representative tasks
- Pass@k metrics recommended for robust evaluation

Reference:
    Paper: https://arxiv.org/abs/2506.07982
    Data: https://github.com/sierra-research/tau2-bench

Usage:
    # Run with default agent on retail domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain retail --limit 5

    # Run with smolagents on airline domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain airline --limit 5

    # Run with langgraph on telecom domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework langgraph --domain telecom --limit 5

    # Run with multiple repeats for Pass@k computation
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain retail --repeats 4

    # Run with a specific model
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain airline --model gpt-5-mini
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from google.genai import Client as GoogleGenAIClient
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI as OpenAIClient
from pydantic import Field, create_model
from smolagents import FinalAnswerTool, OpenAIServerModel, ToolCallingAgent
from smolagents import Tool as SmolagentsTool
from typing_extensions import Annotated, TypedDict

from maseval import AgentAdapter, Task, User
from maseval.benchmark.tau2 import (
    DefaultAgentTau2Benchmark,
    Tau2Benchmark,
    Tau2Environment,
    Tau2User,
    compute_benchmark_metrics,
    compute_pass_at_k,
    configure_model_ids,
    ensure_data_exists,
    load_tasks,
)
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin
from maseval.interface.agents.langgraph import LangGraphAgentAdapter
from maseval.interface.agents.smolagents import SmolAgentAdapter
from maseval.interface.inference import OpenAIModelAdapter
from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

from dotenv import load_dotenv

load_dotenv()

FrameworkType = Literal["default", "smolagents", "langgraph"]


# =============================================================================
# Model Setup
# =============================================================================

_google_client: Optional[GoogleGenAIClient] = None


def get_google_client() -> GoogleGenAIClient:
    """Get or create the shared Google GenAI client."""
    global _google_client
    if _google_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        _google_client = GoogleGenAIClient(api_key=api_key)
    return _google_client


_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create the shared OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAIClient(api_key=api_key)
    return _openai_client


def create_model_adapter(model_id: str = "gemini-2.5-flash") -> GoogleGenAIModelAdapter:
    """Create a Google GenAI model adapter."""
    return GoogleGenAIModelAdapter(get_google_client(), model_id=model_id)


def get_provider_from_model(model_id: str) -> Literal["openai", "google", "anthropic"]:
    """Determine the provider from a model ID."""
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "chatgpt"]):
        return "openai"
    if any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "google"
    if any(x in model_lower for x in ["claude"]):
        return "anthropic"

    return "openai"


# =============================================================================
# Default Framework Implementation
# =============================================================================


class DefaultTau2User(Tau2User):
    """Tau2 User that provides a simple callable for the default agent."""

    def get_tool(self) -> Dict[str, Any]:
        """Return user tool info for the default agent."""

        def ask_user(question: str) -> str:
            """Ask the customer a question to clarify their request or get additional information."""
            return self.respond(question)

        return {"ask_user": ask_user}


class GoogleGenAITau2Benchmark(DefaultAgentTau2Benchmark):
    """Tau2 Benchmark using Google GenAI for the default agent."""

    def __init__(self, model_id: str = "gemini-2.5-flash", **kwargs: Any):
        agent_data = kwargs.pop("agent_data", {})
        agent_data["model_id"] = model_id
        super().__init__(agent_data=agent_data, **kwargs)
        self._model_id = model_id

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> GoogleGenAIModelAdapter:
        """Create a Google GenAI model adapter with tool calling support."""
        adapter = GoogleGenAIModelAdapter(get_google_client(), model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> DefaultTau2User:
        """Create user simulator with tool support for default agent."""
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        return DefaultTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
            tools=environment.create_user_tools(),
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[DefaultTau2User],
    ):
        """Create the default agent with user tool support."""
        agents_to_run, agents_dict = super().setup_agents(agent_data, environment, task, user)

        if user is not None:
            agent = agents_dict["default_agent"]._agent
            user_tools = user.get_tool()
            agent.tools.update(user_tools)

        return agents_to_run, agents_dict


class OpenAITau2Benchmark(DefaultAgentTau2Benchmark):
    """Tau2 Benchmark using OpenAI for the default agent."""

    def __init__(self, model_id: str = "gpt-5-mini", **kwargs: Any):
        agent_data = kwargs.pop("agent_data", {})
        agent_data["model_id"] = model_id
        super().__init__(agent_data=agent_data, **kwargs)
        self._model_id = model_id

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> OpenAIModelAdapter:
        """Create an OpenAI model adapter with tool calling support."""
        adapter = OpenAIModelAdapter(get_openai_client(), model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> DefaultTau2User:
        """Create user simulator with tool support for default agent."""
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        return DefaultTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
            tools=environment.create_user_tools(),
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[DefaultTau2User],
    ):
        """Create the default agent with user tool support."""
        agents_to_run, agents_dict = super().setup_agents(agent_data, environment, task, user)

        if user is not None:
            agent = agents_dict["default_agent"]._agent
            user_tools = user.get_tool()
            agent.tools.update(user_tools)

        return agents_to_run, agents_dict


# =============================================================================
# Smolagents Framework Implementation
# =============================================================================


class SmolagentsToolWrapper(SmolagentsTool, ConfigurableMixin, TraceableMixin):
    """Smolagents wrapper for Tau2 real tools."""

    skip_forward_signature_validation = True

    def __init__(self, name: str, func: callable, description: str, inputs: Dict[str, Any]):
        self._tool_func = func
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = "string"
        self._call_count = 0
        super().__init__()

    def forward(self, **kwargs) -> str:
        self._call_count += 1
        try:
            result = self._tool_func(**kwargs)
            if hasattr(result, "model_dump"):
                return str(result.model_dump())
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def gather_traces(self) -> Dict[str, Any]:
        return {"tool_name": self.name, "call_count": self._call_count}

    def gather_config(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


class SmolagentsTau2User(Tau2User):
    """Tau2 User with smolagents tool integration."""

    def get_tool(self):
        """Return a smolagents-compatible user input tool."""
        user = self

        class UserInputTool(SmolagentsTool):
            name = "user_input"
            description = "Ask the customer a question to clarify their request or get additional information."
            inputs = {"question": {"type": "string", "description": "The question to ask the customer."}}
            output_type = "string"

            def forward(self, question: str) -> str:
                return user.respond(question)

        return UserInputTool()


class SmolagentsTau2Benchmark(Tau2Benchmark):
    """Tau2 Benchmark implementation for smolagents."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID."""
        adapter = create_model_adapter(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> SmolagentsTau2User:
        """Create smolagents-compatible user simulator."""
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        return SmolagentsTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create smolagents customer service agent."""
        model_id = agent_data.get("model_id", "gemini-2.5-flash")
        smol_model = OpenAIServerModel(
            model_id=model_id,
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        tools: List[SmolagentsTool] = []
        for name, func in environment.tools.items():
            tool_meta = environment.toolkit.get_tool_metadata(name)
            wrapper = SmolagentsToolWrapper(
                name=name,
                func=func,
                description=tool_meta.get("description", f"Execute {name}"),
                inputs=tool_meta.get("inputs", {}),
            )
            tools.append(wrapper)
            self.register("tools", name, wrapper)

        tools.append(FinalAnswerTool())

        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                tools.append(user_tool)

        system_prompt = f"""You are a customer service agent. Follow these policies:

{environment.policy}

Be helpful, accurate, and follow all policies strictly."""

        agent = ToolCallingAgent(
            model=smol_model,
            tools=tools,
            name="customer_service_agent",
            system_prompt=system_prompt,
            max_steps=25,
            verbosity_level=0,
        )

        adapter = SmolAgentAdapter(agent, name="customer_service_agent")
        return [adapter], {"customer_service_agent": adapter}


# =============================================================================
# LangGraph Framework Implementation
# =============================================================================


def _create_langgraph_tool(name: str, func: callable, description: str, inputs: Dict[str, Any]) -> StructuredTool:
    """Create a LangGraph StructuredTool from a Tau2 tool."""

    def tool_func(**kwargs) -> str:
        try:
            result = func(**kwargs)
            if hasattr(result, "model_dump"):
                return str(result.model_dump())
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    tool_func.__name__ = name
    tool_func.__doc__ = description

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    field_definitions = {}
    for param_name, param_info in inputs.items():
        param_type = type_mapping.get(param_info.get("type", "string"), str)
        param_desc = param_info.get("description", f"Parameter {param_name}")
        field_definitions[param_name] = (param_type, Field(description=param_desc))

    if field_definitions:
        args_schema = create_model(f"{name}Input", **field_definitions)
    else:
        args_schema = None

    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=description,
        args_schema=args_schema,
    )


class LangGraphTau2User(Tau2User):
    """Tau2 User with LangGraph tool integration."""

    def get_tool(self):
        """Return a LangGraph-compatible user input tool."""

        def user_input(question: str) -> str:
            """Ask the customer a question to clarify their request."""
            return self.respond(question)

        return StructuredTool.from_function(
            func=user_input,
            name="user_input",
            description="Ask the customer a question to clarify requirements or get additional information.",
        )


class AgentState(TypedDict):
    """LangGraph agent state."""

    messages: Annotated[list, add_messages]


class LangGraphTau2Benchmark(Tau2Benchmark):
    """Tau2 Benchmark implementation for langgraph."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID."""
        adapter = create_model_adapter(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
    ) -> LangGraphTau2User:
        """Create langgraph-compatible user simulator."""
        user_data = task.user_data
        instructions = user_data.get("instructions", {})

        if isinstance(instructions, str):
            scenario = instructions
        elif isinstance(instructions, dict):
            parts = []
            if instructions.get("reason_for_call"):
                parts.append(f"Reason for call: {instructions['reason_for_call']}")
            if instructions.get("known_info"):
                parts.append(f"Known info: {instructions['known_info']}")
            if instructions.get("task_instructions"):
                parts.append(f"Task: {instructions['task_instructions']}")
            scenario = "\n".join(parts)
        else:
            scenario = ""

        persona = user_data.get("persona")
        if persona:
            scenario = f"Persona: {persona}\n\n{scenario}"

        user_model_id = self._get_user_model_id(task)
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        return LangGraphTau2User(
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Tau2Environment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create langgraph customer service agent."""
        model_id = agent_data.get("model_id", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        tools: List[StructuredTool] = []
        for name, func in environment.tools.items():
            tool_meta = environment.toolkit.get_tool_metadata(name)
            lg_tool = _create_langgraph_tool(
                name=name,
                func=func,
                description=tool_meta.get("description", f"Execute {name}"),
                inputs=tool_meta.get("inputs", {}),
            )
            tools.append(lg_tool)

        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                tools.append(user_tool)

        llm_with_tools = llm.bind_tools(tools)

        system_prompt = f"""You are a customer service agent. Follow these policies:

{environment.policy}

Be helpful, accurate, and follow all policies strictly."""

        def call_agent(state: AgentState):
            messages = state["messages"]
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                system_msg = SystemMessage(content=system_prompt)
                messages = [system_msg] + list(messages)
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("chatbot", call_agent)
        graph.add_node("tools", ToolNode(tools=tools))
        graph.add_conditional_edges("chatbot", tools_condition)
        graph.add_edge("tools", "chatbot")
        graph.set_entry_point("chatbot")

        # Recursion limit of 50 allows ~25 tool calls
        compiled_graph = graph.compile().with_config({"recursion_limit": 50})
        adapter = LangGraphAgentAdapter(compiled_graph, name="customer_service_agent")

        return [adapter], {"customer_service_agent": adapter}


# =============================================================================
# Benchmark Class Selection
# =============================================================================


def get_benchmark_class(framework: FrameworkType, model_id: str = "gemini-2.5-flash") -> type:
    """Get the benchmark class for the specified framework."""
    if framework == "default":
        provider = get_provider_from_model(model_id)
        if provider == "google":
            return GoogleGenAITau2Benchmark
        elif provider == "openai":
            return OpenAITau2Benchmark
        else:
            raise ValueError(f"Provider '{provider}' is not yet supported. Supported: google, openai")
    elif framework == "smolagents":
        return SmolagentsTau2Benchmark
    elif framework == "langgraph":
        return LangGraphTau2Benchmark
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_benchmark(
    framework: FrameworkType,
    domain: Literal["airline", "retail", "telecom"],
    model_id: str = "gemini-2.5-flash",
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run the Tau2 benchmark."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Ensuring Tau2 data is available...")
    ensure_data_exists(domain=domain)

    print(f"Loading {domain} domain tasks...")
    tasks = load_tasks(domain=domain, split="base", limit=limit)
    print(f"Loaded {len(tasks)} tasks")

    configure_model_ids(
        tasks,
        user_model_id=model_id,
        evaluator_model_id=model_id,
    )

    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"tau2_{domain}_{framework}_{{timestamp}}.jsonl",
    )

    BenchmarkClass = get_benchmark_class(framework, model_id)

    benchmark_kwargs: Dict[str, Any] = {
        "callbacks": [logger],
        "n_task_repeats": n_task_repeats,
        "fail_on_setup_error": True,
        "fail_on_task_error": framework == "default",
        "fail_on_evaluation_error": True,
    }

    if framework == "default":
        benchmark_kwargs["agent_data"] = {
            "model_id": model_id,
            "verbose": 2,
            "llm_args": {"temperature": temperature},
        }
        benchmark = BenchmarkClass(model_id=model_id, **benchmark_kwargs)
    else:
        benchmark_kwargs["agent_data"] = {"model_id": model_id}
        benchmark = BenchmarkClass(**benchmark_kwargs)

    print(f"\nRunning {framework} benchmark on {domain} domain...")
    print(f"Model: {model_id}")
    results = benchmark.run(tasks=tasks)

    summary = compute_benchmark_metrics(results)

    if n_task_repeats > 1:
        k_values = list(range(1, min(n_task_repeats + 1, 5)))
        pass_at_k = compute_pass_at_k(results, k_values=k_values)
        summary["pass_at_k"] = pass_at_k

    print("\n" + "=" * 60)
    print(f"TAU2 BENCHMARK SUMMARY ({framework})")
    print("=" * 60)
    print(f"Framework: {framework}")
    print(f"Domain: {domain}")
    print(f"Model: {model_id}")
    total = summary.get("total_tasks", len(results))
    print(f"Total Tasks: {total}")
    print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
    print(f"Mean Reward: {summary.get('mean_reward', 0):.4f}")

    if "pass_at_k" in summary:
        print("\nPass@k Metrics:")
        for k, score in summary["pass_at_k"].items():
            print(f"  {k}: {score:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)

    return summary


def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the Tau2 benchmark with default, smolagents, or langgraph framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default agent on retail domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain retail --limit 5

    # Run with smolagents on airline domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework smolagents --domain airline --limit 5

    # Run with langgraph on telecom domain
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework langgraph --domain telecom --limit 5

    # Run with 4 repetitions per task for Pass@k
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain retail --repeats 4

    # Run with a specific model
    uv run python examples/tau2_benchmark/tau2_benchmark.py --framework default --domain airline --model gpt-5-mini
        """,
    )

    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["default", "smolagents", "langgraph"],
        help="Agent framework to use",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["airline", "retail", "telecom"],
        help="Tau2 domain to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each task for Pass@k (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0 for deterministic output)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_benchmark(
        framework=args.framework,
        domain=args.domain,
        model_id=args.model,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
