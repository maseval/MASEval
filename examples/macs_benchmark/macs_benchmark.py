"""MACS Benchmark Example.

This example demonstrates running the AWS Multi-Agent Collaboration Scenarios (MACS)
benchmark with either smolagents or langgraph frameworks.

The MACS benchmark evaluates multi-agent collaboration in three enterprise domains:
- Travel: 10 agents, 52 tools (flight booking, hotels, weather, etc.) - 2-level hierarchy
- Mortgage: 6 agents, 35 tools (loan processing, document handling, etc.) - 2-level hierarchy
- Software: 8 agents, 4 tools (code review, issue tracking, etc.) - 3-level hierarchy

Agent Hierarchy:
    Travel/Mortgage: supervisor -> specialist agents
    Software: supervisor -> deploy_agent -> infrastructure_agent, application_agent

Reference:
    Paper: https://arxiv.org/abs/2412.05449
    Data: https://github.com/aws-samples/multiagent-collab-scenario-benchmark

Usage:
    # Run with smolagents on travel domain
    uv run python examples/macs_benchmark.py --framework smolagents --domain travel --limit 5

    # Run with langgraph on mortgage domain
    uv run python examples/macs_benchmark.py --framework langgraph --domain mortgage --limit 5

    # Run a single task by ID for debugging
    uv run python examples/macs_benchmark.py --framework smolagents --domain travel --task-id task_001
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Third-party imports (both frameworks will be installed)
from google.genai import Client as GoogleGenAIClient

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, FinalAnswerTool
from smolagents import Tool as SmolagentsTool

# langgraph imports
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated

# MASEval imports
from maseval import AgentAdapter, Environment, Task, User
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin
from maseval.interface.agents.smolagents import SmolAgentAdapter
from maseval.interface.agents.langgraph import LangGraphAgentAdapter
from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

from maseval.benchmark.macs import (
    MACSBenchmark,
    MACSEnvironment,
    MACSGenericTool,
    MACSUser,
    compute_benchmark_metrics,
    configure_model_ids,
    ensure_data_exists,
    load_agent_config,
    load_tasks,
)


# =============================================================================
# Model Setup
# =============================================================================

# Shared client for all model adapters (reuses connection)
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


def create_model(
    model_id: str = "gemini-2.5-flash",
) -> GoogleGenAIModelAdapter:
    """Create a Google GenAI model adapter with separate tracing.

    All adapters share the same underlying client connection for efficiency,
    but each has its own trace logs for debugging.

    Args:
        model_id: Model identifier (default: gemini-2.5-flash)

    Returns:
        Configured GoogleGenAIModelAdapter with independent trace logs
    """
    return GoogleGenAIModelAdapter(get_google_client(), model_id=model_id)


# =============================================================================
# Smolagents Implementation
# =============================================================================


class SmolagentsToolWrapper(SmolagentsTool, ConfigurableMixin, TraceableMixin):
    """Smolagents wrapper for MACSGenericTool."""

    skip_forward_signature_validation = True

    def __init__(self, generic_tool: MACSGenericTool):
        self.generic_tool = generic_tool
        self.name = generic_tool.name
        self.description = generic_tool.description
        self.inputs = generic_tool.inputs
        self.output_type = generic_tool.output_type
        super().__init__()

    def forward(self, **kwargs) -> str:
        return self.generic_tool(**kwargs)

    def gather_traces(self) -> Dict[str, Any]:
        return self.generic_tool.gather_traces()

    def gather_config(self) -> Dict[str, Any]:
        return self.generic_tool.gather_config()


class SmolagentsMACSUser(MACSUser):
    """MACS User with smolagents tool integration."""

    def get_tool(self):
        """Return a smolagents-compatible user input tool."""
        user = self

        class UserInputTool(SmolagentsTool):
            name = "user_input"
            description = "Asks for user's input on a specific question."
            inputs = {"question": {"type": "string", "description": "The question to ask the user."}}
            output_type = "string"

            def forward(self, question: str) -> str:
                return user.respond(question)

        return UserInputTool()


class SmolagentsMACSBenchmark(MACSBenchmark):
    """MACS Benchmark implementation for smolagents with multi-agent hierarchy."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID.

        Each component (tool, user, evaluator) gets its own adapter for separate tracing.
        Registration is handled via kwargs passed by the base class.

        Args:
            model_id: The model identifier (e.g., "gemini-2.5-flash")
            **kwargs: Optional registration info (register_name)

        Returns:
            Configured GoogleGenAIModelAdapter
        """
        adapter = create_model(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
    ) -> SmolagentsMACSUser:
        """Create smolagents-compatible user simulator.

        Extends base MACSUser with smolagents-specific tool integration.
        Model ID is read from task.user_data["model_id"].
        """
        scenario = task.metadata.get("scenario", "")
        user_model_id = self._get_user_model_id(task)

        # Create dedicated model for user via get_model_adapter
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        user = SmolagentsMACSUser(
            name="Simulated User",
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

        # Register the user's simulator for tracing
        self.register("simulators", "user_simulator", user.simulator)

        return user

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,  # type: ignore[override]
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create smolagents multi-agent hierarchy.

        Implements the exact agent topology from agents.json:
        - Travel/Mortgage: 2-level hierarchy (supervisor -> specialists)
        - Software: 3-level hierarchy (supervisor -> deploy_agent -> infra/app agents)

        Each tool has its own ModelAdapter for separate tracing.
        """
        # Create smolagents model
        smol_model = OpenAIServerModel(
            model_id="gemini-2.5-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Build agent lookup
        agents_config = agent_data.get("agents", [])
        agent_lookup = {a["agent_id"]: a for a in agents_config}
        primary_agent_id = agent_data.get("primary_agent_id", "supervisor")

        # Wrap all generic tools for smolagents and register them for tracing
        # Each tool has its own model from MACSEnvironment.create_tools()
        # Models are already registered by the environment via get_model_adapter()
        tool_wrappers: Dict[str, SmolagentsToolWrapper] = {}
        for name, tool in environment.tools.items():
            wrapper = SmolagentsToolWrapper(tool)
            tool_wrappers[name] = wrapper
            self.register("tools", name, wrapper)
            # Register the tool's simulator for tracing
            self.register("simulators", f"simulator_tool_{name}", tool.simulator)

        # Helper to get tools for an agent
        def get_agent_tools(agent_spec: Dict[str, Any]) -> List["SmolagentsTool"]:
            """Get wrapped tools for an agent based on its tool groups."""
            agent_tools = environment.get_tools_for_agent(agent_spec)
            return [tool_wrappers[name] for name in agent_tools if name in tool_wrappers]

        # Recursive function to build agent hierarchy
        def build_agent(agent_id: str, depth: int = 0) -> ToolCallingAgent:
            """Build an agent with its sub-agents (managed_agents)."""
            agent_spec = agent_lookup.get(agent_id, {})

            # Get this agent's tools
            agent_tools: List["SmolagentsTool"] = get_agent_tools(agent_spec)
            agent_tools.append(FinalAnswerTool())

            # Build managed agents from reachable_agents
            managed_agents = []
            reachable = agent_spec.get("reachable_agents", [])

            for reachable_spec in reachable:
                sub_agent_id = reachable_spec.get("agent_id")
                if sub_agent_id and sub_agent_id in agent_lookup:
                    sub_agent = build_agent(sub_agent_id, depth + 1)
                    managed_agents.append(sub_agent)

            # Create the agent
            agent = ToolCallingAgent(
                model=smol_model,
                tools=agent_tools,
                managed_agents=managed_agents if managed_agents else None,
                name=agent_spec.get("agent_name", agent_id),
                description=agent_spec.get("agent_instruction", ""),
                max_steps=25,  # Allow more steps for complex multi-agent tasks
                verbosity_level=0,
            )

            return agent

        # Build the primary agent with full hierarchy
        primary_agent = build_agent(primary_agent_id)

        # Add user tool to primary agent if user simulator is available
        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                primary_agent.tools[user_tool.name] = user_tool

        # Wrap with adapter
        adapter = SmolAgentAdapter(primary_agent, name=primary_agent_id)

        return [adapter], {primary_agent_id: adapter}


# =============================================================================
# LangGraph Implementation
# =============================================================================


def _create_pydantic_model_from_inputs(name: str, inputs: Dict[str, Any]):
    """Create a Pydantic model from MACS tool inputs specification.

    Args:
        name: Name for the generated model class
        inputs: Dict mapping param names to {type, description}

    Returns:
        A Pydantic BaseModel class with the specified fields
    """
    from pydantic import Field, create_model

    # Map MACS types to Python types
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields = {}
    for param_name, param_spec in inputs.items():
        param_type = param_spec.get("type", "string")
        python_type = type_mapping.get(param_type, str)
        description = param_spec.get("description", "")

        # All fields are optional with empty string default for flexibility
        fields[param_name] = (python_type, Field(default="", description=description))

    return create_model(f"{name}Input", **fields)


def _create_langgraph_tool(generic_tool: MACSGenericTool) -> StructuredTool:
    """Create a LangGraph StructuredTool from a MACSGenericTool.

    This creates a proper Pydantic schema from the tool's inputs specification,
    which LangGraph/LangChain requires for tool calling.

    Args:
        generic_tool: The MACS generic tool to wrap

    Returns:
        A LangGraph-compatible StructuredTool
    """
    # Create the args schema from tool inputs
    args_schema = _create_pydantic_model_from_inputs(generic_tool.name, generic_tool.inputs)

    # Create a wrapper function that calls the generic tool
    def tool_func(**kwargs) -> str:
        return generic_tool(**kwargs)

    # Set function metadata for better tool descriptions
    tool_func.__name__ = generic_tool.name
    tool_func.__doc__ = generic_tool.description

    return StructuredTool(
        name=generic_tool.name,
        description=generic_tool.description,
        func=tool_func,
        args_schema=args_schema,
    )


class LangGraphToolWrapper(ConfigurableMixin, TraceableMixin):
    """LangGraph wrapper for MACSGenericTool.

    This wrapper creates a LangGraph-compatible StructuredTool from a MACSGenericTool
    by dynamically generating a Pydantic schema from the tool's input specification.
    """

    def __init__(self, generic_tool: MACSGenericTool):
        self.generic_tool = generic_tool
        self.tool = _create_langgraph_tool(generic_tool)

    def __call__(self, *args, **kwargs) -> str:
        return self.generic_tool(**kwargs)

    def gather_traces(self) -> Dict[str, Any]:
        return self.generic_tool.gather_traces()

    def gather_config(self) -> Dict[str, Any]:
        return self.generic_tool.gather_config()


class LangGraphMACSUser(MACSUser):
    """MACS User with LangGraph tool integration."""

    def get_tool(self):
        """Return a LangGraph-compatible user input tool."""

        def user_input(question: str) -> str:
            """Ask the user a question to understand their complete requirements."""
            return self.respond(question)

        return StructuredTool.from_function(
            func=user_input,
            name="user_input",
            description="Ask the user a question. Use this to clarify requirements or get additional information.",
        )


# LangGraph agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class LangGraphMACSBenchmark(MACSBenchmark):
    """MACS Benchmark implementation for langgraph with multi-agent hierarchy."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given model ID.

        Each component (tool, user, evaluator) gets its own adapter for separate tracing.
        Registration is handled via kwargs passed by the base class.

        Args:
            model_id: The model identifier (e.g., "gemini-2.5-flash")
            **kwargs: Optional registration info (register_name)

        Returns:
            Configured GoogleGenAIModelAdapter
        """
        adapter = create_model(model_id=model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
    ) -> LangGraphMACSUser:
        """Create langgraph-compatible user simulator.

        Extends base MACSUser with LangGraph-specific tool integration.
        Model ID is read from task.user_data["model_id"].
        """
        scenario = task.metadata.get("scenario", "")
        user_model_id = self._get_user_model_id(task)

        # Create dedicated model for user via get_model_adapter
        user_model = self.get_model_adapter(user_model_id, register_name="user_simulator")

        user = LangGraphMACSUser(
            name="Simulated User",
            model=user_model,
            scenario=scenario,
            initial_query=task.query,
        )

        # Register the user's simulator for tracing
        self.register("simulators", "user_simulator", user.simulator)

        return user

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,  # type: ignore[override]
        task: Task,
        user: Optional[User],
    ) -> Tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create langgraph multi-agent hierarchy.

        Uses subgraphs to implement the agent hierarchy from agents.json.
        Each tool has its own ModelAdapter for separate tracing.
        """
        # Create LangChain model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        # Build agent lookup
        agents_config = agent_data.get("agents", [])
        agent_lookup = {a["agent_id"]: a for a in agents_config}
        primary_agent_id = agent_data.get("primary_agent_id", "supervisor")

        # Wrap all generic tools for LangGraph and register them for tracing
        # Each tool has its own model from MACSEnvironment.create_tools()
        # Models are already registered by the environment via get_model_adapter()
        tool_wrappers: Dict[str, LangGraphToolWrapper] = {}
        for name, tool in environment.tools.items():
            wrapper = LangGraphToolWrapper(tool)
            tool_wrappers[name] = wrapper
            self.register("tools", name, wrapper)
            # Register the tool's simulator for tracing
            self.register("simulators", f"simulator_tool_{name}", tool.simulator)

        # Helper to get tools for an agent
        def get_agent_tools(agent_spec: Dict[str, Any]) -> List[StructuredTool]:
            """Get wrapped tools for an agent based on its tool groups."""
            agent_tools = environment.get_tools_for_agent(agent_spec)
            return [tool_wrappers[name].tool for name in agent_tools if name in tool_wrappers]

        # Build agent graph recursively
        def build_agent_graph(agent_id: str) -> StateGraph:
            """Build a LangGraph for an agent with potential sub-agents."""
            agent_spec = agent_lookup.get(agent_id, {})

            # Get this agent's tools
            agent_tools = get_agent_tools(agent_spec)

            # Build sub-agent tools from reachable_agents
            reachable = agent_spec.get("reachable_agents", [])

            for reachable_spec in reachable:
                sub_agent_id = reachable_spec.get("agent_id")
                if sub_agent_id and sub_agent_id in agent_lookup:
                    sub_spec = agent_lookup[sub_agent_id]
                    sub_graph = build_agent_graph(sub_agent_id).compile()

                    # Create a tool that invokes the sub-agent
                    def make_sub_agent_tool(graph, name, description):
                        def invoke_sub_agent(query: str) -> str:
                            """Delegate task to sub-agent."""
                            from langchain_core.messages import HumanMessage

                            result = graph.invoke({"messages": [HumanMessage(content=query)]})
                            if result["messages"]:
                                return result["messages"][-1].content
                            return "No response from sub-agent."

                        return StructuredTool.from_function(
                            func=invoke_sub_agent,
                            name=name,
                            description=description,
                        )

                    sub_tool = make_sub_agent_tool(
                        sub_graph,
                        sub_spec.get("agent_name", sub_agent_id),
                        reachable_spec.get("scenario", sub_spec.get("agent_instruction", "")),
                    )
                    agent_tools.append(sub_tool)

            # Build this agent's graph
            agent_name = agent_spec.get("agent_name", agent_id)
            agent_instruction = agent_spec.get("agent_instruction", "")

            if agent_tools:
                llm_with_tools = llm.bind_tools(agent_tools)
            else:
                llm_with_tools = llm

            def call_agent(state: AgentState):
                messages = state["messages"]
                has_system = any(isinstance(m, SystemMessage) for m in messages)
                if not has_system:
                    system_msg = SystemMessage(content=f"You are {agent_name}. {agent_instruction}")
                    messages = [system_msg] + list(messages)
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}

            graph = StateGraph(AgentState)
            graph.add_node("chatbot", call_agent)

            if agent_tools:
                graph.add_node("tools", ToolNode(tools=agent_tools))
                graph.add_conditional_edges("chatbot", tools_condition)
                graph.add_edge("tools", "chatbot")
            else:
                graph.add_edge("chatbot", END)

            graph.set_entry_point("chatbot")
            return graph

        # Build primary agent graph
        primary_spec = agent_lookup.get(primary_agent_id, {})
        primary_tools: List[StructuredTool] = get_agent_tools(primary_spec)

        # Add user tool if available
        if user and hasattr(user, "get_tool"):
            user_tool = user.get_tool()
            if user_tool:
                primary_tools.append(user_tool)

        # Build sub-agent tools for primary agent
        reachable = primary_spec.get("reachable_agents", [])
        for reachable_spec in reachable:
            sub_agent_id = reachable_spec.get("agent_id")
            if sub_agent_id and sub_agent_id in agent_lookup:
                sub_spec = agent_lookup[sub_agent_id]
                sub_graph = build_agent_graph(sub_agent_id).compile()

                def make_sub_agent_tool(graph, name, description):
                    def invoke_sub_agent(query: str) -> str:
                        """Delegate task to sub-agent."""
                        from langchain_core.messages import HumanMessage

                        result = graph.invoke({"messages": [HumanMessage(content=query)]})
                        if result["messages"]:
                            return result["messages"][-1].content
                        return "No response from sub-agent."

                    return StructuredTool.from_function(
                        func=invoke_sub_agent,
                        name=name,
                        description=description,
                    )

                sub_tool = make_sub_agent_tool(
                    sub_graph,
                    sub_spec.get("agent_name", sub_agent_id),
                    reachable_spec.get("scenario", sub_spec.get("agent_instruction", "")),
                )
                primary_tools.append(sub_tool)

        # Build primary agent graph
        if primary_tools:
            llm_with_tools = llm.bind_tools(primary_tools)
        else:
            llm_with_tools = llm

        primary_name = primary_spec.get("agent_name", primary_agent_id)
        primary_instruction = primary_spec.get("agent_instruction", "")

        def call_primary(state: AgentState):
            messages = state["messages"]
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                system_msg = SystemMessage(content=f"You are {primary_name}. {primary_instruction}")
                messages = [system_msg] + list(messages)
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("chatbot", call_primary)

        if primary_tools:
            graph.add_node("tools", ToolNode(tools=primary_tools))
            graph.add_conditional_edges("chatbot", tools_condition)
            graph.add_edge("tools", "chatbot")
        else:
            graph.add_edge("chatbot", END)

        graph.set_entry_point("chatbot")
        compiled_graph = graph.compile()

        # Wrap with adapter
        adapter = LangGraphAgentAdapter(compiled_graph, name=primary_agent_id)

        return [adapter], {primary_agent_id: adapter}


# =============================================================================
# Main Entry Point
# =============================================================================


def get_benchmark_class(framework: Literal["smolagents", "langgraph"]) -> type:
    """Get the benchmark class for the specified framework.

    Args:
        framework: Either "smolagents" or "langgraph"

    Returns:
        The appropriate MACSBenchmark subclass
    """
    if framework == "smolagents":
        return SmolagentsMACSBenchmark
    elif framework == "langgraph":
        return LangGraphMACSBenchmark
    else:
        raise ValueError(f"Unsupported framework: {framework}. Choose 'smolagents' or 'langgraph'.")


def run_benchmark(
    framework: Literal["smolagents", "langgraph"],
    domain: Literal["travel", "mortgage", "software"],
    limit: Optional[int] = None,
    task_id: Optional[str] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the MACS benchmark.

    Args:
        framework: Agent framework to use
        domain: MACS domain (travel, mortgage, or software)
        limit: Maximum number of tasks to run (None for all)
        task_id: Specific task ID to run (for debugging)
        n_task_repeats: Number of times to repeat each task
        output_dir: Directory for results (default: examples/results/)

    Returns:
        Summary metrics from the benchmark run
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure data is downloaded
    print("Ensuring MACS data is available...")
    ensure_data_exists(verbose=1)

    # Load data
    print(f"Loading {domain} domain tasks...")
    tasks = load_tasks(domain, limit=limit)

    # Filter to specific task if requested
    if task_id:
        tasks = [t for t in tasks if str(t.id) == task_id]
        if not tasks:
            raise ValueError(f"Task with ID '{task_id}' not found in {domain} domain")
        print(f"Running single task: {task_id}")

    # Configure model IDs for all benchmark components
    # This sets model_id in environment_data, user_data, and evaluation_data
    # for each task. The benchmark reads these when setting up components.
    configure_model_ids(
        tasks,
        tool_model_id="gemini-2.5-flash",
        user_model_id="gemini-2.5-flash",
        evaluator_model_id="gemini-2.5-flash",
    )

    agent_config = load_agent_config(domain)

    # Print agent hierarchy info
    agents_count = len(agent_config.get("agents", []))
    primary_agent_id = agent_config.get("primary_agent_id", "unknown")
    print(f"Loaded {len(tasks)} tasks with {agents_count}-agent hierarchy")
    print(f"Primary agent: {primary_agent_id}")

    # Setup callback for logging results
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"{domain}_{framework}_{{timestamp}}.jsonl",
    )

    # Get benchmark class and instantiate
    BenchmarkClass = get_benchmark_class(framework)
    benchmark = BenchmarkClass(
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
        fail_on_setup_error=True,
        fail_on_task_error=True,
        fail_on_evaluation_error=True,
    )

    # Run benchmark
    print(f"\nRunning {framework} benchmark on {domain} domain...")
    results = benchmark.run(tasks=tasks, agent_data=agent_config)

    # Compute summary metrics
    summary = compute_benchmark_metrics(results)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Framework: {framework}")
    print(f"Domain: {domain}")
    total = summary["total_tasks"]
    print(f"Total Tasks: {total}")
    print(f"Scored Tasks: {summary['scored_tasks']}")
    print(f"Successful Tasks (Overall GSR=1.0): {summary['successful_tasks']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")

    # Show status breakdown with counts and proportions
    print("\nStatus Breakdown:")
    status_counts = summary.get("status_counts", {})
    for status, count in sorted(status_counts.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {status:<25} {count:>4} ({pct:5.1f}%)")

    # Show excluded tasks if any
    excluded = summary.get("excluded", {})
    total_excluded = sum(excluded.values())
    if total_excluded > 0:
        pct_excluded = (total_excluded / total * 100) if total > 0 else 0
        print(f"\nExcluded from scoring: {total_excluded} ({pct_excluded:.1f}%)")
        for status, count in sorted(excluded.items()):
            if count > 0:
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {status:<25} {count:>4} ({pct:5.1f}%)")

    print("\nMean Metrics (scored tasks only):")
    for metric, value in summary["mean_metrics"].items():
        print(f"  {metric:<25} {value:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 50)

    return summary


def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the MACS benchmark with smolagents or langgraph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with smolagents on travel domain
    uv run python examples/macs_benchmark.py --framework smolagents --domain travel

    # Run with langgraph on mortgage domain, limited to 5 tasks
    uv run python examples/macs_benchmark.py --framework langgraph --domain mortgage --limit 5

    # Run a single task by ID for debugging
    uv run python examples/macs_benchmark.py --framework smolagents --domain travel --task-id task_001

    # Run with 3 repetitions per task
    uv run python examples/macs_benchmark.py --framework smolagents --domain software --repeats 3
        """,
    )

    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        choices=["smolagents", "langgraph"],
        help="Agent framework to use",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["travel", "mortgage", "software"],
        help="MACS domain to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run a single task by ID (for debugging)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each task (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: examples/macs_benchmark/results/)",
    )

    args = parser.parse_args()

    run_benchmark(
        framework=args.framework,
        domain=args.domain,
        limit=args.limit,
        task_id=args.task_id,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
