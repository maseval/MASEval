"""
5-A-Day Benchmark - Framework-Agnostic Agent Evaluation

Educational example showcasing the maseval library through 5 diverse evaluation tasks.
Demonstrates framework-agnostic tools, multiple evaluation types, and single vs multi-agent patterns.

Key Features:
- Framework-agnostic BaseTools with conversion methods for smolagents, langgraph, llamaindex
- 7 evaluation types: unit tests, LLM judges, optimization, pattern matching, static analysis
- Environment loaded from tasks.json with automatic tool instantiation
- Both single-agent and orchestrator+specialist multi-agent configurations

Tasks:
0. Email & Banking: Verify payment and draft email (financial accuracy, privacy)
1. Finance Calculation: Calculate stock inheritance split (arithmetic, tool validation)
2. Code Generation: Implement DP algorithm (unit tests, complexity analysis, quality)
3. Calendar Scheduling: Find meeting overlaps using MCP (slot matching, logic validation)
4. Hotel Optimization: Select best hotel by criteria (ranking, search strategy, reasoning)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

from utils import sanitize_name  # type: ignore[unresolved-import]

from maseval import Benchmark, Environment, Evaluator, Task, TaskQueue, AgentAdapter, ModelAdapter, SeedGenerator
from maseval.core.callbacks.result_logger import FileResultLogger

# Import tool implementations
from tools import (  # type: ignore[unresolved-import]
    EmailToolCollection,
    BankingToolCollection,
    CalculatorToolCollection,
    CodeExecutionToolCollection,
    FamilyInfoToolCollection,
    StockPriceToolCollection,
    CalendarToolCollection,
    RunningAppToolCollection,
    GymTrackerToolCollection,
    HotelSearchToolCollection,
    MCPCalendarToolCollection,
    filter_tool_adapters_by_prefix,
    get_states,
)

# Import all evaluators dynamically
import evaluators  # type: ignore[unresolved-import]

# ============================================================================
# Parse command-line arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="5-A-Day Benchmark - Framework-Agnostic Agent Evaluation",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--framework", type=str, default="smolagents", choices=["smolagents", "langgraph", "llamaindex"], help="Agent framework to use"
)
parser.add_argument(
    "--config-type", type=str, default="single", choices=["single", "multi"], help="Agent configuration type (single-agent or multi-agent)"
)
parser.add_argument("--limit", type=int, default=None, help="Number of tasks to run (default: all tasks)")
parser.add_argument("--task", type=int, default=None, help="Run only a specific task by index (default: all tasks)")
parser.add_argument("--model-id", type=str, default="gemini-2.5-flash", help="Model identifier to use")
parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None for non-deterministic)")


# ============================================================================
# Global Model Factory
# ============================================================================


def get_model(model_id: str, framework: str, temperature: float = 0.7, seed: Optional[int] = None) -> Any:
    """Get a model instance for the specified framework.

    Args:
        model_id: Model identifier (e.g., 'gemini-2.5-flash')
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        temperature: Model temperature (default: 0.7)
        seed: Random seed for reproducibility (default: None)

    Returns:
        Framework-specific model instance
    """
    if framework == "smolagents":
        from smolagents import LiteLLMModel

        return LiteLLMModel(
            model_id=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,
        )

    elif framework == "langgraph":
        from langchain_litellm import ChatLiteLLM

        return ChatLiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,  # type: ignore[arg-type]
        )

    elif framework == "llamaindex":
        from llama_index.llms.litellm import LiteLLM

        return LiteLLM(
            model=f"gemini/{model_id}",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported framework: {framework}")


# ============================================================================
# Environment
# ============================================================================


class FiveADayEnvironment(Environment):
    """Environment that creates tools from task data and converts them to framework-specific types.

    Loads tool specifications from tasks.json, instantiates the appropriate tool classes
    with task-specific data, and converts them to the target framework (smolagents, langgraph, llamaindex).
    """

    def __init__(self, task_data: Dict[str, Any], framework: str, callbacks: Optional[List] = None):
        """Initialize environment with framework info.

        Args:
            task_data: Task configuration dictionary
            framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
            callbacks: Optional callback handlers
        """
        self.framework = framework
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize environment state from task data.

        Creates state objects for tools that require them (e.g., EmailState, BankingState).
        State objects are stored alongside raw environment data for tool initialization.
        """

        env_data = task_data["environment_data"].copy()
        tool_names = env_data.get("tools", [])

        # Initialize state objects for tools that need them
        states = get_states(tool_names, env_data)
        env_data.update(states)

        return env_data

    def create_tools(self) -> Dict[str, Any]:
        """Create tool instances from environment_data and convert to framework-specific types.

        Returns:
            Dict mapping tool names to framework-specific tool objects
        """
        tools_dict: Dict[str, Any] = {}

        # Map tool names to tool collection classes and their initialization data
        tool_mapping = {
            "email": (EmailToolCollection, lambda: (self.state["email_state"],)),
            "banking": (BankingToolCollection, lambda: (self.state["banking_state"],)),
            "calculator": (CalculatorToolCollection, lambda: ()),
            "python_executor": (CodeExecutionToolCollection, lambda: (self.state["python_executor_state"],)),
            "family_info": (FamilyInfoToolCollection, lambda: (self.state["family_info"],)),
            "stock_price": (StockPriceToolCollection, lambda: (self.state["stock_price_lookup"],)),
            "calendar": (CalendarToolCollection, lambda: (self.state["calendar_state"],)),
            "running_app": (RunningAppToolCollection, lambda: (self.state["running_app_state"],)),
            "gym_tracker": (GymTrackerToolCollection, lambda: (self.state["gym_tracker_state"],)),
            "hotel_search": (HotelSearchToolCollection, lambda: (self.state["hotel_search_state"],)),
            "my_calendar_mcp": (MCPCalendarToolCollection, lambda: (self.state["my_calendar_mcp_state"],)),
            "other_calendar_mcp": (MCPCalendarToolCollection, lambda: (self.state["other_calendar_mcp_state"],)),
        }

        for tool_name in self.state["tools"]:
            if tool_name in tool_mapping:
                ToolClass, get_init_args = tool_mapping[tool_name]
                init_args = get_init_args()

                assert isinstance(init_args, tuple), f"Tool {tool_name} init_args must be a tuple"
                tool_instance = ToolClass(*init_args)

                # Get all sub-tools from the collection
                base_tools = tool_instance.get_sub_tools()

                # Convert each base tool to framework-specific tool
                for base_tool in base_tools:
                    framework_tool = self._convert_tool(base_tool)
                    # Use the base tool's name as the key
                    tool_key = getattr(base_tool, "name", None) or str(type(base_tool).__name__)
                    tools_dict[tool_key] = framework_tool

        return tools_dict

    def _convert_tool(self, base_tool):
        """Convert BaseTool to framework-specific tool adapter.

        Args:
            base_tool: BaseTool instance

        Returns:
            Tool adapter with gather_traces() for tracing support.
            For smolagents: SmolagentsToolAdapter (has .tool attribute for raw Tool)
            For langgraph: LangGraphToolAdapter (has .tool attribute for StructuredTool)
            For llamaindex: LlamaIndexToolAdapter (has .tool attribute for FunctionTool)
        """
        if self.framework == "smolagents":
            return base_tool.to_smolagents()
        elif self.framework == "langgraph":
            return base_tool.to_langgraph()
        elif self.framework == "llamaindex":
            return base_tool.to_llamaindex()
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")


# ============================================================================
# Framework-Specific Agent Builders
# ============================================================================


def build_smolagents_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build a single smolagents agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        seeds: Optional dict mapping agent_id to seed

    Returns:
        Tuple of (primary_adapter, all_adapters_dict) for consistent interface
    """
    from smolagents import ToolCallingAgent
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    model = get_model(model_id, "smolagents", temperature, seed)
    tool_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters.values()]
    sanitized_name = sanitize_name(primary_spec["agent_name"])

    agent = ToolCallingAgent(
        model=model,
        tools=tools,
        name=sanitized_name,
        instructions=primary_spec["agent_instruction"],
        verbosity_level=0,
    )

    adapter = SmolAgentAdapter(agent, primary_spec["agent_id"])
    return adapter, {primary_spec["agent_id"]: adapter}


def build_langgraph_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build a single langgraph agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        seeds: Optional dict mapping agent_id to seed

    Returns:
        Tuple of (primary_adapter, all_adapters_dict) for consistent interface
    """
    from langchain_core.messages import SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing_extensions import TypedDict, Annotated
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    model = get_model(model_id, "langgraph", temperature, seed)
    tool_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters.values()]

    class AgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]

    llm_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState):
        messages = state["messages"]
        has_system = any(isinstance(m, SystemMessage) for m in messages)
        if not has_system and primary_spec["agent_instruction"]:
            system_message = SystemMessage(content=primary_spec["agent_instruction"])
            messages = [system_message] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    graph = workflow.compile()
    adapter = LangGraphAgentAdapter(graph, primary_spec["agent_id"])
    return adapter, {primary_spec["agent_id"]: adapter}


def build_llamaindex_single_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build a single llamaindex agent.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: Empty list for single-agent (ignored)
        seeds: Optional dict mapping agent_id to seed

    Returns:
        Tuple of (primary_adapter, all_adapters_dict) for consistent interface
    """
    from llama_index.core.agent.workflow.react_agent import ReActAgent
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

    seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    model = get_model(model_id, "llamaindex", temperature, seed)
    tool_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, primary_spec["tools"])
    tools = [adapter.tool for adapter in tool_adapters.values()]

    agent = ReActAgent(
        tools=tools,
        llm=model,
        verbose=True,
        max_iterations=10,
        system_prompt=primary_spec.get("agent_instruction"),
    )

    adapter = LlamaIndexAgentAdapter(agent, primary_spec["agent_id"])
    return adapter, {primary_spec["agent_id"]: adapter}


def build_smolagents_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build smolagents multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        seeds: Optional dict mapping agent_id to seed

    Returns:
        Tuple of (primary_adapter, all_adapters_dict) where all_adapters_dict
        includes the orchestrator and all specialists for trace registration.
    """
    from smolagents import ToolCallingAgent, FinalAnswerTool
    from maseval.interface.agents.smolagents import SmolAgentAdapter

    specialist_agents = []
    specialist_adapters_dict: Dict[str, Any] = {}

    for agent_spec in specialist_specs:
        specialist_seed = seeds.get(agent_spec["agent_id"]) if seeds else None
        specialist_model = get_model(model_id, "smolagents", temperature, specialist_seed)
        specialist_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters.values()]
        specialist_tools.append(FinalAnswerTool())
        sanitized_name = sanitize_name(agent_spec["agent_name"])

        specialist = ToolCallingAgent(
            model=specialist_model,
            tools=specialist_tools,
            name=sanitized_name,
            description=agent_spec["agent_instruction"],
            instructions=agent_spec["agent_instruction"],
            verbosity_level=0,
        )
        specialist_agents.append(specialist)
        # Create adapter for each specialist for trace registration
        specialist_adapters_dict[agent_spec["agent_id"]] = SmolAgentAdapter(specialist, agent_spec["agent_id"])

    primary_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, primary_spec["tools"])
    primary_tools = [adapter.tool for adapter in primary_adapters.values()]
    primary_tools.append(FinalAnswerTool())
    sanitized_primary_name = sanitize_name(primary_spec["agent_name"])
    primary_seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    primary_model = get_model(model_id, "smolagents", temperature, primary_seed)

    agent = ToolCallingAgent(
        model=primary_model,
        tools=primary_tools,
        managed_agents=specialist_agents if specialist_agents else None,
        name=sanitized_primary_name,
        instructions=primary_spec["agent_instruction"],
        verbosity_level=0,
    )

    primary_adapter = SmolAgentAdapter(agent, primary_spec["agent_id"])

    # Return primary adapter and dict of all adapters (including primary) for trace registration
    all_adapters = {primary_spec["agent_id"]: primary_adapter, **specialist_adapters_dict}
    return primary_adapter, all_adapters


def build_langgraph_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build langgraph multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        seeds: Optional dict mapping agent_id to seed values

    Returns:
        Tuple of (primary_adapter, all_adapters_dict). Note: LangGraph multi-agent
        compiles specialists into graph nodes, so only the graph is traceable.
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool as create_tool
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing_extensions import TypedDict, Annotated
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    class MultiAgentState(TypedDict):
        messages: Annotated[List[Any], add_messages]

    # Create specialist subgraphs
    specialist_subgraphs = {}
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_instruction = agent_spec["agent_instruction"]
        specialist_seed = seeds.get(agent_id) if seeds else None
        specialist_model = get_model(model_id, "langgraph", temperature, specialist_seed)
        specialist_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters.values()]

        def make_specialist_node(spec_instruction, spec_tools, spec_model):
            def specialist_node(state: MultiAgentState):
                messages = state["messages"]

                # Extract delegated query from orchestrator's tool call
                delegated_query = None
                if messages and isinstance(messages[-1], AIMessage):
                    last_msg = messages[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        tool_call = last_msg.tool_calls[0]
                        if "args" in tool_call and "query" in tool_call["args"]:
                            delegated_query = tool_call["args"]["query"]

                if not delegated_query:
                    for msg in reversed(messages):
                        if hasattr(msg, "type") and msg.type == "human":
                            delegated_query = msg.content
                            break

                if not delegated_query:
                    delegated_query = "Please help with the task."

                specialist_messages = []
                if spec_instruction:
                    specialist_messages.append(SystemMessage(content=spec_instruction))
                specialist_messages.append(HumanMessage(content=delegated_query))

                if spec_tools:
                    specialist_model_with_tools = spec_model.bind_tools(spec_tools)
                    current_messages = specialist_messages

                    for _ in range(5):
                        response = specialist_model_with_tools.invoke(current_messages)
                        current_messages = current_messages + [response]

                        if hasattr(response, "tool_calls") and response.tool_calls:
                            tool_node = ToolNode(spec_tools)
                            tool_results = tool_node.invoke({"messages": [response]})
                            current_messages = current_messages + tool_results["messages"]
                        else:
                            return {"messages": [response]}

                    return {"messages": [current_messages[-1]]}
                else:
                    response = spec_model.invoke(specialist_messages)
                    return {"messages": [response]}

            return specialist_node

        specialist_subgraphs[agent_id] = make_specialist_node(agent_instruction, specialist_tools, specialist_model)

    # Create handoff tools
    handoff_tools = []
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_name = sanitize_name(agent_spec["agent_name"])
        agent_description = agent_spec["agent_instruction"]

        def make_handoff_tool(spec_id, spec_name, spec_description):
            @create_tool
            def handoff_tool(query: str) -> str:
                """Delegate query to specialist."""
                return f"Delegating to {spec_name}: {query}"

            handoff_tool.name = f"ask_{spec_id}"
            handoff_tool.description = f"Delegate to {spec_name}: {spec_description}"
            handoff_tool._target_agent = spec_id  # type: ignore[attr-defined]
            return handoff_tool

        handoff_tools.append(make_handoff_tool(agent_id, agent_name, agent_description))

    # Create orchestrator
    primary_instruction = primary_spec["agent_instruction"]
    primary_seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    primary_model = get_model(model_id, "langgraph", temperature, primary_seed)
    orchestrator_model = primary_model.bind_tools(handoff_tools)

    def orchestrator_node(state: MultiAgentState):
        messages = state["messages"]
        if primary_instruction and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=primary_instruction)] + messages
        response = orchestrator_model.invoke(messages)
        return {"messages": [response]}

    def route_after_orchestrator(state: MultiAgentState):
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            if tool_name.startswith("ask_"):
                specialist_id = tool_name[4:]
                return specialist_id

        return END

    # Build graph
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("orchestrator", orchestrator_node)

    for agent_id, node_fn in specialist_subgraphs.items():
        workflow.add_node(agent_id, node_fn)

    workflow.set_entry_point("orchestrator")

    specialist_routes = {agent_id: agent_id for agent_id in specialist_subgraphs.keys()}
    specialist_routes[END] = END
    workflow.add_conditional_edges("orchestrator", route_after_orchestrator, specialist_routes)

    for agent_id in specialist_subgraphs.keys():
        workflow.add_edge(agent_id, "orchestrator")

    graph = workflow.compile()
    adapter = LangGraphAgentAdapter(graph, primary_spec["agent_id"])
    return adapter, {primary_spec["agent_id"]: adapter}


def build_llamaindex_multi_agent(
    model_id: str,
    temperature: float,
    all_tool_adapters: Dict[str, Any],
    primary_spec: Dict[str, Any],
    specialist_specs: List[Dict[str, Any]],
    seeds: Optional[Dict[str, int]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Build llamaindex multi-agent setup with orchestrator and specialists.

    Args:
        model_id: Model identifier
        temperature: Model temperature
        all_tool_adapters: All available tool adapters (dict keyed by name)
        primary_spec: Primary agent specification
        specialist_specs: List of specialist agent specifications
        seeds: Optional dict mapping agent_id to seed values

    Returns:
        Tuple of (primary_adapter, all_adapters_dict). Note: LlamaIndex multi-agent
        uses handoff tools, so only the orchestrator is directly traceable.
    """
    from llama_index.core.agent.workflow.react_agent import ReActAgent
    from llama_index.core.tools import FunctionTool
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    import asyncio

    specialist_agents_dict = {}
    for agent_spec in specialist_specs:
        agent_id = agent_spec["agent_id"]
        agent_name = sanitize_name(agent_spec["agent_name"])
        agent_instruction = agent_spec["agent_instruction"]
        specialist_seed = seeds.get(agent_id) if seeds else None
        specialist_model = get_model(model_id, "llamaindex", temperature, specialist_seed)
        specialist_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, agent_spec["tools"])
        specialist_tools = [adapter.tool for adapter in specialist_adapters.values()]

        specialist_agent = ReActAgent(
            tools=specialist_tools,
            llm=specialist_model,
            verbose=True,
            max_iterations=10,
            system_prompt=agent_instruction,
        )

        specialist_agents_dict[agent_id] = {
            "agent": specialist_agent,
            "name": agent_name,
            "description": agent_instruction,
        }

    def make_handoff_tool(specialist_id: str, specialist_info: dict):
        def handoff_to_specialist(task: str) -> str:
            """Delegate task to specialist agent."""
            specialist_agent = specialist_info["agent"]

            async def run_specialist():
                handler = specialist_agent.run(user_msg=task)
                result = await handler
                return result

            result = asyncio.run(run_specialist())
            if hasattr(result, "response"):
                return str(result.response)
            return str(result)

        return FunctionTool.from_defaults(
            fn=handoff_to_specialist,
            name=f"ask_{specialist_id}",
            description=f"Delegate to {specialist_info['name']}: {specialist_info['description']}",
        )

    orchestrator_tools = [make_handoff_tool(spec_id, spec_info) for spec_id, spec_info in specialist_agents_dict.items()]
    primary_adapters = filter_tool_adapters_by_prefix(all_tool_adapters, primary_spec["tools"])
    primary_tools = [adapter.tool for adapter in primary_adapters.values()]
    orchestrator_tools.extend(primary_tools)

    primary_seed = seeds.get(primary_spec["agent_id"]) if seeds else None
    primary_model = get_model(model_id, "llamaindex", temperature, primary_seed)

    orchestrator = ReActAgent(
        tools=orchestrator_tools,
        llm=primary_model,
        verbose=True,
        max_iterations=15,
        system_prompt=primary_spec.get("agent_instruction"),
    )

    adapter = LlamaIndexAgentAdapter(orchestrator, primary_spec["agent_id"])
    return adapter, {primary_spec["agent_id"]: adapter}


def get_agent_builder(framework: str, agent_type: str):
    """Get the appropriate agent builder function based on framework and agent type."""
    if agent_type == "single":
        if framework == "smolagents":
            return build_smolagents_single_agent
        elif framework == "langgraph":
            return build_langgraph_single_agent
        elif framework == "llamaindex":
            return build_llamaindex_single_agent
    elif agent_type == "multi":
        if framework == "smolagents":
            return build_smolagents_multi_agent
        elif framework == "langgraph":
            return build_langgraph_multi_agent
        elif framework == "llamaindex":
            return build_llamaindex_multi_agent

    raise ValueError(f"Unsupported combination of framework '{framework}' and agent_type '{agent_type}'")


# ============================================================================
# Benchmark
# ============================================================================


class FiveADayBenchmark(Benchmark):
    """5-A-Day benchmark with framework-agnostic tools.

    Demonstrates the maseval library through 5 diverse tasks with different evaluation approaches.
    Supports single-agent and multi-agent (orchestrator+specialist) configurations.
    """

    def setup_environment(self, agent_data: Dict[str, Any], task: Task, seed_generator: SeedGenerator) -> Environment:
        """Create environment from task data."""
        # Pass full task data to environment
        task_data = {
            "environment_data": task.environment_data,
            "query": task.query,
            "evaluation_data": task.evaluation_data,
            "metadata": task.metadata,
        }

        framework = agent_data["framework"]
        environment = FiveADayEnvironment(task_data, framework)

        # Register all tools with the benchmark for tracing
        for tool_name, tool_adapter in environment.get_tools().items():
            self.register("tools", tool_name, tool_adapter)

        return environment

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user,
        seed_generator: SeedGenerator,
    ) -> tuple[List[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create framework-specific agent with tools from environment.

        Returns:
            Tuple of (agents_to_run, agents_dict):
            - agents_to_run: List of adapters for agents that should be executed
            - agents_dict: Dict of all adapters for trace registration (includes specialists)
        """
        framework = agent_data["framework"]
        agent_type = agent_data["agent_type"]
        model_id = agent_data["model_config"]["model_id"]
        temperature = agent_data["model_config"]["temperature"]
        primary_agent_id = agent_data["primary_agent_id"]
        agents_specs = agent_data["agents"]
        all_tool_adapters = environment.get_tools()

        # Extract primary and specialist specs
        primary_spec = next(a for a in agents_specs if a["agent_id"] == primary_agent_id)
        specialist_specs = [a for a in agents_specs if a["agent_id"] != primary_agent_id]

        # Derive seeds for agents using seed_generator
        # Use child("agents") to create logical paths like "agents/primary_agent"
        # derive_seed() returns None if seeding is disabled
        agent_gen = seed_generator.child("agents")
        seeds = {primary_spec["agent_id"]: agent_gen.derive_seed(primary_spec["agent_id"])}
        for spec in specialist_specs:
            seeds[spec["agent_id"]] = agent_gen.derive_seed(spec["agent_id"])

        # Build agent using unified interface - now returns (primary_adapter, all_adapters_dict)
        builder = get_agent_builder(framework, agent_type)
        primary_adapter, all_adapters_dict = builder(model_id, temperature, all_tool_adapters, primary_spec, specialist_specs, seeds)

        # Return primary adapter to run, and all adapters for trace registration
        return [primary_adapter], all_adapters_dict

    def setup_evaluators(self, environment, task, agents, user, seed_generator: SeedGenerator) -> Sequence[Evaluator]:
        """Create evaluators based on task's evaluation_data.evaluators list."""
        if not task.evaluation_data["evaluators"]:
            return []

        # Dynamically instantiate evaluators
        evaluator_instances = []
        for name in task.evaluation_data["evaluators"]:
            # Get evaluator class from evaluators module
            evaluator_class = getattr(evaluators, name)

            # Instantiate with standard arguments
            evaluator_instances.append(evaluator_class(task, environment, user))

        return evaluator_instances

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment, query: str) -> Sequence[Any]:
        """Execute agents and return their final answers."""
        answers = [agent.run(query) for agent in agents]
        return answers

    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """Return a model adapter for benchmark components that need LLM access.

        This benchmark doesn't use simulated tools, user simulators, or LLM judges,
        so this method is not called during execution.
        """
        raise NotImplementedError("This benchmark doesn't use model adapters for tools/users/evaluators.")

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate agent performance."""
        results = []

        for evaluator in evaluators:
            # ensure the evaluator is only called with relevant traces
            filtered_traces = evaluator.filter_traces(traces)
            results.append(evaluator(filtered_traces, final_answer))

        return results


# ============================================================================
# Data Loading
# ============================================================================


def load_benchmark_data(
    config_type: str,
    framework: str,
    model_id: str,
    temperature: float,
    limit: Optional[int] = None,
    specific_task: Optional[int] = None,
) -> tuple[TaskQueue, List[Dict[str, Any]]]:
    """Load tasks and agent configurations with validation.

    Args:
        config_type: Agent configuration type ('single' or 'multi')
        framework: Target framework ('smolagents', 'langgraph', 'llamaindex')
        limit: Optional limit on number of tasks/configs to load
        specific_task: Optional index to load only a specific task
        model_id: Model identifier
        temperature: Model temperature

    Returns:
        Tuple of (TaskQueue, agent_configs_list)
    """
    if limit is not None and specific_task is not None:
        raise ValueError("Cannot specify both limit and specific_task")

    data_dir = Path(__file__).parent / "data"

    # Load raw data from JSON files
    with open(data_dir / "tasks.json", "r") as f:
        tasks_raw = json.load(f)
    with open(data_dir / f"{config_type}agent.json", "r") as f:
        configs_raw = json.load(f)

    # Validate alignment
    if len(tasks_raw) != len(configs_raw):
        raise ValueError(f"Mismatch: {len(tasks_raw)} tasks vs {len(configs_raw)} configs")

    # Determine indices to load
    if specific_task is not None:
        indices = [specific_task]
    elif limit is not None:
        indices = list(range(limit))
    else:
        indices = list(range(len(tasks_raw)))

    # Build tasks and configs in parallel
    tasks_data = []
    configs_data = []

    for idx in indices:
        task_dict = tasks_raw[idx]
        config = configs_raw[idx]
        task_id = task_dict["metadata"]["task_id"]
        task_dict["environment_data"]["agent_framework"] = framework

        # Create task with id from metadata
        tasks_data.append(
            Task(
                query=task_dict["query"],
                id=task_id,
                environment_data=task_dict["environment_data"],
                evaluation_data=task_dict["evaluation_data"],
                metadata=task_dict["metadata"],
            )
        )

        # Enrich config with framework and model info
        config["framework"] = framework
        config["model_config"] = {"model_id": model_id, "temperature": temperature}

        configs_data.append(config)

    print(f"Loaded {len(tasks_data)} tasks and {len(configs_data)} agent configs\n")

    return TaskQueue(tasks_data), configs_data


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    args = parser.parse_args()

    print("Running 5-A-Day Benchmark")
    print(f"Framework: {args.framework}")
    print(f"Config: {args.config_type}agent")
    print(f"Model: {args.model_id}")
    print(f"Temperature: {args.temperature}")
    print(f"Seed: {args.seed if args.seed is not None else 'None (non-deterministic)'}")
    print(f"Task limit: {args.limit or 'all'}")
    print(f"Specific task: {args.task if args.task is not None else 'all'}\n")

    # Load benchmark data
    tasks, agent_configs = load_benchmark_data(
        config_type=args.config_type,
        framework=args.framework,
        model_id=args.model_id,
        temperature=args.temperature,
        limit=args.limit,
        specific_task=args.task,
    )

    logger = FileResultLogger(
        output_dir=str(Path(__file__).parent / "results"),
        filename_pattern=f"{args.framework}_{args.config_type}agent_{{timestamp}}.jsonl",
        validate_on_completion=False,
    )

    benchmark = FiveADayBenchmark(
        callbacks=[logger],
        seed=args.seed,  # Use benchmark's seeding system
        fail_on_setup_error=True,
        fail_on_task_error=True,
        fail_on_evaluation_error=True,
    )
    results = benchmark.run(tasks=tasks, agent_data=agent_configs)

    print("\n--- Benchmark Complete ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Results saved to: {logger.output_dir}")
