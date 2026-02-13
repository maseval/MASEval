"""LangGraph integration for MASEval.

This module requires langgraph to be installed:
    pip install maseval[langgraph]
"""

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

from maseval import AgentAdapter, MessageHistory, LLMUser

__all__ = ["LangGraphAgentAdapter", "LangGraphLLMUser"]

# Only import langgraph types for type checking, not at runtime
if TYPE_CHECKING:
    from langgraph.graph import StateGraph
    from langchain_core.messages import BaseMessage
else:
    StateGraph = None
    BaseMessage = None


def _check_langgraph_installed():
    """Check if langgraph is installed and raise a helpful error if not."""
    try:
        import langgraph  # noqa: F401
    except ImportError as e:
        raise ImportError("langgraph is not installed. Install it with: pip install maseval[langgraph]") from e


class LangGraphAgentAdapter(AgentAdapter):
    """An AgentAdapter for LangGraph CompiledGraph agents.

    This adapter integrates LangGraph's compiled graphs with MASEval's benchmarking framework,
    converting LangChain/LangGraph message types to OpenAI-compatible MessageHistory format.
    It preserves tool calls, tool responses, multi-modal content, and supports both stateless
    and stateful (checkpointed) graph execution.

    LangGraph graphs can operate in two modes:

    - **Stateless**: Messages from invoke() result are cached in the adapter for access
    - **Stateful**: With checkpointer and thread_id, messages are fetched from persistent state

    The adapter automatically handles both modes, preferring persistent state when available
    and falling back to cached results for stateless graphs.

    How to use:
        1. **Create a LangGraph graph** with state and nodes
        2. **Compile the graph** (optionally with checkpointer for state persistence)
        3. **Wrap with LangGraphAgentAdapter** to enable MASEval integration
        4. **Use in benchmarks** or call directly for testing
        5. **Access traces and config** for analysis and debugging

        Example workflow:
            ```python
            from maseval.interface.agents.langgraph import LangGraphAgentAdapter
            from langgraph.graph import StateGraph, MessagesState
            from langgraph.checkpoint.memory import MemorySaver

            # Define your graph
            def chatbot(state: MessagesState):
                # Your agent logic
                return {"messages": [response]}

            # Build graph
            graph = StateGraph(MessagesState)
            graph.add_node("chatbot", chatbot)
            graph.set_entry_point("chatbot")
            graph.set_finish_point("chatbot")

            # Compile (stateless)
            compiled_graph = graph.compile()
            agent_adapter = LangGraphAgentAdapter(compiled_graph, "agent_name")

            # Or compile with checkpointer (stateful)
            memory = MemorySaver()
            compiled_graph = graph.compile(checkpointer=memory)
            config = {"configurable": {"thread_id": "session_1"}}
            agent_adapter = LangGraphAgentAdapter(
                compiled_graph,
                "agent_name",
                config=config
            )

            # Run agent
            result = agent_adapter.run("What's the weather?")

            # Access message history in OpenAI format
            for msg in agent_adapter.get_messages():
                print(f"{msg['role']}: {msg['content']}")

            # Gather execution traces
            traces = agent_adapter.gather_traces()
            if 'total_tokens' in traces:
                print(f"Total tokens: {traces['total_tokens']}")

            # Use in benchmark
            benchmark = MyBenchmark(agent_data={"agent": agent_adapter})
            results = benchmark.run(tasks)
            ```

        For stateful graphs, the adapter preserves conversation context across multiple
        calls using the same thread_id, enabling multi-turn interactions.

    Message Format:
        LangGraph uses LangChain message types. The adapter converts to `maseval` / OpenAI format.

        Tool calls are preserved with metadata and converted to OpenAI's tool call format.

    Token Usage:
        If LangChain messages include `usage_metadata`, the adapter automatically extracts
        and aggregates token counts. This is available for models that provide usage information.

    Requires:
        langgraph to be installed: `pip install maseval[langgraph]`
    """

    def __init__(self, agent_instance, name: str, callbacks=None, config=None):
        """Initialize the LangGraph adapter.

        Args:
            agent_instance: Compiled LangGraph graph
            name: Agent name
            callbacks: Optional list of callbacks
            config: Optional LangGraph config dict (for stateful graphs with checkpointer)
                   Should include 'configurable': {'thread_id': '...'} for persistent state
        """
        super().__init__(agent_instance, name, callbacks)
        self._langgraph_config = config
        self._last_result = None

    def get_messages(self) -> MessageHistory:
        """Get message history from LangGraph.

        For stateful graphs (with checkpointer and thread_id), fetches from graph state.
        For stateless graphs, returns cached messages from last run.

        Returns:
            MessageHistory with converted messages
        """
        _check_langgraph_installed()

        # If we have a config with thread_id and the graph has get_state, use it
        if self._langgraph_config and hasattr(self.agent, "get_state"):
            try:
                state = self.agent.get_state(self._langgraph_config)
                messages = state.values.get("messages", [])
                if messages:
                    return self._convert_langchain_messages(messages)
            except Exception:
                # If get_state fails, fall back to cached result
                pass

        # Fall back to cached result from last run
        if self._last_result:
            messages = self._last_result.get("messages", [])
            return self._convert_langchain_messages(messages)

        # No messages available
        return MessageHistory()

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this LangGraph agent.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: Agent name
            - agent_type: CompiledGraph or similar
            - adapter_type: LangGraphAgentAdapter
            - callbacks: List of callback class names
            - has_checkpointer: Whether the graph has state persistence
            - config: LangGraph config dict (with sensitive data removed)
            - graph_info: Information about the graph structure (if available)
        """
        base_config = super().gather_config()
        _check_langgraph_installed()

        # Add LangGraph-specific config
        langgraph_config = {}

        # Check if graph has checkpointer (state persistence)
        if hasattr(self.agent, "checkpointer"):
            langgraph_config["has_checkpointer"] = self.agent.checkpointer is not None

        # Include config (but remove sensitive data like thread_id which might be user-specific)
        if self._langgraph_config:
            safe_config = {}
            for key, value in self._langgraph_config.items():
                if key != "configurable":  # Skip configurable section which may have thread_id
                    safe_config[key] = value
                else:
                    # Include configurable but sanitize
                    safe_config["configurable"] = {"has_thread_id": "thread_id" in value if isinstance(value, dict) else False}
            langgraph_config["config"] = safe_config

        # Try to get graph structure info
        if hasattr(self.agent, "get_graph"):
            try:
                graph = self.agent.get_graph()
                if graph:
                    langgraph_config["graph_info"] = {
                        "num_nodes": len(graph.nodes) if hasattr(graph, "nodes") else None,
                        "num_edges": len(graph.edges) if hasattr(graph, "edges") else None,
                    }
            except Exception:
                pass

        if langgraph_config:
            base_config["langgraph_config"] = langgraph_config

        return base_config

    def _run_agent(self, query: str) -> Any:
        _check_langgraph_installed()
        from langchain_core.messages import HumanMessage

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            # Initialize the state with the user query
            initial_state = {"messages": [HumanMessage(content=query)]}

            # Invoke the graph (with config if provided)
            if self._langgraph_config:
                result = self.agent.invoke(initial_state, config=self._langgraph_config)
            else:
                result = self.agent.invoke(initial_state)

            # Cache the result for stateless graphs
            self._last_result = result
            duration = time.time() - start_time

            # Log successful execution
            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "query": query,
                "query_length": len(query),
                "duration_seconds": duration,
                "status": "success",
            }

            # Extract state information if available
            if isinstance(result, dict):
                log_entry["state_keys"] = list(result.keys())
                messages = result.get("messages", [])
                log_entry["message_count"] = len(messages) if messages else 0

                # Try to extract token usage from messages if available
                # (LangChain messages may have usage_metadata)
                total_input_tokens = 0
                total_output_tokens = 0
                for msg in messages:
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        # usage_metadata can be dict or object
                        if isinstance(msg.usage_metadata, dict):
                            total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
                            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)
                        else:
                            total_input_tokens += getattr(msg.usage_metadata, "input_tokens", 0)
                            total_output_tokens += getattr(msg.usage_metadata, "output_tokens", 0)

                if total_input_tokens > 0 or total_output_tokens > 0:
                    log_entry["input_tokens"] = total_input_tokens
                    log_entry["output_tokens"] = total_output_tokens
                    log_entry["total_tokens"] = total_input_tokens + total_output_tokens

            # For stateful graphs with checkpointer, get state snapshot metadata
            if self._langgraph_config and hasattr(self.agent, "get_state"):
                try:
                    state_snapshot = self.agent.get_state(self._langgraph_config)
                    if state_snapshot.metadata:
                        log_entry["checkpoint_metadata"] = {
                            "source": state_snapshot.metadata.get("source"),
                            "step": state_snapshot.metadata.get("step"),
                        }
                    if state_snapshot.created_at:
                        log_entry["checkpoint_created_at"] = state_snapshot.created_at
                except Exception:
                    # If get_state fails, just skip metadata
                    pass

            self.logs.append(log_entry)

            # Extract and return the final answer from the graph's result
            # LangGraph typically returns dict with 'messages' key, extract the last AI message
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                # Return the content of the last message as the final answer
                return getattr(last_message, "content", str(last_message))

            return None

        except Exception as e:
            duration = time.time() - start_time

            # Log failed execution
            self.logs.append(
                {
                    "timestamp": timestamp,
                    "query": query,
                    "query_length": len(query),
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

            raise

    def _convert_langchain_messages(self, lc_messages: list) -> MessageHistory:
        """Convert LangChain messages to MASEval MessageHistory format.

        LangChain uses typed message classes (HumanMessage, AIMessage, etc.).
        This method converts them to OpenAI-compatible dict format.

        Args:
            lc_messages: List of LangChain message objects

        Returns:
            MessageHistory with converted messages
        """
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

        history_messages = []

        for msg in lc_messages:
            if isinstance(msg, HumanMessage):
                history_messages.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, AIMessage):
                message_dict = {
                    "role": "assistant",
                    "content": msg.content if msg.content else "",
                }
                # Include tool calls if present
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    message_dict["tool_calls"] = tool_calls
                history_messages.append(message_dict)
            elif isinstance(msg, ToolMessage):
                message_dict = {
                    "role": "tool",
                    "content": msg.content,
                }
                # Include tool metadata
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    message_dict["tool_call_id"] = tool_call_id
                tool_name = getattr(msg, "name", None)
                if tool_name:
                    message_dict["name"] = tool_name
                history_messages.append(message_dict)
            elif isinstance(msg, SystemMessage):
                history_messages.append(
                    {
                        "role": "system",
                        "content": msg.content,
                    }
                )

        return MessageHistory(history_messages)


class LangGraphLLMUser(LLMUser):
    """A LangGraph-specific LLM user that provides a tool for user interaction.

    Extends LLMUser to provide a LangChain-compatible tool via get_tool().
    Requires langgraph to be installed.

    Example:
        ```python
        from maseval.interface.agents.langgraph import LangGraphLLMUser

        user = LangGraphLLMUser(...)
        tool = user.get_tool()  # Returns a LangChain tool
        ```
    """

    def get_tool(self):
        """Get a LangChain-compatible tool for user interaction."""
        _check_langgraph_installed()
        from langchain_core.tools import tool

        user_instance = self

        @tool
        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Args:
                question: The question to ask the user

            Returns:
                The user's response
            """
            return user_instance.respond(question)

        return ask_user
