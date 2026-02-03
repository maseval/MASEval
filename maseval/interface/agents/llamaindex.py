"""LlamaIndex integration for MASEval.

This module requires llama-index-core to be installed:
    pip install maseval[llamaindex]
"""

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from maseval import AgentAdapter, MessageHistory, LLMUser

__all__ = ["LlamaIndexAgentAdapter", "LlamaIndexLLMUser"]

# Only import LlamaIndex types for type checking, not at runtime
if TYPE_CHECKING:
    from llama_index.core.agent.workflow import AgentWorkflow, BaseWorkflowAgent
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools.types import BaseTool
else:
    AgentWorkflow = None
    BaseWorkflowAgent = None
    ChatMessage = None
    BaseTool = None


def _check_llamaindex_installed():
    """Check if llama-index-core is installed and raise a helpful error if not."""
    try:
        import llama_index.core  # noqa: F401
    except ImportError as e:
        raise ImportError("llama-index-core is not installed. Install it with: pip install maseval[llamaindex]") from e


class LlamaIndexAgentAdapter(AgentAdapter):
    """An AgentAdapter for LlamaIndex workflow-based agents.

    This adapter integrates LlamaIndex's workflow-based agent system with MASEval's benchmarking
    framework, converting LlamaIndex's ChatMessage format to OpenAI-compatible MessageHistory format.
    It handles both AgentWorkflow and BaseWorkflowAgent instances, automatically managing async
    execution in synchronous contexts.

    LlamaIndex agents are async-first, using workflows that must be awaited. This adapter handles
    the async-to-sync conversion automatically, supporting both agents with persistent memory and
    stateless execution modes. It seamlessly integrates with MASEval's synchronous benchmarking API.

    How to use:
        1. **Create a LlamaIndex workflow agent** with tools and LLM
        2. **Wrap with LlamaIndexAgentAdapter** to enable MASEval integration
        3. **Use in benchmarks** or call directly for testing
        4. **Access traces and config** for analysis and debugging

        Example workflow:
            ```python
            from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
            from llama_index.core.agent.workflow import AgentWorkflow
            from llama_index.core.llms import OpenAI
            from llama_index.core.tools import FunctionTool

            # Define a tool
            def search(query: str) -> str:
                \"\"\"Search for information.\"\"\"
                return f\"Results for: {query}\"

            search_tool = FunctionTool.from_defaults(fn=search)

            # Create a LlamaIndex workflow
            workflow = AgentWorkflow.from_tools_or_functions(
                tools_or_functions=[search_tool],
                llm=OpenAI(model=\"gpt-4\"),
                system_prompt=\"You are a helpful research assistant\"
            )

            # Wrap with adapter
            agent_adapter = LlamaIndexAgentAdapter(workflow, \"research_agent\")

            # Run agent (async handled automatically)
            result = agent_adapter.run(\"What are the latest developments in quantum computing?\")\n\n            # Access message history in OpenAI format\n            for msg in agent_adapter.get_messages():\n                print(f\"{msg['role']}: {msg['content']}\")\n\n            # Gather configuration including tools and system prompt\n            config = agent_adapter.gather_config()\n            print(f\"System prompt: {config['llamaindex_config']['system_prompt']}\")\n            print(f\"Tools: {config['llamaindex_config']['tools']}\")\n\n            # Gather execution traces with timing\n            traces = agent_adapter.gather_traces()\n            if 'total_tokens' in traces:\n                print(f\"Total tokens: {traces['total_tokens']}\")\n\n            # Use in benchmark\n            benchmark = MyBenchmark(agent_data={\"agent\": agent_adapter})\n            results = benchmark.run(tasks)\n            ```

        The adapter works with various LlamaIndex agent types including AgentWorkflow,\n        FunctionAgent (tool calling), ReActAgent, and CodeActAgent.

    Message Format:
        LlamaIndex uses ChatMessage objects with MessageRole enums. The adapter converts to `maseval` / OpenAI format.

        Tool calls are preserved in the `additional_kwargs` field and converted to OpenAI's
        tool call format when available.

    Async Handling:
        LlamaIndex agents return a WorkflowHandler from `.run()` which must be awaited.
        The adapter handles this automatically:

        - Checks for `run_sync()` method first (for compatibility)
        - Falls back to `asyncio.run()` to execute the async `run()` method
        - Works seamlessly in synchronous benchmarking contexts

        This allows you to use async-first LlamaIndex agents in MASEval's sync API without
        any additional configuration.

    Supported Agent Types:
        - **AgentWorkflow**: Multi-agent workflow orchestrator
        - **FunctionAgent**: Function-calling based agent (for LLMs with tool calling)
        - **ReActAgent**: ReAct prompting pattern agent
        - **CodeActAgent**: Code execution based agent

    Token Usage:
        Token usage is extracted from LLM responses when available. If the LLM response
        includes usage metadata, it's automatically captured in execution traces.

    Requires:
        llama-index-core to be installed: `pip install maseval[llamaindex]`
    """

    def __init__(self, agent_instance, name: str, callbacks=None):
        """Initialize the LlamaIndex adapter.

        Args:
            agent_instance: LlamaIndex AgentWorkflow or BaseWorkflowAgent instance
            name: Agent name
            callbacks: Optional list of callbacks
        """
        super().__init__(agent_instance, name, callbacks)
        self._last_result = None
        self._message_cache: List[Dict[str, Any]] = []

    def get_messages(self) -> MessageHistory:
        """Get message history from LlamaIndex.

        For agents with accessible memory, fetches from the agent's memory.
        Otherwise, returns cached messages from the last run.

        Returns:
            MessageHistory with converted messages
        """
        _check_llamaindex_installed()

        # Try to extract from agent memory if available
        if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "get_all"):
            try:
                messages = self.agent.memory.get_all()
                if messages:
                    return self._convert_llamaindex_messages(messages)
            except Exception:
                # If memory access fails, fall back to cache
                pass

        # Fall back to cached messages
        return MessageHistory(self._message_cache)

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this LlamaIndex agent.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: Agent name
            - agent_type: Underlying agent class name
            - adapter_type: LlamaIndexAgentAdapter
            - callbacks: List of callback class names
            - llamaindex_config: LlamaIndex-specific configuration (if available)
        """
        base_config = super().gather_config()
        _check_llamaindex_installed()

        # Add LlamaIndex-specific config
        llamaindex_config: Dict[str, Any] = {}

        # Try to get workflow/agent configuration
        if hasattr(self.agent, "name"):
            llamaindex_config["agent_name"] = self.agent.name

        if hasattr(self.agent, "description"):
            llamaindex_config["agent_description"] = self.agent.description

        if hasattr(self.agent, "system_prompt"):
            llamaindex_config["system_prompt"] = self.agent.system_prompt

        # Get tool information
        if hasattr(self.agent, "tools"):
            tools = getattr(self.agent, "tools", [])
            if tools:
                llamaindex_config["tools"] = [
                    {
                        "name": getattr(tool, "metadata", {}).get("name", str(tool)),
                        "description": getattr(tool, "metadata", {}).get("description", ""),
                    }
                    for tool in tools
                ]

        # Check if it's a workflow
        if hasattr(self.agent, "get_config"):
            try:
                workflow_config = self.agent.get_config()
                if workflow_config:
                    llamaindex_config["workflow_config"] = workflow_config
            except Exception:
                pass

        if llamaindex_config:
            base_config["llamaindex_config"] = llamaindex_config

        return base_config

    def _run_agent(self, query: str) -> Any:
        """Run the LlamaIndex agent and cache execution state.

        Args:
            query: The user query to send to the agent

        Returns:
            The agent's final answer

        Raises:
            Exception: If agent execution fails
        """
        _check_llamaindex_installed()

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        try:
            # Run the agent (handles async automatically)
            result = self._run_agent_sync(query)

            # Cache the result
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

            # Extract and cache messages from result
            messages = self._extract_messages_from_result(result, query)
            self._message_cache = messages

            log_entry["message_count"] = len(messages)

            # Try to extract token usage if available
            if hasattr(result, "raw") and result.raw:
                if hasattr(result.raw, "usage"):
                    usage = result.raw.usage
                    if hasattr(usage, "total_tokens"):
                        log_entry["total_tokens"] = usage.total_tokens
                    if hasattr(usage, "prompt_tokens"):
                        log_entry["input_tokens"] = usage.prompt_tokens
                    if hasattr(usage, "completion_tokens"):
                        log_entry["output_tokens"] = usage.completion_tokens

            self.logs.append(log_entry)

            # Extract and return the final answer
            return self._extract_final_answer(result)

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

    def _run_agent_sync(self, query: str) -> Any:
        """Run agent in sync context.

        Supports multiple LlamaIndex agent types:
        - AgentWorkflow: .run() returns a WorkflowHandler that must be awaited
        - ReActAgent/FunctionAgent: .chat() for synchronous execution
        - Legacy agents: .run_sync() or .query() methods

        Args:
            query: The user query

        Returns:
            The agent's result
        """
        # Check if agent has a synchronous run_sync method (for mocks and older patterns)
        if hasattr(self.agent, "run_sync") and callable(self.agent.run_sync):
            return self.agent.run_sync(input=query)

        # Check if agent has .chat() method (ReActAgent, FunctionAgent, etc.)
        if hasattr(self.agent, "chat") and callable(self.agent.chat):
            return self.agent.chat(query)

        # Check if agent has .query() method (some LlamaIndex agents)
        if hasattr(self.agent, "query") and callable(self.agent.query):
            return self.agent.query(query)

        # LlamaIndex .run() returns a WorkflowHandler that must be awaited (AgentWorkflow)
        # We need to run this in an async context
        async def run_async():
            handler = self.agent.run(user_msg=query)
            result = await handler
            return result

        # Run the async function using asyncio.run()
        # This properly sets up the event loop for LlamaIndex's internal async operations
        return asyncio.run(run_async())

    def _extract_messages_from_result(self, result, query: str) -> List[Dict[str, Any]]:
        """Extract messages from agent result.

        Args:
            result: The agent's execution result
            query: Original query

        Returns:
            List of message dictionaries
        """
        from llama_index.core.base.llms.types import ChatMessage

        messages: List[Dict[str, Any]] = []

        # Add the user query
        messages.append({"role": "user", "content": query})

        # Try to extract from result.response (which is a ChatMessage)
        if hasattr(result, "response"):
            response_msg = result.response
            if isinstance(response_msg, ChatMessage):
                converted = self._convert_single_message(response_msg)
                messages.append(converted)
            elif hasattr(response_msg, "content"):
                # It's a message-like object
                messages.append({"role": "assistant", "content": str(response_msg.content)})

        # If no messages extracted, create a simple assistant response
        if len(messages) == 1:  # Only user message
            final_answer = self._extract_final_answer(result)
            if final_answer:
                messages.append({"role": "assistant", "content": str(final_answer)})

        return messages

    def _extract_final_answer(self, result) -> str:
        """Extract the final answer from agent result.

        Args:
            result: The agent's execution result

        Returns:
            Final answer as string
        """
        # Try to extract from result.response
        if hasattr(result, "response"):
            response = result.response
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        # Fall back to string representation
        return str(result)

    def _convert_llamaindex_messages(self, messages: List) -> MessageHistory:
        """Convert LlamaIndex messages to MASEval MessageHistory format.

        LlamaIndex uses ChatMessage objects with MessageRole enums. This method
        normalizes them to OpenAI-compatible format with string literal roles.

        Args:
            messages: List of ChatMessage objects from LlamaIndex

        Returns:
            MessageHistory with converted messages
        """
        converted_messages = [self._convert_single_message(msg) for msg in messages]
        return MessageHistory(converted_messages)

    def _convert_single_message(self, msg) -> Dict[str, Any]:
        """Convert a single LlamaIndex ChatMessage to OpenAI format.

        Args:
            msg: ChatMessage object

        Returns:
            Message dictionary in OpenAI format
        """
        from llama_index.core.base.llms.types import MessageRole

        # Extract role and convert enum to string
        role = getattr(msg, "role", MessageRole.ASSISTANT)
        if hasattr(role, "value"):
            role_str = role.value.lower()
        else:
            role_str = str(role).lower()

        # Map LlamaIndex roles to OpenAI roles
        role_mapping = {
            "user": "user",
            "assistant": "assistant",
            "system": "system",
            "tool": "tool",
            "function": "assistant",  # Map function to assistant
        }
        role_str = role_mapping.get(role_str, "assistant")

        # Extract content
        content = getattr(msg, "content", "")

        # Build the converted message
        converted_msg: Dict[str, Any] = {
            "role": role_str,
            "content": content or "",
        }

        # Handle tool calls if present in additional_kwargs
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            additional_kwargs = msg.additional_kwargs
            if "tool_calls" in additional_kwargs:
                converted_msg["tool_calls"] = additional_kwargs["tool_calls"]
            if "tool_call_id" in additional_kwargs:
                converted_msg["tool_call_id"] = additional_kwargs["tool_call_id"]
            if role_str == "tool" and "name" in additional_kwargs:
                converted_msg["name"] = additional_kwargs["name"]

        return converted_msg


class LlamaIndexLLMUser(LLMUser):
    """A LlamaIndex-specific LLM user that provides a tool for user interaction.

    Extends LLMUser to provide a LlamaIndex-compatible tool via get_tool().
    Requires llama-index-core to be installed.

    Example:
        ```python
        from maseval.interface.agents.llamaindex import LlamaIndexLLMUser

        user = LlamaIndexLLMUser(...)
        tool = user.get_tool()  # Returns a LlamaIndex FunctionTool
        ```
    """

    def get_tool(self):
        """Get a LlamaIndex-compatible tool for user interaction.

        Returns:
            LlamaIndex FunctionTool that wraps the respond method.
        """
        _check_llamaindex_installed()
        from llama_index.core.tools import FunctionTool

        user_instance = self

        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Args:
                question: The question to ask the user

            Returns:
                The user's response
            """
            return user_instance.respond(question)

        return FunctionTool.from_defaults(
            fn=ask_user,
            name="ask_user",
            description="Ask the user a question and get their response. Use this when you need clarification or additional information from the user.",
        )
