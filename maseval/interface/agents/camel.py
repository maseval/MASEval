"""CAMEL-AI integration for MASEval.

This module requires camel-ai to be installed:
    pip install maseval[camel]

Components:
    CamelAgentAdapter: Wraps CAMEL ChatAgent for MASEval evaluation
    CamelLLMUser: LLM-simulated user with CAMEL-compatible tool
    CamelAgentUser: User backed by a CAMEL ChatAgent (for agent-to-agent evaluation)
    camel_role_playing_execution_loop: Execution loop using CAMEL RolePlaying semantics
    CamelRolePlayingTracer: Collects orchestration traces from RolePlaying
    CamelWorkforceTracer: Collects orchestration traces from Workforce
"""

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from maseval import AgentAdapter, MessageHistory, LLMUser, User
from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin

__all__ = [
    "CamelAgentAdapter",
    "CamelLLMUser",
    "CamelAgentUser",
    "camel_role_playing_execution_loop",
    "CamelRolePlayingTracer",
    "CamelWorkforceTracer",
]

# Only import camel types for type checking, not at runtime
if TYPE_CHECKING:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.responses import ChatAgentResponse
else:
    ChatAgent = None
    BaseMessage = None
    ChatAgentResponse = None

# Unified role mapping for CAMEL to OpenAI format conversion
_CAMEL_ROLE_MAPPING = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "assistant",
    "critic": "assistant",  # CAMEL-specific role
    "embodiment": "assistant",  # CAMEL-specific role
    "default": "assistant",
}


def _check_camel_installed():
    """Check if camel-ai is installed and raise a helpful error if not."""
    try:
        import camel  # noqa: F401
    except ImportError as e:
        raise ImportError("camel-ai is not installed. Install it with: pip install maseval[camel]") from e


def _extract_response_content(response) -> str:
    """Extract content string from a CAMEL ChatAgentResponse.

    Args:
        response: ChatAgentResponse from agent.step()

    Returns:
        The response content as a string.
    """
    # ChatAgentResponse has msgs attribute with response messages
    if hasattr(response, "msgs") and response.msgs:
        msg = response.msgs[0]
        if hasattr(msg, "content"):
            return str(msg.content)
        return str(msg)

    # Fallback: try msg attribute (singular)
    if hasattr(response, "msg") and response.msg:
        msg = response.msg
        if hasattr(msg, "content"):
            return str(msg.content)
        return str(msg)

    # Last resort: string representation
    return str(response)


class CamelAgentAdapter(AgentAdapter):
    """An AgentAdapter for CAMEL-AI ChatAgent.

    This adapter integrates CAMEL-AI's ChatAgent with MASEval's benchmarking framework,
    converting CAMEL's message format to OpenAI-compatible MessageHistory format.
    It leverages CAMEL's native memory system and response info as the source of truth
    for conversation history and execution traces, ensuring accurate tracking of
    multi-turn interactions without duplicating CAMEL's internal state.

    CAMEL-AI is a modular framework for building intelligent multi-agent systems.
    The ChatAgent is its core component for single-agent interactions, supporting
    tool calling, memory management, and various LLM backends.

    How to use:
        1. **Create a CAMEL ChatAgent** with system message and optional tools
        2. **Wrap with CamelAgentAdapter** to enable MASEval integration
        3. **Use in benchmarks** or call directly for testing
        4. **Access traces and config** for analysis and debugging

        Example workflow:
            ```python
            from maseval.interface.agents.camel import CamelAgentAdapter
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType, ModelType

            # Create a CAMEL model
            model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
            )

            # Create a CAMEL ChatAgent
            agent = ChatAgent(
                system_message="You are a helpful assistant.",
                model=model,
            )

            # Wrap with adapter
            agent_adapter = CamelAgentAdapter(agent, name="assistant")

            # Run agent
            result = agent_adapter.run("What is the capital of France?")

            # Access message history in OpenAI format
            for msg in agent_adapter.get_messages():
                print(f"{msg['role']}: {msg['content']}")

            # Gather execution traces with token usage and tool calls
            traces = agent_adapter.gather_traces()
            print(f"Total tokens: {traces['total_tokens']}")
            print(f"Tool calls: {traces['total_tool_calls']}")

            # Gather configuration
            config = agent_adapter.gather_config()
            print(f"Model: {config.get('camel_config', {}).get('model_type')}")

            # Use in benchmark
            benchmark = MyBenchmark(agent_data={"agent": agent_adapter})
            results = benchmark.run(tasks)
            ```

    Message Format:
        CAMEL uses BaseMessage objects with role_name, role_type, and content.
        The adapter converts these to OpenAI-compatible format via the agent's
        memory system, which already provides messages in a compatible structure.

    Memory as Source of Truth:
        Following MASEval's adapter pattern (similar to SmolAgentAdapter), this adapter
        uses CAMEL's native memory and ChatAgentResponse info as the single source of
        truth. The `logs` property dynamically extracts execution data from stored
        responses rather than manually tracking metrics.

    Execution Model:
        CAMEL's ChatAgent uses a `step()` method for execution, which processes
        one turn of conversation and returns a ChatAgentResponse containing:
        - msgs: Response messages
        - terminated: Whether the conversation should end
        - info: Dict with usage stats, tool_calls, termination_reasons, etc.

    Requires:
        camel-ai to be installed: `pip install maseval[camel]`
    """

    def __init__(self, agent_instance, name: str, callbacks=None):
        """Initialize the CAMEL adapter.

        Note: We don't call super().__init__() to avoid initializing self.logs as a list,
        since we override it as a property that dynamically fetches from stored responses.

        Args:
            agent_instance: CAMEL ChatAgent instance
            name: Agent name for identification
            callbacks: Optional list of AgentCallback instances
        """
        self.agent = agent_instance
        self.name = name
        self.callbacks = callbacks or []
        self.messages = None
        # Store responses from each step() call
        self._responses: List[Any] = []
        # Store errors that occur during execution (for comprehensive logging)
        self._errors: List[Dict[str, Any]] = []

    @property
    def logs(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        """Dynamically generate logs from CAMEL's ChatAgentResponse info.

        Extracts execution data from stored ChatAgentResponse objects,
        including token usage, tool calls, and termination reasons.
        This follows the same pattern as SmolAgentAdapter.

        Returns:
            List of log dictionaries with comprehensive step information
        """
        _check_camel_installed()

        logs_list: List[Dict[str, Any]] = []

        for step_num, response in enumerate(self._responses, start=1):
            log_entry: Dict[str, Any] = {
                "step_number": step_num,
                "status": "success",
            }

            # Extract termination status
            if hasattr(response, "terminated"):
                log_entry["terminated"] = response.terminated

            # Extract response message count
            if hasattr(response, "msgs") and response.msgs:
                log_entry["response_count"] = len(response.msgs)
                # Include response content preview
                if response.msgs[0] and hasattr(response.msgs[0], "content"):
                    content = str(response.msgs[0].content)
                    log_entry["response_preview"] = content[:200] if len(content) > 200 else content

            # Extract info dict (contains usage, tool_calls, termination_reasons, etc.)
            if hasattr(response, "info") and response.info:
                info = response.info

                # Token usage
                if "usage" in info and isinstance(info["usage"], dict):
                    usage = info["usage"]
                    log_entry["input_tokens"] = usage.get("prompt_tokens", 0)
                    log_entry["output_tokens"] = usage.get("completion_tokens", 0)
                    log_entry["total_tokens"] = usage.get("total_tokens", 0)

                # Context tokens
                if "num_tokens" in info:
                    log_entry["context_tokens"] = info["num_tokens"]

                # Termination reasons
                if "termination_reasons" in info:
                    log_entry["termination_reasons"] = info["termination_reasons"]

                # Response/session ID
                if "id" in info:
                    log_entry["response_id"] = info["id"]

                # Tool calls - extract from ToolCallingRecord objects
                if "tool_calls" in info and info["tool_calls"]:
                    tool_calls_data = []
                    for tc in info["tool_calls"]:
                        tc_entry: Dict[str, Any] = {}
                        # ToolCallingRecord has: tool_name, args, result, tool_call_id, images
                        if hasattr(tc, "tool_name"):
                            tc_entry["name"] = tc.tool_name
                        if hasattr(tc, "args"):
                            tc_entry["arguments"] = tc.args
                        if hasattr(tc, "result"):
                            # Truncate large results
                            result_str = str(tc.result)
                            tc_entry["result"] = result_str[:500] if len(result_str) > 500 else result_str
                        if hasattr(tc, "tool_call_id"):
                            tc_entry["id"] = tc.tool_call_id
                        if hasattr(tc, "images") and tc.images:
                            tc_entry["images_count"] = len(tc.images)
                        if tc_entry:
                            tool_calls_data.append(tc_entry)
                    if tool_calls_data:
                        log_entry["tool_calls"] = tool_calls_data

                # External tool call requests
                if "external_tool_call_requests" in info and info["external_tool_call_requests"]:
                    log_entry["external_tool_call_requests"] = [str(req) for req in info["external_tool_call_requests"]]

            logs_list.append(log_entry)

        # Also include any errors that occurred during execution
        logs_list.extend(self._errors)

        return logs_list

    def get_messages(self) -> MessageHistory:
        """Get message history from CAMEL's memory system.

        Dynamically fetches messages from the agent's memory, converting
        them to MASEval's MessageHistory format. CAMEL's memory.get_context()
        returns messages in OpenAI-compatible format.

        Returns:
            MessageHistory with converted messages
        """
        _check_camel_installed()

        messages: List[Dict[str, Any]] = []

        # Try to get messages from agent's memory
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            try:
                # get_context() returns (messages_list, token_count)
                context = self.agent.memory.get_context()
                if isinstance(context, tuple) and len(context) >= 1:
                    memory_messages = context[0]
                    if isinstance(memory_messages, list):
                        messages = self._convert_memory_messages(memory_messages)
            except Exception:
                # If memory access fails, fall back to empty
                pass

        return MessageHistory(messages)

    def _convert_memory_messages(self, memory_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert CAMEL memory messages to MASEval format.

        CAMEL's memory.get_context() typically returns messages already in
        OpenAI-compatible format with 'role' and 'content' keys.

        Args:
            memory_messages: List of message dicts from CAMEL memory

        Returns:
            List of message dicts in MASEval/OpenAI format
        """
        converted = []
        for msg in memory_messages:
            if isinstance(msg, dict):
                # Already in dict format, normalize role names
                role = msg.get("role", "assistant")
                content = msg.get("content", "")

                # Normalize role to standard values
                normalized_role = _CAMEL_ROLE_MAPPING.get(role.lower() if isinstance(role, str) else "assistant", "assistant")

                converted_msg: Dict[str, Any] = {
                    "role": normalized_role,
                    "content": content,
                }

                # Preserve tool-related fields if present
                if "tool_calls" in msg:
                    converted_msg["tool_calls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    converted_msg["tool_call_id"] = msg["tool_call_id"]
                if normalized_role == "tool" and "name" in msg:
                    converted_msg["name"] = msg["name"]

                converted.append(converted_msg)
            else:
                # Handle BaseMessage objects
                converted.append(self._convert_base_message(msg))

        return converted

    def _convert_base_message(self, msg) -> Dict[str, Any]:
        """Convert a CAMEL BaseMessage to MASEval format.

        Args:
            msg: CAMEL BaseMessage object

        Returns:
            Message dict in MASEval/OpenAI format
        """
        # Extract role from role_type enum or role_name
        role = "assistant"
        if hasattr(msg, "role_type"):
            role_type = msg.role_type
            if hasattr(role_type, "value"):
                role = role_type.value.lower()
            elif hasattr(role_type, "name"):
                role = role_type.name.lower()

        normalized_role = _CAMEL_ROLE_MAPPING.get(role, "assistant")

        # Extract content
        content = getattr(msg, "content", "")

        return {
            "role": normalized_role,
            "content": content,
        }

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this CAMEL agent.

        Extends the base class to include CAMEL-specific execution data
        with aggregated statistics from all responses.

        Returns:
            Dictionary containing:
            - Base traces (type, gathered_at, name, messages, logs, etc.)
            - total_steps: Number of step() calls made
            - total_input_tokens: Aggregated input tokens
            - total_output_tokens: Aggregated output tokens
            - total_tokens: Aggregated total tokens
            - total_tool_calls: Total number of tool calls made
            - last_terminated: Whether the last response indicated termination
        """
        base_traces = super().gather_traces()
        _check_camel_installed()

        # Calculate aggregated statistics from responses
        total_input_tokens = 0
        total_output_tokens = 0
        total_tool_calls = 0
        last_terminated = False

        for response in self._responses:
            if hasattr(response, "terminated"):
                last_terminated = response.terminated

            if hasattr(response, "info") and response.info:
                info = response.info

                # Aggregate token usage
                if "usage" in info and isinstance(info["usage"], dict):
                    usage = info["usage"]
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)

                # Count tool calls
                if "tool_calls" in info and info["tool_calls"]:
                    total_tool_calls += len(info["tool_calls"])

        # Add aggregated statistics
        base_traces.update(
            {
                "total_steps": len(self._responses),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "total_tool_calls": total_tool_calls,
                "last_terminated": last_terminated,
            }
        )

        return base_traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this CAMEL agent.

        Returns:
            Dictionary containing:
            - Base config (type, gathered_at, name, agent_type, adapter_type, callbacks)
            - camel_config: CAMEL-specific configuration including:
                - system_message: The agent's system prompt
                - model_type: The model being used
                - tools: List of configured tools
                - memory_type: Type of memory being used
        """
        base_config = super().gather_config()
        _check_camel_installed()

        camel_config: Dict[str, Any] = {}

        # Get system message
        if hasattr(self.agent, "system_message"):
            sys_msg = self.agent.system_message
            if hasattr(sys_msg, "content"):
                camel_config["system_message"] = sys_msg.content
            elif isinstance(sys_msg, str):
                camel_config["system_message"] = sys_msg

        # Get model information
        if hasattr(self.agent, "model_backend"):
            model = self.agent.model_backend
            if hasattr(model, "model_type"):
                model_type = model.model_type
                if hasattr(model_type, "value"):
                    camel_config["model_type"] = model_type.value
                else:
                    camel_config["model_type"] = str(model_type)
            if hasattr(model, "model_config_dict"):
                camel_config["model_config"] = model.model_config_dict
        elif hasattr(self.agent, "model"):
            # Fallback for older CAMEL versions
            model = self.agent.model
            if hasattr(model, "model_type"):
                model_type = model.model_type
                if hasattr(model_type, "value"):
                    camel_config["model_type"] = model_type.value
                else:
                    camel_config["model_type"] = str(model_type)
            if hasattr(model, "model_config_dict"):
                camel_config["model_config"] = model.model_config_dict

        # Get tools information from tool_list or tools attribute
        tool_list = None
        if hasattr(self.agent, "tool_list"):
            tl = self.agent.tool_list
            if isinstance(tl, (list, tuple)):
                tool_list = tl
        if tool_list is None and hasattr(self.agent, "tools"):
            tl = self.agent.tools
            if isinstance(tl, (list, tuple)):
                tool_list = tl

        if tool_list:
            tools_info = []
            for tool in tool_list:
                tool_info: Dict[str, Any] = {}
                if hasattr(tool, "get_function_name"):
                    tool_info["name"] = tool.get_function_name()
                elif hasattr(tool, "name"):
                    tool_info["name"] = tool.name
                if hasattr(tool, "get_function_description"):
                    tool_info["description"] = tool.get_function_description()
                elif hasattr(tool, "description"):
                    tool_info["description"] = tool.description
                if tool_info:
                    tools_info.append(tool_info)
            if tools_info:
                camel_config["tools"] = tools_info

        # Get memory type
        if hasattr(self.agent, "memory") and self.agent.memory is not None:
            camel_config["memory_type"] = type(self.agent.memory).__name__

        if camel_config:
            base_config["camel_config"] = camel_config

        return base_config

    def _run_agent(self, query: str) -> str:
        """Execute the CAMEL agent with the given query.

        Uses CAMEL's step() method to process one turn of conversation.
        The response is stored for later extraction via the logs property.

        Args:
            query: The user query to send to the agent

        Returns:
            The agent's response content as a string

        Raises:
            Exception: If agent execution fails
        """
        _check_camel_installed()
        from camel.messages import BaseMessage

        # Create user message for CAMEL
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=query,
        )

        try:
            # Execute one step - CAMEL tracks everything internally
            response = self.agent.step(user_msg)

            # Store the response for traces (logs property extracts data from here)
            self._responses.append(response)

            # Extract and return the final answer
            return self._extract_final_answer(response)

        except Exception as e:
            # Log the error for traceability
            self._errors.append(
                {
                    "step_number": len(self._responses) + 1,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query[:500] if len(query) > 500 else query,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            raise

    def _extract_final_answer(self, response) -> str:
        """Extract the final answer from a CAMEL ChatAgentResponse.

        Args:
            response: ChatAgentResponse from agent.step()

        Returns:
            The assistant's response content as a string
        """
        return _extract_response_content(response)


class CamelLLMUser(LLMUser):
    """A CAMEL-specific LLM user that provides a tool for user interaction.

    Extends LLMUser to provide a CAMEL-compatible FunctionTool that wraps
    the respond method, allowing CAMEL agents to interact with users
    during benchmarking.

    Requires camel-ai to be installed.

    Example:
        ```python
        from maseval.interface.agents.camel import CamelLLMUser
        from maseval.interface.inference import OpenAIModelAdapter

        # Create a model for user simulation
        model = OpenAIModelAdapter(model_id="gpt-4o-mini")

        # Create the user
        user = CamelLLMUser(
            name="customer",
            model=model,
            user_profile={"name": "John", "preferences": ["fast service"]},
            scenario="Customer seeking help with a product return",
            initial_query="I need to return a product I bought last week.",
        )

        # Get the tool for use with CAMEL agent
        tool = user.get_tool()

        # Create CAMEL agent with the user tool
        from camel.agents import ChatAgent
        agent = ChatAgent(
            system_message="You are a helpful customer service agent.",
            tools=[tool],
        )
        ```
    """

    def get_tool(self):
        """Get a CAMEL-compatible tool for user interaction.

        Returns a CAMEL FunctionTool that wraps the respond method,
        allowing agents to ask the user questions during execution.

        Returns:
            CAMEL FunctionTool instance for user interaction
        """
        _check_camel_installed()
        from camel.toolkits import FunctionTool

        user_instance = self

        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Use this tool when you need to ask the user for clarification,
            additional information, or to confirm something.

            Args:
                question: The question to ask the user

            Returns:
                The user's response to the question
            """
            return user_instance.respond(question)

        return FunctionTool(func=ask_user)


class CamelAgentUser(User):
    """User backed by a CAMEL ChatAgent.

    Wraps a CAMEL ChatAgent to act as the user in MASEval's evaluation loop,
    enabling agent-to-agent evaluation where one agent acts as the user.

    Unlike `CamelLLMUser` which uses MASEval's LLM simulator, this class
    delegates directly to a CAMEL ChatAgent for generating responses.

    Example:
        ```python
        from camel.agents import ChatAgent
        from camel.models import ModelFactory
        from camel.types import ModelPlatformType, ModelType
        from maseval.interface.agents.camel import CamelAgentUser

        # Create a ChatAgent to act as the user
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
        )
        user_agent = ChatAgent(
            system_message="You are a customer seeking help with an order.",
            model=model,
        )

        # Wrap as MASEval user
        user = CamelAgentUser(
            user_agent=user_agent,
            initial_query="I need help with my order",
            max_turns=5,
        )
        ```
    """

    def __init__(
        self,
        user_agent,
        initial_query: str,
        name: str = "camel_agent_user",
        max_turns: int = 10,
    ):
        """Initialize CamelAgentUser.

        Args:
            user_agent: CAMEL ChatAgent instance to use as the user.
            initial_query: The opening message to start the conversation.
            name: Name for this user (used in traces). Defaults to "camel_agent_user".
            max_turns: Maximum number of response turns. Defaults to 10.
        """
        _check_camel_installed()
        self.user_agent = user_agent
        self._initial_query = initial_query
        self.name = name
        self._turn_count = 0
        self._max_turns = max_turns
        self.logs: List[Dict[str, Any]] = []

    def get_initial_query(self) -> str:
        """Return the initial query to start the conversation.

        Returns:
            The initial query provided at construction.
        """
        return self._initial_query

    def respond(self, message: str) -> str:
        """Forward the message to the CAMEL agent and return its response.

        Args:
            message: The agent's message to respond to.

        Returns:
            The CAMEL agent's response, or empty string if done.
        """
        if self.is_done():
            return ""

        _check_camel_installed()
        from camel.messages import BaseMessage

        self._turn_count += 1
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        log_entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "turn": self._turn_count,
            "message": message[:500],  # Truncate for logging
            "status": "success",
        }

        try:
            # Create message for CAMEL agent
            user_msg = BaseMessage.make_user_message(
                role_name="User",
                content=message,
            )

            # Get response from CAMEL agent
            response = self.user_agent.step(user_msg)
            content = _extract_response_content(response)

            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["response_preview"] = content[:200]
            self.logs.append(log_entry)

            return content

        except Exception as e:
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(e)
            log_entry["error_type"] = type(e).__name__
            self.logs.append(log_entry)
            raise

    def is_done(self) -> bool:
        """Check if the user interaction should terminate.

        Returns:
            True if max_turns has been reached.
        """
        return self._turn_count >= self._max_turns

    def get_tool(self):
        """Return a CAMEL FunctionTool for agent-to-user interaction.

        Returns:
            CAMEL FunctionTool wrapping the respond method.
        """
        _check_camel_installed()
        from camel.toolkits import FunctionTool

        user_instance = self

        def ask_user(question: str) -> str:
            """Ask the user a question and get their response.

            Use this tool when you need to ask the user for clarification,
            additional information, or to confirm something.

            Args:
                question: The question to ask the user

            Returns:
                The user's response to the question
            """
            return user_instance.respond(question)

        return FunctionTool(func=ask_user)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this user.

        Returns:
            Dictionary containing trace information.
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "turns_used": self._turn_count,
            "max_turns": self._max_turns,
            "logs": self.logs,
            "agent_type": type(self.user_agent).__name__,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this user.

        Returns:
            Dictionary containing configuration information.
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "max_turns": self._max_turns,
            "initial_query": self._initial_query[:200],  # Truncate for config
            "agent_type": type(self.user_agent).__name__,
        }


def camel_role_playing_execution_loop(
    role_playing,
    task,
    max_steps: int = 10,
    tracer: Optional["CamelRolePlayingTracer"] = None,
) -> Any:
    """Execution loop for benchmarks using CAMEL's RolePlaying.

    CAMEL's RolePlaying manages its own agent-user coordination: it alternates
    between an assistant agent and a user agent via `step()` calls, handling
    turn-taking and termination internally. This differs from MASEval's default
    execution loop, which coordinates between an `AgentAdapter` and a `User`.

    This function bridges the two: call it from your benchmark's `execution_loop`
    override to let RolePlaying handle the interaction while MASEval handles
    the evaluation lifecycle.

    Args:
        role_playing: The CAMEL RolePlaying instance.
        task: Current MASEval task (passed for interface consistency).
        max_steps: Maximum number of RolePlaying steps. Defaults to 10.
        tracer: Optional `CamelRolePlayingTracer` to record step data.

    Returns:
        Final answer from the assistant agent, or `None` if no response.

    Example:
        ```python
        class CamelRolePlayingBenchmark(Benchmark):
            def setup_agents(self, agent_data, environment, task, user):
                self._role_playing = RolePlaying(
                    assistant_role_name="Assistant",
                    user_role_name="User",
                    task_prompt=task.query,
                )

                # Wrap both agents for tracing
                assistant = CamelAgentAdapter(
                    self._role_playing.assistant_agent, "assistant"
                )
                user_agent = CamelAgentAdapter(
                    self._role_playing.user_agent, "user_agent"
                )

                # Optional: create tracer
                self._tracer = CamelRolePlayingTracer(self._role_playing)
                self.register(self._tracer)

                return [assistant], {"assistant": assistant, "user_agent": user_agent}

            def execution_loop(self, agents, task, environment, user):
                return camel_role_playing_execution_loop(
                    self._role_playing, task, tracer=self._tracer
                )
        ```
    """
    _check_camel_installed()

    # Initialize chat if RolePlaying supports it
    if hasattr(role_playing, "init_chat"):
        role_playing.init_chat()

    final_answer = None
    for step in range(max_steps):
        # Execute one step of RolePlaying
        assistant_response, user_response = role_playing.step()

        # Record step in tracer if provided
        if tracer is not None:
            tracer.record_step(assistant_response, user_response)

        # Extract final answer from assistant response
        final_answer = _extract_response_content(assistant_response)

        # Check for termination
        assistant_terminated = getattr(assistant_response, "terminated", False)
        user_terminated = getattr(user_response, "terminated", False)

        if assistant_terminated or user_terminated:
            break

    return final_answer


class CamelRolePlayingTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL RolePlaying.

    RolePlaying is a CAMEL-AI component that orchestrates turn-based interaction
    between two ChatAgents (an assistant and a simulated user).

    When using RolePlaying, you typically wrap both agents with `CamelAgentAdapter`
    to trace their individual message histories and token usage. However, this
    misses orchestration-level data that no single agent owns: how many
    back-and-forth steps occurred, which agent terminated the conversation, etc.

    This tracer fills that gap by capturing RolePlaying's orchestration state,
    giving you the complete picture alongside individual agent traces.

    Register with benchmark to include in trace collection:

        tracer = CamelRolePlayingTracer(role_playing)
        self.register(tracer)

    Then call record_step() after each RolePlaying.step():

        assistant_response, user_response = role_playing.step()
        tracer.record_step(assistant_response, user_response)

    Example:
        ```python
        class MyBenchmark(Benchmark):
            def setup_agents(self, agent_data, environment, task, user):
                self._role_playing = RolePlaying(...)
                self._tracer = CamelRolePlayingTracer(self._role_playing)
                self.register(self._tracer)
                return [...]

            def execution_loop(self, agents, task, environment, user):
                for _ in range(10):
                    assistant_response, user_response = self._role_playing.step()
                    self._tracer.record_step(assistant_response, user_response)
                    if assistant_response.terminated:
                        break
                return final_answer
        ```
    """

    def __init__(self, role_playing, name: str = "role_playing"):
        """Initialize the RolePlaying tracer.

        Args:
            role_playing: CAMEL RolePlaying instance to trace.
            name: Name for this tracer in traces. Defaults to "role_playing".
        """
        _check_camel_installed()
        self.role_playing = role_playing
        self.name = name
        self._step_count = 0
        self._termination_reason: Optional[str] = None
        self._step_logs: List[Dict[str, Any]] = []

    def record_step(self, assistant_response, user_response) -> None:
        """Record data from a RolePlaying step.

        Call this after each role_playing.step() to track progress.

        Args:
            assistant_response: ChatAgentResponse from the assistant.
            user_response: ChatAgentResponse from the user agent.
        """
        self._step_count += 1

        # Record step details
        step_log: Dict[str, Any] = {
            "step": self._step_count,
            "timestamp": datetime.now().isoformat(),
        }

        # Check termination
        assistant_terminated = getattr(assistant_response, "terminated", False)
        user_terminated = getattr(user_response, "terminated", False)

        step_log["assistant_terminated"] = assistant_terminated
        step_log["user_terminated"] = user_terminated

        if assistant_terminated and self._termination_reason is None:
            self._termination_reason = "assistant_terminated"
        elif user_terminated and self._termination_reason is None:
            self._termination_reason = "user_terminated"

        self._step_logs.append(step_log)

    def gather_traces(self) -> Dict[str, Any]:
        """Gather orchestration traces from RolePlaying.

        Returns:
            Dictionary containing:
            - name: Tracer name
            - type: "role_playing_orchestration"
            - step_count: Number of steps executed
            - termination_reason: Why the interaction ended
            - step_logs: Per-step termination data
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "trace_type": "role_playing_orchestration",
            "step_count": self._step_count,
            "termination_reason": self._termination_reason,
            "step_logs": self._step_logs,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from RolePlaying.

        Returns:
            Dictionary containing RolePlaying configuration.
        """
        config: Dict[str, Any] = {
            **super().gather_config(),
            "name": self.name,
        }

        # Extract task prompt if available
        if hasattr(self.role_playing, "task_prompt"):
            task_prompt = self.role_playing.task_prompt
            config["task_prompt"] = str(task_prompt)[:500] if task_prompt else None

        # Extract role names if available
        if hasattr(self.role_playing, "assistant_role_name"):
            config["assistant_role"] = self.role_playing.assistant_role_name
        if hasattr(self.role_playing, "user_role_name"):
            config["user_role"] = self.role_playing.user_role_name

        return config


class CamelWorkforceTracer(TraceableMixin, ConfigurableMixin):
    """Collects orchestration traces from CAMEL Workforce.

    Workforce is a CAMEL-AI component that manages task decomposition, worker
    assignment, and retry strategies for complex multi-agent collaboration.

    When using Workforce, you typically wrap individual workers with
    `CamelAgentAdapter` to trace their message histories. However, this misses
    orchestration-level data that no single worker owns: how the problem was
    decomposed into subtasks, which worker was assigned to each task, task
    dependencies, and completion status.

    This tracer fills that gap by capturing Workforce's orchestration state,
    giving you the complete picture alongside individual worker traces.

    Note: This tracer accesses Workforce internal attributes (`_children`,
    `_assignees`, `_pending_tasks`, etc.) which may change with CAMEL updates.

    Register with benchmark to include in trace collection:

        tracer = CamelWorkforceTracer(workforce)
        self.register(tracer)

    Example:
        ```python
        class MyBenchmark(Benchmark):
            def setup_agents(self, agent_data, environment, task, user):
                workforce = Workforce(...)
                self._workforce = workforce

                # Create tracer and register it
                tracer = CamelWorkforceTracer(workforce)
                self.register(tracer)

                # Wrap individual workers for message tracing
                worker_adapters = {}
                for worker in workforce._children:
                    adapter = CamelAgentAdapter(worker.agent, name=worker.name)
                    worker_adapters[worker.name] = adapter

                return [], worker_adapters
        ```
    """

    def __init__(self, workforce, name: str = "workforce"):
        """Initialize the Workforce tracer.

        Args:
            workforce: CAMEL Workforce instance to trace.
            name: Name for this tracer in traces. Defaults to "workforce".
        """
        _check_camel_installed()
        self.workforce = workforce
        self.name = name

    def gather_traces(self) -> Dict[str, Any]:
        """Gather orchestration traces from Workforce.

        Extracts task decomposition, worker assignments, and task lifecycle
        information from the Workforce's internal state.

        Returns:
            Dictionary containing:
            - name: Tracer name
            - type: "workforce_orchestration"
            - task_decomposition: Task dependency graph
            - worker_assignments: Which worker handled which task
            - completed_tasks: List of completed task summaries
            - pending_tasks: Count of pending tasks
        """
        traces: Dict[str, Any] = {
            **super().gather_traces(),
            "name": self.name,
            "trace_type": "workforce_orchestration",
        }

        # Extract task dependency graph
        traces["task_decomposition"] = self._extract_task_graph()

        # Extract worker assignments
        assignees = getattr(self.workforce, "_assignees", {})
        traces["worker_assignments"] = {str(k): str(v) for k, v in assignees.items()} if assignees else {}

        # Extract completed tasks
        traces["completed_tasks"] = self._extract_completed_tasks()

        # Extract pending task count
        pending = getattr(self.workforce, "_pending_tasks", [])
        traces["pending_tasks_count"] = len(pending) if pending else 0

        return traces

    def _extract_task_graph(self) -> Dict[str, Any]:
        """Extract task dependency graph from Workforce.

        Returns:
            Dictionary mapping task IDs to their dependencies.
        """
        deps = getattr(self.workforce, "_task_dependencies", {})
        if not deps:
            return {}

        graph: Dict[str, Any] = {}
        for task_id, dep_ids in deps.items():
            try:
                graph[str(task_id)] = [str(d) for d in dep_ids] if dep_ids else []
            except (TypeError, ValueError):
                graph[str(task_id)] = []

        return graph

    def _extract_completed_tasks(self) -> List[Dict[str, Any]]:
        """Extract completed task summaries from Workforce.

        Returns:
            List of task summaries with id and content preview.
        """
        completed = getattr(self.workforce, "_completed_tasks", [])
        if not completed:
            return []

        tasks: List[Dict[str, Any]] = []
        for task in completed:
            task_info: Dict[str, Any] = {}

            # Try to get task ID
            if hasattr(task, "id"):
                task_info["id"] = str(task.id)

            # Try to get task content (truncated)
            if hasattr(task, "content"):
                content = str(task.content)
                task_info["content"] = content[:200] if len(content) > 200 else content

            # Try to get task result
            if hasattr(task, "result"):
                result = str(task.result) if task.result else None
                task_info["result"] = result[:200] if result and len(result) > 200 else result

            if task_info:
                tasks.append(task_info)

        return tasks

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from Workforce.

        Returns:
            Dictionary containing Workforce configuration.
        """
        config: Dict[str, Any] = {
            **super().gather_config(),
            "name": self.name,
        }

        # Extract workforce mode if available
        if hasattr(self.workforce, "mode"):
            config["mode"] = str(self.workforce.mode)

        # Extract worker names
        children = getattr(self.workforce, "_children", [])
        if children:
            worker_names = []
            for child in children:
                name = getattr(child, "name", None) or type(child).__name__
                worker_names.append(str(name))
            config["workers"] = worker_names

        return config
