"""Smolagents integration for MASEval.

This module requires smolagents to be installed:
    pip install maseval[smolagents]
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from maseval import AgentAdapter, MessageHistory, LLMUser
from maseval.core.usage import CostCalculator, TokenUsage, Usage

__all__ = ["SmolAgentAdapter", "SmolAgentLLMUser"]

# Only import smolagents types for type checking, not at runtime
if TYPE_CHECKING:
    from smolagents import UserInputTool
else:
    # Lazy import with helpful error message if smolagents is not installed
    UserInputTool = None


def _check_smolagents_installed():
    """Check if smolagents is installed and raise a helpful error if not."""
    try:
        import smolagents  # noqa: F401
    except ImportError as e:
        raise ImportError("smolagents is not installed. Install it with: pip install maseval[smolagents]") from e


class SmolAgentAdapter(AgentAdapter):
    """An AgentAdapter for HuggingFace smolagents MultiStepAgent.

    This adapter integrates smolagents' MultiStepAgent with MASEval's benchmarking framework,
    converting smolagents' internal message format to OpenAI-compatible MessageHistory format.
    It automatically tracks tool calls, tool responses, agent reasoning steps, and provides
    comprehensive execution monitoring through smolagents' built-in memory system.

    The adapter leverages smolagents' native memory storage as the source of truth, dynamically
    fetching messages, logs, and execution traces from the agent's internal state. This ensures
    accurate tracking of tool usage, timing, and token consumption without additional overhead.

    How to use:
        1. **Create a smolagents agent** with tools and configuration
        2. **Wrap with SmolAgentAdapter** to enable MASEval integration
        3. **Use in benchmarks** or call directly for testing
        4. **Access traces and config** for analysis and debugging

        Example workflow:
            ```python
            from maseval.interface.agents.smolagents import SmolAgentAdapter
            from smolagents import MultiStepAgent, ToolCallingAgent
            from smolagents.tools import DuckDuckGoSearchTool

            # Create a smolagents agent
            agent = ToolCallingAgent(
                tools=[DuckDuckGoSearchTool()],
                model="gpt-4",
                max_steps=10
            )

            # Wrap with adapter
            agent_adapter = SmolAgentAdapter(agent, name="search_agent")

            # Run agent
            result = agent_adapter.run("What's the latest news on AI?")

            # Access message history in OpenAI format
            for msg in agent_adapter.get_messages():
                print(f"{msg['role']}: {msg['content']}")

            # Gather aggregated usage
            usage = agent_adapter.gather_usage()
            print(f"Total tokens: {usage.total_tokens}")

            # Gather execution traces with timing
            traces = agent_adapter.gather_traces()
            print(f"Total duration: {traces['total_duration_seconds']}s")

            # Use in benchmark
            benchmark = MyBenchmark(agent_data={"agent": agent_adapter})
            results = benchmark.run(tasks)
            ```

        The adapter automatically converts smolagents' ActionStep and PlanningStep objects
        into structured logs, preserving timing, token usage, tool calls, and error information.

    Message Format:
        smolagents uses its own message format. The adapter converts to `maseval` / OpenAI format.

        Tool calls are preserved with their IDs, names, and arguments.

    Execution Monitoring:
        The adapter provides comprehensive monitoring through `gather_traces()`:

        - **Token usage**: Input, output, and total tokens per step and aggregated
        - **Timing**: Duration per step and total execution time
        - **Tool calls**: Complete tool call history with arguments and results
        - **Errors**: Error tracking with type and message
        - **Observations**: Tool outputs and agent observations

    Requires:
        smolagents to be installed: `pip install maseval[smolagents]`
    """

    def __init__(
        self,
        agent_instance: Any,
        name: str,
        callbacks: Any = None,
        cost_calculator: Optional[CostCalculator] = None,
        model_id: Optional[str] = None,
    ):
        """Initialize the Smolagent adapter.

        Note: We don't call super().__init__() to avoid initializing self.logs as a list,
        since we override it as a property that dynamically fetches from agent.memory.

        Args:
            agent_instance: smolagents MultiStepAgent or similar
            name: Agent name for identification
            callbacks: Optional list of AgentCallback instances
            cost_calculator: Optional cost calculator. If not provided, a
                ``LiteLLMCostCalculator`` is created automatically when litellm
                is available.
            model_id: Optional model ID for cost calculation. If not provided,
                auto-detected from ``agent.model.model_id``.
        """
        self.agent = agent_instance
        self.name = name
        self.callbacks = callbacks or []
        self.messages = None
        self._cost_calculator = cost_calculator
        self._model_id = model_id
        self._auto_calculator: Optional[CostCalculator] = None
        self._auto_attempted = False

    @property
    def logs(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        """Dynamically generate logs from smolagents' internal memory.

        Converts smolagents' ActionStep and PlanningStep objects into log entries
        compatible with the AgentAdapter contract, including all available properties.

        Returns:
            List of log dictionaries with comprehensive step information
        """
        _check_smolagents_installed()
        from smolagents.memory import ActionStep, PlanningStep, TaskStep

        logs_list: List[Dict[str, Any]] = []

        if not hasattr(self.agent, "memory") or not hasattr(self.agent.memory, "steps"):
            return logs_list

        for step in self.agent.memory.steps:
            if isinstance(step, ActionStep):
                log_entry: Dict[str, Any] = {
                    "step_type": "ActionStep",
                    "step_number": step.step_number,
                    "status": self._determine_step_status(step),
                }

                # Timing information
                if hasattr(step, "timing") and step.timing:
                    log_entry["start_time"] = step.timing.start_time
                    log_entry["end_time"] = step.timing.end_time
                    log_entry["duration_seconds"] = step.timing.duration

                # Token usage information
                if hasattr(step, "token_usage") and step.token_usage:
                    log_entry["input_tokens"] = step.token_usage.input_tokens
                    log_entry["output_tokens"] = step.token_usage.output_tokens
                    log_entry["total_tokens"] = step.token_usage.total_tokens

                # Model input messages - convert to MASEval format
                if hasattr(step, "model_input_messages") and step.model_input_messages:
                    log_entry["model_input_messages"] = self._convert_smolagents_messages(step.model_input_messages).to_list()

                # Tool calls (ToolCall objects)
                if hasattr(step, "tool_calls") and step.tool_calls:
                    log_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in step.tool_calls
                    ]

                # Error information
                if step.error:
                    log_entry["error"] = str(step.error)
                    log_entry["error_type"] = type(step.error).__name__

                # Model output message - convert to MASEval format
                if hasattr(step, "model_output_message") and step.model_output_message:
                    converted = self._convert_smolagents_messages([step.model_output_message])
                    if len(converted) > 0:
                        log_entry["model_output_message"] = converted[0]

                # Model output (raw)
                if hasattr(step, "model_output") and step.model_output is not None:
                    log_entry["model_output"] = step.model_output

                # Code action (for CodeAgent)
                if hasattr(step, "code_action") and step.code_action:
                    log_entry["code_action"] = step.code_action

                # Observations
                if hasattr(step, "observations") and step.observations:
                    log_entry["observations"] = step.observations

                # Observations images
                if hasattr(step, "observations_images") and step.observations_images:
                    log_entry["observations_images_count"] = len(step.observations_images)

                # Action output
                if hasattr(step, "action_output") and step.action_output is not None:
                    # Convert to string if it's not JSON-serializable
                    try:
                        log_entry["action_output"] = step.action_output
                    except (TypeError, ValueError):
                        log_entry["action_output"] = str(step.action_output)

                # Is final answer flag
                if hasattr(step, "is_final_answer"):
                    log_entry["is_final_answer"] = step.is_final_answer

                logs_list.append(log_entry)

            elif isinstance(step, PlanningStep):
                log_entry = {
                    "step_type": "PlanningStep",
                }

                # Timing information
                if hasattr(step, "timing") and step.timing:
                    log_entry["start_time"] = step.timing.start_time
                    log_entry["end_time"] = step.timing.end_time
                    log_entry["duration_seconds"] = step.timing.duration

                # Token usage information
                if hasattr(step, "token_usage") and step.token_usage:
                    log_entry["input_tokens"] = step.token_usage.input_tokens
                    log_entry["output_tokens"] = step.token_usage.output_tokens
                    log_entry["total_tokens"] = step.token_usage.total_tokens

                # Model input messages - convert to MASEval format
                if hasattr(step, "model_input_messages") and step.model_input_messages:
                    log_entry["model_input_messages"] = self._convert_smolagents_messages(step.model_input_messages).to_list()

                # Model output message - convert to MASEval format
                if hasattr(step, "model_output_message") and step.model_output_message:
                    converted = self._convert_smolagents_messages([step.model_output_message])
                    if len(converted) > 0:
                        log_entry["model_output_message"] = converted[0]

                # Plan
                if hasattr(step, "plan") and step.plan:
                    log_entry["plan"] = step.plan

                logs_list.append(log_entry)

            elif isinstance(step, TaskStep):
                # Log task initiation
                log_entry = {
                    "step_type": "TaskStep",
                    "task": step.task,
                }

                # Task images if present
                if hasattr(step, "task_images") and step.task_images:
                    log_entry["task_images_count"] = len(step.task_images)

                logs_list.append(log_entry)

        return logs_list

    def gather_traces(self) -> dict:
        """Gather traces including message history and monitoring data.

        Extends the base class to include smolagents' per-step monitoring data
        (token usage, timing, actions, observations). Aggregated usage totals
        are available via ``gather_usage()``.

        Returns:
            Dict containing messages and per-step monitoring statistics.
        """
        base_logs = super().gather_traces()
        _check_smolagents_installed()

        # Extract monitoring data from agent's memory steps
        if hasattr(self.agent, "memory") and hasattr(self.agent.memory, "steps"):
            steps_stats = []
            total_duration = 0.0

            from smolagents.memory import ActionStep, PlanningStep

            for step in self.agent.memory.steps:
                # Process ActionStep and PlanningStep (both have token_usage and timing)
                if isinstance(step, (ActionStep, PlanningStep)):
                    step_info: Dict[str, Any] = {
                        "step_number": getattr(step, "step_number", None),
                    }

                    # Add timing information
                    if hasattr(step, "timing") and step.timing:
                        step_info["duration_seconds"] = step.timing.duration
                        if step.timing.duration is not None:
                            total_duration += step.timing.duration

                    # Add per-step token usage
                    if hasattr(step, "token_usage") and step.token_usage:
                        step_info["input_tokens"] = step.token_usage.input_tokens
                        step_info["output_tokens"] = step.token_usage.output_tokens
                        step_info["total_tokens"] = step.token_usage.total_tokens

                    # Add action details for ActionStep
                    if isinstance(step, ActionStep):
                        if hasattr(step, "observations") and step.observations:
                            step_info["observations"] = step.observations
                        if hasattr(step, "action_output") and step.action_output:
                            step_info["action_output"] = str(step.action_output)
                        if hasattr(step, "error") and step.error:
                            step_info["error"] = str(step.error)

                    # Add plan for PlanningStep
                    elif isinstance(step, PlanningStep):
                        if hasattr(step, "plan") and step.plan:
                            step_info["plan"] = step.plan

                    steps_stats.append(step_info)

            base_logs.update(
                {
                    "total_steps": len(steps_stats),
                    "total_duration_seconds": total_duration,
                    "steps_detail": steps_stats,
                }
            )

        return base_logs

    def _resolve_model_id(self) -> Optional[str]:
        """Auto-detect model ID from smolagents agent.

        All smolagents model classes (LiteLLMModel, OpenAIServerModel,
        TransformersModel, etc.) inherit from ``Model`` which stores
        ``model_id`` on the instance.
        """
        try:
            return self.agent.model.model_id
        except AttributeError:
            return None

    def _resolve_cost_calculator(self) -> Optional[CostCalculator]:
        """Return the cost calculator, auto-creating one if litellm is available."""
        from maseval.interface.agents._cost import resolve_auto_cost_calculator

        calculator, self._auto_calculator, self._auto_attempted = resolve_auto_cost_calculator(
            self._cost_calculator, self._auto_calculator, self._auto_attempted
        )
        return calculator

    def _gather_usage(self) -> Usage:
        """Gather aggregated token usage across all agent steps.

        Walks smolagents' memory steps (ActionStep and PlanningStep) and sums
        their ``token_usage`` into a single ``TokenUsage``.

        Returns:
            Aggregated token usage, or empty ``Usage`` if no steps or no usage data.
        """
        _check_smolagents_installed()

        if not (hasattr(self.agent, "memory") and hasattr(self.agent.memory, "steps")):
            return Usage()

        from smolagents.memory import ActionStep, PlanningStep

        total_input = 0
        total_output = 0
        has_usage = False

        for step in self.agent.memory.steps:
            if isinstance(step, (ActionStep, PlanningStep)):
                if hasattr(step, "token_usage") and step.token_usage:
                    total_input += step.token_usage.input_tokens
                    total_output += step.token_usage.output_tokens
                    has_usage = True

        if not has_usage:
            return Usage()

        return TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
        )

    def gather_config(self) -> dict[str, Any]:
        """Gather configuration from this SmolAgent.

        Integrates with smolagents' native configuration system by accessing
        the agent's to_dict() method which includes comprehensive config data.

        Returns:
            Dictionary containing:
            - type: Component class name
            - gathered_at: ISO timestamp
            - name: Agent name
            - agent_type: Underlying agent class name
            - adapter_type: SmolAgentAdapter
            - callbacks: List of callback class names
            - smolagents_config: Full configuration from agent.to_dict() including:
                - model: Model configuration with class and parameters
                - tools: List of tool configurations
                - max_steps: Maximum number of steps
                - planning_interval: Planning interval (if set)
                - verbosity_level: Logging verbosity
                - additional_authorized_imports: Additional imports (CodeAgent only)
                - executor_type: Code executor type (CodeAgent only)
                - managed_agents: List of managed agent configs (if any)
        """
        base_config = super().gather_config()
        _check_smolagents_installed()

        # Get comprehensive config from smolagents' native to_dict() method.
        # No try/except: if to_dict() exists but fails, the error should
        # propagate so it's visible in the registry's error output.
        if hasattr(self.agent, "to_dict"):
            base_config["smolagents_config"] = self.agent.to_dict()

        return base_config

    def get_messages(self) -> MessageHistory:
        """Get message history by converting from smolagents memory.

        This method dynamically fetches messages from the agent's internal memory
        and converts them to MASEval format.

        Returns:
            MessageHistory with converted messages from smolagents
        """
        _check_smolagents_installed()

        # Get messages from smolagents memory
        smol_messages = self.agent.write_memory_to_messages()

        # Convert and return
        return self._convert_smolagents_messages(smol_messages)

    def _run_agent(self, query: str) -> str:
        _check_smolagents_installed()

        # Run the agent (this updates the agent's internal memory and returns the final answer)
        # All execution details are tracked in agent.memory.steps automatically
        final_answer = self.agent.run(query)

        # Return the final answer (traces are captured via get_messages() and gather_traces())
        return final_answer

    @staticmethod
    def _determine_step_status(step) -> str:
        """Determine the status of an ActionStep.

        Detects crashed steps where smolagents raised ``AgentGenerationError``
        before setting ``step.error``. That is the only ``AgentError`` subclass
        that is re-raised instead of being recorded on the step, so a step with
        no output fields (``model_output_message``, ``tool_calls``,
        ``observations``, ``action_output``) and ``is_final_answer=False``
        is always an error — not a success.

        Returns:
            ``"error"`` if ``step.error`` is set or the step has no output fields,
            ``"success"`` otherwise.
        """
        if step.error:
            return "error"

        has_output = (
            (hasattr(step, "model_output_message") and step.model_output_message is not None)
            or (hasattr(step, "tool_calls") and step.tool_calls)
            or (hasattr(step, "observations") and step.observations)
            or (hasattr(step, "action_output") and step.action_output is not None)
            or (hasattr(step, "is_final_answer") and step.is_final_answer)
        )

        if not has_output:
            return "error"

        return "success"

    def _convert_smolagents_messages(self, smol_messages: list) -> MessageHistory:
        """Convert smolagents message format to MASEval MessageHistory.

        Smolagents uses ChatMessage objects with MessageRole enums. This method
        normalizes them to OpenAI-compatible format with string literal roles
        while preserving tool call information.

        Args:
            smol_messages: List of ChatMessage objects from smolagents

        Returns:
            MessageHistory with converted messages
        """
        converted_messages = []

        for msg in smol_messages:
            # smolagents messages are ChatMessage objects with role and content attributes
            # Handle both dict format (from dict conversion) and ChatMessage objects
            if isinstance(msg, dict):
                role = msg.get("role", "assistant")
                content = msg.get("content", "")
            else:
                # ChatMessage object with MessageRole enum
                role = getattr(msg, "role", "assistant")
                content = getattr(msg, "content", "")

            # Convert MessageRole enum to string literal if needed
            if hasattr(role, "value"):
                # It's an enum, extract the string value
                role = role.value  # type: ignore
            elif not isinstance(role, str):
                # Convert to string if it's something else
                role = str(role).lower()

            # Build the converted message
            converted_msg = {
                "role": role,
                "content": content,
            }

            # Handle tool calls if present
            if isinstance(msg, dict):
                if "tool_calls" in msg:
                    converted_msg["tool_calls"] = msg["tool_calls"]
                if "tool_call_id" in msg:
                    converted_msg["tool_call_id"] = msg["tool_call_id"]
                if role == "tool" and "name" in msg:
                    converted_msg["name"] = msg["name"]
                if "metadata" in msg:
                    converted_msg["metadata"] = msg["metadata"]
            else:
                # ChatMessage object - check for attributes
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    converted_msg["tool_calls"] = msg.tool_calls
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    converted_msg["tool_call_id"] = msg.tool_call_id
                if role == "tool" and hasattr(msg, "name") and msg.name:
                    converted_msg["name"] = msg.name

            converted_messages.append(converted_msg)

        return MessageHistory(converted_messages)


class SmolAgentLLMUser(LLMUser):
    """A smolagents-specific LLM user that provides a tool for user interaction.

    Extends LLMUser to provide a smolagents-compatible tool via get_tool().
    Requires smolagents to be installed.

    Example:
        ```python
        from maseval.interface.agents.smolagents import SmolAgentLLMUser

        user = SmolAgentLLMUser(...)
        tool = user.get_tool()  # Returns a SmolAgentUserSimulationInputTool
        ```
    """

    def get_tool(self) -> Any:
        """Get a smolagents-compatible tool for user interaction.

        Returns a `SmolAgentUserSimulationInputTool` instance that wraps this user
        and can be passed directly to a smolagents agent.

        Returns:
            A tool instance compatible with smolagents that simulates user responses.

        Example:
            ```python
            user = SmolAgentLLMUser(model=model, persona="...", scenario="...")
            tool = user.get_tool()
            agent = CodeAgent(tools=[tool, ...], model=model)
            ```
        """
        _check_smolagents_installed()
        from maseval.interface.agents.smolagents_optional import SmolAgentUserSimulationInputTool

        return SmolAgentUserSimulationInputTool(user=self)
