"""MACS Benchmark - Multi-Agent Collaboration Scenarios.

Framework-agnostic implementation of the AWS MACS benchmark for evaluating
multi-agent collaboration in enterprise applications.

Reference: https://arxiv.org/abs/2412.05449
Dataset: https://github.com/aws-samples/multiagent-collab-scenario-benchmark

Usage:
    from maseval.benchmark.macs import (
        MACSBenchmark, MACSEnvironment, MACSEvaluator, MACSGenericTool,
        load_tasks, load_agent_config, configure_model_ids,
    )

    # Load data and configure model IDs for components
    tasks = load_tasks("travel", limit=5)
    configure_model_ids(
        tasks,
        tool_model_id="gemini-2.5-flash",
        user_model_id="gemini-2.5-flash",
        evaluator_model_id="gemini-2.5-flash",
    )
    agent_config = load_agent_config("travel")

    # Create your framework-specific benchmark subclass
    class MyMACSBenchmark(MACSBenchmark):
        def setup_agents(self, agent_data, environment, task, user):
            # Your framework-specific agent creation
            ...

        def get_model_adapter(self, model_id, **kwargs):
            # Create and optionally register model adapters
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyMACSBenchmark()
    results = benchmark.run(tasks, agent_data=agent_config)
"""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from maseval import (
    AgentAdapter,
    Benchmark,
    User,
    Environment,
    Evaluator,
    MessageHistory,
    ModelAdapter,
    Task,
    TaskExecutionStatus,
    ToolInvocationHistory,
    ToolLLMSimulator,
    LLMUser,
    AgentError,
    EnvironmentError,
    validate_arguments_from_schema,
)
from maseval.core.config import ConfigurableMixin
from maseval.core.tracing import TraceableMixin


# Statuses where agent is accountable (included in scoring)
# Note: task_timeout is included - timeouts count as failures in MACS
SCOREABLE_STATUSES = frozenset(
    {
        TaskExecutionStatus.SUCCESS.value,
        TaskExecutionStatus.AGENT_ERROR.value,
        TaskExecutionStatus.TASK_TIMEOUT.value,
    }
)


# =============================================================================
# Tool
# =============================================================================


class MACSGenericTool(TraceableMixin, ConfigurableMixin):
    """Framework-agnostic tool with LLM-based response simulation.

    This tool does not inherit from any framework-specific Tool class.
    Users wrap it for their framework using composition. Example for smolagents:

        class MySmolagentsTool(smolagents.Tool):
            skip_forward_signature_validation = True

            def __init__(self, generic_tool: MACSGenericTool):
                self.generic_tool = generic_tool
                self.name = generic_tool.name
                self.description = generic_tool.description
                self.inputs = generic_tool.inputs
                self.output_type = "string"
                super().__init__()

            def forward(self, **kwargs) -> str:
                return self.generic_tool(**kwargs)

    Error Classification:
        - AgentError: Raised when agent provides invalid arguments (wrong types,
          missing required args, constraint violations). Agent's fault.
        - EnvironmentError: Raised when tool infrastructure fails after input
          validation (LLM simulator fails, internal error). Not agent's fault.
    """

    def __init__(self, spec: Dict[str, Any], model: ModelAdapter):
        """Initialize tool from specification.

        Args:
            spec: Tool specification with 'name', 'description', 'input_schema'
            model: ModelAdapter for LLM-based response simulation
        """
        super().__init__()
        self.name = spec["name"]
        self.description = spec.get("description", "")
        self.input_schema = spec.get("input_schema", {})
        self.output_type = "string"
        self.history = ToolInvocationHistory()

        # Convert schema to inputs format
        self.inputs = self._schema_to_inputs(self.input_schema)

        # Create simulator
        self.simulator = ToolLLMSimulator(
            model=model,
            tool_name=self.name,
            tool_description=self.description,
            tool_inputs=self.inputs,
        )

    @staticmethod
    def _schema_to_inputs(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON schema to inputs format."""
        inputs = {}
        for k, prop in schema.get("properties", {}).items():
            dtype = prop.get("data_type") or prop.get("type", "string")
            inputs[k] = {
                "type": dtype if isinstance(dtype, str) else "string",
                "description": prop.get("description", ""),
            }
        return inputs

    def __call__(self, **kwargs) -> str:
        """Execute the tool with simulated response.

        Args:
            **kwargs: Tool arguments provided by the agent.

        Returns:
            Simulated tool response string.

        Raises:
            AgentError: If agent provides invalid arguments (wrong types, missing
                required args). This is the agent's fault.
            EnvironmentError: If tool infrastructure fails after validation (LLM
                simulator fails, internal error). Not the agent's fault.
        """
        # 1. VALIDATE INPUTS (agent's responsibility to get this right)
        try:
            validate_arguments_from_schema(
                kwargs,
                self.input_schema,
                component=self.name,
                strict=False,  # Allow extra args (some agents add metadata)
            )
        except AgentError:
            # Re-raise AgentError as-is
            raise
        except (TypeError, ValueError, KeyError) as e:
            # Convert other validation errors to AgentError
            raise AgentError(
                f"Invalid arguments for tool '{self.name}': {e}",
                component=self.name,
            ) from e

        # 2. EXECUTE (our responsibility - if this fails after validation, it's on us)
        try:
            # ToolLLMSimulator raises ToolSimulatorError (subclass of EnvironmentError)
            # on failure, so it's automatically classified correctly
            response, details = self.simulator(actual_inputs=kwargs)
        except EnvironmentError:
            # Re-raise EnvironmentError as-is (includes ToolSimulatorError)
            raise
        except Exception as e:
            # Any other error in our tool code is our fault
            raise EnvironmentError(
                f"Tool '{self.name}' internal error: {e}",
                component=self.name,
            ) from e

        # 3. RECORD INVOCATION
        self.history.add_invocation(
            inputs=kwargs,
            outputs=response,
            status="success",
            meta=details,
        )
        return response

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces."""
        return {
            **super().gather_traces(),
            "name": self.name,
            "invocations": self.history.to_list(),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration."""
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}: {v['type']}" for k, v in self.inputs.items())
        return f"{self.__class__.__name__}({self.name}({args}) -> {self.output_type})"


# =============================================================================
# Evaluator
# =============================================================================


class MACSEvaluator(Evaluator):
    """LLM-based assertion evaluator for GSR metrics.

    Follows AWS paper methodology for Goal Success Rate (GSR) evaluation:
    - user: Evaluates user-observable behaviors (conversation only)
    - system: Evaluates internal behaviors (tool calls, agent actions)
    """

    DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"

    def __init__(
        self,
        model: ModelAdapter,
        task: Task,
        gsr_type: Literal["user", "system"] = "user",
        template: Optional[str] = None,
    ):
        """Initialize the evaluator.

        Args:
            model: ModelAdapter for LLM evaluation
            task: Task being evaluated (contains assertions)
            gsr_type: Either "user" or "system"
            template: Optional custom prompt template (uses default if None)
        """
        # Note: base Evaluator.__init__ does nothing, so we skip calling it
        # to avoid needing a real Environment instance
        self.model = model
        self.task = task
        self.gsr_type = gsr_type

        # Load template
        if template is None:
            template_file = "user.txt" if gsr_type == "user" else "system.txt"
            template_path = self.DEFAULT_TEMPLATES_DIR / template_file
            self.template = template_path.read_text()
        else:
            self.template = template

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter traces based on gsr_type.

        For user evaluation: Use user trace which contains the user-observable
        conversation by construction (what the user sees: queries, agent questions,
        user answers, and final answers).

        For system evaluation: Full traces including all agent messages and
        tool invocations (internal behaviors not visible to users).

        Args:
            traces: Full execution traces dict containing 'agents', 'tools', 'user', etc.

        Returns:
            Filtered dict with 'messages' and optionally 'tool_traces'
        """
        if self.gsr_type == "user":
            # User trace contains the user-observable conversation by construction
            user_trace = traces.get("user", {})
            return {"messages": user_trace.get("messages", [])}
        else:
            # System evaluation needs full agent messages and tool traces
            primary_agent_id = next(iter(traces.get("agents", {}).keys()), None)
            if primary_agent_id:
                agent_trace = traces["agents"][primary_agent_id]
                all_messages = agent_trace.get("messages", [])
            else:
                all_messages = []

            return {
                "messages": all_messages,
                "tool_traces": traces.get("tools", {}),
            }

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate the trace against assertions.

        Args:
            traces: Filtered traces dict containing 'messages' and optionally 'tool_traces'
            final_answer: Final answer from agents (unused in MACS evaluation)

        Returns:
            Dict with: gsr, partial_gsr, report (list of assertion judgments)
        """
        # Extract message history and tool traces from the traces dict
        trace = traces.get("messages", MessageHistory())
        tool_traces = traces.get("tool_traces")

        # Parse assertions for this evaluation type
        all_assertions = self.task.evaluation_data.get("assertions", [])
        assertions = self._parse_assertions(all_assertions)

        if not assertions:
            return {"gsr": 1.0, "partial_gsr": 1.0, "report": []}

        # Format conversation history
        history = self._format_conversation_history(trace)

        # Get scenario description
        scenario = self.task.metadata.get("scenario", "")
        if not scenario:
            raise ValueError("Task metadata must include 'scenario' for GSR evaluation")

        # Build prompt
        if self.gsr_type == "user":
            prompt = (
                self.template.replace("{{scenario}}", scenario).replace("{{history}}", history).replace("{{assertions}}", "\n".join(assertions))
            )
        else:
            invocations = self._format_tool_invocations(tool_traces or {})
            prompt = (
                self.template.replace("{{scenario}}", scenario)
                .replace("{{history}}", history)
                .replace("{{invocations}}", invocations)
                .replace("{{assertions}}", "\n".join(assertions))
            )

        # Get LLM judgment
        response = self.model.generate(prompt).strip()
        response = response.strip("```").strip("json").strip()

        try:
            report = json.loads(response)

            # Handle wrapped responses
            for key in ["assertions", "results"]:
                if isinstance(report, dict) and key in report:
                    report = report[key]
                    break

            if isinstance(report, dict):
                report = [report]

            gsr, partial_gsr = self._compute_gsr(report)

            for item in report:
                item["assertion_type"] = self.gsr_type

            return {"gsr": gsr, "partial_gsr": partial_gsr, "report": report}

        except json.JSONDecodeError as e:
            return {
                "gsr": 0.0,
                "partial_gsr": 0.0,
                "report": [],
                "error": f"JSON decode error: {e}",
                "raw_response": response,
            }

    def _parse_assertions(self, assertions: List[str]) -> List[str]:
        """Parse assertions and filter by type."""
        parsed = []
        user_prefix, system_prefix = "user:", "agent:"

        for assertion in assertions:
            assertion = assertion.strip()

            if self.gsr_type == "user":
                if assertion.lower().startswith(user_prefix):
                    parsed.append(assertion[len(user_prefix) :].strip())
                elif not assertion.lower().startswith(system_prefix):
                    # No prefix means user assertion (AWS default)
                    parsed.append(assertion)
            else:
                if assertion.lower().startswith(system_prefix):
                    parsed.append(assertion[len(system_prefix) :].strip())

        return parsed

    def _format_conversation_history(self, trace: MessageHistory) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for msg in trace:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, list):
                content = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _format_tool_invocations(self, tool_traces: Dict[str, Any]) -> str:
        """Format tool invocations for system-side evaluation."""
        lines = []

        for tool_name, tool_data in tool_traces.items():
            invocations = tool_data.get("invocations", [])
            for inv in invocations:
                lines.append(
                    f"Tool: {tool_name}\n"
                    f"  Inputs: {inv.get('inputs', {})}\n"
                    f"  Outputs: {inv.get('outputs', '')}\n"
                    f"  Status: {inv.get('status', 'Unknown')}"
                )

        return "\n".join(lines) if lines else "No tool invocations recorded"

    def _compute_gsr(self, report: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Compute GSR metrics.

        Returns:
            (gsr, partial_gsr) where:
            - gsr: 1.0 if all assertions True, else 0.0
            - partial_gsr: Percentage of True assertions
        """
        if not report:
            return 1.0, 1.0

        true_count = sum(1 for item in report if str(item.get("answer", "")).lower() == "true")
        total = len(report)

        gsr = 1.0 if true_count == total else 0.0
        partial_gsr = true_count / total if total > 0 else 1.0

        return gsr, partial_gsr


# =============================================================================
# User Simulator
# =============================================================================


class MACSUser(LLMUser):
    """MACS-specific user simulator with conversation limits.

    Extends the LLMUser class with MACS-specific behavior:
    - Maximum 5 turns of interaction (as per MACS paper)
    - </stop> token detection for natural conversation ending
    - User profile and scenario-aware responses
    - LLM-based satisfaction evaluation

    The simulator maintains a conversation history and uses an LLM to generate
    responses that are consistent with the user's profile and scenario.

    Note: This is a base class. Framework-specific subclasses should override
    get_tool() to return a compatible tool (e.g., SmolAgentUserSimulationInputTool).
    """

    DEFAULT_MAX_TURNS = 5
    DEFAULT_STOP_TOKENS = ["</stop>"]
    DEFAULT_EARLY_STOPPING_CONDITION = "ALL goals have been satisfactorily addressed by the assistant"

    def __init__(
        self,
        model: ModelAdapter,
        scenario: str,
        initial_query: str,
        name: str = "Simulated User",
        template: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        stop_tokens: Optional[List[str]] = None,
        early_stopping_condition: str = DEFAULT_EARLY_STOPPING_CONDITION,
    ):
        """Initialize MACS user simulator.

        Args:
            model: ModelAdapter for LLM-based response generation
            scenario: Full scenario text (contains goals and user background)
            initial_query: The initial query to the agent
            name: User name for identification (default: "Simulated User")
            template: Optional custom prompt template (uses base UserLLMSimulator template)
            max_turns: Maximum conversation turns (default: 5, per MACS paper)
            stop_tokens: Tokens indicating user satisfaction (default: ["</stop>"])
            early_stopping_condition: Description of when to emit stop token
                (default: "ALL goals have been satisfactorily addressed by the assistant")
        """
        # Extract user profile from scenario text
        user_profile = self._extract_user_profile(scenario)

        # Use default stop tokens if not provided
        if stop_tokens is None:
            stop_tokens = self.DEFAULT_STOP_TOKENS.copy()

        super().__init__(
            name=name,
            model=model,
            user_profile=user_profile,
            scenario=scenario,
            initial_query=initial_query,
            template=template,
            max_turns=max_turns,
            stop_tokens=stop_tokens,
            early_stopping_condition=early_stopping_condition,
        )

    def get_tool(self) -> Any:
        """Return a tool for agent interaction.

        This base implementation raises NotImplementedError.
        Framework-specific subclasses should override this method.

        For smolagents, use SmolAgentMACSUser which provides a smolagents-compatible tool.
        For langgraph, use LangGraphMACSUser which provides a langchain-compatible tool.

        Raises:
            NotImplementedError: Always, as this must be implemented by subclass.
        """
        raise NotImplementedError(
            "MACSUser.get_tool() must be overridden by framework-specific subclass. "
            "Use SmolAgentMACSUser for smolagents or LangGraphMACSUser for langgraph."
        )

    def reset(self) -> None:
        """Reset the conversation state for a new interaction."""
        self._stopped = False
        # Keep only the initial user message
        if len(self.messages) > 0:
            initial = self.messages[0]
            self.messages = MessageHistory([initial])
            self._turn_count = 1  # Initial message counts as first turn
        else:
            self.messages = MessageHistory()
            self._turn_count = 0

    @staticmethod
    def _extract_user_profile(scenario: str) -> Dict[str, Any]:
        """Extract user profile from scenario text.

        The MACS scenarios contain user background info after "Background:" marker.

        Args:
            scenario: Full scenario text with goals and background

        Returns:
            Dict with user profile fields
        """
        profile: Dict[str, Any] = {}

        # Find the Background section
        if "Background:" in scenario:
            background_section = scenario.split("Background:")[-1].strip()

            # Parse bullet points (* User's name is ...)
            for line in background_section.split("\n"):
                line = line.strip().lstrip("*").strip()
                if line.lower().startswith("user"):
                    # Try to extract key-value pairs
                    if " is " in line.lower():
                        key_part, value_part = line.split(" is ", 1)
                        key = key_part.lower().replace("user's ", "").replace("user ", "").strip()
                        profile[key] = value_part.strip().rstrip(".")
                    elif " has " in line.lower():
                        key_part, value_part = line.split(" has ", 1)
                        key = key_part.lower().replace("user's ", "").replace("user ", "").strip()
                        profile[key] = value_part.strip().rstrip(".")

        # Include full scenario as fallback context
        profile["full_scenario"] = scenario

        return profile

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces with MACS-specific information."""
        traces = super().gather_traces()
        traces.update(
            {
                "max_turns": self.max_turns,
                "turns_used": self._turn_count,
                "stopped_by_user": self._stopped,
            }
        )
        return traces


# =============================================================================
# Environment
# =============================================================================


class MACSEnvironment(Environment):
    """Unified environment for all MACS domains.

    Creates MACSGenericTool instances from task's environment_data.
    Tools are stored in a dict keyed by name for efficient lookup.
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        model_factory: Callable[[str], ModelAdapter],
        callbacks: Optional[List[Any]] = None,
    ):
        """Initialize environment.

        Args:
            task_data: Task data containing environment_data with tool specs
            model_factory: Factory function that creates a ModelAdapter for a given model_name
            callbacks: Optional callbacks
        """
        self._model_factory = model_factory
        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state from task data."""
        return {
            "tool_specs": task_data.get("environment_data", {}).get("tools", []),
        }

    def create_tools(self) -> Dict[str, MACSGenericTool]:  # type: ignore[override]
        """Create tools from task specifications.

        Each tool gets its own ModelAdapter instance for separate tracing.

        Returns:
            Dict mapping tool names to MACSGenericTool instances
        """
        tools: Dict[str, MACSGenericTool] = {}
        for tool_group in self.state["tool_specs"]:
            for action in tool_group.get("actions", []):
                name = action.get("name")
                if name and name not in tools:
                    # Each tool gets its own model adapter for separate traces
                    model = self._model_factory(f"tool_{name}")
                    tools[name] = MACSGenericTool(action, model)
        return tools

    def get_tools_for_agent(self, agent_spec: Dict[str, Any]) -> Dict[str, MACSGenericTool]:
        """Get tools for a specific agent based on its configuration.

        Args:
            agent_spec: Agent specification dict with 'tools' key containing tool group names

        Returns:
            Dict of MACSGenericTool instances assigned to this agent, keyed by name
        """
        tool_groups = agent_spec.get("tools", [])
        result: Dict[str, MACSGenericTool] = {}
        for tool_group in self.state["tool_specs"]:
            if tool_group.get("tool_name") in tool_groups:
                for action in tool_group.get("actions", []):
                    name = action.get("name")
                    if name and name in self.tools:
                        result[name] = self.tools[name]
        return result


# =============================================================================
# Benchmark
# =============================================================================


class MACSBenchmark(Benchmark):
    """MACS Benchmark - Framework-agnostic base class.

    This base class handles:
    - Environment setup with MACSEnvironment
    - Dual evaluator setup (user-side + system-side)
    - GSR metric aggregation

    Users must subclass and implement:
    - setup_agents() for their agent framework
    - get_model_adapter() to provide model adapters

    Model IDs for components (tools, user, evaluators) are read from task data:
    - task.environment_data["model_id"] for tool simulators
    - task.user_data["model_id"] for user simulator
    - task.evaluation_data["model_id"] for evaluators

    Use configure_model_ids() to set these values after loading tasks:

        from maseval.benchmark.macs import load_tasks, configure_model_ids

        tasks = load_tasks("travel")
        configure_model_ids(
            tasks,
            tool_model_id="gemini-2.5-flash",
            user_model_id="gemini-2.5-flash",
            evaluator_model_id="gemini-2.5-flash",
        )
    """

    def __init__(
        self,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        max_invocations: int = 5,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            callbacks: Benchmark callbacks
            n_task_repeats: Repetitions per task
            max_invocations: Maximum agent-user interaction rounds (default: 5 per MACS paper)
        """
        super().__init__(callbacks=callbacks, n_task_repeats=n_task_repeats, max_invocations=max_invocations, **kwargs)

    def _get_tool_model_id(self, task: Task) -> str:
        """Get tool simulator model ID from task.environment_data.

        Raises:
            ValueError: If model_id not configured in task.environment_data
        """
        model_id = task.environment_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Tool simulator model_id not configured in task.environment_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.macs import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('travel')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        tool_model_id='gemini-2.5-flash',\n"
                "        user_model_id='gemini-2.5-flash',\n"
                "        evaluator_model_id='gemini-2.5-flash',\n"
                "    )"
            )
        return model_id

    def _get_user_model_id(self, task: Task) -> str:
        """Get user simulator model ID from task.user_data.

        Raises:
            ValueError: If model_id not configured in task.user_data
        """
        model_id = task.user_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "User simulator model_id not configured in task.user_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.macs import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('travel')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        tool_model_id='gemini-2.5-flash',\n"
                "        user_model_id='gemini-2.5-flash',\n"
                "        evaluator_model_id='gemini-2.5-flash',\n"
                "    )"
            )
        return model_id

    def _get_evaluator_model_id(self, task: Task) -> str:
        """Get evaluator model ID from task.evaluation_data.

        Raises:
            ValueError: If model_id not configured in task.evaluation_data
        """
        model_id = task.evaluation_data.get("model_id")
        if model_id is None:
            raise ValueError(
                "Evaluator model_id not configured in task.evaluation_data.\n"
                "Use configure_model_ids() after loading tasks:\n\n"
                "    from maseval.benchmark.macs import load_tasks, configure_model_ids\n\n"
                "    tasks = load_tasks('travel')\n"
                "    configure_model_ids(\n"
                "        tasks,\n"
                "        tool_model_id='gemini-2.5-flash',\n"
                "        user_model_id='gemini-2.5-flash',\n"
                "        evaluator_model_id='gemini-2.5-flash',\n"
                "    )"
            )
        return model_id

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> MACSEnvironment:
        """Create environment for a task.

        Uses get_model_adapter() to create separate model adapters for each tool,
        enabling independent tracing per tool.

        Model ID is read from task.environment_data["model_id"].
        """
        tool_model_id = self._get_tool_model_id(task)

        # Create a factory that captures the model_id from task data
        # tool_name is passed by create_tools() with "tool_" prefix
        def tool_model_factory(tool_name: str) -> ModelAdapter:
            return self.get_model_adapter(
                tool_model_id,
                register_name=tool_name,
            )

        return MACSEnvironment(
            task_data={"environment_data": task.environment_data},
            model_factory=tool_model_factory,
        )

    def setup_user(  # type: ignore[invalid-method-override]
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,
        task: Task,
    ) -> MACSUser:
        """Create MACS user simulator.

        Creates a MACSUser with scenario and query from the task.
        The user profile is automatically extracted from the scenario text.
        Model ID is read from task.user_data["model_id"].

        Note: MACSUser.get_tool() raises NotImplementedError.
        Framework-specific subclasses in examples should wrap this user
        or override setup_user() to return a user with get_tool() implemented.

        Args:
            agent_data: Agent configuration
            environment: The task environment
            task: Current task with scenario and user profile

        Returns:
            MACSUser instance
        """
        scenario = task.metadata.get("scenario", "")
        user_model_id = self._get_user_model_id(task)
        return MACSUser(
            model=self.get_model_adapter(
                user_model_id,
                register_name="user_simulator",
            ),
            scenario=scenario,
            initial_query=task.query,
        )

    @abstractmethod
    def setup_agents(  # type: ignore[invalid-method-override]
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,
        task: Task,
        user: Optional[User],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create agents for this task. Must be implemented by subclass.

        Args:
            agent_data: Agent configuration with hierarchy spec
            environment: MACSEnvironment with tools
            task: Current task
            user: Optional user simulator

        Returns:
            Tuple of (ordered agent list, agent dict keyed by ID)
        """
        pass

    def setup_evaluators(  # type: ignore[invalid-method-override]
        self,
        environment: MACSEnvironment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
    ) -> Sequence[Evaluator]:
        """Create user-side and system-side evaluators.

        Each evaluator gets its own model adapter for separate tracing.
        Model ID is read from task.evaluation_data["model_id"].
        """
        evaluator_model_id = self._get_evaluator_model_id(task)
        return [
            MACSEvaluator(
                self.get_model_adapter(
                    evaluator_model_id,
                    register_name="evaluator_user_gsr",
                ),
                task,
                gsr_type="user",
            ),
            MACSEvaluator(
                self.get_model_adapter(
                    evaluator_model_id,
                    register_name="evaluator_system_gsr",
                ),
                task,
                gsr_type="system",
            ),
        ]

    def run_agents(  # type: ignore[invalid-method-override]
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: MACSEnvironment,
        query: str = "",
    ) -> Any:
        """Execute agents and return final answer."""
        answers = [agent.run(query) for agent in agents]
        return answers[0] if len(answers) == 1 else answers

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate using both evaluators and aggregate GSR metrics.

        Uses each evaluator's filter_traces() method to extract relevant data,
        then calls the evaluator with the filtered traces.

        Returns AWS paper format:
        - user_gsr, system_gsr, overall_gsr, supervisor_gsr
        - user_partial_gsr, system_partial_gsr, overall_partial_gsr
        - report: Combined assertion judgments
        """
        # Run evaluators - each handles its own trace filtering
        results = []
        for evaluator in evaluators:
            # Use the evaluator's filter_traces method to get the right data
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)

        # Combine results
        user_result = results[0] if results else {"gsr": 0.0, "partial_gsr": 0.0, "report": []}
        system_result = results[1] if len(results) > 1 else {"gsr": 0.0, "partial_gsr": 0.0, "report": []}

        user_report: List[Dict[str, Any]] = user_result.get("report", [])  # type: ignore[assignment]
        system_report: List[Dict[str, Any]] = system_result.get("report", [])  # type: ignore[assignment]
        combined_report = user_report + system_report

        # Compute overall metrics per AWS paper
        overall_gsr = 1.0 if (user_result.get("gsr", 0.0) == 1.0 and system_result.get("gsr", 0.0) == 1.0) else 0.0

        # Supervisor GSR: success if overall passes OR user-side passes
        supervisor_gsr = 1.0 if (overall_gsr == 1.0 or user_result.get("gsr", 0.0) == 1.0) else 0.0

        # Overall partial GSR
        if combined_report:
            total_true = sum(1 for item in combined_report if str(item.get("answer", "")).lower() == "true")
            overall_partial_gsr = total_true / len(combined_report)
        else:
            overall_partial_gsr = 1.0

        return [
            {
                "user_gsr": user_result.get("gsr", 0.0),
                "user_partial_gsr": user_result.get("partial_gsr", 0.0),
                "system_gsr": system_result.get("gsr", 0.0),
                "system_partial_gsr": system_result.get("partial_gsr", 0.0),
                "overall_gsr": overall_gsr,
                "overall_partial_gsr": overall_partial_gsr,
                "supervisor_gsr": supervisor_gsr,
                "report": combined_report,
            }
        ]


# =============================================================================
# Utility Functions
# =============================================================================


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    Infrastructure errors (environment errors, user simulator errors, evaluation errors,
    unknown errors) are excluded from scoring metrics to ensure fair evaluation. Only
    tasks that completed execution (successfully or with agent errors) are included in
    the success rate and mean metric calculations.

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with:
            - total_tasks: Total number of tasks attempted
            - scored_tasks: Tasks included in scoring (excludes infrastructure errors)
            - successful_tasks: Tasks with overall_gsr=1.0
            - success_rate: successful_tasks / scored_tasks
            - mean_metrics: Mean of each metric across scored tasks
            - excluded: Dict with counts of excluded tasks by category
            - status_counts: Dict with counts of each status type
    """
    if not results:
        return {
            "total_tasks": 0,
            "scored_tasks": 0,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "mean_metrics": {},
            "excluded": {},
            "status_counts": {},
        }

    total_tasks = len(results)
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    successful_tasks = 0
    scored_tasks = 0
    status_counts: Dict[str, int] = {}
    excluded_counts: Dict[str, int] = {}

    for res in results:
        status = res.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        # Skip infrastructure failures from scoring (use module-level SCOREABLE_STATUSES)
        if status not in SCOREABLE_STATUSES:
            excluded_counts[status] = excluded_counts.get(status, 0) + 1
            continue

        # Task is included in scoring
        scored_tasks += 1
        evals = res.get("eval") or []
        found_success = False

        for entry in evals:
            for k, v in entry.items():
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1

            if not found_success and entry.get("overall_gsr", 0.0) == 1.0:
                found_success = True

        if found_success:
            successful_tasks += 1

    success_rate = successful_tasks / scored_tasks if scored_tasks > 0 else 0.0
    mean_metrics = {k: metric_sums[k] / metric_counts[k] if metric_counts[k] else 0.0 for k in metric_sums}

    return {
        "total_tasks": total_tasks,
        "scored_tasks": scored_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": success_rate,
        "mean_metrics": mean_metrics,
        "excluded": excluded_counts,
        "status_counts": status_counts,
    }
