from .model import ModelAdapter
from .simulator import UserLLMSimulator, AgenticUserLLMSimulator
from .tracing import TraceableMixin
from .config import ConfigurableMixin
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import time
import inspect
from .history import MessageHistory


class TerminationReason(Enum):
    """Reason why user interaction terminated.

    Attributes:
        NOT_TERMINATED: Interaction is still ongoing.
        MAX_TURNS: Maximum turn count reached.
        USER_TERMINATED: User emitted a stop token (e.g., expressing satisfaction).
        MAX_TURNS_AND_USER_TERMINATED: Both max turns reached and stop token detected.
    """

    NOT_TERMINATED = "not_terminated"
    MAX_TURNS = "max_turns"
    USER_TERMINATED = "user_terminated"
    MAX_TURNS_AND_USER_TERMINATED = "max_turns_and_user_terminated"


class User(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract interface for user interaction during evaluation.

    A user represents the entity that interacts with agents during evaluation.
    This could be an LLM simulating a human, a scripted response sequence,
    a real human, or another agent system.

    Subclasses must implement:

    - `get_initial_query()` - Return the opening message to start the conversation
    - `respond()` - Generate responses to agent messages
    - `is_done()` - Determine when the interaction should end

    The optional `get_tool()` method can be overridden for frameworks that use
    tool-based user interaction (e.g., smolagents, CAMEL).
    """

    @abstractmethod
    def get_initial_query(self) -> str:
        """Return the initial query to start the conversation.

        Returns:
            The opening message from the user to begin the interaction.
        """
        ...

    @abstractmethod
    def respond(self, message: str) -> str:
        """Respond to a message from the agent.

        Args:
            message: The agent's message or question.

        Returns:
            The user's response.
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the user interaction should terminate.

        Returns:
            True if the user is done interacting, False to continue.
        """
        ...

    def get_tool(self) -> Any:
        """Return a framework-compatible tool for agent interaction.

        Some frameworks (smolagents, CAMEL) use a tool-based pattern where
        agents invoke an AskUser tool to interact with the user. Override
        this in subclasses for frameworks that need it.

        Returns:
            Framework-specific tool, or `None` if not applicable.
        """
        return None


class LLMUser(User):
    """User simulated by a language model.

    Uses an LLM to generate realistic user responses based on a user profile
    and scenario description. Maintains conversation history and supports
    multi-turn interaction with configurable termination conditions.

    The user only has access to the conversation history and does not see
    the full environment state, ensuring partial observability.

    Multi-Turn Interaction:
        By default, users support single-turn interaction (max_turns=1). For
        benchmarks that require multiple agent-user exchanges, set max_turns > 1.

    Early Stopping:
        For benchmarks where termination depends on user satisfaction rather than
        a fixed turn count, configure `stop_tokens`. When the user's response contains
        any of these tokens, `is_done()` returns True. The MACS benchmark uses
        `"</stop>"` to signal satisfaction.

    Attributes:
        name: User identifier.
        model: Language model for generating responses.
        user_profile: Dictionary describing the user's persona and preferences.
        scenario: Description of the task the user is trying to accomplish.
        simulator: The LLM simulator instance generating responses.
        messages: Conversation history between user and agent.
        max_turns: Maximum number of user response turns.
        stop_tokens: Tokens that trigger early stopping when detected (empty list if disabled).
        early_stopping_condition: Description of when to emit a stop token, or None.
    """

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        initial_query: Optional[str] = None,
        template: Optional[str] = None,
        max_try: int = 3,
        max_turns: int = 1,
        stop_tokens: Optional[List[str]] = None,
        early_stopping_condition: Optional[str] = None,
    ):
        """Initialize the LLMUser.

        Args:
            name: The name of the user.
            model: The language model to be used for generating responses.
            user_profile: A dictionary describing the user's persona,
                preferences, and other relevant information.
            scenario: A description of the situation or task the user is trying to
                accomplish.
            initial_query: A pre-set query to start the conversation. If provided,
                it becomes the first user message. If None, call get_initial_query()
                to generate one from the model based on the user profile and scenario.
                Defaults to None.
            template: A custom prompt template for the user simulator. Defaults to None.
            max_try: The maximum number of attempts for the simulator to
                generate a valid response. Defaults to 3.
            max_turns: Maximum number of user messages in the conversation.
                Each user message counts as one turn, including the initial_query.
                Use max_turns=1 for single-turn benchmarks, or higher values for
                multi-turn interaction. Defaults to 1.
            stop_tokens: List of tokens that signal user satisfaction, enabling
                early termination. When the user's LLM-generated response contains
                any of these tokens, is_done() returns True regardless of remaining
                turns. The matched token is stripped from the response.
                Defaults to None (early stopping disabled).
            early_stopping_condition: A description of when the user should stop
                the conversation (e.g., "all goals have been accomplished").
                Used with stop_tokens to instruct the LLM when to emit a stop token.
                Must be provided if stop_tokens is set. Defaults to None.

        Raises:
            ValueError: If stop_tokens is set but early_stopping_condition is not provided.
        """
        # Normalize stop_tokens to list
        stop_tokens = stop_tokens or []

        # Validate early stopping configuration
        has_stop_tokens = len(stop_tokens) > 0
        if has_stop_tokens != (early_stopping_condition is not None):
            raise ValueError(
                "stop_tokens and early_stopping_condition must both be set or both be None. "
                f"Got stop_tokens={stop_tokens!r}, early_stopping_condition={early_stopping_condition!r}"
            )

        self.name = name
        self.model = model
        self.user_profile = user_profile
        self.scenario = scenario

        # Pass first stop token to simulator (it only uses one for prompt instructions)
        primary_stop_token = stop_tokens[0] if stop_tokens else None
        self.simulator = UserLLMSimulator(
            model=self.model,
            user_profile=self.user_profile,
            scenario=self.scenario,
            template=template,
            max_try=max_try,
            stop_token=primary_stop_token,
            early_stopping_condition=early_stopping_condition,
        )
        # Initialize message history - empty or with initial query
        if initial_query is not None:
            self.messages = MessageHistory([{"role": "user", "content": initial_query}])
            self._initial_turn_count = 1  # Initial query counts as first turn
        else:
            self.messages = MessageHistory()
            self._initial_turn_count = 0
        self.logs: List[Dict[str, Any]] = []

        # Multi-turn configuration
        self.max_turns = max_turns
        self.stop_tokens = stop_tokens
        self.early_stopping_condition = early_stopping_condition
        self._turn_count = self._initial_turn_count
        self._stopped = False
        self._stop_reason: Optional[str] = None  # Which token triggered the stop

    def respond(self, message: str) -> str:
        """Respond to a message from the agent using LLM simulation.

        This method appends the agent's message to the conversation history,
        generates a response using the LLM simulator, appends the response
        to the history, and returns it.

        If the user is already done (max_turns reached or stop_token detected),
        returns an empty string without making an LLM call. If a stop_token is
        detected in the response, triggers early stopping.

        Args:
            message: The message from the agent to which the user should respond.

        Returns:
            The user's response, or empty string if done.
        """
        # Check if already done - saves LLM call
        if self.is_done():
            return ""

        # Record the assistant prompt and ask simulator. MessageHistory is iterable
        # and can be converted to a list for the simulator.
        self.messages.add_message("assistant", message)
        start_time = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": message,
            "status": "success",
        }

        try:
            response = self.simulator(conversation_history=self.messages.to_list())
        except Exception as exc:  # pragma: no cover - propagate but log failure
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            log_entry["error_type"] = type(exc).__name__
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["response_preview"] = self._summarize_response(response)
        self.logs.append(log_entry)

        # Check for stop token and clean response if needed
        _, clean_response = self._check_stop_token(response)

        self.messages.add_message("user", clean_response)
        self.increment_turn()
        return clean_response

    def get_initial_query(self) -> str:
        """Get the initial query for the conversation.

        If an initial_query was provided at construction, returns it.
        Otherwise, generates one using the LLM simulator based on the user's
        profile and scenario.

        This method:
        - Returns the existing initial query if one was provided
        - Or calls the LLM simulator to generate one
        - Ensures the query is in the message history
        - Counts the initial query as the first turn

        Returns:
            The initial query (either pre-set or LLM-generated).

        Raises:
            RuntimeError: If called after conversation has progressed beyond
                the initial message.
        """
        # If we already have an initial query in messages, return it
        if len(self.messages) > 0:
            first_message = self.messages[0]
            if first_message.get("role") == "user":
                return first_message.get("content", "")
            raise RuntimeError("Cannot get initial query: conversation has progressed. Use respond() for subsequent turns.")

        # Generate initial query via LLM
        start_time = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "question": "[initial_query]",
            "status": "success",
        }

        try:
            response = self.simulator(conversation_history=[])
        except Exception as exc:  # pragma: no cover
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            log_entry["error_type"] = type(exc).__name__
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["response_preview"] = self._summarize_response(response)
        self.logs.append(log_entry)

        # Check for stop token (user might be immediately satisfied with scenario)
        _, clean_response = self._check_stop_token(response)

        # Add as initial user message and count as first turn
        self.messages.add_message("user", clean_response)
        self.increment_turn()
        return clean_response

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this user.

        Output fields:

        - `name` - User identifier
        - `profile` - User profile data
        - `message_count` - Number of messages in history
        - `messages` - Full conversation history
        - `logs` - Execution logs with timing
        - `termination_reason` - Why interaction ended (see `TerminationReason`)
        - `stop_reason` - Which stop token triggered termination, if any
        - `max_turns` - Maximum allowed turns
        - `turns_used` - Actual turns used
        - `stopped_by_user` - Whether user emitted a stop token

        Returns:
            Dictionary containing user state and interaction data.
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "profile": self.user_profile,
            "message_count": len(self.messages),
            "messages": self.messages.to_list(),
            "logs": self.logs,
            "termination_reason": self.termination_reason.value,
            "stop_reason": self._stop_reason,
            "max_turns": self.max_turns,
            "turns_used": self._turn_count,
            "stopped_by_user": self._stopped,
        }

    @staticmethod
    def _summarize_response(response: str) -> str:
        return response[:2000]

    @property
    def termination_reason(self) -> TerminationReason:
        """Get the reason why the user interaction terminated.

        Returns:
            Why `is_done()` returns True, or `NOT_TERMINATED` if still ongoing.
        """
        max_turns_reached = self._turn_count >= self.max_turns
        user_terminated = self._stopped

        if max_turns_reached and user_terminated:
            return TerminationReason.MAX_TURNS_AND_USER_TERMINATED
        elif max_turns_reached:
            return TerminationReason.MAX_TURNS
        elif user_terminated:
            return TerminationReason.USER_TERMINATED
        else:
            return TerminationReason.NOT_TERMINATED

    def is_done(self) -> bool:
        """Check if the user interaction should end.

        Checks:
        1. If max_turns has been reached
        2. If the user previously indicated termination (via stop_token)

        Subclasses can override to add custom termination logic (e.g., LLM-based
        satisfaction checks) by calling super().is_done() first.

        Returns:
            True if the user is done interacting, False to continue.
        """
        return self.termination_reason != TerminationReason.NOT_TERMINATED

    def _check_stop_token(self, response: str) -> tuple[bool, str]:
        """Check if response contains any stop token and clean it up.

        Args:
            response: The user's response to check.

        Returns:
            Tuple of (should_stop, cleaned_response).
        """
        response_lower = response.lower()
        for token in self.stop_tokens:
            if token.lower() in response_lower:
                self._stopped = True
                self._stop_reason = token  # Track which token triggered the stop
                # Remove the stop token from the response
                cleaned = response.replace(token, "").replace(token.lower(), "").strip()
                return True, cleaned if cleaned else "Thank you, that's all I needed!"
        return False, response

    def increment_turn(self) -> None:
        """Increment the turn counter.

        Call this after recording a user response in the message history.
        """
        self._turn_count += 1

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this user.

        Output fields:

        - `name` - User identifier
        - `profile` - User profile data
        - `scenario` - Task scenario description
        - `max_turns` - Maximum interaction turns
        - `stop_tokens` - Early stopping tokens (empty list if disabled)

        Returns:
            Dictionary containing user configuration.
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "profile": self.user_profile,
            "scenario": self.scenario,
            "max_turns": self.max_turns,
            "stop_tokens": self.stop_tokens,
        }


class AgenticLLMUser(LLMUser):
    """LLM-simulated user with access to tools.

    Extends LLMUser with the ability to use tools (e.g., check order status,
    lookup information) during the conversation. Uses a ReAct-style loop
    to iteratively call tools and generate responses.
    """

    def __init__(
        self,
        name: str,
        model: ModelAdapter,
        user_profile: Dict[str, Any],
        scenario: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_internal_steps: int = 5,
        **kwargs,
    ):
        """Initialize AgenticLLMUser.

        Args:
            name: The name of the user.
            model: The language model to be used for generating responses.
            user_profile: A dictionary describing the user's persona.
            scenario: A description of the task the user is trying to accomplish.
            tools: Dictionary of tools available to the user.
            max_internal_steps: Maximum number of tool execution loops per turn.
            **kwargs: Arguments passed to LLMUser.__init__
        """
        self.tools = tools or {}
        self.max_internal_steps = max_internal_steps

        # Initialize the parent, but we will replace the simulator
        super().__init__(name=name, model=model, user_profile=user_profile, scenario=scenario, **kwargs)

        # Generate tool definitions
        tool_definitions = self._generate_tool_definitions() if self.tools else None

        # Replace the standard simulator with AgenticUserLLMSimulator
        # Use first stop token for simulator (for prompt instructions)
        primary_stop_token = self.stop_tokens[0] if self.stop_tokens else None
        self.simulator = AgenticUserLLMSimulator(
            model=self.model,
            user_profile=self.user_profile,
            scenario=self.scenario,
            template=kwargs.get("template"),
            max_try=kwargs.get("max_try", 3),
            stop_token=primary_stop_token,
            early_stopping_condition=kwargs.get("early_stopping_condition"),
            tools=tool_definitions,
        )

    def _generate_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generate tool definitions from the tools dictionary."""
        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = func.__doc__ or ""

            inputs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                param_type = "string"  # Default
                if param.annotation is not inspect.Parameter.empty:
                    if param.annotation is int:
                        param_type = "integer"
                    elif param.annotation is float:
                        param_type = "number"
                    elif param.annotation is bool:
                        param_type = "boolean"

                inputs[param_name] = {"type": param_type, "description": f"Parameter {param_name}"}

            definitions.append({"name": name, "description": doc.strip(), "inputs": inputs})
        return definitions

    def respond(self, message: str) -> str:
        """Respond to a message, potentially executing tools in a loop.

        Uses a ReAct-style loop where the LLM can call tools and reason
        about the results before generating a final response.

        Args:
            message: The message from the agent to respond to.

        Returns:
            The user's final response after any tool execution.
        """
        if self.is_done():
            return ""

        # Start with the current shared history
        self.messages.add_message("assistant", message)

        # Internal history for the ReAct loop (scratchpad)
        # We start with a copy of the shared conversation history
        internal_history = list(self.messages.to_list())

        start_time = time.time()
        log_entry: Dict[str, Any] = {"timestamp": datetime.now().isoformat(), "question": message, "status": "success", "internal_steps": []}

        final_response = ""
        steps = 0

        try:
            while steps < self.max_internal_steps:
                steps += 1

                # Call simulator
                # Note: AgenticUserLLMSimulator returns (text, tool_calls)
                text, tool_calls = self.simulator(conversation_history=internal_history)

                # Log the step
                step_log = {"step": steps, "thought": text, "tool_calls": tool_calls}

                if not tool_calls:
                    # No tools called, this is the final response
                    final_response = text
                    log_entry["internal_steps"].append(step_log)
                    break

                # Tools called - execute them
                # Append the thought to internal history
                # We use 'user' role for thoughts as it's the user speaking to themselves/environment
                internal_history.append({"role": "user", "content": text})

                tool_outputs = []
                for call in tool_calls:
                    name = call.get("name")  # type: ignore[union-attr]
                    args = call.get("arguments", {})  # type: ignore[union-attr]
                    result_str = ""

                    if name in self.tools:
                        try:
                            result = self.tools[name](**args)
                            result_str = str(result)
                            status = "success"
                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                            status = "error"
                    else:
                        result_str = f"Error: Tool '{name}' not found"
                        status = "error"

                    tool_outputs.append({"name": name, "arguments": args, "result": result_str, "status": status})

                    # Append observation to internal history
                    # Format: Tool Output [name]: result
                    internal_history.append(
                        {
                            "role": "user",  # Using user role for simplicity, or we could use 'system'
                            "content": f"Tool Output [{name}]: {result_str}",
                        }
                    )

                step_log["tool_outputs"] = tool_outputs
                log_entry["internal_steps"].append(step_log)

            if not final_response and steps >= self.max_internal_steps:
                # Forced termination of loop
                final_response = "I need to stop checking things now."
                log_entry["status"] = "max_internal_steps_reached"

        except Exception as exc:
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            self.logs.append(log_entry)
            raise

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["final_response"] = final_response[:200]
        self.logs.append(log_entry)

        # Clean stop token from final response
        _, clean_response = self._check_stop_token(final_response)

        # Update the shared history with the final text only
        self.messages.add_message("user", clean_response)
        self.increment_turn()

        return clean_response
