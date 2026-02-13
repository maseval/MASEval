from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict

from .callback import AgentCallback
from .history import MessageHistory
from .tracing import TraceableMixin
from .config import ConfigurableMixin


class AgentAdapter(ABC, TraceableMixin, ConfigurableMixin):
    """Wraps an agent from any framework to provide a standard interface.

    This Adapter provides:
    - Unified execution interface via `run()`
    - Callback hooks for monitoring
    - Message history management via getter/setter
    - Framework-agnostic tracing
    """

    def __init__(self, agent_instance: Any, name: str, callbacks: Optional[List[AgentCallback]] = None):
        self.agent = agent_instance
        self.name = name
        self.callbacks = callbacks or []
        self.messages: Optional[MessageHistory] = None
        self.logs: List[Dict[str, Any]] = []

    def run(self, query: str) -> Any:
        """Executes the agent and returns the result."""
        for cb in self.callbacks:
            cb.on_run_start(self)

        result = self._run_agent(query)

        for cb in self.callbacks:
            cb.on_run_end(self, result)

        return result

    @abstractmethod
    def _run_agent(self, query: str) -> Any:
        """Framework-specific agent execution logic.

        Subclasses should:
        1. Execute the agent with the given query
        2. Extract and return the final answer/result from the agent's execution
        3. Store message history internally for tracing (via set_message_history or get_messages)

        The return value should be the agent's final answer or output, NOT the full message trace.
        Message traces are captured automatically through the tracing system via get_messages().

        Args:
            query: The user query/prompt to send to the agent

        Common return patterns:

        - String containing the final answer
        - Dict with structured output
        - Any framework-specific result object

        Returns:
            The agent's final answer/result.

        Example:
            ```python
            def _run_agent(self, query: str) -> str:
                # Run agent (updates internal state)
                self.agent.run(query)

                # Extract final answer from last message or tool call
                messages = self.agent.get_messages()
                final_answer = self._extract_final_answer(messages)

                return final_answer  # Return answer, not full trace
            ```
        """
        pass

    def get_messages(self) -> MessageHistory:
        """Get the current message history as an iterable MessageHistory object.

        The returned MessageHistory can be:
        - Iterated: `for msg in agent.get_messages(): ...`
        - Indexed: `agent.get_messages()[0]`
        - Converted to list: `list(agent.get_messages())` or `agent.get_messages().to_list()`
        - Checked for emptiness: `if agent.get_messages(): ...`

        Returns:
            MessageHistory object (empty if no messages yet)

        Example:
            ```python
            # Iterate directly
            for msg in agent.get_messages():
                print(msg['role'], msg['content'])

            # Convert to list
            messages = agent.get_messages().to_list()
            messages = list(agent.get_messages())

            # Check if empty
            if agent.get_messages():
                print("Agent has messages")
            ```
        """
        return self.messages if self.messages is not None else MessageHistory()

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this agent.

        Collects comprehensive information about the agent's execution including
        message history, callback information, and agent metadata.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `name` - Agent name
        - `agent_type` - Underlying agent framework class name
        - `message_count` - Number of messages in history
        - `messages` - Full message history as list of dicts
        - `callbacks` - List of callback class names attached to this agent

        Returns:
            Dictionary containing agent execution traces.

        How to use:
            This method is automatically called by Benchmark during trace collection.
            Framework-specific adapters can extend this to include additional data:

            ```python
            def gather_traces(self) -> Dict[str, Any]:
                return {
                    **super().gather_traces(),
                    "framework_specific_metric": self.agent.some_metric
                }
            ```
        """
        history = self.get_messages()
        return {
            **super().gather_traces(),
            "name": self.name,
            "agent_type": type(self.agent).__name__,
            "message_count": len(history),
            "messages": history.to_list() if history else [],
            "callbacks": [type(cb).__name__ for cb in self.callbacks],
            "logs": self.logs,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this agent.

        Collects comprehensive configuration information about the agent including
        its name, type, and callback configuration.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `name` - Agent name
        - `agent_type` - Underlying agent framework class name
        - `adapter_type` - The specific adapter class (e.g., `SmolAgentAdapter`)
        - `callbacks` - List of callback class names attached to this agent

        Returns:
            Dictionary containing agent configuration.

        How to use:
            This method is automatically called by Benchmark during config collection.
            Framework-specific adapters can extend this to include additional data:

            ```python
            def gather_config(self) -> Dict[str, Any]:
                return {
                    **super().gather_config(),
                    "framework_specific_setting": self.agent.some_setting
                }
            ```
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "agent_type": type(self.agent).__name__,
            "adapter_type": type(self).__name__,
            "callbacks": [type(cb).__name__ for cb in self.callbacks],
        }

    def __repr__(self):
        return f"AgentAdapter(name={self.name}, agent_type={type(self.agent).__name__})"
