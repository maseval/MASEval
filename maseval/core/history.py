from typing import List, Any, Optional, Union, Dict, Literal, Iterator, overload
from datetime import datetime
import uuid


ContentType = Literal["text", "image_url", "image_file", "file", "audio"]
RoleType = Literal["user", "assistant", "system", "tool"]


class MessageHistory:
    """A message history class for storing conversation messages with multi-modal support.

    Behaves like a list - can be iterated, indexed, and checked for length.
    Use list() or .to_list() to get the raw list of messages.

    This class supports:
    - Simple text messages
    - Multi-modal content (images, files, audio)
    - Tool calls and responses
    - Rich metadata and timestamps

    The message format is OpenAI-compatible for maximum interoperability.

    Example:
        ```python
        # Simple text message
        history = MessageHistory()
        history.add_message("user", "Hello!")

        # Iterate directly
        for msg in history:
            print(msg['role'], msg['content'])

        # Convert to list
        msg_list = history.to_list()
        msg_list = list(history)  # Also works

        # Check if empty
        if history:
            print("Has messages")

        # Access by index
        first = history[0]
        last = history[-1]

        # Multi-modal message
        history.add_message(
            "user",
            [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://..."}}
            ]
        )

        # Tool call message
        history.add_tool_call(
            tool_calls=[{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}]
        )

        # Tool response message
        history.add_tool_response(
            tool_call_id="call_123",
            content="Temperature: 72°F",
            name="get_weather"
        )
        ```
    """

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None):
        """Initialize message history.

        Args:
            messages: List of message dictionaries in OpenAI format
        """
        self._messages: List[Dict[str, Any]] = messages or []

    # Make it iterable
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over messages."""
        return iter(self._messages)

    # Make it indexable
    @overload
    def __getitem__(self, index: int) -> Dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> List[Dict[str, Any]]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Access messages by index or slice."""
        return self._messages[index]

    # Make len() work
    def __len__(self) -> int:
        """Get number of messages."""
        return len(self._messages)

    # Make it behave like a list in boolean contexts
    def __bool__(self) -> bool:
        """Check if history has messages."""
        return len(self._messages) > 0

    def __repr__(self) -> str:
        """String representation."""
        return f"MessageHistory({len(self._messages)} messages)"

    def to_list(self) -> List[Dict[str, Any]]:
        """Get the raw list of messages.

        Returns:
            List of message dictionaries in OpenAI format
        """
        return self._messages

    def add_message(
        self,
        role: RoleType,
        content: Union[str, List[Dict[str, Any]]],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a simple message to the history.

        Args:
            role: The role of the message sender ("user", "assistant", "system", "tool")
            content: The message content (string or list of content parts for multi-modal)
            name: Optional name for the message sender (useful for multi-agent scenarios)
            metadata: Optional metadata dictionary
            timestamp: Optional ISO format timestamp (auto-generated if not provided)
        """
        message: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if name is not None:
            message["name"] = name
        if metadata is not None:
            message["metadata"] = metadata
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        message["timestamp"] = timestamp

        self._messages.append(message)

    def add_tool_call(
        self,
        tool_calls: List[Dict[str, Any]],
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add an assistant message with tool calls.

        Args:
            tool_calls: List of tool call dictionaries. Each should have:
                - id: Unique identifier for the tool call
                - type: Usually "function"
                - function: Dict with "name" and "arguments" (JSON string)
            content: Optional text content accompanying the tool calls
            metadata: Optional metadata dictionary
            timestamp: Optional ISO format timestamp (auto-generated if not provided)
        """
        message: Dict[str, Any] = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if content is not None:
            message["content"] = content
        if metadata is not None:
            message["metadata"] = metadata
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        message["timestamp"] = timestamp

        self._messages.append(message)

    def add_tool_response(
        self,
        tool_call_id: str,
        content: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a tool response message.

        Args:
            tool_call_id: The ID of the tool call this is responding to
            content: The tool's output/response
            name: Optional name of the tool
            metadata: Optional metadata dictionary
            timestamp: Optional ISO format timestamp (auto-generated if not provided)
        """
        message: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        if name is not None:
            message["name"] = name
        if metadata is not None:
            message["metadata"] = metadata
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        message["timestamp"] = timestamp

        self._messages.append(message)

    def clear(self) -> None:
        """Clear all messages from the history."""
        self._messages = []

    def filter_by_role(self, role: RoleType) -> List[Dict[str, Any]]:
        """Get all messages with a specific role.

        Args:
            role: The role to filter by

        Returns:
            List of messages with the specified role
        """
        return [m for m in self._messages if m.get("role") == role]

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the most recent message.

        Returns:
            The last message or None if history is empty
        """
        return self._messages[-1] if self._messages else None

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI API format (strips metadata/timestamps).

        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        for m in self._messages:
            # Only include fields that OpenAI API expects
            msg = {"role": m["role"]}
            if "content" in m:
                msg["content"] = m["content"]
            if "name" in m:
                msg["name"] = m["name"]
            if "tool_calls" in m:
                msg["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                msg["tool_call_id"] = m["tool_call_id"]
            openai_messages.append(msg)
        return openai_messages


class ToolInvocationHistory:
    """A small history container for tool / invocation events.

    Tool calls tend to be structured (inputs, outputs, timestamps). This class
    stores invocation records separately from conversational messages.
    """

    def __init__(self, records: Optional[List[dict]] = None):
        self.logs: List[dict] = records or []

    def add_invocation(self, inputs: Any, outputs: Any, status: str, timestamp: Optional[str] = None, meta: Optional[dict] = None) -> None:
        self.logs.append(
            {
                "id": str(uuid.uuid4()),
                "inputs": inputs,
                "outputs": outputs,
                "status": status,
                "timestamp": timestamp or datetime.now().isoformat(),
                "meta": meta or {},
            }
        )

    def __len__(self) -> int:
        """Get number of invocations."""
        return len(self.logs)

    def to_list(self) -> List[dict]:
        return self.logs
