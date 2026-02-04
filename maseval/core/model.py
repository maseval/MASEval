"""Core model adapter abstractions for LLM inference.

This module provides the base `ModelAdapter` class that all model adapters must
implement. It defines a consistent interface for interacting with LLMs across
different providers (OpenAI, Anthropic, Google, HuggingFace, LiteLLM, etc.).

See `maseval.interface.inference` for concrete implementations.

Example:
    ```python
    from maseval.interface.inference import LiteLLMModelAdapter

    # Create adapter
    model = LiteLLMModelAdapter(model_id="gpt-4")

    # Simple text generation
    response = model.generate("What is 2+2?")
    print(response)  # "4"

    # Chat with messages
    response = model.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ])
    print(response.content)  # "4"

    # Chat with tools
    response = model.chat(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    )
    if response.tool_calls:
        print(response.tool_calls[0]["function"]["name"])  # "get_weather"
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
import time

from .tracing import TraceableMixin
from .config import ConfigurableMixin
from .history import MessageHistory


@dataclass
class ChatResponse:
    """Response from a chat completion.

    When the model generates a response, it returns either text content,
    tool calls, or both. Use this class to access the response data.

    Attributes:
        content: The text content of the response. May be None if the model
            only returned tool calls.
        tool_calls: List of tool calls the model wants to execute. Each tool
            call is a dict with 'id', 'type', and 'function' keys. The
            'function' contains 'name' and 'arguments' (JSON string).
            None if no tools were called.
        role: The role of the response message. Always "assistant".
        usage: Token usage statistics if available. Dict with keys like
            'input_tokens', 'output_tokens', 'total_tokens'.
        model: The model ID that generated this response, if available.
        stop_reason: Why the model stopped generating. Common values:
            'end_turn', 'tool_use', 'max_tokens', 'stop_sequence'.

    Example:
        ```python
        response = model.chat([{"role": "user", "content": "Hello"}])

        # Text response
        if response.content:
            print(response.content)

        # Tool call response
        if response.tool_calls:
            for call in response.tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                result = execute_tool(name, args)
        ```
    """

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    role: str = "assistant"
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_message(self) -> Dict[str, Any]:
        """Convert this response to an OpenAI-compatible message dict.

        Use this to append the assistant's response to your message history
        before continuing the conversation.

        Returns:
            Dict with 'role', 'content', and optionally 'tool_calls'.

        Example:
            ```python
            messages = [{"role": "user", "content": "Hello"}]
            response = model.chat(messages)

            # Add assistant response to history
            messages.append(response.to_message())

            # Continue conversation
            messages.append({"role": "user", "content": "Tell me more"})
            response = model.chat(messages)
            ```
        """
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class ModelAdapter(ABC, TraceableMixin, ConfigurableMixin):
    """Abstract base class for model adapters.

    ModelAdapter provides a consistent interface for LLM inference across
    different providers. All adapters implement the same methods, so you
    can swap providers without changing your code.

    To use a model adapter:
        1. Create an instance with provider-specific configuration
        2. Call `chat()` for message-based conversations
        3. Call `generate()` for simple text-in/text-out

    The adapter automatically tracks all calls for tracing and evaluation.

    Implementing a custom adapter:
        Subclass ModelAdapter and implement:
        - `model_id` property: Return the model identifier string
        - `_chat_impl()`: The actual chat completion logic

    See maseval.interface.inference for concrete implementations:
        - AnthropicModelAdapter
        - GoogleGenAIModelAdapter
        - HuggingFaceModelAdapter
        - LiteLLMModelAdapter
        - OpenAIModelAdapter

    Seeding:
        Pass a `seed` parameter to enable deterministic generation. This seed
        is passed to the underlying provider API if supported. If a seed is
        provided but the provider doesn't support seeding, the adapter should
        raise `SeedingError` from `maseval.core.seeding`.

        User-provided generation_params["seed"] takes precedence over the
        adapter's seed parameter.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the model adapter with call tracing.

        Args:
            seed: Seed for deterministic generation. Passed to the underlying
                provider API if supported. If the provider doesn't support
                seeding, subclasses should raise SeedingError.
        """
        super().__init__()
        self._seed = seed
        self.logs: List[Dict[str, Any]] = []

    @property
    def seed(self) -> Optional[int]:
        """Seed for deterministic generation, or None if unseeded."""
        return self._seed

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The identifier for the underlying model.

        Returns:
            A string identifying the model (e.g., "gpt-4", "claude-sonnet-4-5",
            "gemini-pro"). Used for tracing and configuration.
        """

    def chat(
        self,
        messages: Union[List[Dict[str, Any]], MessageHistory],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send messages to the model and get a response.

        This is the primary method for interacting with the model. Pass a
        conversation history and receive the model's response.

        Args:
            messages: The conversation history. Either a list of message dicts
                in OpenAI format, or a MessageHistory object. Each message
                has 'role' ('system', 'user', 'assistant', 'tool') and
                'content' keys.
            generation_params: Model parameters like temperature, max_tokens,
                top_p, etc. Provider-specific parameters are also accepted.
            tools: Tool definitions the model can use. Each tool is a dict
                with 'type' (usually 'function') and 'function' containing
                'name', 'description', and 'parameters' (JSON Schema).
            tool_choice: How the model should use tools:
                - "auto": Model decides whether to use tools (default)
                - "none": Model won't use tools
                - "required": Model must use a tool
                - {"type": "function", "function": {"name": "..."}}: Use specific tool
            **kwargs: Additional provider-specific arguments.

        Returns:
            ChatResponse containing the model's response (text and/or tool calls).

        Raises:
            Exception: Provider-specific errors are logged and re-raised.

        Example:
            ```python
            # Simple conversation
            response = model.chat([
                {"role": "user", "content": "Hello!"}
            ])
            print(response.content)

            # With system prompt
            response = model.chat([
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Hello!"}
            ])

            # With tools
            response = model.chat(
                messages=[{"role": "user", "content": "What's 2+2?"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate math expressions",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"]
                        }
                    }
                }]
            )
            ```
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Convert MessageHistory to list if needed
        if isinstance(messages, MessageHistory):
            messages_list = messages.to_openai_format()
        else:
            messages_list = messages

        try:
            result = self._chat_impl(
                messages_list,
                generation_params=generation_params,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            duration = time.time() - start_time

            self.logs.append(
                {
                    "timestamp": timestamp,
                    "message_count": len(messages_list),
                    "response_type": "tool_call" if result.tool_calls else "text",
                    "response_length": len(result.content) if result.content else 0,
                    "tool_calls_count": len(result.tool_calls) if result.tool_calls else 0,
                    "duration_seconds": duration,
                    "status": "success",
                    "generation_params": generation_params or {},
                    "tools_provided": len(tools) if tools else 0,
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.logs.append(
                {
                    "timestamp": timestamp,
                    "message_count": len(messages_list),
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "generation_params": generation_params or {},
                    "tools_provided": len(tools) if tools else 0,
                    "kwargs": {k: str(v) for k, v in kwargs.items()},
                }
            )

            raise

    @abstractmethod
    def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Internal chat implementation to be overridden by subclasses.

        Implement this method to call your provider's API. The base class
        handles tracing, timing, and error logging.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions, if any.
            tool_choice: Tool choice setting, if any.
            **kwargs: Additional provider-specific arguments.

        Returns:
            ChatResponse with the model's output.
        """

    def generate(
        self,
        prompt: str,
        generation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a simple prompt.

        This is a convenience method that wraps the prompt in a user message
        and calls `chat()`. Use this for simple text-in/text-out scenarios.

        For conversations or tool use, use `chat()` directly.

        Args:
            prompt: The input prompt.
            generation_params: Generation parameters (temperature, max_tokens, etc.).
            **kwargs: Additional provider-specific arguments.

        Returns:
            The model's text response.

        Example:
            ```python
            response = model.generate("What is the capital of France?")
            print(response)  # "Paris"
            ```
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, generation_params=generation_params, **kwargs)
        return response.content or ""

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this model adapter.

        Called automatically by Benchmark to collect execution data for
        evaluation. Returns comprehensive statistics about all calls made
        to this adapter.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `model_id` - Model identifier
        - `total_calls` - Number of chat/generate calls
        - `successful_calls` - Number of successful calls
        - `failed_calls` - Number of failed calls
        - `total_duration_seconds` - Total time spent in calls
        - `average_duration_seconds` - Average time per call
        - `logs` - List of individual call records

        Returns:
            Dictionary containing model execution traces.
        """
        total_calls = len(self.logs)
        successful_calls = sum(1 for call in self.logs if call["status"] == "success")
        failed_calls = total_calls - successful_calls
        total_duration = sum(call["duration_seconds"] for call in self.logs)
        avg_duration = total_duration / total_calls if total_calls > 0 else 0

        return {
            **super().gather_traces(),
            "model_id": self.model_id,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "total_duration_seconds": total_duration,
            "average_duration_seconds": avg_duration,
            "logs": self.logs,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this model adapter.

        Called automatically by Benchmark to collect configuration for
        reproducibility. Returns identifying information about this adapter.

        Output fields:

        - `type` - Component class name
        - `gathered_at` - ISO timestamp
        - `model_id` - Model identifier
        - `adapter_type` - The specific adapter class name
        - `seed` - Seed for deterministic generation, or None if unseeded

        Returns:
            Dictionary containing model configuration.
        """
        return {
            **super().gather_config(),
            "model_id": self.model_id,
            "adapter_type": type(self).__name__,
            "seed": self._seed,
        }
