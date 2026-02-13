"""Anthropic model adapter.

This adapter works with the official Anthropic Python SDK for accessing
Claude models directly.

Requires anthropic to be installed:
    pip install maseval[anthropic]

Example:
    ```python
    from anthropic import Anthropic
    from maseval.interface.inference import AnthropicModelAdapter

    # Create client (uses ANTHROPIC_API_KEY env var)
    client = Anthropic()

    # Create adapter
    model = AnthropicModelAdapter(
        client=client,
        model_id="claude-sonnet-4-5-20250514"
    )

    # Simple generation
    response = model.generate("Hello!")

    # Chat with messages
    response = model.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ])

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
    ```
"""

import json
from typing import Any, Optional, Dict, List, Union

from maseval.core.model import ModelAdapter, ChatResponse
from maseval.core.seeding import SeedingError


class AnthropicModelAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models.

    Works with Claude models through the official Anthropic Python SDK.
    Pass any model ID supported by the Anthropic API.

    The adapter accepts OpenAI-style messages and converts them to Anthropic's
    format internally. Key differences handled automatically:

    - System messages are passed separately (not in messages array)
    - Tool definitions are converted to Anthropic format
    - Tool responses are converted to tool_result content blocks
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
    ):
        """Initialize Anthropic model adapter.

        Args:
            client: An anthropic.Anthropic client instance.
            model_id: The model identifier (e.g., "claude-sonnet-4-5-20250514").
            default_generation_params: Default parameters for all calls.
                Common parameters: temperature, top_p, top_k.
            max_tokens: Maximum tokens to generate. Anthropic requires this
                parameter. Default is 4096.
            seed: Seed for deterministic generation. Note: Anthropic does NOT
                support seeding. Providing a seed will raise SeedingError.

        Raises:
            SeedingError: If seed is provided (Anthropic doesn't support seeding).
        """
        if seed is not None:
            raise SeedingError(
                f"Anthropic models do not support seeding. "
                f"Model '{model_id}' cannot use seed={seed}. "
                f"Remove the seed parameter or use a provider that supports seeding."
            )
        super().__init__(seed=seed)
        self._client = client
        self._model_id = model_id
        self._default_generation_params = default_generation_params or {}
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Call Anthropic Messages API.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions for function calling (OpenAI format).
            tool_choice: Tool choice setting.
            **kwargs: Additional Anthropic parameters.

        Returns:
            ChatResponse with the model's output.
        """
        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Extract and set max_tokens
        max_tokens = params.pop("max_tokens", self._max_tokens)

        # Convert messages (extract system, convert tool responses)
        system_prompt, converted_messages = self._convert_messages(messages)

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = self._convert_tools(tools)

        # Handle tool_choice
        anthropic_tool_choice = None
        if tool_choice is not None:
            anthropic_tool_choice = self._convert_tool_choice(tool_choice)

        # Build request
        request_params = {
            "model": self._model_id,
            "max_tokens": max_tokens,
            "messages": converted_messages,
            **params,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if anthropic_tools:
            request_params["tools"] = anthropic_tools

        if anthropic_tool_choice:
            request_params["tool_choice"] = anthropic_tool_choice

        # Call API
        response = self._client.messages.create(**request_params)

        return self._parse_response(response)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert OpenAI messages to Anthropic format.

        Anthropic separates system messages and uses different format for
        tool responses.

        Args:
            messages: OpenAI-format messages.

        Returns:
            Tuple of (system_prompt, converted_messages).
        """
        system_prompt = None
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Anthropic takes system as separate parameter
                system_prompt = content

            elif role == "tool":
                # Convert to Anthropic tool_result format
                # Tool results in Anthropic are user messages with tool_result content
                tool_call_id = msg.get("tool_call_id", "")
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": content,
                            }
                        ],
                    }
                )

            elif role == "assistant":
                # Check if this message has tool_calls (from previous response)
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Convert to Anthropic format with tool_use content blocks
                    content_blocks = []

                    # Add text content if present
                    if msg.get("content"):
                        content_blocks.append({"type": "text", "text": msg["content"]})

                    # Add tool use blocks
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}

                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "input": args,
                            }
                        )

                    converted.append({"role": "assistant", "content": content_blocks})
                else:
                    # Simple text message
                    converted.append({"role": "assistant", "content": content})

            else:
                # User message
                converted.append({"role": "user", "content": content})

        return system_prompt, converted

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format.

        Args:
            tools: OpenAI-format tool definitions.

        Returns:
            Anthropic-format tool definitions.
        """
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                    }
                )

        return anthropic_tools

    def _convert_tool_choice(self, tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OpenAI tool_choice to Anthropic format.

        Args:
            tool_choice: OpenAI-format tool choice.

        Returns:
            Anthropic-format tool choice.
        """
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            # Anthropic doesn't have a direct "none" - we just don't pass tools
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            return {"type": "tool", "name": tool_choice["function"]["name"]}
        else:
            return {"type": "auto"}

    def _parse_response(self, response: Any) -> ChatResponse:
        """Parse Anthropic response into ChatResponse.

        Args:
            response: The raw response from Anthropic.

        Returns:
            ChatResponse with extracted data.
        """
        # Extract content (may be text and/or tool_use blocks)
        content = None
        tool_calls = None

        if hasattr(response, "content") and response.content:
            text_parts = []
            tool_use_parts = []

            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_use_parts.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )

            if text_parts:
                content = "".join(text_parts)

            if tool_use_parts:
                tool_calls = tool_use_parts

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": (getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)),
            }

        # Extract stop reason
        stop_reason = None
        if hasattr(response, "stop_reason"):
            stop_reason = response.stop_reason

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            role="assistant",
            usage=usage,
            model=getattr(response, "model", self._model_id),
            stop_reason=stop_reason,
        )

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this Anthropic model adapter.

        Returns:
            Dictionary containing model configuration.
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "max_tokens": self._max_tokens,
                "client_type": type(self._client).__name__,
            }
        )

        return base_config
