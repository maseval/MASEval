"""OpenAI and OpenAI-compatible model adapter.

This adapter works with the official OpenAI Python SDK and any OpenAI-compatible
API (like Azure OpenAI, local models with OpenAI-compatible servers, etc.).

Requires openai to be installed:
    pip install maseval[openai]

Example:
    ```python
    from openai import OpenAI
    from maseval.interface.inference import OpenAIModelAdapter

    # Standard OpenAI usage
    client = OpenAI()  # Uses OPENAI_API_KEY env var
    model = OpenAIModelAdapter(client=client, model_id="gpt-4")

    # Simple generation
    response = model.generate("Hello!")

    # Chat with messages
    response = model.chat([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ])

    # Chat with tools
    response = model.chat(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {...}
            }
        }]
    )

    # Azure OpenAI
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint="https://your-resource.openai.azure.com",
        api_version="2024-02-15-preview"
    )
    model = OpenAIModelAdapter(client=client, model_id="gpt-4")
    ```
"""

from typing import Any, Optional, Dict, List, Union

from maseval.core.model import ModelAdapter, ChatResponse


class OpenAIModelAdapter(ModelAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs.

    Works with:
        - OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
        - Azure OpenAI
        - Any OpenAI-compatible server (vLLM, LocalAI, etc.)

    The adapter expects an OpenAI client instance. API keys and configuration
    should be set on the client before passing it to the adapter.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize OpenAI model adapter.

        Args:
            client: An OpenAI client instance (openai.OpenAI or openai.AzureOpenAI).
                The client should already be configured with API keys.
            model_id: The model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
            default_generation_params: Default parameters for all calls.
                Common parameters: temperature, max_tokens, top_p.
            seed: Seed for deterministic generation. OpenAI supports this natively.
                Note: Determinism is best-effort, not guaranteed by OpenAI.
        """
        super().__init__(seed=seed)
        self._client = client
        self._model_id = model_id
        self._default_generation_params = default_generation_params or {}

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
        """Call OpenAI chat completions API.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions for function calling.
            tool_choice: Tool choice setting.
            **kwargs: Additional OpenAI parameters.

        Returns:
            ChatResponse with the model's output.
        """
        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Add seed if set and not already in params (user params take precedence)
        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed

        # Add tools if provided
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        # Call OpenAI API
        # Try the modern client interface first
        if hasattr(self._client, "chat") and hasattr(self._client.chat, "completions"):
            response = self._client.chat.completions.create(model=self._model_id, messages=messages, **params)
        else:
            # Fallback for older or custom clients
            response = self._call_legacy_client(messages, params)

        return self._parse_response(response)

    def _call_legacy_client(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Any:
        """Handle older client interfaces or callables.

        Args:
            messages: Messages to send.
            params: Parameters to pass.

        Returns:
            Response from the client.
        """
        # Try common method names
        for method_name in ("create", "complete", "chat", "generate"):
            if hasattr(self._client, method_name):
                method = getattr(self._client, method_name)
                try:
                    return method(model=self._model_id, messages=messages, **params)
                except TypeError:
                    # Try without model parameter
                    try:
                        return method(messages=messages, **params)
                    except TypeError:
                        continue

        # Last resort: try calling directly
        if callable(self._client):
            return self._client(model=self._model_id, messages=messages, **params)

        raise TypeError(
            f"Unable to call client of type {type(self._client).__name__}. Expected an OpenAI client with chat.completions.create() method."
        )

    def _parse_response(self, response: Any) -> ChatResponse:
        """Parse OpenAI response into ChatResponse.

        Args:
            response: The raw response from OpenAI.

        Returns:
            ChatResponse with extracted data.
        """
        # Handle dict responses (from mocks or legacy clients)
        if isinstance(response, dict):
            return self._parse_dict_response(response)

        # Handle modern OpenAI response objects
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

        # Extract usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            role=getattr(message, "role", "assistant"),
            usage=usage,
            model=getattr(response, "model", self._model_id),
            stop_reason=getattr(choice, "finish_reason", None),
        )

    def _parse_dict_response(self, response: Dict[str, Any]) -> ChatResponse:
        """Parse dict response (from mocks or legacy APIs).

        Args:
            response: Dict response in OpenAI format.

        Returns:
            ChatResponse with extracted data.
        """
        if "choices" not in response or not response["choices"]:
            # Simple string response wrapped in dict
            return ChatResponse(content=str(response))

        choice = response["choices"][0]

        # Handle chat-style response
        if "message" in choice:
            message = choice["message"]
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            role = message.get("role", "assistant")
        # Handle completion-style response
        elif "text" in choice:
            content = choice["text"]
            tool_calls = None
            role = "assistant"
        else:
            content = str(choice)
            tool_calls = None
            role = "assistant"

        # Extract usage if present
        usage = None
        if "usage" in response:
            usage = {
                "input_tokens": response["usage"].get("prompt_tokens", 0),
                "output_tokens": response["usage"].get("completion_tokens", 0),
                "total_tokens": response["usage"].get("total_tokens", 0),
            }

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            role=role,
            usage=usage,
            model=response.get("model", self._model_id),
            stop_reason=choice.get("finish_reason"),
        )

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this OpenAI model adapter.

        Returns:
            Dictionary containing model configuration and client settings.
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "client_type": type(self._client).__name__,
            }
        )

        # Extract client configuration
        client_config = {}

        if hasattr(self._client, "timeout"):
            timeout = self._client.timeout
            if hasattr(timeout, "connect"):
                client_config["timeout"] = {
                    "connect": timeout.connect,
                    "read": timeout.read,
                    "write": timeout.write,
                    "pool": timeout.pool,
                }
            else:
                client_config["timeout"] = timeout

        if hasattr(self._client, "max_retries"):
            client_config["max_retries"] = self._client.max_retries

        if client_config:
            base_config["client_config"] = client_config

        return base_config
