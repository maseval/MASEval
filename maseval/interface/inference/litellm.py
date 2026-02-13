"""LiteLLM model adapter.

LiteLLM provides a unified interface for 100+ LLM APIs using OpenAI-compatible
syntax. This adapter wraps LiteLLM to provide consistent behavior within MASEval.

Requires litellm to be installed:
    pip install maseval[litellm]

Example:
    ```python
    from maseval.interface.inference import LiteLLMModelAdapter

    # OpenAI models
    model = LiteLLMModelAdapter(model_id="gpt-4")

    # Anthropic models
    model = LiteLLMModelAdapter(model_id="claude-3-opus-20240229")

    # Azure OpenAI
    model = LiteLLMModelAdapter(
        model_id="azure/gpt-4",
        api_base="https://your-resource.openai.azure.com"
    )

    # AWS Bedrock
    model = LiteLLMModelAdapter(model_id="bedrock/anthropic.claude-v2")

    # Simple generation
    response = model.generate("Hello!")

    # Chat with messages
    response = model.chat([
        {"role": "user", "content": "Hello!"}
    ])

    # Chat with tools
    response = model.chat(
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[{"type": "function", "function": {...}}]
    )
    ```
"""

from typing import Any, Optional, Dict, List, Union

from maseval.core.model import ModelAdapter, ChatResponse


class LiteLLMModelAdapter(ModelAdapter):
    """Adapter for LiteLLM unified interface.

    LiteLLM provides a consistent API for calling multiple LLM providers
    (OpenAI, Anthropic, Cohere, Azure, AWS Bedrock, Google, etc.) using
    OpenAI-compatible syntax.

    For supported providers see https://docs.litellm.ai/docs/providers.

    API keys are read from environment variables by default:
        - OPENAI_API_KEY for OpenAI
        - ANTHROPIC_API_KEY for Anthropic
        - etc.

    Or pass api_key directly to the constructor.
    """

    def __init__(
        self,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize LiteLLM model adapter.

        Args:
            model_id: The model identifier in LiteLLM format. Examples:
                - "gpt-4" (OpenAI)
                - "claude-3-opus-20240229" (Anthropic)
                - "azure/gpt-4" (Azure OpenAI)
                - "bedrock/anthropic.claude-v2" (AWS Bedrock)
                See https://docs.litellm.ai/docs/providers for full list.
            default_generation_params: Default parameters for all calls.
                Common parameters: temperature, max_tokens, top_p.
            api_key: API key for the provider. If not provided, LiteLLM
                reads from environment variables.
            api_base: Custom API base URL for self-hosted or Azure endpoints.
            seed: Seed for deterministic generation. LiteLLM passes this to
                the underlying provider. Note: Not all providers support seeding.
        """
        super().__init__(seed=seed)
        self._model_id = model_id
        self._default_generation_params = default_generation_params or {}
        self._api_key = api_key
        self._api_base = api_base

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
        """Call LiteLLM completion API.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions for function calling.
            tool_choice: Tool choice setting.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            ChatResponse with the model's output.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError("LiteLLM is not installed. Install with: pip install maseval[litellm]") from e

        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Add seed if set and not already in params (user params take precedence)
        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed

        # Add API credentials if provided
        if self._api_key:
            params["api_key"] = self._api_key
        if self._api_base:
            params["api_base"] = self._api_base

        # Add tools if provided
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice

        # Call LiteLLM
        response = litellm.completion(model=self._model_id, messages=messages, **params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        # Build tool_calls list if present
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

        # Build usage dict if present
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
            role=message.role if hasattr(message, "role") else "assistant",
            usage=usage,
            model=getattr(response, "model", self._model_id),
            stop_reason=getattr(choice, "finish_reason", None),
        )

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this LiteLLM model adapter.

        Returns:
            Dictionary containing model configuration and LiteLLM settings.
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
            }
        )

        # Extract LiteLLM global configuration
        try:
            import litellm

            litellm_config = {}

            if hasattr(litellm, "num_retries"):
                litellm_config["num_retries"] = litellm.num_retries

            if hasattr(litellm, "drop_params"):
                litellm_config["drop_params"] = litellm.drop_params

            if hasattr(litellm, "verbose"):
                litellm_config["verbose"] = litellm.verbose

            if litellm_config:
                base_config["litellm_global_config"] = litellm_config

        except ImportError:
            pass

        return base_config
