"""Google Generative AI model adapter.

This adapter works with Google's Generative AI SDK (google-genai) for accessing
Gemini models.

Requires google-genai to be installed:
    pip install maseval[google-genai]

Example:
    ```python
    from google import genai
    from maseval.interface.inference import GoogleGenAIModelAdapter

    # Create client
    client = genai.Client(api_key="your-api-key")
    # Or set GOOGLE_API_KEY environment variable

    # Create adapter
    model = GoogleGenAIModelAdapter(
        client=client,
        model_id="gemini-2.0-flash"
    )

    # Simple generation
    response = model.generate("Hello!")

    # Chat with messages
    response = model.chat([
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
    ```
"""

from typing import Any, Optional, Dict, List, Union

from maseval.core.model import ModelAdapter, ChatResponse


class GoogleGenAIModelAdapter(ModelAdapter):
    """Adapter for Google Generative AI (Gemini models).

    Works with Google's Gemini models through the google-genai SDK.
    Pass any model ID supported by the Google GenAI API.

    The adapter converts OpenAI-style messages to Google's format internally,
    so you can use the same message format across all adapters.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        default_generation_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize Google GenAI model adapter.

        Args:
            client: A google.genai.Client instance.
            model_id: The model identifier (e.g., "gemini-2.0-flash").
            default_generation_params: Default parameters for all calls.
                Common parameters: temperature, max_output_tokens, top_p.
            seed: Seed for deterministic generation. Google GenAI supports this.
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
        """Call Google GenAI API.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions for function calling.
            tool_choice: Tool choice setting.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with the model's output.
        """
        from google import genai

        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Add seed if set and not already in params (user params take precedence)
        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed

        # Convert messages to Google format
        system_instruction, contents = self._convert_messages(messages)

        # Build config
        config_params = dict(params)
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        # Convert tools to Google format
        if tools:
            config_params["tools"] = self._convert_tools(tools)

        # Handle tool_choice
        if tool_choice is not None:
            if tool_choice == "none":
                config_params["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
            elif tool_choice == "auto":
                config_params["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
            elif tool_choice == "required":
                config_params["tool_config"] = {"function_calling_config": {"mode": "ANY"}}
            elif isinstance(tool_choice, dict) and "function" in tool_choice:
                config_params["tool_config"] = {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [tool_choice["function"]["name"]],
                    }
                }

        # Build generation config
        generation_config = genai.types.GenerateContentConfig(**config_params) if config_params else None

        # Call API
        response = self._client.models.generate_content(model=self._model_id, contents=contents, config=generation_config)

        return self._parse_response(response)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert OpenAI messages to Google format.

        Google uses 'contents' with 'parts', and separates system instructions.
        Roles are 'user' and 'model' (not 'assistant').

        Args:
            messages: OpenAI-format messages.

        Returns:
            Tuple of (system_instruction, contents).
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                # Handle assistant messages with or without tool calls
                parts = []
                if content:
                    parts.append({"text": content})
                # Convert tool_calls to Google's function_call format
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    import json

                    for tc in tool_calls:
                        if tc.get("type") == "function":
                            func = tc.get("function", {})
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except json.JSONDecodeError:
                                args = {}
                            parts.append({"function_call": {"name": func.get("name", ""), "args": args}})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                # Tool response in Google format
                tool_call_id = msg.get("tool_call_id", "")
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.get("name", tool_call_id),
                                    "response": {"result": content},
                                }
                            }
                        ],
                    }
                )
            else:
                # User message
                contents.append({"role": "user", "parts": [{"text": content}]})

        return system_instruction, contents

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Google format.

        Args:
            tools: OpenAI-format tool definitions.

        Returns:
            Google-format tool definitions.
        """
        google_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                google_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": func.get("name", ""),
                                "description": func.get("description", ""),
                                "parameters": func.get("parameters", {}),
                            }
                        ]
                    }
                )

        return google_tools

    def _parse_response(self, response: Any) -> ChatResponse:
        """Parse Google GenAI response into ChatResponse.

        Args:
            response: The raw response from Google.

        Returns:
            ChatResponse with extracted data.
        """
        # Extract text content
        content = None
        if hasattr(response, "text"):
            content = response.text

        # Extract tool calls (function calls in Google terminology)
        tool_calls = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part.function_call
                        # Convert args to JSON string
                        import json

                        args = dict(fc.args) if fc.args else {}
                        tool_calls.append(
                            {
                                "id": f"call_{fc.name}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        )

        # Extract usage
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", 0),
                "output_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        # Extract stop reason
        stop_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                stop_reason = str(candidate.finish_reason)

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            role="assistant",
            usage=usage,
            model=self._model_id,
            stop_reason=stop_reason,
        )

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this Google GenAI model adapter.

        Returns:
            Dictionary containing model configuration.
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "client_type": type(self._client).__name__,
            }
        )

        return base_config
