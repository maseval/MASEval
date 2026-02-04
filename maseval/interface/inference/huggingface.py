"""HuggingFace model adapter.

This adapter works with HuggingFace transformers pipelines and models.
It supports both simple callable models and full pipeline objects.

Requires transformers to be installed:
    pip install maseval[transformers]

Example:
    ```python
    from transformers import pipeline
    from maseval.interface.inference import HuggingFaceModelAdapter

    # Using a pipeline
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
    model = HuggingFaceModelAdapter(model=pipe, model_id="llama-3.1-8b")

    # Simple generation
    response = model.generate("Hello!")

    # Chat with messages (uses chat template if available)
    response = model.chat([
        {"role": "user", "content": "Hello!"}
    ])
    ```

Note on tool calling:
    HuggingFace models have varying support for tool calling. This adapter
    will raise an exception if tools are passed but the model's chat template
    does not support them. Use LiteLLMModelAdapter for more reliable tool
    calling with a wider range of models.
"""

from typing import Any, Optional, Dict, List, Callable, Union

from maseval.core.model import ModelAdapter, ChatResponse


class ToolCallingNotSupportedError(Exception):
    """Raised when tool calling is requested but not supported by the model."""

    pass


class HuggingFaceModelAdapter(ModelAdapter):
    """Adapter for HuggingFace transformers models and pipelines.

    Works with:
        - transformers.pipeline() objects
        - Any callable that accepts a prompt and returns text

    For chat functionality, the adapter uses the tokenizer's chat template
    if available. This provides proper formatting for instruction-tuned models.

    Tool calling support:
        Tool calling is only supported if the model's chat template explicitly
        supports it. If you pass tools and the model doesn't support them,
        a ToolCallingNotSupportedError is raised. For reliable tool calling,
        consider using LiteLLMModelAdapter instead.
    """

    def __init__(
        self,
        model: Callable[[str], str],
        model_id: Optional[str] = None,
        default_generation_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize HuggingFace model adapter.

        Args:
            model: A callable that generates text. Can be:
                - A transformers pipeline (e.g., pipeline("text-generation", ...))
                - Any callable that takes a prompt string and returns text
            model_id: Identifier for the model. If not provided, attempts to
                extract from the model's name_or_path attribute.
            default_generation_params: Default parameters for all calls.
                Common parameters: max_new_tokens, temperature, top_p, do_sample.
            seed: Seed for deterministic generation. Sets the random seed before
                each generation call using transformers.set_seed().
        """
        super().__init__(seed=seed)
        self._model = model
        self._model_id = model_id or getattr(model, "name_or_path", "huggingface:unknown")
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
        """Generate response using HuggingFace model.

        Args:
            messages: List of message dicts in OpenAI format.
            generation_params: Generation parameters (temperature, etc.).
            tools: Tool definitions. Raises ToolCallingNotSupportedError if
                provided but not supported by the model's chat template.
            tool_choice: Tool choice setting (ignored if tools not supported).
            **kwargs: Additional parameters passed to the model.

        Returns:
            ChatResponse with the model's output.

        Raises:
            ToolCallingNotSupportedError: If tools are provided but the model
                doesn't support tool calling.
        """
        # Merge parameters
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        # Try to use chat template if available
        tokenizer = self._get_tokenizer()

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return self._chat_with_template(messages, params, tools, tool_choice, tokenizer)
        else:
            # Fallback: convert messages to simple prompt
            if tools:
                raise ToolCallingNotSupportedError(
                    f"Model {self._model_id} does not have a chat template that supports tools. "
                    "Tool calling requires a model with an appropriate chat template. "
                    "Consider using LiteLLMModelAdapter for reliable tool calling."
                )
            return self._chat_without_template(messages, params)

    def _get_tokenizer(self) -> Any:
        """Get the tokenizer from the model/pipeline if available.

        Returns:
            The tokenizer, or None if not available.
        """
        # Pipeline objects have a tokenizer attribute
        if hasattr(self._model, "tokenizer"):
            return self._model.tokenizer

        # Some models expose the tokenizer directly
        if hasattr(self._model, "model") and hasattr(self._model.model, "tokenizer"):
            return self._model.model.tokenizer

        return None

    def _chat_with_template(
        self,
        messages: List[Dict[str, Any]],
        params: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
        tokenizer: Any,
    ) -> ChatResponse:
        """Generate using the tokenizer's chat template.

        Args:
            messages: Messages to send.
            params: Generation parameters.
            tools: Tool definitions.
            tool_choice: Tool choice setting.
            tokenizer: The tokenizer with chat template.

        Returns:
            ChatResponse with the model's output.
        """
        # Check if tools are requested but not supported
        if tools:
            # Try to apply template with tools to check support
            try:
                # The template should accept tools parameter if it supports them
                prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
            except TypeError:
                # Template doesn't accept tools parameter
                raise ToolCallingNotSupportedError(
                    f"Model {self._model_id} chat template does not support tools. "
                    "The apply_chat_template() method does not accept a 'tools' parameter. "
                    "Consider using LiteLLMModelAdapter for reliable tool calling."
                )
        else:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Generate response
        response_text = self._call_model(prompt, params)

        # Parse tool calls from response if tools were provided
        tool_calls = None
        content = response_text

        if tools:
            # Attempt to parse tool calls from the response
            # Different models format tool calls differently
            tool_calls, content = self._parse_tool_calls(response_text)

        return ChatResponse(
            content=content if content else None,
            tool_calls=tool_calls,
            role="assistant",
            model=self._model_id,
        )

    def _chat_without_template(self, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> ChatResponse:
        """Generate without a chat template (simple prompt concatenation).

        Args:
            messages: Messages to convert to prompt.
            params: Generation parameters.

        Returns:
            ChatResponse with the model's output.
        """
        # Simple conversion: concatenate messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        prompt = "\n".join(prompt_parts) + "\nassistant:"

        response_text = self._call_model(prompt, params)

        return ChatResponse(
            content=response_text,
            role="assistant",
            model=self._model_id,
        )

    def _call_model(self, prompt: str, params: Dict[str, Any]) -> str:
        """Call the underlying model with a prompt.

        Args:
            prompt: The formatted prompt.
            params: Generation parameters.

        Returns:
            The generated text.
        """
        # Set seed before generation if configured
        if self._seed is not None:
            try:
                from transformers import set_seed

                set_seed(self._seed)
            except ImportError:
                pass  # transformers not available, skip seeding

        try:
            result = self._model(prompt, **params)
        except TypeError:
            # Fallback: call without params
            result = self._model(prompt)

        # Extract text from various response formats
        if isinstance(result, str):
            return result
        elif isinstance(result, list) and len(result) > 0:
            # Pipeline returns list of dicts
            item = result[0]
            if isinstance(item, dict):
                # Text generation pipeline format
                if "generated_text" in item:
                    generated = item["generated_text"]
                    # Remove the prompt from the response if it's included
                    if generated.startswith(prompt):
                        return generated[len(prompt) :].strip()
                    return generated
                return str(item)
            return str(item)
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            return str(result)
        else:
            return str(result)

    def _parse_tool_calls(self, response: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Parse tool calls from model response.

        Different models format tool calls differently. This method attempts
        to parse common formats.

        Args:
            response: The raw model response.

        Returns:
            Tuple of (tool_calls, remaining_content).
        """
        import json
        import re

        # Try to find JSON tool calls in the response
        # Common patterns: <tool_call>...</tool_call>, ```json...```, etc.

        tool_calls = []
        remaining_content = response

        # Pattern 1: <tool_call> tags (used by some models)
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                call_data = json.loads(match.strip())
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": call_data.get("name", ""),
                            "arguments": json.dumps(call_data.get("arguments", {})),
                        },
                    }
                )
                remaining_content = remaining_content.replace(f"<tool_call>{match}</tool_call>", "")
            except json.JSONDecodeError:
                continue

        # Pattern 2: Function call JSON blocks
        function_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
        for match in re.finditer(function_pattern, response):
            try:
                name = match.group(1)
                args = match.group(2)
                # Validate JSON
                json.loads(args)
                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args,
                        },
                    }
                )
            except (json.JSONDecodeError, IndexError):
                continue

        remaining_content = remaining_content.strip()

        return (tool_calls if tool_calls else None, remaining_content if remaining_content else None)

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this HuggingFace model adapter.

        Returns:
            Dictionary containing model configuration.
        """
        base_config = super().gather_config()
        base_config.update(
            {
                "default_generation_params": self._default_generation_params,
                "callable_type": type(self._model).__name__,
            }
        )

        # Extract pipeline configuration
        pipeline_config = {}

        if hasattr(self._model, "task"):
            pipeline_config["task"] = self._model.task

        if hasattr(self._model, "device"):
            device = self._model.device
            pipeline_config["device"] = str(device) if device is not None else None

        if hasattr(self._model, "framework"):
            pipeline_config["framework"] = self._model.framework

        if pipeline_config:
            base_config["pipeline_config"] = pipeline_config

        return base_config
