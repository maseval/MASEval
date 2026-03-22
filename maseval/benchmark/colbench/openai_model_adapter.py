"""
OpenAI-compatible ModelAdapter for ColBench.

Implements ``ModelAdapter._chat_impl()`` using an ``openai.OpenAI`` client.
Works with any OpenAI-compatible API server (vLLM, TGI, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from maseval.core.model import ModelAdapter, ChatResponse

logger = logging.getLogger(__name__)


class OpenAIModelAdapter(ModelAdapter):
    """ModelAdapter backed by an OpenAI-compatible API client.

    Usage::

        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        adapter = OpenAIModelAdapter(client, model_id="meta-llama/Llama-3.1-70B-Instruct")

        # Use via ModelAdapter interface
        response = adapter.chat([{"role": "user", "content": "Hello"}])
        print(response.content)

        # Or simple generation
        text = adapter.generate("What is 2+2?")
    """

    def __init__(
        self,
        client: Any,  # openai.OpenAI
        model_id: str,
        default_max_tokens: int = 4096,
        default_temperature: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self._client = client
        self._model_id = model_id
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

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
        """Call the OpenAI-compatible chat completions API."""
        params = generation_params or {}
        max_tokens = params.pop("max_tokens", self.default_max_tokens)
        temperature = params.pop("temperature", self.default_temperature)

        api_kwargs: Dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **params,
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = tools
        if tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice
        if self._seed is not None and "seed" not in api_kwargs:
            api_kwargs["seed"] = self._seed

        completion = self._client.chat.completions.create(**api_kwargs)
        choice = completion.choices[0]
        message = choice.message

        # Parse tool calls if present
        parsed_tool_calls = None
        if message.tool_calls:
            parsed_tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        # Parse usage
        usage = None
        if completion.usage:
            usage = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return ChatResponse(
            content=message.content,
            tool_calls=parsed_tool_calls,
            role=message.role or "assistant",
            usage=usage,
            model=completion.model,
            stop_reason=choice.finish_reason,
        )

    def gather_config(self) -> Dict[str, Any]:
        return {
            **super().gather_config(),
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature,
        }