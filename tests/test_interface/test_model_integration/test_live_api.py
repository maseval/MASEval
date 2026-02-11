"""Live API tests for model adapters.

These tests call real LLM APIs with minimal token usage to validate that
the adapter → SDK → API → ChatResponse chain works end-to-end with live
services.

Each test validates:

- ``ChatResponse`` fields are populated correctly (content, role, usage, stop_reason)
- Tool calling produces properly structured ``tool_calls`` dicts
- The adapter's format conversions survive a real API round-trip

These tests require API keys and incur small costs (~$0.001 per run).
They are marked ``credentialed`` and excluded from default test runs.

Run with::

    pytest tests/test_interface/test_model_integration/test_live_api.py -v

Prerequisites::

    export OPENAI_API_KEY=...      # OpenAI and LiteLLM tests
    export ANTHROPIC_API_KEY=...   # Anthropic tests
    export GOOGLE_API_KEY=...      # Google GenAI tests
"""

import json
import os

import pytest

pytestmark = [pytest.mark.interface, pytest.mark.credentialed]

requires_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
requires_anthropic = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
requires_google = pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")

# Shared tool definition used across all provider tests.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}


# =============================================================================
# OpenAI
# =============================================================================


class TestOpenAILiveAPI:
    """Validate OpenAI adapter against live API (gpt-4o-mini)."""

    @requires_openai
    def test_text_response(self):
        """Text completion returns a valid ChatResponse with all fields."""
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        client = OpenAI()
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o-mini")
        response = adapter.chat(
            [{"role": "user", "content": "Say 'test' and nothing else."}],
            generation_params={"max_tokens": 10},
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.role == "assistant"
        assert response.model is not None
        assert response.stop_reason is not None

        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @requires_openai
    def test_tool_call_response(self):
        """Tool calling returns properly structured tool_calls."""
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        client = OpenAI()
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o-mini")
        response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris? You must use the get_weather tool."}],
            tools=[WEATHER_TOOL],
            generation_params={"max_tokens": 50},
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1

        tc = response.tool_calls[0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"

        args = json.loads(tc["function"]["arguments"])
        assert isinstance(args, dict)
        assert "city" in args


# =============================================================================
# Anthropic
# =============================================================================


class TestAnthropicLiveAPI:
    """Validate Anthropic adapter against live API (claude-3-5-haiku)."""

    @requires_anthropic
    def test_text_response(self):
        """Text completion returns a valid ChatResponse with all fields."""
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        client = Anthropic()
        adapter = AnthropicModelAdapter(client=client, model_id="claude-3-5-haiku-20241022", max_tokens=10)
        response = adapter.chat(
            [{"role": "user", "content": "Say 'test' and nothing else."}],
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.role == "assistant"
        assert response.model is not None
        assert response.stop_reason is not None

        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @requires_anthropic
    def test_tool_use_response(self):
        """Tool use returns properly structured tool_calls."""
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        client = Anthropic()
        adapter = AnthropicModelAdapter(client=client, model_id="claude-3-5-haiku-20241022", max_tokens=100)
        response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris? You must use the get_weather tool."}],
            tools=[WEATHER_TOOL],
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1

        tc = response.tool_calls[0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"

        args = json.loads(tc["function"]["arguments"])
        assert isinstance(args, dict)
        assert "city" in args


# =============================================================================
# Google GenAI
# =============================================================================


class TestGoogleGenAILiveAPI:
    """Validate Google GenAI adapter against live API (gemini-2.0-flash)."""

    @requires_google
    def test_text_response(self):
        """Text completion returns a valid ChatResponse with all fields."""
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        client = genai.Client()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-2.0-flash")
        response = adapter.chat(
            [{"role": "user", "content": "Say 'test' and nothing else."}],
            generation_params={"max_output_tokens": 10},
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.role == "assistant"

        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @requires_google
    def test_function_call_response(self):
        """Function calling returns properly structured tool_calls."""
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        client = genai.Client()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-2.0-flash")
        response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris? You must use the get_weather tool."}],
            tools=[WEATHER_TOOL],
            generation_params={"max_output_tokens": 50},
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1

        tc = response.tool_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"

        args = json.loads(tc["function"]["arguments"])
        assert isinstance(args, dict)
        assert "city" in args


# =============================================================================
# LiteLLM (routes through OpenAI)
# =============================================================================


class TestLiteLLMLiveAPI:
    """Validate LiteLLM adapter routes through a real provider correctly.

    Uses OpenAI (gpt-4o-mini) as the underlying provider since LiteLLM is
    a routing layer.  Requires ``OPENAI_API_KEY``.
    """

    @requires_openai
    def test_text_response(self):
        """LiteLLM routes to OpenAI and returns a valid ChatResponse."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")
        response = adapter.chat(
            [{"role": "user", "content": "Say 'test' and nothing else."}],
            generation_params={"max_tokens": 10},
        )

        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.role == "assistant"
        assert response.model is not None

        assert response.usage is not None
        assert response.usage["input_tokens"] > 0
        assert response.usage["output_tokens"] > 0

    @requires_openai
    def test_tool_call_response(self):
        """LiteLLM routes tool calling through OpenAI correctly."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")
        response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris? You must use the get_weather tool."}],
            tools=[WEATHER_TOOL],
            generation_params={"max_tokens": 50},
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1

        tc = response.tool_calls[0]
        assert "id" in tc
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"

        args = json.loads(tc["function"]["arguments"])
        assert isinstance(args, dict)
        assert "city" in args
