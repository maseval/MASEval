"""Live API tests for model adapters.

These tests call real LLM APIs with minimal token usage to validate that
the adapter → SDK → API → ChatResponse chain works end-to-end with live
services.

Each test validates:

- ``ChatResponse`` fields are populated correctly (content, role, usage, stop_reason)
- Tool calling produces properly structured ``tool_calls`` dicts
- Structured output via ``response_model`` returns validated Pydantic instances
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
from pydantic import BaseModel, Field

pytestmark = [pytest.mark.interface, pytest.mark.credentialed]

requires_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
requires_anthropic = pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
requires_google = pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")

# Model IDs used across tests. Update here when models are rotated.
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-4-5"
GOOGLE_MODEL = "gemini-2.0-flash"
LITELLM_MODEL = "gpt-4o-mini"


# Shared response model used across all structured output tests.
class Capital(BaseModel):
    """A country's capital city."""

    city: str = Field(description="Name of the capital city")
    country: str = Field(description="Name of the country")


# Nested response model — Pydantic v2 generates additionalProperties: false for these,
# which Gemini's structured output API rejects. Used to verify GENAI_TOOLS mode handles it.
class Location(BaseModel):
    """A geographic location."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class WeatherReport(BaseModel):
    """A weather report for a location."""

    location: Location = Field(description="The location")
    temperature_celsius: float = Field(description="Temperature in Celsius")
    conditions: str = Field(description="Weather conditions (e.g. sunny, cloudy)")


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
        adapter = OpenAIModelAdapter(client=client, model_id=OPENAI_MODEL)
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
        adapter = OpenAIModelAdapter(client=client, model_id=OPENAI_MODEL)
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

    @requires_openai
    def test_structured_output(self):
        """Structured output via response_model returns a validated Pydantic instance."""
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        client = OpenAI()
        adapter = OpenAIModelAdapter(client=client, model_id=OPENAI_MODEL)
        response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
            generation_params={"max_tokens": 50},
        )

        assert isinstance(response.structured_response, Capital)
        assert response.structured_response.city.lower() == "paris"
        assert response.structured_response.country.lower() == "france"
        assert response.content is not None  # JSON serialization of the model


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
        adapter = AnthropicModelAdapter(client=client, model_id=ANTHROPIC_MODEL, max_tokens=10)
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
        adapter = AnthropicModelAdapter(client=client, model_id=ANTHROPIC_MODEL, max_tokens=100)
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

    @requires_anthropic
    def test_structured_output(self):
        """Structured output via response_model returns a validated Pydantic instance."""
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        client = Anthropic()
        adapter = AnthropicModelAdapter(client=client, model_id=ANTHROPIC_MODEL, max_tokens=100)
        response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
        )

        assert isinstance(response.structured_response, Capital)
        assert response.structured_response.city.lower() == "paris"
        assert response.structured_response.country.lower() == "france"
        assert response.content is not None


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
        adapter = GoogleGenAIModelAdapter(client=client, model_id=GOOGLE_MODEL)
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
        adapter = GoogleGenAIModelAdapter(client=client, model_id=GOOGLE_MODEL)
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

    @requires_google
    def test_structured_output(self):
        """Structured output via response_model returns a validated Pydantic instance."""
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        client = genai.Client()
        adapter = GoogleGenAIModelAdapter(client=client, model_id=GOOGLE_MODEL)
        response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
        )

        assert isinstance(response.structured_response, Capital)
        assert response.structured_response.city.lower() == "paris"
        assert response.structured_response.country.lower() == "france"
        assert response.content is not None

    @requires_google
    def test_structured_output_nested_model(self):
        """Nested Pydantic models work despite additionalProperties in JSON schema.

        Pydantic v2 emits additionalProperties: false for nested models.
        Gemini's native JSON schema output (GENAI_STRUCTURED_OUTPUTS) rejects this,
        but GENAI_TOOLS (function calling) handles it correctly via schema conversion.
        """
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        client = genai.Client()
        adapter = GoogleGenAIModelAdapter(client=client, model_id=GOOGLE_MODEL)
        response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris, France right now? Estimate the temperature."}],
            response_model=WeatherReport,
        )

        assert isinstance(response.structured_response, WeatherReport)
        assert isinstance(response.structured_response.location, Location)
        assert response.structured_response.location.city.lower() == "paris"
        assert response.content is not None


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

        adapter = LiteLLMModelAdapter(model_id=LITELLM_MODEL)
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

        adapter = LiteLLMModelAdapter(model_id=LITELLM_MODEL)
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

    @requires_openai
    def test_structured_output(self):
        """Structured output via response_model returns a validated Pydantic instance."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id=LITELLM_MODEL)
        response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
            generation_params={"max_tokens": 50},
        )

        assert isinstance(response.structured_response, Capital)
        assert response.structured_response.city.lower() == "paris"
        assert response.structured_response.country.lower() == "france"
        assert response.content is not None


# =============================================================================
# Cross-provider parameterized tests
# =============================================================================


def _make_openai_adapter(**kwargs):
    from openai import OpenAI
    from maseval.interface.inference.openai import OpenAIModelAdapter

    return OpenAIModelAdapter(client=OpenAI(), model_id=OPENAI_MODEL, **kwargs)


def _make_anthropic_adapter(**kwargs):
    from anthropic import Anthropic
    from maseval.interface.inference.anthropic import AnthropicModelAdapter

    return AnthropicModelAdapter(client=Anthropic(), model_id=ANTHROPIC_MODEL, max_tokens=100, **kwargs)


def _make_google_adapter(**kwargs):
    from google import genai
    from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

    return GoogleGenAIModelAdapter(client=genai.Client(), model_id=GOOGLE_MODEL, **kwargs)


def _make_litellm_adapter(**kwargs):
    pytest.importorskip("litellm")
    from maseval.interface.inference.litellm import LiteLLMModelAdapter

    return LiteLLMModelAdapter(model_id=LITELLM_MODEL, **kwargs)


# Each entry: (factory, env_var, max_tokens_param_name, supports_seed)
_ADAPTER_CONFIGS = [
    pytest.param(_make_openai_adapter, "OPENAI_API_KEY", "max_tokens", True, id="openai"),
    pytest.param(_make_anthropic_adapter, "ANTHROPIC_API_KEY", "max_tokens", False, id="anthropic"),
    pytest.param(_make_google_adapter, "GOOGLE_API_KEY", "max_output_tokens", True, id="google"),
    pytest.param(_make_litellm_adapter, "OPENAI_API_KEY", "max_tokens", True, id="litellm"),
]


class TestCrossProviderStructuredOutput:
    """Parameterized structured output tests across all adapters."""

    @pytest.mark.parametrize("factory,env_var,max_tok_key,supports_seed", _ADAPTER_CONFIGS)
    def test_structured_output_with_generation_params(self, factory, env_var, max_tok_key, supports_seed):
        """Structured output works with temperature and seed across all providers."""
        if not os.environ.get(env_var):
            pytest.skip(f"{env_var} not set")
        adapter = factory(seed=42) if supports_seed else factory()
        response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
            generation_params={"temperature": 0.0, max_tok_key: 100},
        )
        assert isinstance(response.structured_response, Capital)
        assert response.structured_response.city.lower() == "paris"
        assert response.structured_response.country.lower() == "france"

    @pytest.mark.parametrize("factory,env_var,max_tok_key,supports_seed", _ADAPTER_CONFIGS)
    def test_tool_call_then_structured_output(self, factory, env_var, max_tok_key, supports_seed):
        """Tool calling and structured output both work on the same adapter instance."""
        if not os.environ.get(env_var):
            pytest.skip(f"{env_var} not set")
        adapter = factory()

        # Tool call
        tool_response = adapter.chat(
            [{"role": "user", "content": "What is the weather in Paris? You must use the get_weather tool."}],
            tools=[WEATHER_TOOL],
            generation_params={max_tok_key: 100},
        )
        assert tool_response.tool_calls is not None
        assert len(tool_response.tool_calls) >= 1
        assert tool_response.tool_calls[0]["function"]["name"] == "get_weather"

        args = json.loads(tool_response.tool_calls[0]["function"]["arguments"])
        assert isinstance(args, dict)
        assert "city" in args

        # Structured output on the same adapter
        structured_response = adapter.chat(
            [{"role": "user", "content": "What is the capital of France?"}],
            response_model=Capital,
            generation_params={max_tok_key: 100},
        )
        assert isinstance(structured_response.structured_response, Capital)
        assert structured_response.structured_response.city.lower() == "paris"
