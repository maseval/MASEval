"""API contract tests for model adapters.

These tests validate that model adapters correctly parse responses from real
SDK clients by mocking HTTP responses with realistic API payloads.  They test
the full chain:

    adapter → SDK client → HTTP (mocked by respx) → SDK response parsing → ChatResponse

Unlike the SDK-level mocks in test_model_adapters.py (which use hand-crafted
duck-typed mock objects), these tests use *real* SDK clients with intercepted
HTTP, catching issues where SDK updates change response structures.

These tests do NOT require API keys or network access.

Run with::

    pytest tests/test_interface/test_model_integration/test_api_contracts.py -v
"""

import json

import pytest
import respx

pytestmark = [pytest.mark.interface]


# =============================================================================
# Response Fixtures — OpenAI
# =============================================================================


OPENAI_TEXT_RESPONSE = {
    "id": "chatcmpl-contract-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_contract_test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "refusal": None,
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 9,
        "total_tokens": 21,
    },
}

OPENAI_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-contract-tools",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "system_fingerprint": "fp_contract_test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
                "refusal": None,
            },
            "logprobs": None,
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 82,
        "completion_tokens": 18,
        "total_tokens": 100,
    },
}


# =============================================================================
# Response Fixtures — Anthropic
# =============================================================================


ANTHROPIC_TEXT_RESPONSE = {
    "id": "msg_contract_test",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! How can I help you today?",
        }
    ],
    "model": "claude-sonnet-4-5-20250514",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 9,
    },
}

ANTHROPIC_TOOL_USE_RESPONSE = {
    "id": "msg_contract_tools",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ],
    "model": "claude-sonnet-4-5-20250514",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 82,
        "output_tokens": 18,
    },
}


# =============================================================================
# Response Fixtures — Google GenAI
# =============================================================================


GOOGLE_TEXT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello! How can I help you today?"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 12,
        "candidatesTokenCount": 9,
        "totalTokenCount": 21,
    },
    "modelVersion": "gemini-2.0-flash",
}

GOOGLE_FUNCTION_CALL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                        }
                    }
                ],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 82,
        "candidatesTokenCount": 18,
        "totalTokenCount": 100,
    },
}


# =============================================================================
# OpenAI Contract Tests
# =============================================================================


class TestOpenAIApiContracts:
    """Validate OpenAI adapter against realistic HTTP responses.

    Uses real ``openai.OpenAI`` client with HTTP intercepted by respx.
    """

    @respx.mock
    def test_text_response(self):
        """Adapter correctly parses a text completion."""
        pytest.importorskip("openai")
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        respx.post("https://api.openai.com/v1/chat/completions").respond(200, json=OPENAI_TEXT_RESPONSE)

        client = OpenAI(api_key="test-key-not-real")
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o-mini")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Hello! How can I help you today?"
        assert response.role == "assistant"
        assert response.stop_reason == "stop"
        assert response.model == "gpt-4o-mini"
        assert response.tool_calls is None

        assert response.usage is not None
        assert response.usage["input_tokens"] == 12
        assert response.usage["output_tokens"] == 9
        assert response.usage["total_tokens"] == 21

    @respx.mock
    def test_tool_call_response(self):
        """Adapter correctly parses tool calls."""
        pytest.importorskip("openai")
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        respx.post("https://api.openai.com/v1/chat/completions").respond(200, json=OPENAI_TOOL_CALL_RESPONSE)

        client = OpenAI(api_key="test-key-not-real")
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o-mini")
        response = adapter.chat(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert response.content is None
        assert response.stop_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

        assert response.usage is not None
        assert response.usage["input_tokens"] == 82
        assert response.usage["output_tokens"] == 18

    @respx.mock
    def test_seed_propagation(self):
        """Seed is included in the HTTP request body."""
        pytest.importorskip("openai")
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        route = respx.post("https://api.openai.com/v1/chat/completions").respond(200, json=OPENAI_TEXT_RESPONSE)

        client = OpenAI(api_key="test-key-not-real")
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o-mini", seed=42)
        adapter.chat([{"role": "user", "content": "Hello"}])

        request = route.calls[0].request
        body = json.loads(request.content)
        assert body.get("seed") == 42


# =============================================================================
# Anthropic Contract Tests
# =============================================================================


class TestAnthropicApiContracts:
    """Validate Anthropic adapter against realistic HTTP responses.

    Uses real ``anthropic.Anthropic`` client with HTTP intercepted by respx.
    """

    @respx.mock
    def test_text_response(self):
        """Adapter correctly parses a text message."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        respx.post("https://api.anthropic.com/v1/messages").respond(200, json=ANTHROPIC_TEXT_RESPONSE)

        client = Anthropic(api_key="test-key-not-real")
        adapter = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5-20250514")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Hello! How can I help you today?"
        assert response.role == "assistant"
        assert response.stop_reason == "end_turn"
        assert response.model == "claude-sonnet-4-5-20250514"
        assert response.tool_calls is None

        assert response.usage is not None
        assert response.usage["input_tokens"] == 12
        assert response.usage["output_tokens"] == 9
        assert response.usage["total_tokens"] == 21  # computed by adapter

    @respx.mock
    def test_tool_use_response(self):
        """Adapter correctly parses tool use blocks."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        respx.post("https://api.anthropic.com/v1/messages").respond(200, json=ANTHROPIC_TOOL_USE_RESPONSE)

        client = Anthropic(api_key="test-key-not-real")
        adapter = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5-20250514")
        response = adapter.chat(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert response.content is None
        assert response.stop_reason == "tool_use"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert tc["id"] == "toolu_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

        assert response.usage is not None
        assert response.usage["input_tokens"] == 82
        assert response.usage["output_tokens"] == 18

    @respx.mock
    def test_system_message_sent_separately(self):
        """System message is sent as top-level ``system`` parameter, not in messages."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        route = respx.post("https://api.anthropic.com/v1/messages").respond(200, json=ANTHROPIC_TEXT_RESPONSE)

        client = Anthropic(api_key="test-key-not-real")
        adapter = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5-20250514")
        adapter.chat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        )

        request = route.calls[0].request
        body = json.loads(request.content)
        assert body["system"] == "You are a helpful assistant."
        assert all(m["role"] != "system" for m in body["messages"])


# =============================================================================
# Google GenAI Contract Tests
# =============================================================================


class TestGoogleGenAIApiContracts:
    """Validate Google GenAI adapter against realistic HTTP responses.

    Uses real ``google.genai.Client`` with HTTP intercepted by respx.
    """

    @respx.mock
    def test_text_response(self):
        """Adapter correctly parses a text response."""
        pytest.importorskip("google.genai")
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        respx.route(
            method="POST",
            url__regex=r".*generativelanguage\.googleapis\.com.*models.*generateContent.*",
        ).respond(200, json=GOOGLE_TEXT_RESPONSE)

        client = genai.Client(api_key="test-key-not-real", http_options={"api_version": "v1beta"})
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-2.0-flash")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Hello! How can I help you today?"
        assert response.role == "assistant"
        assert response.tool_calls is None

        assert response.usage is not None
        assert response.usage["input_tokens"] == 12
        assert response.usage["output_tokens"] == 9
        assert response.usage["total_tokens"] == 21

    @respx.mock
    def test_function_call_response(self):
        """Adapter correctly parses function calls."""
        pytest.importorskip("google.genai")
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        respx.route(
            method="POST",
            url__regex=r".*generativelanguage\.googleapis\.com.*models.*generateContent.*",
        ).respond(200, json=GOOGLE_FUNCTION_CALL_RESPONSE)

        client = genai.Client(api_key="test-key-not-real", http_options={"api_version": "v1beta"})
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-2.0-flash")
        response = adapter.chat(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

        assert response.usage is not None
        assert response.usage["input_tokens"] == 82
        assert response.usage["output_tokens"] == 18


# =============================================================================
# LiteLLM Contract Tests
# =============================================================================


class TestLiteLLMApiContracts:
    """Validate LiteLLM adapter response parsing.

    LiteLLM is a routing layer that wraps provider SDKs, so HTTP-level
    mocking is not practical.  Instead we mock ``litellm.completion`` with
    response objects that match LiteLLM's actual return types.
    """

    def test_text_response(self):
        """Adapter correctly parses a text completion from LiteLLM."""
        litellm = pytest.importorskip("litellm")
        from unittest.mock import patch
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        mock_response = litellm.ModelResponse(
            id="chatcmpl-contract-test",
            choices=[
                litellm.Choices(
                    index=0,
                    message=litellm.Message(
                        role="assistant",
                        content="Hello! How can I help you today?",
                    ),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o-mini",
            usage=litellm.Usage(
                prompt_tokens=12,
                completion_tokens=9,
                total_tokens=21,
            ),
        )

        with patch("litellm.completion", return_value=mock_response):
            adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")
            response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Hello! How can I help you today?"
        assert response.role == "assistant"
        assert response.stop_reason == "stop"
        assert response.model == "gpt-4o-mini"
        assert response.tool_calls is None

        assert response.usage is not None
        assert response.usage["input_tokens"] == 12
        assert response.usage["output_tokens"] == 9
        assert response.usage["total_tokens"] == 21

    def test_tool_call_response(self):
        """Adapter correctly parses tool calls from LiteLLM."""
        pytest.importorskip("litellm")
        from unittest.mock import patch, MagicMock
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        # LiteLLM tool call objects have the same structure as OpenAI
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.type = "function"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Paris"}'

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.role = "assistant"
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 82
        mock_response.usage.completion_tokens = 18
        mock_response.usage.total_tokens = 100

        with patch("litellm.completion", return_value=mock_response):
            adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")
            response = adapter.chat(
                [{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

        assert response.content is None
        assert response.stop_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tc = response.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}

        assert response.usage is not None
        assert response.usage["input_tokens"] == 82
        assert response.usage["output_tokens"] == 18


# =============================================================================
# Usage Extraction Contract Tests
# =============================================================================
#
# These tests verify that each adapter correctly extracts ALL usage fields
# (including cache tokens, reasoning tokens, provider cost) from realistic
# API response payloads, and that the cost calculator produces correct costs.
# =============================================================================


# -- OpenAI usage-rich fixture ------------------------------------------------

OPENAI_USAGE_RICH_RESPONSE = {
    "id": "chatcmpl-usage-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "system_fingerprint": "fp_usage_test",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello!",
                "refusal": None,
            },
            "logprobs": None,
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "total_tokens": 700,
        "prompt_tokens_details": {
            "cached_tokens": 300,
        },
        "completion_tokens_details": {
            "reasoning_tokens": 80,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    },
}


# -- Anthropic usage-rich fixture --------------------------------------------

ANTHROPIC_USAGE_RICH_RESPONSE = {
    "id": "msg_usage_test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-sonnet-4-5-20250514",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_input_tokens": 600,
        "cache_creation_input_tokens": 100,
    },
}


# -- Google usage-rich fixture -----------------------------------------------

GOOGLE_USAGE_RICH_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello!"}],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 500,
        "candidatesTokenCount": 200,
        "totalTokenCount": 700,
        "thoughtsTokenCount": 120,
    },
    "modelVersion": "gemini-2.0-flash-thinking",
}


class TestOpenAIUsageExtraction:
    """Verify OpenAI adapter extracts all usage fields correctly."""

    @respx.mock
    def test_extracts_cached_and_reasoning_tokens(self):
        """Cached tokens and reasoning tokens are extracted from nested details."""
        pytest.importorskip("openai")
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter

        respx.post("https://api.openai.com/v1/chat/completions").respond(200, json=OPENAI_USAGE_RICH_RESPONSE)

        client = OpenAI(api_key="test-key-not-real")
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.usage is not None
        assert response.usage["input_tokens"] == 500
        assert response.usage["output_tokens"] == 200
        assert response.usage["total_tokens"] == 700
        assert response.usage["cached_input_tokens"] == 300
        assert response.usage["reasoning_tokens"] == 80

    @respx.mock
    def test_cost_calculation_with_cached_tokens(self):
        """Full pipeline: OpenAI adapter + StaticPricingCalculator with caching.

        input_tokens=500, cached_input_tokens=300
        Non-cached: 200 * $2.5e-6 = $0.0005
        Cached: 300 * $1.25e-6 = $0.000375
        Output: 200 * $10e-6 = $0.002
        Total = $0.002875
        """
        pytest.importorskip("openai")
        from openai import OpenAI
        from maseval.interface.inference.openai import OpenAIModelAdapter
        from maseval.core.usage import StaticPricingCalculator, TokenUsage

        respx.post("https://api.openai.com/v1/chat/completions").respond(200, json=OPENAI_USAGE_RICH_RESPONSE)

        calc = StaticPricingCalculator(
            {
                "gpt-4o": {
                    "input": 2.5e-6,
                    "output": 10e-6,
                    "cached_input": 1.25e-6,
                },
            }
        )

        client = OpenAI(api_key="test-key-not-real")
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4o", cost_calculator=calc)
        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.input_tokens == 500
        assert total.cached_input_tokens == 300
        assert total.reasoning_tokens == 80
        assert total.cost == pytest.approx(0.002875)


class TestAnthropicUsageExtraction:
    """Verify Anthropic adapter extracts all usage fields correctly."""

    @respx.mock
    def test_extracts_cache_read_and_creation_tokens(self):
        """Both cache_read and cache_creation tokens are extracted."""
        pytest.importorskip("anthropic")
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        respx.post("https://api.anthropic.com/v1/messages").respond(200, json=ANTHROPIC_USAGE_RICH_RESPONSE)

        client = Anthropic(api_key="test-key-not-real")
        adapter = AnthropicModelAdapter(client=client, model_id="claude-sonnet-4-5-20250514")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.usage is not None
        assert response.usage["input_tokens"] == 1000
        assert response.usage["output_tokens"] == 200
        assert response.usage["total_tokens"] == 1200  # computed by adapter
        assert response.usage["cached_input_tokens"] == 600
        assert response.usage["cache_creation_input_tokens"] == 100

    @respx.mock
    def test_cost_calculation_with_cache_creation(self):
        """Full pipeline: Anthropic adapter + StaticPricingCalculator with cache creation.

        input_tokens=1000, cached=600, cache_creation=100
        Non-cached: (1000 - 600 - 100) = 300 * $3e-6 = $0.0009
        Cached: 600 * $0.3e-6 = $0.00018
        Cache creation: 100 * $3.75e-6 = $0.000375
        Output: 200 * $15e-6 = $0.003
        Total = $0.004455
        """
        pytest.importorskip("anthropic")
        from anthropic import Anthropic
        from maseval.interface.inference.anthropic import AnthropicModelAdapter
        from maseval.core.usage import StaticPricingCalculator, TokenUsage

        respx.post("https://api.anthropic.com/v1/messages").respond(200, json=ANTHROPIC_USAGE_RICH_RESPONSE)

        calc = StaticPricingCalculator(
            {
                "claude-sonnet-4-5-20250514": {
                    "input": 3e-6,
                    "output": 15e-6,
                    "cached_input": 0.3e-6,
                    "cache_creation_input": 3.75e-6,
                },
            }
        )

        client = Anthropic(api_key="test-key-not-real")
        adapter = AnthropicModelAdapter(
            client=client,
            model_id="claude-sonnet-4-5-20250514",
            cost_calculator=calc,
        )
        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.cached_input_tokens == 600
        assert total.cache_creation_input_tokens == 100
        assert total.cost == pytest.approx(0.004455)


class TestGoogleGenAIUsageExtraction:
    """Verify Google GenAI adapter extracts all usage fields correctly."""

    @respx.mock
    def test_extracts_thoughts_as_reasoning_tokens(self):
        """Google's thoughtsTokenCount maps to reasoning_tokens."""
        pytest.importorskip("google.genai")
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        respx.route(
            method="POST",
            url__regex=r".*generativelanguage\.googleapis\.com.*models.*generateContent.*",
        ).respond(200, json=GOOGLE_USAGE_RICH_RESPONSE)

        client = genai.Client(
            api_key="test-key-not-real",
            http_options={"api_version": "v1beta"},
        )
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-2.0-flash-thinking")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.usage is not None
        assert response.usage["input_tokens"] == 500
        assert response.usage["output_tokens"] == 200
        assert response.usage["total_tokens"] == 700
        assert response.usage["reasoning_tokens"] == 120

    @respx.mock
    def test_cost_calculation_basic(self):
        """Full pipeline: Google adapter + StaticPricingCalculator.

        500 input * $0.075e-6 = $0.0000375
        200 output * $0.3e-6 = $0.00006
        Total = $0.0000975
        """
        pytest.importorskip("google.genai")
        from google import genai
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter
        from maseval.core.usage import StaticPricingCalculator, TokenUsage

        respx.route(
            method="POST",
            url__regex=r".*generativelanguage\.googleapis\.com.*models.*generateContent.*",
        ).respond(200, json=GOOGLE_USAGE_RICH_RESPONSE)

        calc = StaticPricingCalculator(
            {
                "gemini-2.0-flash-thinking": {
                    "input": 0.075e-6,
                    "output": 0.3e-6,
                },
            }
        )

        client = genai.Client(
            api_key="test-key-not-real",
            http_options={"api_version": "v1beta"},
        )
        adapter = GoogleGenAIModelAdapter(
            client=client,
            model_id="gemini-2.0-flash-thinking",
            cost_calculator=calc,
        )
        adapter.chat([{"role": "user", "content": "Hello"}])
        total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.reasoning_tokens == 120
        assert total.cost == pytest.approx(0.0000975)


class TestLiteLLMUsageExtraction:
    """Verify LiteLLM adapter extracts all usage fields correctly."""

    def test_extracts_cached_and_cache_creation_tokens(self):
        """LiteLLM's prompt_tokens_details with cached_tokens and cache_creation_tokens."""
        pytest.importorskip("litellm")
        from unittest.mock import patch, MagicMock
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        mock_prompt_details = MagicMock()
        mock_prompt_details.cached_tokens = 400
        mock_prompt_details.cache_creation_tokens = 50

        mock_completion_details = MagicMock()
        mock_completion_details.reasoning_tokens = 60

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 800
        mock_usage.completion_tokens = 150
        mock_usage.total_tokens = 950
        mock_usage.prompt_tokens_details = mock_prompt_details
        mock_usage.completion_tokens_details = mock_completion_details

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "claude-sonnet-4-5-20250514"
        mock_response.usage = mock_usage
        mock_response._hidden_params = {"response_cost": 0.0042}

        with patch("litellm.completion", return_value=mock_response):
            adapter = LiteLLMModelAdapter(model_id="claude-sonnet-4-5-20250514")
            response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.usage is not None
        assert response.usage["input_tokens"] == 800
        assert response.usage["output_tokens"] == 150
        assert response.usage["total_tokens"] == 950
        assert response.usage["cached_input_tokens"] == 400
        assert response.usage["cache_creation_input_tokens"] == 50
        assert response.usage["reasoning_tokens"] == 60

    def test_provider_cost_from_hidden_params(self):
        """LiteLLM's _hidden_params.response_cost is extracted as provider cost.

        Provider cost ($0.0042) should take precedence over calculator.
        """
        pytest.importorskip("litellm")
        from unittest.mock import patch, MagicMock
        from maseval.interface.inference.litellm import LiteLLMModelAdapter
        from maseval.core.usage import StaticPricingCalculator, TokenUsage

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.prompt_tokens_details = None
        mock_usage.completion_tokens_details = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response._hidden_params = {"response_cost": 0.0042}

        # Calculator would compute a different cost — provider should win
        calc = StaticPricingCalculator(
            {
                "gpt-4o": {"input": 0.01, "output": 0.02},
            }
        )

        with patch("litellm.completion", return_value=mock_response):
            adapter = LiteLLMModelAdapter(model_id="gpt-4o", cost_calculator=calc)
            adapter.chat([{"role": "user", "content": "Hello"}])
            total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(0.0042)

    def test_calculator_used_when_no_provider_cost(self):
        """When _hidden_params has no cost, calculator is used.

        100 input * $0.01 + 50 output * $0.02 = $2.00
        """
        pytest.importorskip("litellm")
        from unittest.mock import patch, MagicMock
        from maseval.interface.inference.litellm import LiteLLMModelAdapter
        from maseval.core.usage import StaticPricingCalculator, TokenUsage

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.prompt_tokens_details = None
        mock_usage.completion_tokens_details = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response._hidden_params = {}

        calc = StaticPricingCalculator(
            {
                "gpt-4o": {"input": 0.01, "output": 0.02},
            }
        )

        with patch("litellm.completion", return_value=mock_response):
            adapter = LiteLLMModelAdapter(model_id="gpt-4o", cost_calculator=calc)
            adapter.chat([{"role": "user", "content": "Hello"}])
            total = adapter.gather_usage()

        assert isinstance(total, TokenUsage)
        assert total.cost == pytest.approx(2.00)
