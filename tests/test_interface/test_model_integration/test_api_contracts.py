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
