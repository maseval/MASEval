"""Integration tests for ModelAdapter interface implementations.

Tests specific behavior and integration for each ModelAdapter implementation:
- OpenAIModelAdapter
- GoogleGenAIModelAdapter
- HuggingFaceModelAdapter
- LiteLLMModelAdapter

These tests verify that each adapter correctly wraps its underlying client
and provides adapter-specific configuration.
"""

from unittest.mock import MagicMock, patch

import pytest


# ==================== OpenAI Tests ====================


@pytest.mark.interface
class TestOpenAIModelAdapterIntegration:
    """Test OpenAIModelAdapter specific behavior."""

    def test_openai_adapter_initialization(self):
        """OpenAIModelAdapter initializes with client and model_id."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Mock client with chat.completions.create interface
        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")

        assert adapter.model_id == "gpt-4"

    def test_openai_adapter_generate_with_modern_client(self):
        """OpenAIModelAdapter works with modern client interface."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        # Extract user message content
                        user_msg = next((m for m in messages if m["role"] == "user"), {})
                        content = user_msg.get("content", "")
                        return {"choices": [{"message": {"content": f"Response to: {content}"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        assert "Response to: Test prompt" in result

    def test_openai_adapter_extract_text_from_dict(self):
        """OpenAIModelAdapter extracts text from various response formats."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Chat completion format
        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Chat response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        result = adapter.generate("Test")
        assert result == "Chat response"

    def test_openai_adapter_default_generation_params(self):
        """OpenAIModelAdapter uses default generation parameters."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        captured_params = {}

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        captured_params.update(kwargs)
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(
            client=MockClient(),
            model_id="gpt-4",
            default_generation_params={"temperature": 0.7, "max_tokens": 100},
        )

        adapter.generate("Test")

        assert "temperature" in captured_params
        assert captured_params["temperature"] == 0.7
        assert captured_params["max_tokens"] == 100

    def test_openai_adapter_gather_config_includes_params(self):
        """OpenAIModelAdapter config includes generation parameters."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(
            client=MockClient(),
            model_id="gpt-4",
            default_generation_params={"temperature": 0.9},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert "client_type" in config

    def test_openai_adapter_gather_config_includes_client_config(self):
        """OpenAIModelAdapter config includes client configuration."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        # Create a mock client with configuration attributes
        class MockOpenAIClient:
            def __init__(self):
                self.timeout = 60
                self.max_retries = 3

            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "Response"}}]}

                completions = Completions()

            chat = Chat()

        client = MockOpenAIClient()
        adapter = OpenAIModelAdapter(client=client, model_id="gpt-4")

        config = adapter.gather_config()

        # Should include client configuration that affects behavior
        assert "client_type" in config
        assert config["client_type"] == "MockOpenAIClient"

    def test_openai_adapter_tool_calls_response(self):
        """OpenAIModelAdapter handles tool call responses."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockToolCall:
            id = "call_123"
            type = "function"

            class function:
                name = "get_weather"
                arguments = '{"city": "Paris"}'

        class MockMessage:
            content = None
            role = "assistant"
            tool_calls = [MockToolCall()]

        class MockChoice:
            message = MockMessage()
            finish_reason = "tool_calls"

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        class MockResponse:
            choices = [MockChoice()]
            usage = MockUsage()
            model = "gpt-4"

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return MockResponse()

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Weather?"}])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.usage is not None
        assert response.usage["input_tokens"] == 10
        assert response.stop_reason == "tool_calls"

    def test_openai_adapter_tools_parameter_passing(self):
        """OpenAIModelAdapter passes tools to API."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        captured_kwargs = {}

        class MockMessage:
            content = "I'll check the weather"
            role = "assistant"
            tool_calls = None

        class MockChoice:
            message = MockMessage()
            finish_reason = "stop"

        class MockResponse:
            choices = [MockChoice()]
            model = "gpt-4"

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        captured_kwargs.update(kwargs)
                        return MockResponse()

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        adapter.chat(
            [{"role": "user", "content": "Weather?"}],
            tools=tools,
            tool_choice="auto",
        )

        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"] == tools
        assert captured_kwargs["tool_choice"] == "auto"

    def test_openai_adapter_legacy_client_fallback(self):
        """OpenAIModelAdapter falls back to legacy client interface."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class LegacyClient:
            def create(self, model, messages, **kwargs):
                return {"choices": [{"message": {"content": "Legacy response"}}]}

        adapter = OpenAIModelAdapter(client=LegacyClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Legacy response"

    def test_openai_adapter_callable_client(self):
        """OpenAIModelAdapter falls back to calling client directly."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        def callable_client(model, messages, **kwargs):
            return {"choices": [{"message": {"content": "Callable response"}}]}

        adapter = OpenAIModelAdapter(client=callable_client, model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Callable response"

    def test_openai_adapter_text_format_response(self):
        """OpenAIModelAdapter parses text format responses."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"text": "Completion text"}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Completion text"

    def test_openai_adapter_dict_response_with_tool_calls(self):
        """OpenAIModelAdapter parses dict responses with tool calls."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {"name": "search", "arguments": "{}"},
                                            }
                                        ],
                                    },
                                }
                            ],
                        }

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Search"}])

        assert response.tool_calls is not None
        assert response.tool_calls[0]["function"]["name"] == "search"

    def test_openai_adapter_fallback_without_model_param(self):
        """OpenAIModelAdapter falls back to calling without model param."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class LegacyClient:
            def create(self, messages, **kwargs):
                # Only accepts messages, no model param
                return {"choices": [{"message": {"content": "No model param"}}]}

        adapter = OpenAIModelAdapter(client=LegacyClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "No model param"

    def test_openai_adapter_gather_config_with_timeout(self):
        """OpenAIModelAdapter includes timeout in config."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockTimeout:
            connect = 5.0
            read = 30.0
            write = 30.0
            pool = 10.0

        class MockClient:
            timeout = MockTimeout()
            max_retries = 3

            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"choices": [{"message": {"content": "R"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        config = adapter.gather_config()

        assert "client_config" in config
        assert config["client_config"]["max_retries"] == 3


# ==================== Google GenAI Tests ====================


@pytest.mark.interface
class TestGoogleGenAIModelAdapterIntegration:
    """Test GoogleGenAIModelAdapter specific behavior."""

    def test_google_genai_adapter_initialization(self):
        """GoogleGenAIModelAdapter initializes with client and model_id."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        # Mock client
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-pro")

        assert adapter.model_id == "gemini-pro"

    def test_google_genai_adapter_generate(self):
        """GoogleGenAIModelAdapter generates text."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    # Extract text from contents (first user message)
                    text = ""
                    if contents:
                        for content in contents:
                            if content.get("role") == "user":
                                parts = content.get("parts", [])
                                if parts:
                                    text = parts[0].get("text", "")
                                    break

                    class Response:
                        text: str = ""

                    resp = Response()
                    resp.text = f"Response to: {text}"
                    return resp

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(client=client, model_id="gemini-pro")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        assert "Response to: Test prompt" in result

    def test_google_genai_adapter_default_generation_params(self):
        """GoogleGenAIModelAdapter uses default generation parameters."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        captured_config = None

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    nonlocal captured_config
                    captured_config = config

                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(
            client=client,
            model_id="gemini-pro",
            default_generation_params={"temperature": 0.8},
        )

        adapter.generate("Test")

        # Should pass config to generate_content
        assert captured_config is not None

    def test_google_genai_adapter_gather_config(self):
        """GoogleGenAIModelAdapter config includes parameters."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "Response"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        client = MockClient()
        adapter = GoogleGenAIModelAdapter(
            client=client,
            model_id="gemini-pro",
            default_generation_params={"temperature": 0.9},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert "client_type" in config

    def test_google_genai_adapter_function_call_response(self):
        """GoogleGenAIModelAdapter handles function call responses."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockFunctionCall:
            name = "search_web"
            args = {"query": "test"}

        class MockPart:
            type = "function_call"
            function_call = MockFunctionCall()

        class MockContent:
            parts = [MockPart()]

        class MockCandidate:
            content = MockContent()
            finish_reason = "STOP"

        class MockUsage:
            prompt_token_count = 20
            candidates_token_count = 10
            total_token_count = 30

        class MockResponse:
            text = None
            candidates = [MockCandidate()]
            usage_metadata = MockUsage()

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    return MockResponse()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        response = adapter.chat([{"role": "user", "content": "Search"}])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "search_web"
        assert response.usage is not None
        assert response.usage["input_tokens"] == 20

    def test_google_genai_adapter_tools_conversion(self):
        """GoogleGenAIModelAdapter converts tools to Google format."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        captured_config = None

        class MockResponse:
            text = "Response"
            candidates = []

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    nonlocal captured_config
                    captured_config = config
                    return MockResponse()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        adapter.chat([{"role": "user", "content": "Weather?"}], tools=tools)

        assert captured_config is not None

    def test_google_genai_adapter_tool_choice_options(self):
        """GoogleGenAIModelAdapter handles various tool_choice options."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockResponse:
            text = "Response"
            candidates = []

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    return MockResponse()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        tools = [{"type": "function", "function": {"name": "test"}}]

        # Test different tool_choice values
        for choice in ["none", "auto", "required"]:
            response = adapter.chat(
                [{"role": "user", "content": "Test"}],
                tools=tools,
                tool_choice=choice,
            )
            assert response is not None

        # Test specific function choice
        response = adapter.chat(
            [{"role": "user", "content": "Test"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "test"}},
        )
        assert response is not None


# ==================== HuggingFace Tests ====================


@pytest.mark.interface
class TestHuggingFaceModelAdapterIntegration:
    """Test HuggingFaceModelAdapter specific behavior."""

    def test_huggingface_adapter_initialization(self):
        """HuggingFaceModelAdapter initializes with callable."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return f"Response to: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")

        assert adapter.model_id == "gpt2"

    def test_huggingface_adapter_generate(self):
        """HuggingFaceModelAdapter generates text with message formatting."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return f"Generated: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")
        result = adapter.generate("Test prompt")

        assert isinstance(result, str)
        # Without a tokenizer, the adapter formats messages as "user: content\nassistant:"
        assert "Generated:" in result

    def test_huggingface_adapter_default_generation_params(self):
        """HuggingFaceModelAdapter uses default generation parameters."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        captured_params = {}

        def mock_model(prompt, **kwargs):
            captured_params.update(kwargs)
            return "Response"

        adapter = HuggingFaceModelAdapter(
            model=mock_model,
            model_id="gpt2",
            default_generation_params={"max_length": 50, "temperature": 0.7},
        )

        adapter.generate("Test")

        assert "max_length" in captured_params
        assert captured_params["max_length"] == 50
        assert captured_params["temperature"] == 0.7

    def test_huggingface_adapter_fallback_without_kwargs(self):
        """HuggingFaceModelAdapter falls back to calling without kwargs."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt):
            # Only accepts prompt
            return f"Response: {prompt}"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="gpt2")
        result = adapter.generate("Test")

        # Should still work, just formats the prompt as messages
        assert "Response:" in result

    def test_huggingface_adapter_gather_config(self):
        """HuggingFaceModelAdapter config includes parameters."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt):
            return "Response"

        adapter = HuggingFaceModelAdapter(
            model=mock_model,
            model_id="gpt2",
            default_generation_params={"max_length": 100},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["max_length"] == 100
        assert "callable_type" in config

    def test_huggingface_adapter_gather_config_with_pipeline(self):
        """HuggingFaceModelAdapter config includes pipeline configuration."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        # Mock pipeline object with attributes
        class MockPipeline:
            def __init__(self):
                self.task = "text-generation"
                self.device = "cpu"
                self.framework = "pt"

            def __call__(self, prompt, **kwargs):
                return "Response"

        pipeline = MockPipeline()
        adapter = HuggingFaceModelAdapter(model=pipeline, model_id="gpt2")

        config = adapter.gather_config()

        assert "pipeline_config" in config
        assert config["pipeline_config"]["task"] == "text-generation"
        assert "cpu" in str(config["pipeline_config"]["device"])
        assert config["pipeline_config"]["framework"] == "pt"

    def test_huggingface_adapter_tools_raises_error_without_support(self):
        """HuggingFaceModelAdapter raises error when tools not supported."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import (
            HuggingFaceModelAdapter,
            ToolCallingNotSupportedError,
        )

        def mock_model(prompt, **kwargs):
            return "Response"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="test-model")

        with pytest.raises(ToolCallingNotSupportedError):
            adapter.chat(
                [{"role": "user", "content": "Test"}],
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

    def test_huggingface_adapter_tools_raises_when_template_doesnt_support(self):
        """HuggingFaceModelAdapter raises error when template doesn't support tools."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import (
            HuggingFaceModelAdapter,
            ToolCallingNotSupportedError,
        )

        class MockTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
                if "tools" in kwargs:
                    raise TypeError("Unexpected keyword argument 'tools'")
                return "Formatted prompt"

        class MockPipeline:
            tokenizer = MockTokenizer()

            def __call__(self, prompt, **kwargs):
                return "Response"

        adapter = HuggingFaceModelAdapter(model=MockPipeline(), model_id="test-model")

        with pytest.raises(ToolCallingNotSupportedError):
            adapter.chat(
                [{"role": "user", "content": "Test"}],
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

    def test_huggingface_adapter_chat_template_with_tools(self):
        """HuggingFaceModelAdapter works when template supports tools."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        class MockTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, tools=None, **kwargs):
                return "Formatted with tools"

        class MockPipeline:
            tokenizer = MockTokenizer()

            def __call__(self, prompt, **kwargs):
                return "Response"

        adapter = HuggingFaceModelAdapter(model=MockPipeline(), model_id="test-model")
        response = adapter.chat(
            [{"role": "user", "content": "Test"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        assert response is not None

    def test_huggingface_adapter_parses_tool_calls_from_output(self):
        """HuggingFaceModelAdapter parses tool calls from model output."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        class MockTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, tools=None, **kwargs):
                return "Prompt"

        class MockPipeline:
            tokenizer = MockTokenizer()

            def __call__(self, prompt, **kwargs):
                return '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'

        adapter = HuggingFaceModelAdapter(model=MockPipeline(), model_id="test-model")
        response = adapter.chat(
            [{"role": "user", "content": "Search"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) >= 1
        assert any(tc["function"]["name"] == "search" for tc in response.tool_calls)

    def test_huggingface_adapter_chat_with_tokenizer(self):
        """HuggingFaceModelAdapter uses chat template when available."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        class MockTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
                return "Formatted: " + messages[0]["content"]

        class MockPipeline:
            tokenizer = MockTokenizer()

            def __call__(self, prompt, **kwargs):
                return f"Response to: {prompt}"

        adapter = HuggingFaceModelAdapter(model=MockPipeline(), model_id="test-model")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content is not None

    def test_huggingface_adapter_pipeline_response_format(self):
        """HuggingFaceModelAdapter handles pipeline list response format."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return [{"generated_text": prompt + " Generated"}]

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="test-model")
        response = adapter.chat([{"role": "user", "content": "Test"}])

        assert response.content is not None
        assert "Generated" in response.content

    def test_huggingface_adapter_dict_response_format(self):
        """HuggingFaceModelAdapter handles dict response format."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        def mock_model(prompt, **kwargs):
            return {"generated_text": "Dict response"}

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="test-model")
        response = adapter.chat([{"role": "user", "content": "Test"}])

        assert response.content == "Dict response"

    def test_huggingface_adapter_nested_tokenizer(self):
        """HuggingFaceModelAdapter gets tokenizer from model.model.tokenizer."""
        pytest.importorskip("transformers")
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

        class MockTokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
                return "From nested tokenizer"

        class MockInnerModel:
            tokenizer = MockTokenizer()

        class MockPipeline:
            model = MockInnerModel()

            def __call__(self, prompt, **kwargs):
                return "Response"

        adapter = HuggingFaceModelAdapter(model=MockPipeline(), model_id="test-model")
        response = adapter.chat([{"role": "user", "content": "Test"}])

        assert response is not None


# ==================== LiteLLM Tests ====================


@pytest.mark.interface
class TestLiteLLMModelAdapterIntegration:
    """Test LiteLLMModelAdapter specific behavior."""

    def test_litellm_adapter_initialization(self):
        """LiteLLMModelAdapter initializes with model_id."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-3.5-turbo")

        assert adapter.model_id == "gpt-3.5-turbo"

    def test_litellm_adapter_initialization_with_params(self):
        """LiteLLMModelAdapter initializes with parameters."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="gpt-4",
            default_generation_params={"temperature": 0.7, "max_tokens": 100},
            api_key="test-key",
            api_base="https://test.com",
        )

        assert adapter.model_id == "gpt-4"
        assert adapter._default_generation_params["temperature"] == 0.7
        assert adapter._api_key == "test-key"
        assert adapter._api_base == "https://test.com"

    def test_litellm_adapter_gather_config(self):
        """LiteLLMModelAdapter config includes parameters."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="gpt-4",
            default_generation_params={"temperature": 0.9, "max_tokens": 200},
        )

        config = adapter.gather_config()

        assert "default_generation_params" in config
        assert config["default_generation_params"]["temperature"] == 0.9
        assert config["default_generation_params"]["max_tokens"] == 200
        assert config["model_id"] == "gpt-4"

    def test_litellm_adapter_tool_calls_response(self):
        """LiteLLMModelAdapter handles tool call responses."""
        pytest.importorskip("litellm")
        import litellm
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        class MockToolCall:
            id = "call_456"
            type = "function"

            class function:
                name = "calculator"
                arguments = '{"expression": "2+2"}'

        class MockMessage:
            content = None
            role = "assistant"
            tool_calls = [MockToolCall()]

        class MockChoice:
            message = MockMessage()
            finish_reason = "tool_calls"

        class MockUsage:
            prompt_tokens = 15
            completion_tokens = 8
            total_tokens = 23

        class MockResponse:
            choices = [MockChoice()]
            usage = MockUsage()
            model = "gpt-4"

        original = litellm.completion

        def mock_completion(model, messages, **kwargs):
            return MockResponse()

        litellm.completion = mock_completion

        try:
            adapter = LiteLLMModelAdapter(model_id="gpt-4")
            response = adapter.chat([{"role": "user", "content": "Calculate"}])

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "calculator"
            assert response.usage is not None
            assert response.usage["input_tokens"] == 15
            assert response.stop_reason == "tool_calls"
        finally:
            litellm.completion = original

    def test_litellm_adapter_tools_and_credentials_passing(self):
        """LiteLLMModelAdapter passes tools and credentials."""
        pytest.importorskip("litellm")
        import litellm
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        captured_kwargs = {}

        class MockMessage:
            content = "Response"
            role = "assistant"
            tool_calls = None

        class MockChoice:
            message = MockMessage()
            finish_reason = "stop"

        class MockResponse:
            choices = [MockChoice()]

        original = litellm.completion

        def mock_completion(model, messages, **kwargs):
            captured_kwargs.update(kwargs)
            return MockResponse()

        litellm.completion = mock_completion

        try:
            adapter = LiteLLMModelAdapter(
                model_id="gpt-4",
                api_key="test-key",
                api_base="https://test.api.com",
            )
            tools = [{"type": "function", "function": {"name": "test"}}]
            adapter.chat(
                [{"role": "user", "content": "Test"}],
                tools=tools,
                tool_choice="required",
            )

            assert captured_kwargs["api_key"] == "test-key"
            assert captured_kwargs["api_base"] == "https://test.api.com"
            assert captured_kwargs["tools"] == tools
            assert captured_kwargs["tool_choice"] == "required"
        finally:
            litellm.completion = original


# ==================== Anthropic Tests ====================


@pytest.mark.interface
class TestAnthropicModelAdapterIntegration:
    """Test AnthropicModelAdapter specific behavior."""

    def test_anthropic_adapter_initialization(self):
        """AnthropicModelAdapter initializes with client and model_id."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            pass

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        assert adapter.model_id == "claude-3"

    def test_anthropic_adapter_chat_basic(self):
        """AnthropicModelAdapter handles basic chat."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockTextBlock:
            type = "text"
            text = "Hello! How can I help?"

        class MockUsage:
            input_tokens = 10
            output_tokens = 8

        class MockResponse:
            content = [MockTextBlock()]
            usage = MockUsage()
            model = "claude-3"
            stop_reason = "end_turn"

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        response = adapter.chat([{"role": "user", "content": "Hello"}])

        assert response.content == "Hello! How can I help?"
        assert response.usage is not None
        assert response.usage["input_tokens"] == 10
        assert response.stop_reason == "end_turn"

    def test_anthropic_adapter_tool_use_response(self):
        """AnthropicModelAdapter handles tool use responses."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockToolUseBlock:
            type = "tool_use"
            id = "tool_123"
            name = "get_weather"
            input = {"city": "Paris"}

        class MockUsage:
            input_tokens = 15
            output_tokens = 12

        class MockResponse:
            content = [MockToolUseBlock()]
            usage = MockUsage()
            model = "claude-3"
            stop_reason = "tool_use"

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        response = adapter.chat([{"role": "user", "content": "Weather?"}])

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.stop_reason == "tool_use"

    def test_anthropic_adapter_system_message_extraction(self):
        """AnthropicModelAdapter extracts system message."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        captured_kwargs = {}

        class MockTextBlock:
            type = "text"
            text = "I'm helpful!"

        class MockResponse:
            content = [MockTextBlock()]

        class MockMessages:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        adapter.chat(
            [
                {"role": "system", "content": "You are very helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )

        assert captured_kwargs["system"] == "You are very helpful"
        assert all(m["role"] != "system" for m in captured_kwargs["messages"])

    def test_anthropic_adapter_tools_conversion(self):
        """AnthropicModelAdapter converts tools to Anthropic format."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        captured_kwargs = {}

        class MockTextBlock:
            type = "text"
            text = "Response"

        class MockResponse:
            content = [MockTextBlock()]

        class MockMessages:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        adapter.chat([{"role": "user", "content": "Search"}], tools=tools)

        assert "tools" in captured_kwargs
        assert captured_kwargs["tools"][0]["name"] == "search"
        assert "input_schema" in captured_kwargs["tools"][0]

    def test_anthropic_adapter_tool_choice_conversion(self):
        """AnthropicModelAdapter converts tool_choice options."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        captured_kwargs = {}

        class MockTextBlock:
            type = "text"
            text = "Response"

        class MockResponse:
            content = [MockTextBlock()]

        class MockMessages:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        tools = [{"type": "function", "function": {"name": "test"}}]

        # Test "required" -> "any"
        adapter.chat(
            [{"role": "user", "content": "Test"}],
            tools=tools,
            tool_choice="required",
        )
        assert captured_kwargs["tool_choice"]["type"] == "any"

        # Test specific function
        adapter.chat(
            [{"role": "user", "content": "Test"}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "test"}},
        )
        assert captured_kwargs["tool_choice"]["type"] == "tool"
        assert captured_kwargs["tool_choice"]["name"] == "test"

    def test_anthropic_adapter_tool_result_conversion(self):
        """AnthropicModelAdapter converts tool result messages."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        captured_kwargs = {}

        class MockTextBlock:
            type = "text"
            text = "Final answer"

        class MockResponse:
            content = [MockTextBlock()]

        class MockMessages:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        adapter.chat(
            [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tool_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tool_1", "content": "Sunny, 22°C"},
            ]
        )

        messages = captured_kwargs["messages"]
        tool_result_msg = [m for m in messages if m["role"] == "user" and isinstance(m.get("content"), list)]
        assert len(tool_result_msg) > 0

    def test_anthropic_adapter_mixed_content_response(self):
        """AnthropicModelAdapter handles mixed text and tool_use response."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockTextBlock:
            type = "text"
            text = "Let me check that for you."

        class MockToolUseBlock:
            type = "tool_use"
            id = "tool_456"
            name = "lookup"
            input = {"id": "123"}

        class MockUsage:
            input_tokens = 20
            output_tokens = 15

        class MockResponse:
            content = [MockTextBlock(), MockToolUseBlock()]
            usage = MockUsage()
            model = "claude-3"
            stop_reason = "tool_use"

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            messages = MockMessages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-3")
        response = adapter.chat([{"role": "user", "content": "Look up ID 123"}])

        assert response.content == "Let me check that for you."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "lookup"

    def test_anthropic_adapter_gather_config(self):
        """AnthropicModelAdapter config includes parameters."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            pass

        adapter = AnthropicModelAdapter(
            client=MockClient(),
            model_id="claude-3",
            max_tokens=2048,
            default_generation_params={"temperature": 0.8},
        )
        config = adapter.gather_config()

        assert config["model_id"] == "claude-3"
        assert config["max_tokens"] == 2048
        assert config["default_generation_params"]["temperature"] == 0.8
        assert config["client_type"] == "MockClient"


# ==================== Seed Propagation Tests ====================


@pytest.mark.interface
class TestModelAdapterSeedPropagation:
    """Tests verifying seeds are passed to underlying APIs.

    These tests verify the actual behavior: that seeds set on adapters
    are passed through to the underlying provider API calls.
    """

    def test_openai_adapter_passes_seed_to_api(self):
        """OpenAI adapter includes seed in API call."""
        pytest.importorskip("openai")
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4", seed=42)
        adapter.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("seed") == 42

    def test_openai_adapter_no_seed_when_not_set(self):
        """OpenAI adapter doesn't include seed when not set."""
        pytest.importorskip("openai")
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4")  # No seed
        adapter.chat([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "seed" not in call_kwargs

    def test_openai_user_seed_overrides_adapter_seed(self):
        """User-provided seed in generation_params takes precedence."""
        pytest.importorskip("openai")
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4", seed=42)
        adapter.chat(
            [{"role": "user", "content": "test"}],
            generation_params={"seed": 999},  # Should override adapter seed
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("seed") == 999  # User's seed, not adapter's

    def test_litellm_adapter_passes_seed_to_api(self):
        """LiteLLM adapter includes seed in API call."""
        pytest.importorskip("litellm")
        from unittest.mock import MagicMock
        from maseval.interface.inference import LiteLLMModelAdapter

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            adapter = LiteLLMModelAdapter(model_id="gpt-4", seed=42)
            adapter.chat([{"role": "user", "content": "test"}])

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get("seed") == 42

    def test_litellm_adapter_no_seed_when_not_set(self):
        """LiteLLM adapter doesn't include seed when not set."""
        pytest.importorskip("litellm")
        from unittest.mock import MagicMock
        from maseval.interface.inference import LiteLLMModelAdapter

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            adapter = LiteLLMModelAdapter(model_id="gpt-4")  # No seed
            adapter.chat([{"role": "user", "content": "test"}])

            call_kwargs = mock_completion.call_args[1]
            assert "seed" not in call_kwargs

    def test_litellm_user_seed_overrides_adapter_seed(self):
        """LiteLLM user-provided seed in generation_params takes precedence."""
        pytest.importorskip("litellm")
        from unittest.mock import MagicMock
        from maseval.interface.inference import LiteLLMModelAdapter

        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "test"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_response.model = "gpt-4"

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            adapter = LiteLLMModelAdapter(model_id="gpt-4", seed=42)
            adapter.chat(
                [{"role": "user", "content": "test"}],
                generation_params={"seed": 999},
            )

            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs.get("seed") == 999

    def test_google_adapter_passes_seed_to_config(self):
        """Google GenAI adapter includes seed in generation config."""
        pytest.importorskip("google.genai")
        import sys
        from unittest.mock import MagicMock
        from maseval.interface.inference import GoogleGenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.candidates = []
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        # Track what gets passed to GenerateContentConfig
        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        # Create mock genai module structure
        mock_genai = MagicMock()
        mock_genai.types.GenerateContentConfig = capture_config

        # Temporarily replace the google.genai module
        original_google = sys.modules.get("google")
        original_genai = sys.modules.get("google.genai")

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        sys.modules["google"] = mock_google
        sys.modules["google.genai"] = mock_genai

        try:
            adapter = GoogleGenAIModelAdapter(client=mock_client, model_id="gemini-pro", seed=42)
            adapter.chat([{"role": "user", "content": "test"}])

            # Verify seed was passed to GenerateContentConfig
            assert captured_config_kwargs.get("seed") == 42
        finally:
            # Restore original modules
            if original_google is not None:
                sys.modules["google"] = original_google
            elif "google" in sys.modules:
                del sys.modules["google"]
            if original_genai is not None:
                sys.modules["google.genai"] = original_genai
            elif "google.genai" in sys.modules:
                del sys.modules["google.genai"]

    def test_google_adapter_no_seed_when_not_set(self):
        """Google GenAI adapter doesn't include seed when not set."""
        pytest.importorskip("google.genai")
        import sys
        from unittest.mock import MagicMock
        from maseval.interface.inference import GoogleGenAIModelAdapter

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.candidates = []
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        # Track what gets passed to GenerateContentConfig
        captured_config_kwargs = {}

        def capture_config(**kwargs):
            captured_config_kwargs.update(kwargs)
            return MagicMock()

        # Create mock genai module structure
        mock_genai = MagicMock()
        mock_genai.types.GenerateContentConfig = capture_config

        # Temporarily replace the google.genai module
        original_google = sys.modules.get("google")
        original_genai = sys.modules.get("google.genai")

        mock_google = MagicMock()
        mock_google.genai = mock_genai
        sys.modules["google"] = mock_google
        sys.modules["google.genai"] = mock_genai

        try:
            adapter = GoogleGenAIModelAdapter(client=mock_client, model_id="gemini-pro")  # No seed
            adapter.chat([{"role": "user", "content": "test"}])

            # Verify seed was NOT passed
            assert "seed" not in captured_config_kwargs
        finally:
            # Restore original modules
            if original_google is not None:
                sys.modules["google"] = original_google
            elif "google" in sys.modules:
                del sys.modules["google"]
            if original_genai is not None:
                sys.modules["google.genai"] = original_genai
            elif "google.genai" in sys.modules:
                del sys.modules["google.genai"]


# ==================== Cross-Adapter Tests ====================


@pytest.mark.interface
class TestCrossAdapterConsistency:
    """Test that all adapters follow consistent patterns."""

    def test_all_adapters_expose_model_id(self):
        """All adapters expose model_id property."""
        pytest.importorskip("openai")
        pytest.importorskip("google.genai")
        pytest.importorskip("transformers")
        pytest.importorskip("litellm")

        from maseval.interface.inference.openai import OpenAIModelAdapter
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        # OpenAI - mock with modern interface
        class MockOpenAIClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "R"}}]}

                completions = Completions()

            chat = Chat()

        openai_adapter = OpenAIModelAdapter(client=MockOpenAIClient(), model_id="gpt-4")
        assert openai_adapter.model_id == "gpt-4"

        # Google GenAI
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "R"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        google_adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        assert google_adapter.model_id == "gemini-pro"

        # HuggingFace
        hf_adapter = HuggingFaceModelAdapter(model=lambda p: "R", model_id="gpt2")
        assert hf_adapter.model_id == "gpt2"

        # LiteLLM
        litellm_adapter = LiteLLMModelAdapter(model_id="gpt-3.5-turbo")
        assert litellm_adapter.model_id == "gpt-3.5-turbo"

    def test_all_adapters_include_default_params_in_config(self):
        """All adapters include default_generation_params in config."""
        pytest.importorskip("openai")
        pytest.importorskip("google.genai")
        pytest.importorskip("transformers")
        pytest.importorskip("litellm")

        from maseval.interface.inference.openai import OpenAIModelAdapter
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter
        from maseval.interface.inference.huggingface import HuggingFaceModelAdapter
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        params = {"temperature": 0.7}

        # OpenAI
        class MockOpenAIClient:
            class Chat:
                class Completions:
                    def create(self, model, messages, **kwargs):
                        return {"choices": [{"message": {"content": "R"}}]}

                completions = Completions()

            chat = Chat()

        openai_config = OpenAIModelAdapter(
            client=MockOpenAIClient(),
            model_id="gpt-4",
            default_generation_params=params,
        ).gather_config()
        assert "default_generation_params" in openai_config

        # Google GenAI
        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "R"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        google_config = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro", default_generation_params=params).gather_config()
        assert "default_generation_params" in google_config

        # HuggingFace
        hf_config = HuggingFaceModelAdapter(model=lambda p: "R", model_id="gpt2", default_generation_params=params).gather_config()
        assert "default_generation_params" in hf_config

        # LiteLLM
        litellm_config = LiteLLMModelAdapter(model_id="gpt-3.5-turbo", default_generation_params=params).gather_config()
        assert "default_generation_params" in litellm_config


# ==================== Google GenAI Message Conversion Tests ====================


@pytest.mark.interface
class TestGoogleGenAIMessageConversion:
    """Test GoogleGenAIModelAdapter._convert_messages for edge cases."""

    def _make_adapter(self):
        """Create an adapter with a no-op client for testing _convert_messages."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"
                        candidates = []
                        usage_metadata = None

                    return Response()

            def __init__(self):
                self.models = self.Models()

        return GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")

    def test_system_message_extracted(self):
        """System messages become system_instruction, not contents."""
        adapter = self._make_adapter()
        system, contents = adapter._convert_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert system == "You are helpful."
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_assistant_text_only(self):
        """Assistant messages map to 'model' role."""
        adapter = self._make_adapter()
        _, contents = adapter._convert_messages(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        )
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"] == [{"text": "Hello!"}]

    def test_assistant_with_tool_calls(self):
        """Assistant tool_calls are converted to function_call parts."""
        adapter = self._make_adapter()
        _, contents = adapter._convert_messages(
            [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
            ]
        )
        model_msg = contents[1]
        assert model_msg["role"] == "model"
        assert len(model_msg["parts"]) == 2
        assert model_msg["parts"][0] == {"text": "Let me check."}
        assert model_msg["parts"][1]["function_call"]["name"] == "get_weather"
        assert model_msg["parts"][1]["function_call"]["args"] == {"city": "Paris"}

    def test_assistant_with_invalid_json_arguments(self):
        """Invalid JSON in tool call arguments falls back to empty dict."""
        adapter = self._make_adapter()
        _, contents = adapter._convert_messages(
            [
                {"role": "user", "content": "Go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "do_thing",
                                "arguments": "not json at all",
                            },
                        }
                    ],
                },
            ]
        )
        fc = contents[1]["parts"][0]["function_call"]
        assert fc["args"] == {}

    def test_tool_response_messages(self):
        """Tool messages are converted to function_response parts."""
        adapter = self._make_adapter()
        _, contents = adapter._convert_messages(
            [
                {"role": "user", "content": "Weather?"},
                {"role": "tool", "name": "get_weather", "tool_call_id": "call_1", "content": "72°F"},
            ]
        )
        assert contents[1]["role"] == "function"
        fr = contents[1]["parts"][0]["function_response"]
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"result": "72°F"}

    def test_consecutive_tool_responses_merged(self):
        """Consecutive tool messages are merged into a single function contents entry."""
        adapter = self._make_adapter()
        _, contents = adapter._convert_messages(
            [
                {"role": "user", "content": "Do both"},
                {"role": "tool", "name": "tool_a", "tool_call_id": "c1", "content": "result_a"},
                {"role": "tool", "name": "tool_b", "tool_call_id": "c2", "content": "result_b"},
            ]
        )
        # Should be merged into one function entry
        assert len(contents) == 2  # user + merged function
        assert contents[1]["role"] == "function"
        assert len(contents[1]["parts"]) == 2


# ==================== OpenAI Edge Case Tests ====================


@pytest.mark.interface
class TestOpenAIAdapterEdgeCases:
    """Test OpenAI adapter edge cases for uncovered lines."""

    def test_parse_dict_response_no_choices(self):
        """Dict response without choices returns string representation."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"data": "some raw response"}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content is not None

    def test_parse_dict_response_completion_style(self):
        """Dict response with 'text' in choice (completion-style) is parsed."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"choices": [{"text": "Completion text", "finish_reason": "stop"}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content == "Completion text"

    def test_parse_dict_response_unknown_choice_format(self):
        """Dict response with neither 'message' nor 'text' falls back to str."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"choices": [{"unknown_key": "val"}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content is not None

    def test_callable_client_fallback(self):
        """Adapter falls back to calling client directly if no known methods exist."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class CallableClient:
            """Client without chat.completions but is callable."""

            def __call__(self, **kwargs):
                return {"choices": [{"message": {"content": "Direct call"}}]}

        adapter = OpenAIModelAdapter(client=CallableClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content == "Direct call"

    def test_non_callable_client_raises(self):
        """Adapter raises TypeError if client has no suitable method."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class BadClient:
            """Client with no usable methods and not callable."""

            pass

        adapter = OpenAIModelAdapter(client=BadClient(), model_id="gpt-4")
        with pytest.raises(TypeError, match="Unable to call client"):
            adapter.chat([{"role": "user", "content": "Hi"}])


# ==================== Anthropic Edge Case Tests ====================


@pytest.mark.interface
class TestAnthropicAdapterEdgeCases:
    """Test Anthropic adapter edge cases."""

    def test_convert_tool_choice_unknown_string(self):
        """Unknown tool_choice string defaults to auto."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            class Messages:
                def create(self, **kwargs):
                    class TextBlock:
                        type = "text"
                        text = "ok"

                    class Response:
                        content = [TextBlock()]
                        usage = None
                        model = "claude-sonnet-4-5-20250514"
                        stop_reason = "end_turn"

                    return Response()

            messages = Messages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-sonnet-4-5-20250514")
        result = adapter._convert_tool_choice("some_unknown_value")
        assert result == {"type": "auto"}

    def test_convert_messages_assistant_tool_calls_invalid_json(self):
        """Invalid JSON in assistant tool call arguments falls back to empty dict."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            class Messages:
                def create(self, **kwargs):
                    class TextBlock:
                        type = "text"
                        text = "ok"

                    class Response:
                        content = [TextBlock()]
                        usage = None
                        model = "claude-sonnet-4-5-20250514"
                        stop_reason = "end_turn"

                    return Response()

            messages = Messages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-sonnet-4-5-20250514")
        _, converted = adapter._convert_messages(
            [
                {"role": "user", "content": "Go"},
                {
                    "role": "assistant",
                    "content": "Using tool",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "do_thing",
                                "arguments": "not valid json {{{",
                            },
                        }
                    ],
                },
            ]
        )
        # Should have tool_use block with empty input dict
        assistant_msg = converted[1]
        tool_block = [b for b in assistant_msg["content"] if b["type"] == "tool_use"][0]
        assert tool_block["input"] == {}


# ==================== ModelAdapter.chat() with response_model Tests ====================


@pytest.mark.core
class TestModelAdapterStructuredChat:
    """Test that ModelAdapter.chat() routes to _structured_chat when response_model is given."""

    def test_chat_routes_to_structured_chat(self):
        """chat() with response_model calls _structured_chat, not _chat_impl."""
        from conftest import DummyModelAdapter
        from pydantic import BaseModel

        class SimpleResponse(BaseModel):
            answer: str

        model = DummyModelAdapter(responses=['{"answer": "42"}'])
        response = model.chat(
            [{"role": "user", "content": "What is 6*7?"}],
            response_model=SimpleResponse,
        )

        assert response.structured_response is not None
        assert isinstance(response.structured_response, SimpleResponse)
        assert response.structured_response.answer == "42"

    def test_chat_without_response_model_uses_chat_impl(self):
        """chat() without response_model uses _chat_impl directly."""
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=["plain text"])
        response = model.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "plain text"
        assert response.structured_response is None

    def test_chat_with_message_history(self):
        """chat() accepts MessageHistory and converts to list."""
        from conftest import DummyModelAdapter
        from maseval.core.history import MessageHistory

        history = MessageHistory()
        history.add_message("user", "Hello!")

        model = DummyModelAdapter(responses=["Hi there!"])
        response = model.chat(history)

        assert response.content == "Hi there!"

    def test_base_structured_chat_raises_not_implemented(self):
        """Base ModelAdapter._structured_chat raises NotImplementedError."""
        from maseval.core.model import ModelAdapter, ChatResponse

        class MinimalAdapter(ModelAdapter):
            @property
            def model_id(self):
                return "minimal"

            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                return ChatResponse(content="text")

        adapter = MinimalAdapter()
        with pytest.raises(NotImplementedError, match="does not support response_model"):
            adapter.chat(
                [{"role": "user", "content": "Hi"}],
                response_model=object,
            )


# ==================== Adapter _structured_chat Tests ====================
#
# These tests inject a mock _instructor_client to exercise the adapter's
# _structured_chat code path without requiring real API credentials.
# This covers parameter merging, seed propagation, and response wrapping.
# =========================================================================


def _make_mock_instructor_result():
    """Create a mock Pydantic-like result object."""
    result = MagicMock()
    result.model_dump_json.return_value = '{"answer": "42"}'
    return result


@pytest.mark.interface
class TestOpenAIStructuredChat:
    """Test OpenAIModelAdapter._structured_chat with mocked instructor."""

    def test_structured_chat_returns_chat_response(self):
        """_structured_chat returns ChatResponse with structured_response."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"choices": [{"message": {"content": "ok"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4")

        # Inject mock instructor client
        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        response = adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
            generation_params={"temperature": 0.5},
        )

        assert response.content == '{"answer": "42"}'
        assert response.structured_response is mock_result
        assert response.model == "gpt-4"
        mock_instructor.chat.completions.create.assert_called_once()

    def test_structured_chat_propagates_seed(self):
        """_structured_chat includes seed in params."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class MockClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        return {"choices": [{"message": {"content": "ok"}}]}

                completions = Completions()

            chat = Chat()

        adapter = OpenAIModelAdapter(client=MockClient(), model_id="gpt-4", seed=42)

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("seed") == 42


@pytest.mark.interface
class TestAnthropicStructuredChat:
    """Test AnthropicModelAdapter._structured_chat with mocked instructor."""

    def test_structured_chat_returns_chat_response(self):
        """_structured_chat returns ChatResponse with structured_response."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            class Messages:
                def create(self, **kwargs):
                    class TextBlock:
                        type = "text"
                        text = "ok"

                    class Response:
                        content = [TextBlock()]
                        usage = None
                        model = "claude-sonnet-4-5-20250514"
                        stop_reason = "end_turn"

                    return Response()

            messages = Messages()

        adapter = AnthropicModelAdapter(client=MockClient(), model_id="claude-sonnet-4-5-20250514")

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        response = adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
            generation_params={"temperature": 0.5},
        )

        assert response.content == '{"answer": "42"}'
        assert response.structured_response is mock_result
        assert response.model == "claude-sonnet-4-5-20250514"

    def test_structured_chat_uses_max_tokens(self):
        """_structured_chat passes max_tokens from adapter default."""
        pytest.importorskip("anthropic")
        from maseval.interface.inference.anthropic import AnthropicModelAdapter

        class MockClient:
            class Messages:
                def create(self, **kwargs):
                    class TextBlock:
                        type = "text"
                        text = "ok"

                    class Response:
                        content = [TextBlock()]
                        usage = None
                        model = "claude-sonnet-4-5-20250514"
                        stop_reason = "end_turn"

                    return Response()

            messages = Messages()

        adapter = AnthropicModelAdapter(
            client=MockClient(),
            model_id="claude-sonnet-4-5-20250514",
            max_tokens=500,
        )

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 500


@pytest.mark.interface
class TestGoogleGenAIStructuredChat:
    """Test GoogleGenAIModelAdapter._structured_chat with mocked instructor."""

    def test_structured_chat_returns_chat_response(self):
        """_structured_chat returns ChatResponse with structured_response."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        response = adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
            generation_params={"temperature": 0.5},
        )

        assert response.content == '{"answer": "42"}'
        assert response.structured_response is mock_result
        assert response.model == "gemini-pro"

    def test_structured_chat_propagates_seed(self):
        """_structured_chat includes seed in params."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro", seed=99)

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args
        # Generation params should be nested inside generation_config for the
        # Google GenAI instructor adapter (not passed as top-level kwargs).
        gen_config = call_kwargs.kwargs.get("generation_config", {})
        assert gen_config.get("seed") == 99

    def test_structured_chat_separates_instructor_top_level_keys(self):
        """thinking_config stays top-level, generation params go into generation_config."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro", seed=42)

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
            generation_params={"temperature": 0.5, "thinking_config": {"thinking_budget": 1024}},
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        # thinking_config must be top-level (instructor pops it from kwargs directly)
        assert call_kwargs.get("thinking_config") == {"thinking_budget": 1024}
        # generation params must be nested inside generation_config
        gen_config = call_kwargs.get("generation_config", {})
        assert gen_config.get("temperature") == 0.5
        assert gen_config.get("seed") == 42
        # thinking_config must NOT be in generation_config
        assert "thinking_config" not in gen_config

    def test_structured_chat_uses_genai_structured_outputs_mode(self):
        """Instructor client uses GENAI_STRUCTURED_OUTPUTS mode.

        GENAI_TOOLS has upstream bugs with thinking mode (AttributeError on
        MALFORMED_FUNCTION_CALL, AssertionError on duplicate parts).
        GENAI_STRUCTURED_OUTPUTS parses via completion.text, avoiding both.
        """
        pytest.importorskip("google.genai")
        pytest.importorskip("instructor")
        import instructor
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")
        assert adapter._instructor_client is None

        with patch("instructor.from_genai") as mock_from_genai:
            mock_instructor = MagicMock()
            mock_instructor.chat.completions.create.return_value = _make_mock_instructor_result()
            mock_from_genai.return_value = mock_instructor

            adapter._structured_chat(
                messages=[{"role": "user", "content": "Hi"}],
                response_model=object,
            )

            mock_from_genai.assert_called_once()
            call_kwargs = mock_from_genai.call_args
            assert call_kwargs.kwargs.get("mode") == instructor.Mode.GENAI_STRUCTURED_OUTPUTS

    def test_clean_schema_model_strips_additional_properties(self):
        """_clean_schema_model strips additionalProperties from JSON schema.

        Pydantic v2 emits additionalProperties for Dict fields and nested models.
        Gemini's structured output API rejects it. The clean model must strip it
        while preserving parsing and isinstance() behavior.
        """
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        from typing import Any, Dict

        from pydantic import BaseModel, Field

        class Inner(BaseModel):
            name: str

        class Outer(BaseModel):
            text: str = Field(default="")
            details: Dict[str, Any] = Field(default_factory=dict)
            inner: Inner = Field(default_factory=lambda: Inner(name=""))

        # Original schema has additionalProperties
        original_schema = Outer.model_json_schema()
        assert any("additionalProperties" in str(v) for v in original_schema.values()) or "additionalProperties" in str(
            original_schema.get("$defs", {})
        )

        # Cleaned model's schema does not
        CleanOuter = GoogleGenAIModelAdapter._clean_schema_model(Outer)
        clean_schema = CleanOuter.model_json_schema()  # ty: ignore[unresolved-attribute]
        assert "additionalProperties" not in str(clean_schema)

        # Parsing still works and returns instances of the original class
        instance = CleanOuter.model_validate({"text": "hi", "details": {"k": "v"}, "inner": {"name": "test"}})  # ty: ignore[unresolved-attribute]
        assert isinstance(instance, Outer)
        assert instance.text == "hi"
        assert instance.details == {"k": "v"}
        assert instance.inner.name == "test"

    def test_structured_chat_passes_clean_model_to_instructor(self):
        """_structured_chat wraps the response model to strip additionalProperties."""
        pytest.importorskip("google.genai")
        from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

        from typing import Any, Dict

        from pydantic import BaseModel, Field

        class ToolOutput(BaseModel):
            text: str = Field(default="", description="Description")
            details: Dict[str, Any] = Field(default_factory=dict, description="Structured data")

        class MockClient:
            class Models:
                def generate_content(self, model, contents, config=None):
                    class Response:
                        text = "ok"

                    return Response()

            def __init__(self):
                self.models = self.Models()

        adapter = GoogleGenAIModelAdapter(client=MockClient(), model_id="gemini-pro")

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=ToolOutput,
            generation_params={"temperature": 0.5},
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args.kwargs
        passed_model = call_kwargs["response_model"]
        # Must be a subclass (not the original) with clean schema
        assert passed_model is not ToolOutput
        assert issubclass(passed_model, ToolOutput)
        assert "additionalProperties" not in str(passed_model.model_json_schema())
        # Generation params still wrapped correctly
        gen_config = call_kwargs.get("generation_config", {})
        assert gen_config.get("temperature") == 0.5


@pytest.mark.interface
class TestLiteLLMStructuredChat:
    """Test LiteLLMModelAdapter._structured_chat with mocked instructor."""

    def test_structured_chat_returns_chat_response(self):
        """_structured_chat returns ChatResponse with structured_response."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result

        # Pre-inject to skip _get_instructor_client
        adapter._instructor_client = mock_instructor

        response = adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
            generation_params={"temperature": 0.7},
        )

        assert response.content == '{"answer": "42"}'
        assert response.structured_response is mock_result
        assert response.model == "gpt-4o-mini"

    def test_structured_chat_propagates_api_credentials(self):
        """_structured_chat passes api_key and api_base."""
        pytest.importorskip("litellm")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(
            model_id="gpt-4o-mini",
            api_key="test-key",
            api_base="https://custom.api/v1",
            seed=77,
        )

        mock_result = _make_mock_instructor_result()
        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = mock_result
        adapter._instructor_client = mock_instructor

        adapter._structured_chat(
            messages=[{"role": "user", "content": "Hi"}],
            response_model=object,
        )

        call_kwargs = mock_instructor.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("api_key") == "test-key"
        assert call_kwargs.kwargs.get("api_base") == "https://custom.api/v1"
        assert call_kwargs.kwargs.get("seed") == 77

    def test_get_instructor_client_creates_patched_client(self):
        """_get_instructor_client lazily creates instructor client from litellm."""
        pytest.importorskip("litellm")
        pytest.importorskip("instructor")
        from maseval.interface.inference.litellm import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-4o-mini")
        client = adapter._get_instructor_client()

        assert client is not None
        assert hasattr(client, "chat")
        # Calling again returns same instance (lazy caching)
        assert adapter._get_instructor_client() is client


# ==================== OpenAI Legacy Client Fallback Tests ====================


@pytest.mark.interface
class TestOpenAILegacyClientFallback:
    """Test OpenAI adapter legacy client method fallbacks."""

    def test_legacy_client_with_create_method(self):
        """Adapter uses client.create() if no chat.completions."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class LegacyClient:
            def create(self, model, messages, **kwargs):
                return {"choices": [{"message": {"content": "Legacy create"}}]}

        adapter = OpenAIModelAdapter(client=LegacyClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content == "Legacy create"

    def test_legacy_client_without_model_param(self):
        """Adapter retries without model param on TypeError."""
        pytest.importorskip("openai")
        from maseval.interface.inference.openai import OpenAIModelAdapter

        class LegacyClient:
            def create(self, messages, **kwargs):
                # Accepts messages but not model
                return {"choices": [{"message": {"content": "No model param"}}]}

        adapter = OpenAIModelAdapter(client=LegacyClient(), model_id="gpt-4")
        response = adapter.chat([{"role": "user", "content": "Hi"}])
        assert response.content == "No model param"
