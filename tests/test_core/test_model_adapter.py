"""Test ModelAdapter functionality.

These tests verify the core contracts of the ModelAdapter base class:
- Call logging and tracing infrastructure
- Error handling and logging
- Timing capture
- Trace and config structure
- JSON serializability
- Base mixin functionality (TraceableMixin, ConfigurableMixin)

All tests use DummyModelAdapter from conftest.py to avoid external dependencies.
"""

import pytest
import json
import time
from datetime import datetime
from conftest import DummyModelAdapter
from maseval.core.model import ChatResponse


@pytest.mark.core
class TestModelAdapterBaseContract:
    """Test fundamental ModelAdapter base class behavior."""

    def test_model_adapter_has_abstract_methods(self):
        """ModelAdapter requires subclasses to implement model_id and _chat_impl."""
        from maseval.core.model import ModelAdapter

        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelAdapter()  # type: ignore

    def test_model_adapter_requires_model_id_property(self):
        """Subclasses must implement model_id property."""
        from maseval.core.model import ModelAdapter

        class IncompleteAdapter(ModelAdapter):
            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                return ChatResponse(content="test")

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_model_adapter_requires_chat_impl(self):
        """Subclasses must implement _chat_impl method."""
        from maseval.core.model import ModelAdapter

        class IncompleteAdapter(ModelAdapter):
            @property
            def model_id(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_model_adapter_initializes_logs(self):
        """ModelAdapter initializes empty logs on construction."""
        model = DummyModelAdapter()
        assert hasattr(model, "logs")
        assert isinstance(model.logs, list)
        assert len(model.logs) == 0

    def test_model_adapter_model_id_accessible(self):
        """model_id property is accessible."""
        model = DummyModelAdapter(model_id="test-model-id")
        assert model.model_id == "test-model-id"


@pytest.mark.core
class TestModelAdapterGenerationContract:
    """Test generation method behavior and logging."""

    def test_generate_returns_string(self):
        """generate() returns string output."""
        model = DummyModelAdapter(responses=["Test response"])
        result = model.generate("Test prompt")

        assert isinstance(result, str)
        assert result == "Test response"

    def test_generate_logs_successful_calls(self, dummy_model):
        """generate() logs successful calls with metadata."""
        dummy_model.generate("Test prompt 1")
        dummy_model.generate("Test prompt 2")

        # Check internal logs
        assert len(dummy_model.logs) == 2
        assert dummy_model.logs[0]["status"] == "success"
        assert dummy_model.logs[1]["status"] == "success"

        # Verify required fields
        call = dummy_model.logs[0]
        assert "timestamp" in call
        assert "message_count" in call
        assert "response_length" in call
        assert "duration_seconds" in call
        assert "status" in call
        assert "generation_params" in call
        assert "kwargs" in call

    def test_generate_logs_generation_params(self):
        """generate() logs generation parameters."""
        model = DummyModelAdapter()
        params = {"temperature": 0.7, "max_tokens": 100}
        model.generate("Test", generation_params=params)

        call = model.logs[0]
        assert call["generation_params"] == params

    def test_generate_logs_kwargs(self):
        """generate() logs additional kwargs."""
        model = DummyModelAdapter()
        model.generate("Test", custom_arg="value", another_arg=123)

        call = model.logs[0]
        assert "custom_arg" in call["kwargs"]
        assert "another_arg" in call["kwargs"]
        # kwargs are stringified for JSON serialization
        assert call["kwargs"]["custom_arg"] == "value"
        assert call["kwargs"]["another_arg"] == "123"

    def test_model_adapter_log_accumulation(self, dummy_model):
        """Test that logs accumulate across multiple calls."""
        for i in range(5):
            dummy_model.generate(f"Prompt {i}")

        traces = dummy_model.gather_traces()
        assert traces["total_calls"] == 5
        assert traces["successful_calls"] == 5
        assert len(traces["logs"]) == 5

    def test_generate_captures_timing(self):
        """generate() captures execution duration."""
        model = DummyModelAdapter()
        model.generate("Test")

        call = model.logs[0]
        assert "duration_seconds" in call
        assert isinstance(call["duration_seconds"], (int, float))
        assert call["duration_seconds"] >= 0

    def test_generate_with_empty_prompt(self):
        """generate() handles empty prompt."""
        model = DummyModelAdapter(responses=["Response"])
        result = model.generate("")

        assert isinstance(result, str)
        assert len(model.logs) == 1
        # Empty prompt creates one message
        assert model.logs[0]["message_count"] == 1


@pytest.mark.core
class TestModelAdapterChatContract:
    """Test chat() method behavior."""

    def test_chat_returns_chat_response(self):
        """chat() returns a ChatResponse object."""
        model = DummyModelAdapter(responses=["Test response"])
        result = model.chat([{"role": "user", "content": "Hello"}])

        assert isinstance(result, ChatResponse)
        assert result.content == "Test response"
        assert result.role == "assistant"

    def test_chat_with_multiple_messages(self):
        """chat() accepts multiple messages."""
        model = DummyModelAdapter(responses=["Response"])
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = model.chat(messages)

        assert isinstance(result, ChatResponse)
        assert model.logs[0]["message_count"] == 2

    def test_chat_response_to_message(self):
        """ChatResponse.to_message() returns dict."""
        model = DummyModelAdapter(responses=["Hello!"])
        result = model.chat([{"role": "user", "content": "Hi"}])

        message = result.to_message()
        assert isinstance(message, dict)
        assert message["role"] == "assistant"
        assert message["content"] == "Hello!"

    def test_chat_with_tool_calls(self):
        """chat() returns tool_calls when provided."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]
        model = DummyModelAdapter(responses=[""], tool_calls=[tool_calls])
        result = model.chat([{"role": "user", "content": "Weather?"}])

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "get_weather"


@pytest.mark.core
class TestModelAdapterErrorHandling:
    """Test error handling and error logging."""

    def test_model_adapter_error_handling(self, dummy_model):
        """Test that errors are logged correctly."""

        class FailingModel(DummyModelAdapter):
            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                raise ValueError("Test error")

        model = FailingModel()

        with pytest.raises(ValueError):
            model.generate("Test")

        # Error should be logged
        assert len(model.logs) == 1
        assert model.logs[0]["status"] == "error"
        assert "Test error" in model.logs[0]["error"]

    def test_generate_logs_error_timing(self):
        """generate() logs duration even when errors occur."""

        class FailingModel(DummyModelAdapter):
            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                time.sleep(0.01)  # Small delay
                raise RuntimeError("Fail")

        model = FailingModel()

        with pytest.raises(RuntimeError):
            model.generate("Test")

        call = model.logs[0]
        assert call["duration_seconds"] >= 0.01

    def test_generate_logs_error_metadata(self):
        """generate() logs message count and params even on error."""

        class FailingModel(DummyModelAdapter):
            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                raise Exception("Fail")

        model = FailingModel()
        params = {"temperature": 0.9}

        with pytest.raises(Exception):
            model.generate("Test prompt", generation_params=params, custom="arg")

        call = model.logs[0]
        assert call["message_count"] == 1
        assert call["generation_params"] == params
        assert "custom" in call["kwargs"]

    def test_generate_reraises_original_exception(self):
        """generate() re-raises the original exception type."""

        class CustomError(Exception):
            pass

        class FailingModel(DummyModelAdapter):
            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                raise CustomError("Original error")

        model = FailingModel()

        with pytest.raises(CustomError, match="Original error"):
            model.generate("Test")


@pytest.mark.core
class TestModelAdapterTracing:
    """Test gather_traces() contract."""

    def test_gather_traces_returns_dict(self):
        """gather_traces() returns a dictionary."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        assert isinstance(traces, dict)

    def test_model_adapter_gather_traces_structure(self, dummy_model):
        """Test that gather_traces() has correct structure."""
        dummy_model.generate("Test prompt")

        traces = dummy_model.gather_traces()

        # Base fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces
        # ModelAdapter-specific fields
        assert "model_id" in traces
        assert "total_calls" in traces
        assert "successful_calls" in traces
        assert "failed_calls" in traces
        assert "logs" in traces

    def test_gather_traces_has_base_fields(self):
        """gather_traces() includes base TraceableMixin fields."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        # Base fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces

        assert traces["type"] == "DummyModelAdapter"
        assert isinstance(traces["gathered_at"], str)

        # Validate timestamp format
        datetime.fromisoformat(traces["gathered_at"])

    def test_gather_traces_aggregates_statistics(self):
        """gather_traces() correctly aggregates call statistics."""
        model = DummyModelAdapter(responses=["R1", "R2", "R3"])

        for i in range(3):
            model.generate(f"Prompt {i}")

        traces = model.gather_traces()

        assert traces["total_calls"] == 3
        assert traces["successful_calls"] == 3
        assert traces["failed_calls"] == 0
        assert traces["total_duration_seconds"] >= 0
        assert traces["average_duration_seconds"] >= 0

    def test_gather_traces_counts_failures(self):
        """gather_traces() counts failed calls correctly."""

        class SometimesFailingModel(DummyModelAdapter):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def _chat_impl(self, messages, generation_params=None, tools=None, tool_choice=None, **kwargs):
                self.call_count += 1
                if self.call_count % 2 == 0:
                    raise ValueError("Fail")
                return ChatResponse(content="Success")

        model = SometimesFailingModel()

        # Call 1: success
        model.generate("Test 1")
        # Call 2: fail
        with pytest.raises(ValueError):
            model.generate("Test 2")
        # Call 3: success
        model.generate("Test 3")

        traces = model.gather_traces()

        assert traces["total_calls"] == 3
        assert traces["successful_calls"] == 2
        assert traces["failed_calls"] == 1

    def test_model_adapter_timing_captured(self, dummy_model):
        """Test that timing information is captured."""
        dummy_model.generate("Test")

        traces = dummy_model.gather_traces()
        assert "total_duration_seconds" in traces
        assert "average_duration_seconds" in traces
        assert traces["total_duration_seconds"] >= 0

    def test_gather_traces_handles_zero_calls(self):
        """gather_traces() handles models with no calls."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        assert traces["total_calls"] == 0
        assert traces["successful_calls"] == 0
        assert traces["failed_calls"] == 0
        assert traces["total_duration_seconds"] == 0
        assert traces["average_duration_seconds"] == 0
        assert traces["logs"] == []

    def test_gather_traces_is_json_serializable(self):
        """gather_traces() returns JSON-serializable data."""
        model = DummyModelAdapter()
        model.generate("Test", generation_params={"temp": 0.5}, custom_kwarg="value")

        traces = model.gather_traces()

        # Should not raise
        json_str = json.dumps(traces)
        assert isinstance(json_str, str)

        # Should round-trip
        recovered = json.loads(json_str)
        assert recovered["total_calls"] == traces["total_calls"]

    def test_gather_traces_is_idempotent(self):
        """gather_traces() can be called multiple times."""
        model = DummyModelAdapter()
        model.generate("Test")

        traces1 = model.gather_traces()
        traces2 = model.gather_traces()

        # Type and model_id should be identical
        assert traces1["type"] == traces2["type"]
        assert traces1["model_id"] == traces2["model_id"]
        assert traces1["total_calls"] == traces2["total_calls"]

        # Timestamps might differ
        assert "gathered_at" in traces1
        assert "gathered_at" in traces2


@pytest.mark.core
class TestModelAdapterConfiguration:
    """Test gather_config() contract."""

    def test_gather_config_returns_dict(self):
        """gather_config() returns a dictionary."""
        model = DummyModelAdapter()
        config = model.gather_config()

        assert isinstance(config, dict)

    def test_model_adapter_gather_config(self, dummy_model):
        """Test that gather_config() returns configuration."""
        config = dummy_model.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert config["type"] == "DummyModelAdapter"

    def test_gather_config_has_base_fields(self):
        """gather_config() includes base ConfigurableMixin fields."""
        model = DummyModelAdapter()
        config = model.gather_config()

        # Base fields from ConfigurableMixin
        assert "type" in config
        assert "gathered_at" in config

        assert config["type"] == "DummyModelAdapter"
        assert isinstance(config["gathered_at"], str)

        # Validate timestamp format
        datetime.fromisoformat(config["gathered_at"])

    def test_gather_config_has_model_fields(self):
        """gather_config() includes ModelAdapter-specific fields."""
        model = DummyModelAdapter(model_id="test-model")
        config = model.gather_config()

        # ModelAdapter-specific fields
        assert "model_id" in config
        assert "adapter_type" in config

        assert config["model_id"] == "test-model"
        assert config["adapter_type"] == "DummyModelAdapter"

    def test_gather_config_is_static(self):
        """gather_config() returns same data before and after execution."""
        model = DummyModelAdapter()

        config_before = model.gather_config()
        model.generate("Test")
        config_after = model.gather_config()

        # Remove timestamps for comparison
        def remove_timestamp(d):
            return {k: v for k, v in d.items() if k != "gathered_at"}

        assert remove_timestamp(config_before) == remove_timestamp(config_after)

    def test_gather_config_is_json_serializable(self):
        """gather_config() returns JSON-serializable data."""
        model = DummyModelAdapter(model_id="test-model")
        config = model.gather_config()

        # Should not raise
        json_str = json.dumps(config)
        assert isinstance(json_str, str)

        # Should round-trip
        recovered = json.loads(json_str)
        assert recovered["model_id"] == config["model_id"]

    def test_gather_config_is_idempotent(self):
        """gather_config() can be called multiple times."""
        model = DummyModelAdapter()

        config1 = model.gather_config()
        config2 = model.gather_config()

        # Type and model_id should be identical
        assert config1["type"] == config2["type"]
        assert config1["model_id"] == config2["model_id"]

        # Timestamps might differ
        assert "gathered_at" in config1
        assert "gathered_at" in config2


@pytest.mark.core
class TestModelAdapterMixinIntegration:
    """Test that ModelAdapter correctly inherits from TraceableMixin and ConfigurableMixin."""

    def test_model_adapter_is_traceable(self):
        """ModelAdapter inherits from TraceableMixin."""
        from maseval.core.model import ModelAdapter
        from maseval.core.tracing import TraceableMixin

        assert issubclass(ModelAdapter, TraceableMixin)

        model = DummyModelAdapter()
        assert isinstance(model, TraceableMixin)

    def test_model_adapter_is_configurable(self):
        """ModelAdapter inherits from ConfigurableMixin."""
        from maseval.core.model import ModelAdapter
        from maseval.core.config import ConfigurableMixin

        assert issubclass(ModelAdapter, ConfigurableMixin)

        model = DummyModelAdapter()
        assert isinstance(model, ConfigurableMixin)

    def test_gather_traces_includes_mixin_fields(self):
        """gather_traces() includes fields from TraceableMixin."""
        model = DummyModelAdapter()
        traces = model.gather_traces()

        # Fields from TraceableMixin
        assert "type" in traces
        assert "gathered_at" in traces

    def test_gather_config_includes_mixin_fields(self):
        """gather_config() includes fields from ConfigurableMixin."""
        model = DummyModelAdapter()
        config = model.gather_config()

        # Fields from ConfigurableMixin
        assert "type" in config
        assert "gathered_at" in config


@pytest.mark.core
class TestModelAdapterSeeding:
    """Test ModelAdapter seeding functionality."""

    def test_model_adapter_accepts_seed(self):
        """ModelAdapter accepts seed parameter."""
        model = DummyModelAdapter(seed=42)
        assert model.seed == 42

    def test_model_adapter_seed_none_by_default(self):
        """ModelAdapter seed is None by default."""
        model = DummyModelAdapter()
        assert model.seed is None

    def test_model_adapter_seed_in_config(self):
        """ModelAdapter seed appears in gather_config()."""
        model = DummyModelAdapter(seed=12345)
        config = model.gather_config()

        assert "seed" in config
        assert config["seed"] == 12345

    def test_model_adapter_seed_none_in_config_when_not_set(self):
        """ModelAdapter seed is None in config when not set."""
        model = DummyModelAdapter()
        config = model.gather_config()

        assert "seed" in config
        assert config["seed"] is None

    def test_anthropic_adapter_rejects_seed(self):
        """AnthropicModelAdapter raises SeedingError when seed is provided."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import AnthropicModelAdapter
        from maseval.core.seeding import SeedingError

        mock_client = MagicMock()

        with pytest.raises(SeedingError, match="Anthropic models do not support seeding"):
            AnthropicModelAdapter(client=mock_client, model_id="claude-3", seed=42)

    def test_anthropic_adapter_works_without_seed(self):
        """AnthropicModelAdapter works fine when no seed is provided."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import AnthropicModelAdapter

        mock_client = MagicMock()

        # Should not raise
        adapter = AnthropicModelAdapter(client=mock_client, model_id="claude-3")
        assert adapter.seed is None

    def test_openai_adapter_accepts_seed(self):
        """OpenAIModelAdapter accepts seed parameter."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import OpenAIModelAdapter

        mock_client = MagicMock()
        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4", seed=42)

        assert adapter.seed == 42

    def test_google_adapter_accepts_seed(self):
        """GoogleGenAIModelAdapter accepts seed parameter."""
        from unittest.mock import MagicMock
        from maseval.interface.inference import GoogleGenAIModelAdapter

        mock_client = MagicMock()
        adapter = GoogleGenAIModelAdapter(client=mock_client, model_id="gemini-pro", seed=42)

        assert adapter.seed == 42

    def test_litellm_adapter_accepts_seed(self):
        """LiteLLMModelAdapter accepts seed parameter."""
        from maseval.interface.inference import LiteLLMModelAdapter

        adapter = LiteLLMModelAdapter(model_id="gpt-4", seed=42)

        assert adapter.seed == 42

    def test_huggingface_adapter_accepts_seed(self):
        """HuggingFaceModelAdapter accepts seed parameter."""
        from maseval.interface.inference import HuggingFaceModelAdapter

        def mock_model(x):
            return "response"

        adapter = HuggingFaceModelAdapter(model=mock_model, model_id="llama", seed=42)

        assert adapter.seed == 42
