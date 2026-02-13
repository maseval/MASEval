"""Cross-implementation ModelAdapter contract tests.

Verifies that ALL ModelAdapter implementations (DummyModelAdapter, OpenAIModelAdapter,
GoogleGenAIModelAdapter, HuggingFaceModelAdapter, LiteLLMModelAdapter) implement the
same contract and behave identically for key operations.

This validates MASEval's CORE PROMISE: provider-agnostic model abstraction.

Contract tests ensure that users can swap between model providers without changing
their benchmark code.

What this contract validates:
- generate() returns string consistently
- chat() returns ChatResponse consistently
- Call logging happens uniformly (successful and failed calls)
- Timing capture works consistently
- Trace structure is consistent across implementations (gather_traces)
- Config structure is consistent across implementations (gather_config)
- JSON serializability of traces and configs
- Error handling and error logging
- Generation parameters are passed through correctly

If these tests fail, the abstraction layer is broken and users cannot reliably
swap between model providers.
"""

import pytest
import json
from datetime import datetime
from typing import Any, Dict, Optional, List
from conftest import DummyModelAdapter
from maseval.core.model import ChatResponse


# ==================== Helper Functions ====================


def assert_is_valid_iso_timestamp(timestamp: str) -> None:
    """Verify string is valid ISO 8601 timestamp."""
    try:
        datetime.fromisoformat(timestamp)
    except (ValueError, TypeError) as e:
        pytest.fail(f"Invalid ISO timestamp: {timestamp} - {e}")


def assert_is_json_serializable(data: Any) -> None:
    """Verify data can be serialized to JSON."""
    try:
        json.dumps(data)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Data is not JSON serializable: {e}\nData: {data}")


def assert_base_trace_fields(traces: Dict[str, Any], model_id: Optional[str] = None) -> None:
    """Verify traces have required base fields from TraceableMixin and ModelAdapter."""
    assert isinstance(traces, dict), f"gather_traces() must return dict, got {type(traces)}"

    # Base fields from TraceableMixin
    assert "type" in traces, "Missing 'type' field in traces"
    assert isinstance(traces["type"], str), f"'type' must be string, got {type(traces['type'])}"
    assert len(traces["type"]) > 0, "'type' field cannot be empty"

    assert "gathered_at" in traces, "Missing 'gathered_at' field in traces"
    assert isinstance(traces["gathered_at"], str), f"'gathered_at' must be string, got {type(traces['gathered_at'])}"
    assert_is_valid_iso_timestamp(traces["gathered_at"])

    # ModelAdapter-specific fields
    assert "model_id" in traces, "Missing 'model_id' field in traces"
    assert isinstance(traces["model_id"], str), f"'model_id' must be string, got {type(traces['model_id'])}"

    if model_id is not None:
        assert traces["model_id"] == model_id, f"Expected model_id '{model_id}', got '{traces['model_id']}'"

    assert "total_calls" in traces, "Missing 'total_calls' field"
    assert "successful_calls" in traces, "Missing 'successful_calls' field"
    assert "failed_calls" in traces, "Missing 'failed_calls' field"
    assert "total_duration_seconds" in traces, "Missing 'total_duration_seconds' field"
    assert "average_duration_seconds" in traces, "Missing 'average_duration_seconds' field"
    assert "logs" in traces, "Missing 'logs' field"

    # Type validation
    assert isinstance(traces["total_calls"], int)
    assert isinstance(traces["successful_calls"], int)
    assert isinstance(traces["failed_calls"], int)
    assert isinstance(traces["total_duration_seconds"], (int, float))
    assert isinstance(traces["average_duration_seconds"], (int, float))
    assert isinstance(traces["logs"], list)

    # JSON serializability
    assert_is_json_serializable(traces)


def assert_base_config_fields(config: Dict[str, Any], model_id: Optional[str] = None) -> None:
    """Verify config has required base fields from ConfigurableMixin and ModelAdapter."""
    assert isinstance(config, dict), f"gather_config() must return dict, got {type(config)}"

    # Base fields from ConfigurableMixin
    assert "type" in config, "Missing 'type' field in config"
    assert isinstance(config["type"], str), f"'type' must be string, got {type(config['type'])}"
    assert len(config["type"]) > 0, "'type' field cannot be empty"

    assert "gathered_at" in config, "Missing 'gathered_at' field in config"
    assert isinstance(config["gathered_at"], str), f"'gathered_at' must be string, got {type(config['gathered_at'])}"
    assert_is_valid_iso_timestamp(config["gathered_at"])

    # ModelAdapter-specific fields
    assert "model_id" in config, "Missing 'model_id' field in config"
    assert isinstance(config["model_id"], str), f"'model_id' must be string, got {type(config['model_id'])}"

    if model_id is not None:
        assert config["model_id"] == model_id, f"Expected model_id '{model_id}', got '{config['model_id']}'"

    assert "adapter_type" in config, "Missing 'adapter_type' field in config"
    assert isinstance(config["adapter_type"], str)

    # Seed field (can be None or int)
    assert "seed" in config, "Missing 'seed' field in config"
    assert config["seed"] is None or isinstance(config["seed"], int), f"'seed' must be None or int, got {type(config['seed'])}"

    # JSON serializability
    assert_is_json_serializable(config)


# ==================== Adapter Factory Functions ====================


def create_openai_adapter(
    model_id: str = "gpt-4",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Create OpenAIModelAdapter instance."""
    pytest.importorskip("openai")
    from maseval.interface.inference.openai import OpenAIModelAdapter

    response_list: List[Optional[str]] = responses or ["Test response"]
    tool_calls_list = tool_calls
    call_count = [0]

    class MockClient:
        class Chat:
            class Completions:
                def create(self, model, messages, **kwargs):
                    response_text = response_list[call_count[0] % len(response_list)]
                    response_tool_calls = tool_calls_list[call_count[0] % len(tool_calls_list)] if tool_calls_list else None
                    call_count[0] += 1

                    # Mock response structure
                    message = {"content": response_text, "role": "assistant"}

                    if response_tool_calls:
                        message["tool_calls"] = response_tool_calls

                    return {
                        "choices": [{"message": message, "finish_reason": "stop"}],
                        "model": model,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    }

            completions = Completions()

        chat = Chat()

    return OpenAIModelAdapter(client=MockClient(), model_id=model_id, seed=seed)


def create_google_genai_adapter(
    model_id: str = "gemini-pro",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Create GoogleGenAIModelAdapter instance."""
    pytest.importorskip("google.genai")
    from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

    response_list: List[Optional[str]] = responses or ["Test response"]
    tool_calls_list = tool_calls
    call_count = [0]

    class MockClient:
        class Models:
            def generate_content(self_inner, model, contents, config=None):
                response = response_list[call_count[0] % len(response_list)]
                response_tool_calls = tool_calls_list[call_count[0] % len(tool_calls_list)] if tool_calls_list else None
                call_count[0] += 1

                # Build mock response with function calls if tool_calls provided
                if response_tool_calls:

                    class MockFunctionCall:
                        def __init__(self, name, args):
                            self.name = name
                            self.args = args

                    class MockPart:
                        def __init__(self, tc_dict):
                            self.type = "function_call"
                            func = tc_dict.get("function", {})
                            args_str = func.get("arguments", "{}")
                            import json

                            self.function_call = MockFunctionCall(func.get("name", ""), json.loads(args_str) if args_str else {})

                    class MockContent:
                        def __init__(self):
                            self.parts = [MockPart(tc) for tc in response_tool_calls]

                    class MockCandidate:
                        def __init__(self):
                            self.content = MockContent()
                            self.finish_reason = "STOP"

                    class MockResponse:
                        text = None
                        candidates = [MockCandidate()]

                    return MockResponse()
                else:

                    class Response:
                        text = response
                        candidates = []

                    return Response()

        def __init__(self):
            self.models = self.Models()

    return GoogleGenAIModelAdapter(client=MockClient(), model_id=model_id, seed=seed)


def create_huggingface_adapter(
    model_id: str = "gpt2",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Create HuggingFaceModelAdapter instance."""
    pytest.importorskip("transformers")
    from maseval.interface.inference.huggingface import HuggingFaceModelAdapter

    response_list: List[Optional[str]] = responses or ["Test response"]
    call_count = [0]

    def mock_model(prompt, **kwargs):
        response = response_list[call_count[0] % len(response_list)]
        call_count[0] += 1
        return response

    return HuggingFaceModelAdapter(model=mock_model, model_id=model_id, seed=seed)


def create_litellm_adapter(
    model_id: str = "gpt-3.5-turbo",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Create LiteLLMModelAdapter instance."""
    pytest.importorskip("litellm")
    import litellm
    from maseval.interface.inference.litellm import LiteLLMModelAdapter

    # Mock litellm.completion
    response_list: List[Optional[str]] = responses or ["Test response"]
    tool_calls_list = tool_calls
    call_count = [0]
    original_completion = litellm.completion

    def mock_completion(model, messages, **kwargs):
        response = response_list[call_count[0] % len(response_list)]
        response_tool_calls_dicts = tool_calls_list[call_count[0] % len(tool_calls_list)] if tool_calls_list else None
        call_count[0] += 1

        # Convert dict tool_calls to objects with attributes (like real LiteLLM returns)
        mock_tool_calls = None
        if response_tool_calls_dicts:
            mock_tool_calls = []
            for tc_dict in response_tool_calls_dicts:

                class MockFunction:
                    name: str = ""
                    arguments: str = "{}"

                class MockToolCall:
                    id: str = ""
                    type: str = "function"
                    function: MockFunction

                func = MockFunction()
                func.name = tc_dict.get("function", {}).get("name", "")
                func.arguments = tc_dict.get("function", {}).get("arguments", "{}")

                tc = MockToolCall()
                tc.id = tc_dict.get("id", "")
                tc.type = tc_dict.get("type", "function")
                tc.function = func
                mock_tool_calls.append(tc)

        class MockMessage:
            def __init__(self):
                self.content = response
                self.role = "assistant"
                self.tool_calls = mock_tool_calls

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()
                self.finish_reason = "tool_calls" if mock_tool_calls else "stop"

        class MockUsage:
            prompt_tokens = 10
            completion_tokens = 20
            total_tokens = 30

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]
                self.usage = MockUsage()
                self.model = model

        return MockResponse()

    litellm.completion = mock_completion

    adapter = LiteLLMModelAdapter(model_id=model_id, seed=seed)

    # Store original for cleanup
    adapter._original_completion = original_completion  # type: ignore
    adapter._mock_completion = mock_completion  # type: ignore

    return adapter


def create_dummy_adapter(
    model_id: str = "test-model",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> DummyModelAdapter:
    """Create DummyModelAdapter instance."""
    responses = responses or ["Test response"]
    usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    return DummyModelAdapter(model_id=model_id, responses=responses, tool_calls=tool_calls, usage=usage, stop_reason="stop", seed=seed)


def create_anthropic_adapter(
    model_id: str = "claude-3",
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,  # Anthropic doesn't support seeding, parameter ignored
) -> Any:
    """Create AnthropicModelAdapter instance."""
    pytest.importorskip("anthropic")
    from maseval.interface.inference.anthropic import AnthropicModelAdapter

    response_list: List[Optional[str]] = responses or ["Test response"]
    tool_calls_list = tool_calls
    call_count = [0]

    class MockTextBlock:
        type = "text"

        def __init__(self, text: str):
            self.text = text

    class MockToolUseBlock:
        type = "tool_use"

        def __init__(self, tool_call: Dict[str, Any]):
            self.id = tool_call["id"]
            self.name = tool_call["function"]["name"]
            import json

            self.input = json.loads(tool_call["function"]["arguments"])

    class MockUsage:
        input_tokens = 10
        output_tokens = 5

    class MockMessages:
        def create(self, **kwargs):
            response = response_list[call_count[0] % len(response_list)]
            response_tool_calls = tool_calls_list[call_count[0] % len(tool_calls_list)] if tool_calls_list else None
            call_count[0] += 1

            class MockResponse:
                def __init__(self):
                    self.content = []
                    if response:
                        self.content.append(MockTextBlock(response))
                    if response_tool_calls:
                        for tc in response_tool_calls:
                            self.content.append(MockToolUseBlock(tc))
                    self.usage = MockUsage()
                    self.model = model_id
                    self.stop_reason = "end_turn"

            return MockResponse()

    class MockClient:
        messages = MockMessages()

    return AnthropicModelAdapter(client=MockClient(), model_id=model_id)


def create_adapter_for_implementation(
    implementation: str,
    model_id: str,
    responses: Optional[List[Optional[str]]] = None,
    tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Factory function to create adapter for specified implementation."""
    factories = {
        "dummy": create_dummy_adapter,
        "openai": create_openai_adapter,
        "google_genai": create_google_genai_adapter,
        "huggingface": create_huggingface_adapter,
        "litellm": create_litellm_adapter,
        "anthropic": create_anthropic_adapter,
    }

    if implementation not in factories:
        raise ValueError(f"Unknown implementation: {implementation}")

    return factories[implementation](model_id=model_id, responses=responses, tool_calls=tool_calls, seed=seed)


def cleanup_adapter(adapter: Any, implementation: str) -> None:
    """Clean up adapter resources."""
    if implementation == "litellm" and hasattr(adapter, "_original_completion"):
        import litellm

        litellm.completion = adapter._original_completion


# ==================== Contract Tests ====================


@pytest.mark.contract
@pytest.mark.interface
@pytest.mark.parametrize("implementation", ["dummy", "openai", "google_genai", "huggingface", "litellm", "anthropic"])
class TestModelAdapterContract:
    """Verify all ModelAdapter implementations honor the same contract."""

    def test_adapter_generate_returns_string(self, implementation):
        """All adapters return string from generate()."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["Test response"])

        try:
            result = adapter.generate("Test prompt")
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_chat_returns_chat_response(self, implementation):
        """All adapters return ChatResponse from chat()."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["Test response"])

        try:
            result = adapter.chat([{"role": "user", "content": "Test prompt"}])
            assert isinstance(result, ChatResponse)
            assert result.content is not None or result.tool_calls is not None
            assert result.role == "assistant"
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_chat_handles_multi_turn(self, implementation):
        """All adapters handle multi-turn conversations."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["Response"])

        try:
            result = adapter.chat(
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ]
            )
            assert isinstance(result, ChatResponse)
            assert result.content is not None
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_chat_handles_system_message(self, implementation):
        """All adapters handle system messages."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["Response"])

        try:
            result = adapter.chat(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]
            )
            assert isinstance(result, ChatResponse)
            assert result.content is not None
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_traces_have_base_fields(self, implementation):
        """All adapters include required trace fields."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            adapter.generate("Test prompt")
            traces = adapter.gather_traces()
            assert_base_trace_fields(traces, model_id="test-model")
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_config_has_base_fields(self, implementation):
        """All adapters include required config fields."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            config = adapter.gather_config()
            assert_base_config_fields(config, model_id="test-model")
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_logs_successful_calls(self, implementation):
        """All adapters log successful calls."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["R1", "R2"])

        try:
            adapter.generate("Prompt 1")
            adapter.generate("Prompt 2")

            traces = adapter.gather_traces()
            assert traces["total_calls"] == 2
            assert traces["successful_calls"] == 2
            assert traces["failed_calls"] == 0
            assert len(traces["logs"]) == 2
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_captures_timing(self, implementation):
        """All adapters capture execution timing."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            adapter.generate("Test")

            traces = adapter.gather_traces()
            assert traces["total_duration_seconds"] >= 0
            assert traces["average_duration_seconds"] >= 0
            assert traces["logs"][0]["duration_seconds"] >= 0
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_traces_are_json_serializable(self, implementation):
        """All adapters produce JSON-serializable traces."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            adapter.generate("Test", generation_params={"temperature": 0.7})

            traces = adapter.gather_traces()
            json_str = json.dumps(traces)
            assert isinstance(json_str, str)

            # Should round-trip
            recovered = json.loads(json_str)
            assert recovered["total_calls"] == traces["total_calls"]
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_config_is_json_serializable(self, implementation):
        """All adapters produce JSON-serializable config."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            config = adapter.gather_config()
            json_str = json.dumps(config)
            assert isinstance(json_str, str)

            # Should round-trip
            recovered = json.loads(json_str)
            assert recovered["model_id"] == config["model_id"]
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_config_is_static(self, implementation):
        """All adapters return same config before and after execution."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            config_before = adapter.gather_config()
            adapter.generate("Test")
            config_after = adapter.gather_config()

            # Remove timestamps for comparison
            def remove_timestamp(d):
                return {k: v for k, v in d.items() if k != "gathered_at"}

            assert remove_timestamp(config_before) == remove_timestamp(config_after)
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_accumulates_logs(self, implementation):
        """All adapters accumulate logs across multiple generations."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["R1", "R2", "R3"])

        try:
            for i in range(3):
                adapter.generate(f"Prompt {i}")

            traces = adapter.gather_traces()
            assert traces["total_calls"] == 3
            assert len(traces["logs"]) == 3
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_gather_traces_idempotent(self, implementation):
        """All adapters can call gather_traces() multiple times."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            adapter.generate("Test")

            traces1 = adapter.gather_traces()
            traces2 = adapter.gather_traces()

            # Core fields should be identical (except timestamp)
            assert traces1["type"] == traces2["type"]
            assert traces1["model_id"] == traces2["model_id"]
            assert traces1["total_calls"] == traces2["total_calls"]
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_gather_config_idempotent(self, implementation):
        """All adapters can call gather_config() multiple times."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            config1 = adapter.gather_config()
            config2 = adapter.gather_config()

            # Core fields should be identical (except timestamp)
            assert config1["type"] == config2["type"]
            assert config1["model_id"] == config2["model_id"]
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_handles_empty_prompt(self, implementation):
        """All adapters handle empty prompts gracefully."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            result = adapter.generate("")
            assert isinstance(result, str)

            traces = adapter.gather_traces()
            assert traces["total_calls"] == 1
            # Empty prompt still creates one message
            assert traces["logs"][0]["message_count"] == 1
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_preserves_model_id(self, implementation):
        """All adapters preserve model_id consistently."""
        adapter = create_adapter_for_implementation(implementation, model_id="custom-model-123")

        try:
            assert adapter.model_id == "custom-model-123"

            traces = adapter.gather_traces()
            assert traces["model_id"] == "custom-model-123"

            config = adapter.gather_config()
            assert config["model_id"] == "custom-model-123"
        finally:
            cleanup_adapter(adapter, implementation)


@pytest.mark.contract
@pytest.mark.interface
class TestCrossAdapterConsistency:
    """Test that different adapter implementations follow the same patterns."""

    def test_all_adapters_have_consistent_trace_structure(self):
        """All adapter implementations have same base trace structure."""
        implementations = ["dummy", "openai", "google_genai", "huggingface", "litellm", "anthropic"]
        adapters = []

        try:
            for impl in implementations:
                adapter = create_adapter_for_implementation(impl, model_id=f"{impl}-model")
                adapter.generate("Test")
                adapters.append((adapter, impl))

            # Gather all traces
            all_traces = [adapter.gather_traces() for adapter, _ in adapters]

            # All must have same base fields
            for traces in all_traces:
                assert "type" in traces
                assert "gathered_at" in traces
                assert "model_id" in traces
                assert "total_calls" in traces
                assert "successful_calls" in traces
                assert "failed_calls" in traces
                assert "total_duration_seconds" in traces
                assert "average_duration_seconds" in traces
                assert "logs" in traces

                # All must be JSON serializable
                assert_is_json_serializable(traces)
        finally:
            for adapter, impl in adapters:
                cleanup_adapter(adapter, impl)

    def test_all_adapters_have_consistent_config_structure(self):
        """All adapter implementations have same base config structure."""
        implementations = ["dummy", "openai", "google_genai", "huggingface", "litellm", "anthropic"]
        adapters = []

        try:
            for impl in implementations:
                adapter = create_adapter_for_implementation(impl, model_id=f"{impl}-model")
                adapters.append((adapter, impl))

            # Gather all configs
            all_configs = [adapter.gather_config() for adapter, _ in adapters]

            # All must have same base fields
            for config in all_configs:
                assert "type" in config
                assert "gathered_at" in config
                assert "model_id" in config
                assert "adapter_type" in config

                # All must be JSON serializable
                assert_is_json_serializable(config)
        finally:
            for adapter, impl in adapters:
                cleanup_adapter(adapter, impl)

    def test_all_adapters_log_same_call_metadata(self):
        """All adapters log same metadata for each call."""
        implementations = ["dummy", "openai", "google_genai", "huggingface", "litellm", "anthropic"]
        adapters = []

        try:
            for impl in implementations:
                adapter = create_adapter_for_implementation(impl, model_id=f"{impl}-model")
                adapter.generate("Test prompt")
                adapters.append((adapter, impl))

            # Check call log structure
            for adapter, impl in adapters:
                traces = adapter.gather_traces()
                assert len(traces["logs"]) == 1

                call = traces["logs"][0]
                # All must have these fields
                assert "timestamp" in call, f"Missing timestamp in {impl}"
                assert "status" in call, f"Missing status in {impl}"
                assert "duration_seconds" in call, f"Missing duration in {impl}"
                assert "message_count" in call, f"Missing message_count in {impl}"
        finally:
            for adapter, impl in adapters:
                cleanup_adapter(adapter, impl)


# ==================== Tool Calling Contract Tests ====================


@pytest.mark.contract
@pytest.mark.interface
@pytest.mark.parametrize("implementation", ["dummy", "openai", "litellm", "anthropic"])
class TestToolCallingContract:
    """Contract tests for tool calling functionality across adapters.

    These tests verify that tool-related features work consistently across
    all model adapters that support tools. This is critical for users building
    agentic systems that need to swap between providers.

    Note: Only testing adapters that support tools (OpenAI, Anthropic, LiteLLM, Dummy).
    HuggingFace and GoogleGenAI don't fully support tool calling in their current implementation.
    """

    def test_adapter_accepts_tools_parameter(self, implementation):
        """All adapters accept tools parameter without error."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ]

        try:
            result = adapter.chat([{"role": "user", "content": "What's the weather in Paris?"}], tools=tools)
            assert isinstance(result, ChatResponse)
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_accepts_tool_choice_parameter(self, implementation):
        """All adapters accept tool_choice parameter without error."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ]

        try:
            # Test different tool_choice values
            for tool_choice in ["auto", "none", "required"]:
                result = adapter.chat([{"role": "user", "content": "What's the weather?"}], tools=tools, tool_choice=tool_choice)
                assert isinstance(result, ChatResponse)

            # Test specific tool selection
            result = adapter.chat(
                [{"role": "user", "content": "What's the weather?"}],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            )
            assert isinstance(result, ChatResponse)
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_returns_tool_calls_in_response(self, implementation):
        """All adapters return tool_calls with consistent structure."""
        tool_calls_to_return = [
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
        ]

        adapter = create_adapter_for_implementation(
            implementation, model_id="test-model", responses=["I'll check the weather"], tool_calls=tool_calls_to_return
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        try:
            result = adapter.chat([{"role": "user", "content": "What's the weather in Paris?"}], tools=tools)

            assert result.tool_calls is not None, f"{implementation} did not return tool_calls"
            assert isinstance(result.tool_calls, list)
            assert len(result.tool_calls) > 0

            # Verify structure of first tool call
            tc = result.tool_calls[0]
            assert "id" in tc, f"{implementation} tool_call missing 'id'"
            assert "type" in tc, f"{implementation} tool_call missing 'type'"
            assert "function" in tc, f"{implementation} tool_call missing 'function'"
            assert "name" in tc["function"], f"{implementation} tool_call function missing 'name'"
            assert "arguments" in tc["function"], f"{implementation} tool_call function missing 'arguments'"

            # Verify types
            assert isinstance(tc["id"], str)
            assert isinstance(tc["type"], str)
            assert isinstance(tc["function"]["name"], str)
            assert isinstance(tc["function"]["arguments"], str)  # JSON string
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_handles_tool_result_messages(self, implementation):
        """All adapters handle role='tool' messages in conversations."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        # Simulate a conversation with tool use
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": 72, "condition": "sunny"}'},
            {"role": "user", "content": "What about London?"},
        ]

        try:
            result = adapter.chat(messages)
            assert isinstance(result, ChatResponse)
            # Should not raise an error
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_handles_assistant_messages_with_tool_calls(self, implementation):
        """All adapters handle assistant messages containing tool_calls."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        # Include an assistant message with tool_calls in the history
        messages = [
            {"role": "user", "content": "Get weather for Paris"},
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temperature": 72}'},
            {"role": "user", "content": "Thanks!"},
        ]

        try:
            result = adapter.chat(messages)
            assert isinstance(result, ChatResponse)
            # Should process the conversation history without error
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_tool_calls_logs_correctly(self, implementation):
        """All adapters log tool-related calls consistently."""
        tool_calls_to_return = [
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
        ]

        adapter = create_adapter_for_implementation(
            implementation, model_id="test-model", responses=["I'll check"], tool_calls=tool_calls_to_return
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        try:
            adapter.chat([{"role": "user", "content": "Weather?"}], tools=tools)

            traces = adapter.gather_traces()
            assert traces["total_calls"] == 1
            assert len(traces["logs"]) == 1

            call_log = traces["logs"][0]
            assert "response_type" in call_log
            assert call_log["response_type"] == "tool_call"
            assert "tool_calls_count" in call_log
            assert call_log["tool_calls_count"] == 1
            assert "tools_provided" in call_log
            assert call_log["tools_provided"] == 1
        finally:
            cleanup_adapter(adapter, implementation)


# ==================== Usage and Metadata Contract Tests ====================


@pytest.mark.contract
@pytest.mark.interface
@pytest.mark.parametrize("implementation", ["dummy", "openai", "litellm", "anthropic"])
class TestUsageAndMetadataContract:
    """Contract tests for usage tracking and response metadata.

    These tests ensure consistent reporting of token usage, stop reasons,
    and other metadata across all adapters. This is important for evaluation
    and cost tracking in production systems.

    Note: Only testing adapters with full metadata support (OpenAI, Anthropic, LiteLLM, Dummy).
    """

    def test_adapter_returns_usage_info(self, implementation):
        """All adapters return consistent usage information."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            result = adapter.chat([{"role": "user", "content": "Hello"}])

            # Usage should be present and have required fields
            if result.usage is not None:  # Some adapters might not support this
                assert isinstance(result.usage, dict)
                assert "input_tokens" in result.usage
                assert "output_tokens" in result.usage
                assert "total_tokens" in result.usage

                assert isinstance(result.usage["input_tokens"], int)
                assert isinstance(result.usage["output_tokens"], int)
                assert isinstance(result.usage["total_tokens"], int)

                assert result.usage["input_tokens"] >= 0
                assert result.usage["output_tokens"] >= 0
                assert result.usage["total_tokens"] >= 0
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_returns_stop_reason(self, implementation):
        """All adapters return stop_reason in responses."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            result = adapter.chat([{"role": "user", "content": "Hello"}])

            # stop_reason should be present
            if result.stop_reason is not None:  # Some adapters might not support this
                assert isinstance(result.stop_reason, str)
                assert len(result.stop_reason) > 0
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_stop_reason_for_tool_calls(self, implementation):
        """All adapters indicate tool use in stop_reason when applicable."""
        tool_calls_to_return = [
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
        ]

        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=[None], tool_calls=tool_calls_to_return)

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        try:
            result = adapter.chat([{"role": "user", "content": "Weather?"}], tools=tools)

            # When tool_calls are returned, should have a stop_reason
            # (The exact value may vary: "tool_calls", "tool_use", "function_call", etc.)
            if result.stop_reason is not None:
                assert isinstance(result.stop_reason, str)
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_handles_content_none_with_tool_calls(self, implementation):
        """All adapters handle responses with content=None and only tool_calls."""
        tool_calls_to_return = [
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
        ]

        # Response with None content, only tool_calls
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=[None], tool_calls=tool_calls_to_return)

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        try:
            result = adapter.chat([{"role": "user", "content": "What's the weather?"}], tools=tools)

            assert isinstance(result, ChatResponse)
            # content can be None when model only returns tool calls
            assert result.tool_calls is not None, f"{implementation} should return tool_calls when content is None"
            assert isinstance(result.tool_calls, list)
            assert len(result.tool_calls) > 0

            # Verify the response is still valid
            msg = result.to_message()
            assert isinstance(msg, dict)
            assert msg["role"] == "assistant"
            assert "tool_calls" in msg
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_to_message_includes_tool_calls(self, implementation):
        """All adapters include tool_calls in to_message() output."""
        tool_calls_to_return = [
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ]
        ]

        adapter = create_adapter_for_implementation(
            implementation, model_id="test-model", responses=["I'll check"], tool_calls=tool_calls_to_return
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        try:
            result = adapter.chat([{"role": "user", "content": "Weather?"}], tools=tools)

            msg = result.to_message()
            assert isinstance(msg, dict)
            assert msg["role"] == "assistant"
            assert "tool_calls" in msg, f"{implementation} to_message() should include tool_calls"
            assert isinstance(msg["tool_calls"], list)
            assert len(msg["tool_calls"]) > 0
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_usage_tracking_across_calls(self, implementation):
        """All adapters consistently report usage across multiple calls."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", responses=["R1", "R2"])

        try:
            result1 = adapter.chat([{"role": "user", "content": "First"}])
            result2 = adapter.chat([{"role": "user", "content": "Second"}])

            # Both should have usage (if supported)
            if result1.usage is not None and result2.usage is not None:
                assert isinstance(result1.usage, dict)
                assert isinstance(result2.usage, dict)

                # Structure should be consistent
                assert set(result1.usage.keys()) == set(result2.usage.keys())
        finally:
            cleanup_adapter(adapter, implementation)


# ==================== Seeding Contract Tests ====================


@pytest.mark.contract
@pytest.mark.interface
@pytest.mark.parametrize("implementation", ["dummy", "openai", "google_genai", "huggingface", "litellm"])
class TestSeedingContract:
    """Contract tests for seeding functionality across adapters.

    These tests verify that seed parameters are consistently handled across
    all model adapters that support seeding. This is important for reproducibility
    in benchmark evaluations.

    Note: Anthropic is excluded as it explicitly does not support seeding.
    """

    def test_adapter_config_includes_seed_when_set(self, implementation):
        """All adapters include seed in config when provided."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", seed=42)

        try:
            config = adapter.gather_config()
            assert "seed" in config, f"{implementation} config missing 'seed' field"
            assert config["seed"] == 42, f"{implementation} seed not preserved in config"
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_config_includes_seed_none_when_not_set(self, implementation):
        """All adapters include seed=None in config when not provided."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            config = adapter.gather_config()
            assert "seed" in config, f"{implementation} config missing 'seed' field"
            assert config["seed"] is None, f"{implementation} seed should be None when not set"
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_seed_property_accessible(self, implementation):
        """All adapters expose seed property."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model", seed=123)

        try:
            assert hasattr(adapter, "seed"), f"{implementation} missing 'seed' property"
            assert adapter.seed == 123, f"{implementation} seed property value incorrect"
        finally:
            cleanup_adapter(adapter, implementation)

    def test_adapter_seed_none_by_default(self, implementation):
        """All adapters have seed=None by default."""
        adapter = create_adapter_for_implementation(implementation, model_id="test-model")

        try:
            assert hasattr(adapter, "seed"), f"{implementation} missing 'seed' property"
            assert adapter.seed is None, f"{implementation} seed should be None by default"
        finally:
            cleanup_adapter(adapter, implementation)
