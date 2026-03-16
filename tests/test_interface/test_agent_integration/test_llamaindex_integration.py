"""Integration tests for LlamaIndex.

These tests require llama-index-core to be installed.
Run with: pytest -m llamaindex
"""

import pytest

# Skip entire module if llama-index-core not installed
pytest.importorskip("llama_index.core")

# Mark all tests in this file as requiring llamaindex
pytestmark = [pytest.mark.interface, pytest.mark.llamaindex]


def test_llamaindex_adapter_import():
    """Test that LlamaIndexAgentAdapter can be imported when llama-index-core is installed."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter, LlamaIndexLLMUser

    assert LlamaIndexAgentAdapter is not None
    assert LlamaIndexLLMUser is not None


def test_llamaindex_in_agents_all():
    """Test that llamaindex appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "LlamaIndexAgentAdapter" in maseval.interface.agents.__all__
    assert "LlamaIndexLLMUser" in maseval.interface.agents.__all__


def test_check_llamaindex_installed_function():
    """Test that _check_llamaindex_installed doesn't raise when llama-index-core is installed."""
    from maseval.interface.agents.llamaindex import _check_llamaindex_installed

    # Should not raise
    _check_llamaindex_installed()


def test_llamaindex_adapter_creation():
    """Test that LlamaIndexAgentAdapter can be created."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

    # Create adapter with mock agent
    agent_adapter = LlamaIndexAgentAdapter(agent_instance=object(), name="test_agent")

    assert agent_adapter.name == "test_agent"
    assert agent_adapter.agent is not None


def test_llamaindex_user_creation():
    """Test that LlamaIndexLLMUser can be created."""
    from maseval.interface.agents.llamaindex import LlamaIndexLLMUser
    from unittest.mock import Mock

    # Create user with required parameters
    mock_model = Mock()
    user = LlamaIndexLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="test prompt",
    )

    assert user is not None
    assert user.name == "test_user"


def test_llamaindex_user_get_tool():
    """Test that LlamaIndexLLMUser.get_tool() returns a FunctionTool."""
    from maseval.interface.agents.llamaindex import LlamaIndexLLMUser
    from llama_index.core.tools import FunctionTool
    from unittest.mock import Mock

    mock_model = Mock()
    user = LlamaIndexLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="test prompt",
    )

    tool = user.get_tool()

    # Verify it's a FunctionTool
    assert isinstance(tool, FunctionTool)
    assert tool.metadata.name == "ask_user"
    assert "question" in tool.metadata.description.lower()


def test_llamaindex_adapter_message_conversion():
    """Test that LlamaIndex ChatMessage objects are converted to OpenAI format."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    mock_agent = Mock()
    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Create test messages
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant"),
    ]

    # Convert messages
    history = adapter._convert_llamaindex_messages(messages)

    # Verify conversion
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"
    assert history[2]["role"] == "system"
    assert history[2]["content"] == "You are a helpful assistant"


def test_llamaindex_adapter_single_message_conversion():
    """Test single message conversion with tool calls in additional_kwargs."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    mock_agent = Mock()
    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Create message with tool calls
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Let me check that for you",
        additional_kwargs={
            "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}}]
        },
    )

    # Convert message
    converted = adapter._convert_single_message(message)

    # Verify conversion
    assert converted["role"] == "assistant"
    assert converted["content"] == "Let me check that for you"
    assert "tool_calls" in converted
    assert len(converted["tool_calls"]) == 1
    assert converted["tool_calls"][0]["id"] == "call_123"


def test_llamaindex_adapter_get_messages_from_cache():
    """Test that get_messages returns cached messages when agent has no memory."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    # Agent has no memory attribute
    del mock_agent.memory

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Set cache manually
    adapter._message_cache = [{"role": "user", "content": "Test query"}, {"role": "assistant", "content": "Test response"}]

    # Get messages
    messages = adapter.get_messages()

    # Verify we get cached messages
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_llamaindex_adapter_extract_final_answer():
    """Test final answer extraction from various result formats."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    mock_agent = Mock()
    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Test 1: Result with response attribute containing ChatMessage
    result1 = Mock()
    result1.response = ChatMessage(role=MessageRole.ASSISTANT, content="This is the answer")

    answer1 = adapter._extract_final_answer(result1)
    assert answer1 == "This is the answer"

    # Test 2: Result with response attribute containing simple object
    result2 = Mock()
    result2.response = Mock()
    result2.response.content = "Another answer"

    answer2 = adapter._extract_final_answer(result2)
    assert answer2 == "Another answer"

    # Test 3: Result without response attribute - test string conversion
    result3 = Mock()
    # Delete response attribute first, then set __str__ return value
    del result3.response
    result3.__str__ = Mock(return_value="Fallback answer")

    answer3 = adapter._extract_final_answer(result3)
    assert answer3 == "Fallback answer"


def test_llamaindex_adapter_gather_config():
    """Test that gather_config includes LlamaIndex-specific configuration."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    # Create mock agent with configuration attributes
    mock_agent = Mock()
    mock_agent.name = "test_workflow"
    mock_agent.description = "A test workflow agent"
    mock_agent.system_prompt = "You are a helpful assistant"

    # Mock tools
    mock_tool = Mock()
    mock_tool.metadata = {"name": "calculator", "description": "Performs calculations"}
    mock_agent.tools = [mock_tool]

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Gather config
    config = adapter.gather_config()

    # Verify base config
    assert config["name"] == "test_agent"
    assert config["agent_type"] == "Mock"
    assert config["adapter_type"] == "LlamaIndexAgentAdapter"

    # Verify LlamaIndex-specific config
    assert "llamaindex_config" in config
    llamaindex_config = config["llamaindex_config"]

    assert llamaindex_config["agent_name"] == "test_workflow"
    assert llamaindex_config["agent_description"] == "A test workflow agent"
    assert llamaindex_config["system_prompt"] == "You are a helpful assistant"
    assert "tools" in llamaindex_config
    assert len(llamaindex_config["tools"]) == 1
    assert llamaindex_config["tools"][0]["name"] == "calculator"


def test_llamaindex_adapter_async_execution_handling():
    """Test that async agent execution is handled correctly in sync context."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    # Create mock agent with run_sync method
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.response = Mock()
    mock_result.response.content = "Sync result"
    mock_agent.run_sync = Mock(return_value=mock_result)

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent
    result = adapter._run_agent_sync("test query")

    # Verify run_sync was called
    mock_agent.run_sync.assert_called_once_with(input="test query")
    assert result.response.content == "Sync result"


def test_llamaindex_adapter_async_coroutine_execution():
    """Test that async coroutines are executed correctly in sync context."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    # Create mock agent with async run method
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.response = Mock()
    mock_result.response.content = "Async result"

    async def async_run(user_msg=None, **kwargs):
        """Mock run that returns an awaitable result."""
        return mock_result

    mock_agent.run = async_run
    del mock_agent.run_sync  # No sync wrapper
    del mock_agent.chat  # No chat method
    del mock_agent.query  # No query method

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent (should use asyncio.run internally)
    result = adapter._run_agent_sync("test query")

    # Verify result
    assert result.response.content == "Async result"


def test_llamaindex_adapter_chat_method():
    """Test that adapter uses .chat() method when available (ReActAgent pattern)."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    # Create mock agent with chat method
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.response = ChatMessage(role=MessageRole.ASSISTANT, content="Chat result")
    mock_agent.chat = Mock(return_value=mock_result)
    del mock_agent.run_sync  # No run_sync wrapper

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent (should use .chat() method)
    result = adapter._run_agent_sync("test query")

    # Verify chat was called
    mock_agent.chat.assert_called_once_with("test query")
    assert result.response.content == "Chat result"


def test_llamaindex_adapter_query_method():
    """Test that adapter uses .query() method when available."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    # Create mock agent with query method
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.response = ChatMessage(role=MessageRole.ASSISTANT, content="Query result")
    mock_agent.query = Mock(return_value=mock_result)
    del mock_agent.run_sync  # No run_sync wrapper
    del mock_agent.chat  # No chat method

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent (should use .query() method)
    result = adapter._run_agent_sync("test query")

    # Verify query was called
    mock_agent.query.assert_called_once_with("test query")
    assert result.response.content == "Query result"


def test_llamaindex_adapter_logging():
    """Test that adapter logs execution details."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from unittest.mock import Mock

    # Create mock agent
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.response = ChatMessage(role=MessageRole.ASSISTANT, content="Test response")
    mock_agent.run_sync = Mock(return_value=mock_result)

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent
    adapter.run("test query")

    # Verify log entry was created
    assert len(adapter.logs) == 1
    log_entry = adapter.logs[0]

    assert log_entry["status"] == "success"
    assert log_entry["query"] == "test query"
    assert log_entry["query_length"] == len("test query")
    assert "duration_seconds" in log_entry
    assert "timestamp" in log_entry
    assert log_entry["message_count"] >= 1


def test_llamaindex_adapter_error_logging():
    """Test that adapter logs errors during execution."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    # Create mock agent that raises an error
    mock_agent = Mock()
    mock_agent.run_sync = Mock(side_effect=ValueError("Test error"))

    adapter = LlamaIndexAgentAdapter(mock_agent, "test_agent")

    # Run agent (should raise error)
    with pytest.raises(ValueError, match="Test error"):
        adapter.run("test query")

    # Verify error was logged
    assert len(adapter.logs) == 1
    log_entry = adapter.logs[0]

    assert log_entry["status"] == "error"
    assert log_entry["error"] == "Test error"
    assert log_entry["error_type"] == "ValueError"
    assert "duration_seconds" in log_entry


# =============================================================================
# gather_usage() Tests
# =============================================================================


def test_llamaindex_adapter_gather_usage_with_logs():
    """Test that gather_usage() aggregates token usage from execution logs."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage
    from unittest.mock import Mock

    adapter = LlamaIndexAgentAdapter(Mock(), "test_agent")

    # Simulate logs with token usage (as populated by _run_agent)
    adapter.logs.append(
        {
            "timestamp": "2026-01-01T00:00:00",
            "query": "Query 1",
            "status": "success",
            "duration_seconds": 1.0,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
    )
    adapter.logs.append(
        {
            "timestamp": "2026-01-01T00:00:01",
            "query": "Query 2",
            "status": "success",
            "duration_seconds": 0.5,
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
        }
    )

    usage = adapter.gather_usage()

    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 300  # 100 + 200
    assert usage.output_tokens == 130  # 50 + 80
    assert usage.total_tokens == 430


def test_llamaindex_adapter_gather_usage_no_logs():
    """Test that gather_usage() returns empty Usage with no logs."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from maseval.core.usage import Usage
    from unittest.mock import Mock

    adapter = LlamaIndexAgentAdapter(Mock(), "test_agent")

    usage = adapter.gather_usage()

    assert isinstance(usage, Usage)
    assert usage.cost == 0.0


def test_llamaindex_adapter_gather_usage_logs_without_tokens():
    """Test that gather_usage() returns empty Usage when logs have no token fields."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from maseval.core.usage import Usage
    from unittest.mock import Mock

    adapter = LlamaIndexAgentAdapter(Mock(), "test_agent")

    # Log without token fields (error case, or model didn't report usage)
    adapter.logs.append(
        {
            "timestamp": "2026-01-01T00:00:00",
            "query": "Query",
            "status": "error",
            "duration_seconds": 0.1,
            "error": "Something went wrong",
        }
    )

    usage = adapter.gather_usage()

    assert isinstance(usage, Usage)
    assert usage.cost == 0.0


# =============================================================================
# End-to-End Usage Collection Tests
# (real ReActAgent + AgentWorkflow execution, not pre-populated mock data)
# =============================================================================


def _make_llamaindex_adapter(prompt_tokens_per_call: int = 10, completion_tokens_per_call: int = 20):
    """Create adapter wrapping a ReActAgent + AgentWorkflow with a mock LLM that reports usage.

    The mock LLM returns CompletionResponse with token usage in raw, using
    ReAct format ("Thought: ... Answer: ...") so the output parser works.
    """
    from types import SimpleNamespace
    from typing import Any
    from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen
    from llama_index.core.llms.custom import CustomLLM
    from llama_index.core.llms import LLMMetadata
    from llama_index.core.agent.workflow.react_agent import ReActAgent
    from llama_index.core.agent.workflow import AgentWorkflow
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter

    pt = prompt_tokens_per_call
    ct = completion_tokens_per_call

    class _MockLLMWithUsage(CustomLLM):
        prompt_tokens: int = pt
        completion_tokens: int = ct

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(num_output=256)

        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            usage = SimpleNamespace(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            )
            return CompletionResponse(
                text="Thought: I can answer directly.\nAnswer: Mock answer.",
                raw=SimpleNamespace(usage=usage),
            )

        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
            raise NotImplementedError

    llm = _MockLLMWithUsage()
    agent = ReActAgent(name="test_agent", description="test", llm=llm, tools=[], streaming=False)
    workflow = AgentWorkflow(agents=[agent], root_agent="test_agent")
    return LlamaIndexAgentAdapter(workflow, "test_agent")


def test_e2e_llamaindex_gather_usage_single_run():
    """Run a real ReActAgent → adapter.run() → gather_usage() returns real token counts."""
    from maseval.core.usage import TokenUsage as MasevalTokenUsage

    adapter = _make_llamaindex_adapter(prompt_tokens_per_call=10, completion_tokens_per_call=20)

    result = adapter.run("Hello?")
    assert isinstance(result, str)
    assert len(result) > 0

    usage = adapter.gather_usage()
    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30


def test_e2e_llamaindex_gather_usage_accumulates():
    """Multiple adapter.run() calls accumulate in gather_usage()."""
    from maseval.core.usage import TokenUsage as MasevalTokenUsage

    adapter = _make_llamaindex_adapter(prompt_tokens_per_call=15, completion_tokens_per_call=25)

    adapter.run("First query")
    adapter.run("Second query")

    usage = adapter.gather_usage()
    assert isinstance(usage, MasevalTokenUsage)
    assert usage.input_tokens == 30  # 15 + 15
    assert usage.output_tokens == 50  # 25 + 25
    assert usage.total_tokens == 80


def test_e2e_llamaindex_gather_usage_empty_before_run():
    """Verify gather_usage() returns empty Usage before run, real TokenUsage after."""
    from maseval.core.usage import Usage, TokenUsage as MasevalTokenUsage

    adapter = _make_llamaindex_adapter(prompt_tokens_per_call=50, completion_tokens_per_call=100)

    # Before run: no usage
    usage_before = adapter.gather_usage()
    assert isinstance(usage_before, Usage)
    assert not isinstance(usage_before, MasevalTokenUsage)

    # After run: real usage from the LLM
    adapter.run("test query")
    usage_after = adapter.gather_usage()
    assert isinstance(usage_after, MasevalTokenUsage)
    assert usage_after.input_tokens == 50
    assert usage_after.output_tokens == 100
    assert usage_after.total_tokens == 150


def test_e2e_llamaindex_logs_populated_by_real_execution():
    """Verify adapter.logs is populated by _run_agent, not manually."""
    adapter = _make_llamaindex_adapter(prompt_tokens_per_call=50, completion_tokens_per_call=100)

    assert len(adapter.logs) == 0

    adapter.run("Test query")

    assert len(adapter.logs) == 1
    log = adapter.logs[0]
    assert log["status"] == "success"
    assert log["input_tokens"] == 50
    assert log["output_tokens"] == 100
    assert log["total_tokens"] == 150
    assert "timestamp" in log
    assert "duration_seconds" in log


# =============================================================================
# Cost Calculation Tests
# =============================================================================


def test_llamaindex_adapter_cost_with_explicit_calculator():
    """Test that passing a cost_calculator computes cost from token usage."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from maseval.core.usage import TokenUsage as MasevalTokenUsage, StaticPricingCalculator
    from unittest.mock import Mock

    mock_agent = Mock(spec=[])
    adapter = LlamaIndexAgentAdapter(agent_instance=mock_agent, name="test")
    # Simulate logs from a run
    adapter.logs = [{"input_tokens": 1000, "output_tokens": 500, "status": "success"}]

    calculator = StaticPricingCalculator({"gpt-4": {"input": 0.00003, "output": 0.00006}})
    adapter._cost_calculator = calculator
    adapter._model_id = "gpt-4"

    usage = adapter.gather_usage()
    assert isinstance(usage, MasevalTokenUsage)
    assert usage.cost == pytest.approx(1000 * 0.00003 + 500 * 0.00006)


def test_llamaindex_adapter_resolve_model_id():
    """Test that _resolve_model_id() reads from agent.llm.metadata.model_name."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock()
    mock_agent.llm.metadata.model_name = "gpt-4o-mini"

    adapter = LlamaIndexAgentAdapter(agent_instance=mock_agent, name="test")
    assert adapter._resolve_model_id() == "gpt-4o-mini"


def test_llamaindex_adapter_resolve_model_id_missing():
    """Test that _resolve_model_id() returns None when agent has no llm."""
    from maseval.interface.agents.llamaindex import LlamaIndexAgentAdapter
    from unittest.mock import Mock

    mock_agent = Mock(spec=[])
    adapter = LlamaIndexAgentAdapter(agent_instance=mock_agent, name="test")
    assert adapter._resolve_model_id() is None
