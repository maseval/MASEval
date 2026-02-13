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
