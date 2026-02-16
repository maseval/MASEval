"""Integration tests for langgraph.

These tests require langgraph to be installed.
Run with: pytest -m langgraph
"""

import pytest

# Skip entire module if langgraph not installed
pytest.importorskip("langgraph")

# Mark all tests in this file as requiring langgraph
pytestmark = [pytest.mark.interface, pytest.mark.langgraph]


def test_langgraph_adapter_import():
    """Test that LangGraphAgentAdapter can be imported when langgraph is installed."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter, LangGraphLLMUser

    assert LangGraphAgentAdapter is not None
    assert LangGraphLLMUser is not None


def test_langgraph_in_agents_all():
    """Test that langgraph appears in interface.agents.__all__ when installed."""
    import maseval.interface.agents

    assert "LangGraphAgentAdapter" in maseval.interface.agents.__all__
    assert "LangGraphLLMUser" in maseval.interface.agents.__all__


def test_check_langgraph_installed_function():
    """Test that _check_langgraph_installed doesn't raise when langgraph is installed."""
    from maseval.interface.agents.langgraph import _check_langgraph_installed

    # Should not raise
    _check_langgraph_installed()


def test_langgraph_adapter_logs_after_run():
    """Test that LangGraphAgentAdapter.logs is populated after run().

    This test validates that the manual logging implementation in LangGraphAgentAdapter
    captures all relevant execution information including:
    - Timing information (timestamp, duration)
    - Query information
    - Token usage (extracted from message metadata)
    - Status (success/error)
    - State information (keys, message count)
    - Checkpoint metadata (if available)
    """
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage
    from langchain_core.messages.ai import UsageMetadata
    import time

    # Create a LangGraph agent with token usage metadata
    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        # Create AI message with usage metadata (simulates LLM response)
        # UsageMetadata is a TypedDict, so we create it properly
        response = AIMessage(
            content="Test response",
            usage_metadata=UsageMetadata(
                input_tokens=50,
                output_tokens=30,
                total_tokens=80,
            ),
        )
        return {"messages": messages + [response]}

    graph = StateGraph(State)  # type: ignore[arg-type]  # TypedDict in function scope
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Capture time before run
    time_before = time.time()

    # Run the agent
    adapter.run("Test query")

    # Capture time after run
    time_after = time.time()

    # Access logs
    logs = adapter.logs

    # Verify logs structure
    assert isinstance(logs, list)
    assert len(logs) >= 1  # At least one log entry

    # Get the most recent log entry
    log_entry = logs[-1]

    # Verify required fields
    assert "timestamp" in log_entry
    assert "query" in log_entry
    assert "duration_seconds" in log_entry
    assert "status" in log_entry

    # Verify field values
    assert log_entry["query"] == "Test query"
    assert log_entry["status"] == "success"
    assert log_entry["duration_seconds"] > 0
    assert log_entry["duration_seconds"] < (time_after - time_before) + 0.1  # Reasonable duration

    # Verify state information
    assert "state_keys" in log_entry
    assert "messages" in log_entry["state_keys"]
    assert "message_count" in log_entry
    assert log_entry["message_count"] >= 1

    # Verify token usage is captured from message metadata
    assert "input_tokens" in log_entry
    assert "output_tokens" in log_entry
    assert "total_tokens" in log_entry
    assert log_entry["input_tokens"] == 50
    assert log_entry["output_tokens"] == 30
    assert log_entry["total_tokens"] == 80


def test_langgraph_adapter_logs_multiple_runs():
    """Test that logs accumulate across multiple runs."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        response = AIMessage(content="Response")
        return {"messages": messages + [response]}

    graph = StateGraph(State)  # type: ignore[arg-type]  # TypedDict in function scope
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # First run
    adapter.run("Query 1")
    logs_after_first = adapter.logs
    assert len(logs_after_first) == 1
    assert logs_after_first[0]["query"] == "Query 1"

    # Second run
    adapter.run("Query 2")
    logs_after_second = adapter.logs
    assert len(logs_after_second) == 2
    assert logs_after_second[0]["query"] == "Query 1"
    assert logs_after_second[1]["query"] == "Query 2"


def test_langgraph_adapter_logs_error_handling():
    """Test that logs capture error information when agent execution fails."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: list

    def failing_node(state: State) -> State:
        raise ValueError("Intentional test error")

    graph = StateGraph(State)  # type: ignore[arg-type]  # TypedDict in function scope
    graph.add_node("agent", failing_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")

    # Run should raise an error
    try:
        adapter.run("Test query")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Intentional test error" in str(e)

    # Verify error is logged
    logs = adapter.logs
    assert len(logs) == 1

    log_entry = logs[0]
    assert log_entry["status"] == "error"
    assert "error" in log_entry
    assert "error_type" in log_entry
    assert log_entry["error_type"] == "ValueError"
    assert "Intentional test error" in log_entry["error"]
    assert log_entry["query"] == "Test query"
    assert log_entry["duration_seconds"] >= 0


def test_langgraph_adapter_logs_without_token_metadata():
    """Test that logs work correctly when messages don't have usage metadata."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        # Create response without usage metadata
        response = AIMessage(content="Test response")
        return {"messages": messages + [response]}

    graph = StateGraph(State)  # type: ignore[arg-type]  # TypedDict in function scope
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="test_agent")
    adapter.run("Test query")

    # Verify logs exist but token fields are None or 0
    logs = adapter.logs
    assert len(logs) == 1

    log_entry = logs[0]
    # Token fields should be present but with default values
    assert log_entry.get("input_tokens") in [None, 0]
    assert log_entry.get("output_tokens") in [None, 0]
    assert log_entry.get("total_tokens") in [None, 0]


# =============================================================================
# Stateful Graph Fixture and Tests
# =============================================================================


@pytest.fixture
def langgraph_stateful_setup():
    """Create a stateful LangGraph graph with MemorySaver checkpointer."""
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        messages = state["messages"]
        response = AIMessage(content="Test response")
        return {"messages": messages + [response]}

    graph = StateGraph(State)  # type: ignore[arg-type]
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "test_thread_1"}}

    return compiled, config


def test_langgraph_gather_config_with_checkpointer(langgraph_stateful_setup):
    """Test gather_config() captures checkpointer, sanitized config, and graph structure."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    compiled, config = langgraph_stateful_setup
    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="stateful_agent", config=config)

    result = adapter.gather_config()

    assert result["name"] == "stateful_agent"
    assert result["adapter_type"] == "LangGraphAgentAdapter"
    assert "langgraph_config" in result

    lg_config = result["langgraph_config"]

    # Checkpointer detected
    assert lg_config["has_checkpointer"] is True

    # Config sanitized: thread_id replaced with has_thread_id flag
    assert lg_config["config"]["configurable"]["has_thread_id"] is True
    assert "thread_id" not in lg_config["config"]["configurable"]

    # Graph structure info
    assert "graph_info" in lg_config
    assert lg_config["graph_info"]["num_nodes"] >= 2  # at least agent + __end__
    assert lg_config["graph_info"]["num_edges"] >= 1


def test_langgraph_gather_config_no_checkpointer():
    """Test gather_config() for a stateless graph without checkpointer."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AIMessage

    class State(TypedDict):
        messages: list

    def agent_node(state: State) -> State:
        return {"messages": state["messages"] + [AIMessage(content="Response")]}

    graph = StateGraph(State)  # type: ignore[arg-type]
    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="stateless_agent")
    result = adapter.gather_config()

    assert "langgraph_config" in result
    lg_config = result["langgraph_config"]
    assert lg_config["has_checkpointer"] is False
    # No config provided, so no config key in the output
    assert "config" not in lg_config


def test_langgraph_stateful_get_messages(langgraph_stateful_setup):
    """Test get_messages() fetches from persistent state for stateful graphs."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    compiled, config = langgraph_stateful_setup
    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="stateful_agent", config=config)

    adapter.run("Hello")
    messages = adapter.get_messages()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Test response"


def test_langgraph_run_checkpoint_metadata(langgraph_stateful_setup):
    """Test that logs include checkpoint metadata for stateful graphs."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter

    compiled, config = langgraph_stateful_setup
    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="stateful_agent", config=config)

    adapter.run("Hello")

    assert len(adapter.logs) == 1
    log_entry = adapter.logs[0]
    assert "checkpoint_metadata" in log_entry
    assert "source" in log_entry["checkpoint_metadata"]
    assert "step" in log_entry["checkpoint_metadata"]
    assert "checkpoint_created_at" in log_entry


# =============================================================================
# Message Conversion Tests
# =============================================================================


def test_langgraph_convert_tool_and_system_messages():
    """Test _convert_langchain_messages handles SystemMessage, ToolMessage, and AI with tool_calls."""
    from maseval.interface.agents.langgraph import LangGraphAgentAdapter
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: list

    graph = StateGraph(State)  # type: ignore[arg-type]
    graph.add_node("agent", lambda s: s)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    compiled = graph.compile()

    adapter = LangGraphAgentAdapter(agent_instance=compiled, name="msg_agent")

    lc_messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="What is 2+2?"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "calculator", "args": {"expr": "2+2"}}],
        ),
        ToolMessage(content="4", tool_call_id="call_1", name="calculator"),
        AIMessage(content="The answer is 4."),
    ]

    history = adapter._convert_langchain_messages(lc_messages)

    assert len(history) == 5

    assert history[0]["role"] == "system"
    assert history[0]["content"] == "You are helpful."

    assert history[1]["role"] == "user"
    assert history[1]["content"] == "What is 2+2?"

    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == ""
    assert history[2]["tool_calls"][0]["name"] == "calculator"

    assert history[3]["role"] == "tool"
    assert history[3]["content"] == "4"
    assert history[3]["tool_call_id"] == "call_1"
    assert history[3]["name"] == "calculator"

    assert history[4]["role"] == "assistant"
    assert history[4]["content"] == "The answer is 4."
    assert "tool_calls" not in history[4]


# =============================================================================
# LLMUser.get_tool() Test
# =============================================================================


def test_langgraph_llm_user_get_tool():
    """Test LangGraphLLMUser.get_tool() returns a LangChain tool named ask_user."""
    from maseval.interface.agents.langgraph import LangGraphLLMUser
    from langchain_core.tools import BaseTool
    from unittest.mock import Mock

    mock_model = Mock()
    user = LangGraphLLMUser(
        name="test_user",
        model=mock_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="hello",
    )

    tool = user.get_tool()

    assert isinstance(tool, BaseTool)
    assert tool.name == "ask_user"


# =============================================================================
# Phase 2 Hook: _MASEvalLangChainHandler Tests
# =============================================================================


def test_langgraph_callback_handler_chain_and_tool_events():
    """Test _MASEvalLangChainHandler captures chain, tool, and llm events."""
    from maseval.interface.agents.langgraph import _MASEvalLangChainHandler
    from uuid import uuid4

    trace_buffer = []
    handler = _MASEvalLangChainHandler(trace_buffer)

    run_id_chain = uuid4()
    parent_run_id = uuid4()
    run_id_tool = uuid4()
    run_id_llm = uuid4()

    # chain_start
    handler.on_chain_start(
        serialized={"id": ["langchain", "chains", "MyChain"]},
        inputs={"query": "test"},
        run_id=run_id_chain,
        parent_run_id=parent_run_id,
    )

    # tool_start
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2+2",
        run_id=run_id_tool,
    )

    # tool_end
    handler.on_tool_end(output="4", run_id=run_id_tool)

    # llm_end
    from unittest.mock import Mock

    handler.on_llm_end(response=Mock(), run_id=run_id_llm)

    # chain_end
    handler.on_chain_end(outputs={"result": "4"}, run_id=run_id_chain)

    assert len(trace_buffer) == 5

    # chain_start
    assert trace_buffer[0]["source"] == "langgraph_callback"
    assert trace_buffer[0]["event"] == "chain_start"
    assert trace_buffer[0]["chain_type"] == "MyChain"
    assert trace_buffer[0]["run_id"] == str(run_id_chain)
    assert trace_buffer[0]["parent_run_id"] == str(parent_run_id)

    # tool_start
    assert trace_buffer[1]["event"] == "tool_start"
    assert trace_buffer[1]["tool_name"] == "calculator"
    assert trace_buffer[1]["run_id"] == str(run_id_tool)

    # tool_end
    assert trace_buffer[2]["event"] == "tool_end"
    assert trace_buffer[2]["run_id"] == str(run_id_tool)

    # llm_end
    assert trace_buffer[3]["event"] == "llm_end"
    assert trace_buffer[3]["run_id"] == str(run_id_llm)

    # chain_end
    assert trace_buffer[4]["event"] == "chain_end"
    assert trace_buffer[4]["run_id"] == str(run_id_chain)


def test_langgraph_callback_handler_no_op_methods():
    """Test that no-op handler methods don't append to trace buffer."""
    from maseval.interface.agents.langgraph import _MASEvalLangChainHandler
    from uuid import uuid4

    trace_buffer = []
    handler = _MASEvalLangChainHandler(trace_buffer)
    run_id = uuid4()

    handler.on_chat_model_start(serialized={}, messages=[[]], run_id=run_id)
    handler.on_llm_start(serialized={}, prompts=["test"], run_id=run_id)
    handler.on_chain_error(error=RuntimeError("test"), run_id=run_id)
    handler.on_tool_error(error=RuntimeError("test"), run_id=run_id)
    handler.on_llm_error(error=RuntimeError("test"), run_id=run_id)
    handler.on_llm_new_token(token="hello", run_id=run_id)

    assert len(trace_buffer) == 0
