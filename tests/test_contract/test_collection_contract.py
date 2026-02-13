"""Cross-component tracing and configuration collection contract tests.

Verifies that ALL components (Agent, Environment, User, Model, Simulator, Callback)
implement the same base contract for gather_traces() and gather_config(),
regardless of framework or component type.

This ensures MASEval's universal observability promise - all components are
traceable and configurable in a consistent, predictable way.
"""

import pytest
import json
from datetime import datetime
from typing import Any, Dict, Optional
from conftest import (
    DummyAgent,
    DummyAgentAdapter,
    DummyEnvironment,
    DummyUser,
    DummyModelAdapter,
    FakeSmolagentsModel,
)


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


def assert_base_trace_fields(
    traces: Dict[str, Any],
    component_name: Optional[str] = None,
    require_name: bool = False,
) -> None:
    """Verify traces have required base fields from TraceableMixin.

    Args:
        traces: The traces dict to validate
        component_name: Optional expected name value
        require_name: If True, require 'name' field (for Agent/User). If False, name is optional
                     but will be present after benchmark collection injects it.
    """
    assert isinstance(traces, dict), f"gather_traces() must return dict, got {type(traces)}"

    # Required fields
    assert "type" in traces, "Missing 'type' field in traces"
    assert isinstance(traces["type"], str), f"'type' must be string, got {type(traces['type'])}"
    assert len(traces["type"]) > 0, "'type' field cannot be empty"

    assert "gathered_at" in traces, "Missing 'gathered_at' field in traces"
    assert isinstance(traces["gathered_at"], str), f"'gathered_at' must be string, got {type(traces['gathered_at'])}"
    assert_is_valid_iso_timestamp(traces["gathered_at"])

    # Name is optional at component level, but injected by benchmark during collection
    if require_name or "name" in traces:
        assert "name" in traces, "Missing 'name' field in traces (required for this component type)"
        assert isinstance(traces["name"], str), f"'name' must be string, got {type(traces['name'])}"
        assert len(traces["name"]) > 0, "'name' field cannot be empty"

        if component_name is not None:
            assert traces["name"] == component_name, f"Expected name '{component_name}', got '{traces['name']}'"

    # JSON serializability
    assert_is_json_serializable(traces)


def assert_base_config_fields(
    config: Dict[str, Any],
    component_name: Optional[str] = None,
    require_name: bool = False,
) -> None:
    """Verify config has required base fields from ConfigurableMixin.

    Args:
        config: The config dict to validate
        component_name: Optional expected name value
        require_name: If True, require 'name' field (for Agent/User). If False, name is optional
                     but will be present after benchmark collection injects it.
    """
    assert isinstance(config, dict), f"gather_config() must return dict, got {type(config)}"

    # Required fields
    assert "type" in config, "Missing 'type' field in config"
    assert isinstance(config["type"], str), f"'type' must be string, got {type(config['type'])}"
    assert len(config["type"]) > 0, "'type' field cannot be empty"

    assert "gathered_at" in config, "Missing 'gathered_at' field in config"
    assert isinstance(config["gathered_at"], str), f"'gathered_at' must be string, got {type(config['gathered_at'])}"
    assert_is_valid_iso_timestamp(config["gathered_at"])

    # Name is optional at component level, but injected by benchmark during collection
    if require_name or "name" in config:
        assert "name" in config, "Missing 'name' field in config (required for this component type)"
        assert isinstance(config["name"], str), f"'name' must be string, got {type(config['name'])}"
        assert len(config["name"]) > 0, "'name' field cannot be empty"

        if component_name is not None:
            assert config["name"] == component_name, f"Expected name '{component_name}', got '{config['name']}'"

    # JSON serializability
    assert_is_json_serializable(config)


def remove_timestamp(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove timestamp field for comparison."""
    return {k: v for k, v in data.items() if k != "gathered_at"}


# ==================== Component Factory Functions ====================


def create_agent_for_framework(framework: str):
    """Create agent adapter for specified framework."""
    if framework == "dummy":
        agent = DummyAgent()
        return DummyAgentAdapter(agent, "test_agent")

    elif framework == "smolagents":
        pytest.importorskip("smolagents")
        from smolagents import CodeAgent
        from maseval.interface.agents.smolagents import SmolAgentAdapter

        mock_model = FakeSmolagentsModel(["Response"])
        agent = CodeAgent(tools=[], model=mock_model, max_steps=2)
        return SmolAgentAdapter(agent, "test_agent")

    elif framework == "langgraph":
        pytest.importorskip("langgraph")
        from langgraph.graph import StateGraph, END
        from typing_extensions import TypedDict
        from langchain_core.messages import AIMessage
        from maseval.interface.agents.langgraph import LangGraphAgentAdapter

        class State(TypedDict):
            messages: list

        def agent_node(state: State) -> State:
            return {"messages": state["messages"] + [AIMessage(content="Response")]}

        graph = StateGraph(State)  # type: ignore[arg-type]  # TypedDict in function scope
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

        return LangGraphAgentAdapter(graph.compile(), "test_agent")

    else:
        raise ValueError(f"Unknown framework: {framework}")


def create_environment():
    """Create environment instance."""
    return DummyEnvironment(task_data={})


def create_user():
    """Create user instance."""
    model = DummyModelAdapter()
    return DummyUser("test_user", model)


def create_model():
    """Create model adapter instance."""
    return DummyModelAdapter()


# ==================== Universal Contract Tests ====================


@pytest.mark.contract
@pytest.mark.interface
class TestUniversalTracingContract:
    """Test that ALL components follow the same tracing contract."""

    def test_agent_traces_have_base_fields(self):
        """agent adapters must include base trace fields including name."""
        agent = create_agent_for_framework("dummy")
        agent.run("Test query")

        traces = agent.gather_traces()
        assert_base_trace_fields(traces, "test_agent", require_name=True)

    def test_environment_traces_have_base_fields(self):
        """Environments must include base trace fields (name injected by benchmark)."""
        env = create_environment()

        traces = env.gather_traces()
        assert_base_trace_fields(traces, require_name=False)

    def test_user_traces_have_base_fields(self):
        """Users must include base trace fields including name."""
        user = create_user()

        traces = user.gather_traces()
        assert_base_trace_fields(traces, "test_user", require_name=True)

    def test_model_traces_have_base_fields(self):
        """Models must include base trace fields (name injected by benchmark)."""
        model = create_model()
        model.generate("Test prompt")

        traces = model.gather_traces()
        assert_base_trace_fields(traces, require_name=False)

    def test_traces_are_idempotent(self):
        """Calling gather_traces() multiple times returns consistent data."""
        agent = create_agent_for_framework("dummy")
        agent.run("Test query")

        traces1 = agent.gather_traces()
        traces2 = agent.gather_traces()

        # Type and name must be identical
        assert traces1["type"] == traces2["type"]
        assert traces1["name"] == traces2["name"]

        # Timestamps might differ slightly but should be close
        assert_is_valid_iso_timestamp(traces1["gathered_at"])
        assert_is_valid_iso_timestamp(traces2["gathered_at"])

    def test_traces_never_raise_exceptions(self):
        """gather_traces() must not raise exceptions even with missing data."""
        agent = create_agent_for_framework("dummy")
        # Don't run agent - should still gather traces without error

        try:
            traces = agent.gather_traces()
            assert_base_trace_fields(traces, "test_agent")
        except Exception as e:
            pytest.fail(f"gather_traces() raised exception: {e}")


@pytest.mark.contract
@pytest.mark.interface
class TestUniversalConfigContract:
    """Test that ALL components follow the same configuration contract."""

    def test_agent_config_has_base_fields(self):
        """agent adapters must include base config fields including name."""
        agent = create_agent_for_framework("dummy")

        config = agent.gather_config()
        assert_base_config_fields(config, "test_agent", require_name=True)

    def test_environment_config_has_base_fields(self):
        """Environments must include base config fields (name injected by benchmark)."""
        env = create_environment()

        config = env.gather_config()
        assert_base_config_fields(config, require_name=False)

    def test_user_config_has_base_fields(self):
        """Users must include base config fields including name."""
        user = create_user()

        config = user.gather_config()
        assert_base_config_fields(config, "test_user", require_name=True)

    def test_model_config_has_base_fields(self):
        """Models must include base config fields (name injected by benchmark)."""
        model = create_model()

        config = model.gather_config()
        assert_base_config_fields(config, require_name=False)

    def test_config_is_static(self):
        """Config should not change before/after execution (except timestamp)."""
        agent = create_agent_for_framework("dummy")

        config_before = agent.gather_config()
        agent.run("Test query")
        config_after = agent.gather_config()

        # Remove timestamps for comparison
        config_before_no_ts = remove_timestamp(config_before)
        config_after_no_ts = remove_timestamp(config_after)

        assert config_before_no_ts == config_after_no_ts, "Config should be static (not runtime state)"

    def test_config_is_idempotent(self):
        """Calling gather_config() multiple times returns identical data (except timestamp)."""
        agent = create_agent_for_framework("dummy")

        config1 = agent.gather_config()
        config2 = agent.gather_config()

        # Type and name must be identical
        assert config1["type"] == config2["type"]
        assert config1["name"] == config2["name"]

        # Non-timestamp fields should match
        assert remove_timestamp(config1) == remove_timestamp(config2)

    def test_config_never_raises_exceptions(self):
        """gather_config() must not raise exceptions."""
        agent = create_agent_for_framework("dummy")

        try:
            config = agent.gather_config()
            assert_base_config_fields(config, "test_agent")
        except Exception as e:
            pytest.fail(f"gather_config() raised exception: {e}")


@pytest.mark.contract
@pytest.mark.interface
@pytest.mark.parametrize("framework", ["dummy", "smolagents", "langgraph"])
class TestCrossFrameworkTracingConsistency:
    """Test that agent adapters have consistent tracing across frameworks."""

    def test_all_frameworks_return_same_base_structure(self, framework):
        """All frameworks must return same base trace structure."""
        agent = create_agent_for_framework(framework)
        agent.run("Test query")

        traces = agent.gather_traces()

        # All must have base fields
        assert_base_trace_fields(traces, "test_agent")

        # All agents must have these fields (AgentAdapter contract)
        assert "agent_type" in traces, "Missing 'agent_type' field"
        assert "message_count" in traces, "Missing 'message_count' field"
        assert "messages" in traces, "Missing 'messages' field"
        assert "callbacks" in traces, "Missing 'callbacks' field"

        # Verify types
        assert isinstance(traces["agent_type"], str)
        assert isinstance(traces["message_count"], int)
        assert isinstance(traces["messages"], list)
        assert isinstance(traces["callbacks"], list)

    def test_all_frameworks_return_same_base_config(self, framework):
        """All frameworks must return same base config structure."""
        agent = create_agent_for_framework(framework)

        config = agent.gather_config()

        # All must have base fields
        assert_base_config_fields(config, "test_agent")

        # All agents must have these fields (AgentAdapter contract)
        assert "agent_type" in config, "Missing 'agent_type' field"
        assert "adapter_type" in config, "Missing 'adapter_type' field"
        assert "callbacks" in config, "Missing 'callbacks' field"

        # Verify types
        assert isinstance(config["agent_type"], str)
        assert isinstance(config["adapter_type"], str)
        assert isinstance(config["callbacks"], list)

    def test_all_frameworks_have_json_serializable_traces(self, framework):
        """Traces from all frameworks must be JSON serializable."""
        agent = create_agent_for_framework(framework)
        agent.run("Test query")

        traces = agent.gather_traces()
        assert_is_json_serializable(traces)

    def test_all_frameworks_have_json_serializable_config(self, framework):
        """Config from all frameworks must be JSON serializable."""
        agent = create_agent_for_framework(framework)

        config = agent.gather_config()
        assert_is_json_serializable(config)

    def test_all_frameworks_preserve_name(self, framework):
        """All frameworks must preserve the agent name."""
        agent = create_agent_for_framework(framework)

        traces = agent.gather_traces()
        config = agent.gather_config()

        assert traces["name"] == "test_agent"
        assert config["name"] == "test_agent"
        assert agent.name == "test_agent"


@pytest.mark.contract
@pytest.mark.interface
class TestCrossComponentConsistency:
    """Test that different component types follow the same patterns."""

    def test_all_components_have_consistent_trace_structure(self):
        """All component types must have same base trace structure."""
        agent = create_agent_for_framework("dummy")
        env = create_environment()
        user = create_user()
        model = create_model()

        # Run components to generate traces
        agent.run("Test")
        model.generate("Test")

        # Gather all traces
        all_traces = [
            agent.gather_traces(),
            env.gather_traces(),
            user.gather_traces(),
            model.gather_traces(),
        ]

        # All must have base fields (type, gathered_at always present)
        # Name is present for Agent/User, injected by benchmark for others
        for traces in all_traces:
            assert "type" in traces
            assert "gathered_at" in traces
            # Name present in Agent/User, optional in Environment/Model (injected by benchmark)
            if traces["type"] in ["DummyAgentAdapter", "DummyUser"]:
                assert "name" in traces
                assert isinstance(traces["name"], str)
            assert isinstance(traces["type"], str)
            assert isinstance(traces["gathered_at"], str)
            assert_is_valid_iso_timestamp(traces["gathered_at"])
            assert_is_json_serializable(traces)

    def test_all_components_have_consistent_config_structure(self):
        """All component types must have same base config structure."""
        agent = create_agent_for_framework("dummy")
        env = create_environment()
        user = create_user()
        model = create_model()

        # Gather all configs
        all_configs = [
            agent.gather_config(),
            env.gather_config(),
            user.gather_config(),
            model.gather_config(),
        ]

        # All must have base fields (type, gathered_at always present)
        # Name is present for Agent/User, injected by benchmark for others
        for config in all_configs:
            assert "type" in config
            assert "gathered_at" in config
            # Name present in Agent/User, optional in Environment/Model (injected by benchmark)
            if config["type"] in ["DummyAgentAdapter", "DummyUser"]:
                assert "name" in config
                assert isinstance(config["name"], str)
            assert isinstance(config["type"], str)
            assert isinstance(config["gathered_at"], str)
            assert_is_valid_iso_timestamp(config["gathered_at"])
            assert_is_json_serializable(config)
