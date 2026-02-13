"""Test Environment functionality.

These tests verify that Environment manages state and tools correctly.
"""

import pytest


@pytest.mark.core
class TestEnvironment:
    """Tests for Environment state and tool management."""

    def test_environment_setup_state_called(self, dummy_environment):
        """Test that setup_state is called during initialization."""
        assert dummy_environment.state is not None
        assert "test_key" in dummy_environment.state
        assert dummy_environment.state["test_key"] == "test_value"

    def test_environment_create_tools_called(self, dummy_environment):
        """Test that create_tools is called during initialization."""
        assert dummy_environment.tools is not None
        assert isinstance(dummy_environment.tools, dict)

    def test_environment_get_tools_returns_dict(self, dummy_environment):
        """Test that get_tools() returns the tools dict."""
        tools = dummy_environment.get_tools()
        assert isinstance(tools, dict)
        assert tools is dummy_environment.tools

    def test_environment_get_tool_returns_none_for_missing(self, dummy_environment):
        """Test that get_tool() returns None for missing tools."""
        assert dummy_environment.get_tool("nonexistent") is None

    def test_environment_callbacks_triggered(self):
        """Test that environment callbacks are triggered."""
        from maseval import EnvironmentCallback
        from conftest import DummyEnvironment

        callback_calls = []

        class TrackingCallback(EnvironmentCallback):
            def on_setup_start(self, environment):
                callback_calls.append("start")

            def on_setup_end(self, environment):
                callback_calls.append("end")

        callback = TrackingCallback()
        _ = DummyEnvironment({"test": "data"}, callbacks=[callback])

        assert callback_calls == ["start", "end"]

    def test_environment_gather_traces_includes_tools(self, dummy_environment):
        """Test that gather_traces() includes tool information."""
        traces = dummy_environment.gather_traces()

        assert "type" in traces
        assert "gathered_at" in traces
        assert "tool_count" in traces
        assert "tools" in traces

        assert traces["tool_count"] == 0  # DummyEnvironment has no tools

    def test_environment_tool_history_captured(self):
        """Test that tool invocations can be captured if tools support it."""
        from conftest import DummyEnvironment

        # Create environment (tools would be added by subclass)
        env = DummyEnvironment({"test": "data"})

        # Verify tools dict is available
        assert hasattr(env, "tools")
        assert isinstance(env.tools, dict)

    def test_environment_gather_config(self, dummy_environment):
        """Test that gather_config() returns configuration."""
        config = dummy_environment.gather_config()

        assert "type" in config
        assert "gathered_at" in config
        assert config["type"] == "DummyEnvironment"
