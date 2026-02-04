"""Test trace collection from all components.

These tests verify that the trace collection system gathers execution data
from all registered components correctly.
"""

import pytest
from maseval import TaskQueue


@pytest.mark.core
class TestTraceCollection:
    """Tests for comprehensive trace collection."""

    def test_traces_collected_from_all_components(self):
        """Test that traces are collected from all registered components."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Verify traces have the expected structure
        assert "metadata" in traces
        assert "agents" in traces
        assert "environment" in traces

        # Verify metadata
        assert traces["metadata"]["total_components"] > 0
        assert "timestamp" in traces["metadata"]
        assert "thread_id" in traces["metadata"]

        # Verify agent traces
        assert "test_agent" in traces["agents"]
        agent_trace = traces["agents"]["test_agent"]
        assert "type" in agent_trace
        assert "gathered_at" in agent_trace
        assert "message_count" in agent_trace

        # Verify environment traces
        assert traces["environment"] is not None
        env_trace = traces["environment"]
        assert "type" in env_trace
        assert "gathered_at" in env_trace

    def test_traces_include_message_histories(self):
        """Test that agent traces include complete message histories."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test query", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Get agent trace
        agent_trace = traces["agents"]["test_agent"]

        # Verify message history is included
        assert "messages" in agent_trace
        messages = agent_trace["messages"]
        assert len(messages) > 0

        # Verify message structure
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test query"
        assert messages[1]["role"] == "assistant"

    def test_traces_survive_component_errors(self):
        """Test that trace collection continues even if a component fails."""
        from conftest import DummyBenchmark
        from maseval import AgentAdapter

        class FailingAgentAdapter(AgentAdapter):
            def _run_agent(self, query: str) -> str:
                return "success"

            def gather_traces(self):
                # Simulate an error during trace gathering
                raise RuntimeError("Trace gathering failed")

        class TestBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                from conftest import DummyAgent

                agent = DummyAgent()
                agent_adapter = FailingAgentAdapter(agent, "failing_agent")
                return [agent_adapter], {"failing_agent": agent_adapter}  # type: ignore[return-value]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        # Should complete without raising, with error info in traces
        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Verify trace collection handled the error
        assert "agents" in traces
        agent_trace = traces["agents"]["failing_agent"]
        assert "error" in agent_trace
        assert "Trace gathering failed" in agent_trace["error"]

    def test_model_adapter_traces_logs(self):
        """Test that ModelAdapter traces include logs."""
        from conftest import DummyModelAdapter, DummyBenchmark
        from maseval import AgentAdapter

        model = DummyModelAdapter()

        class ModelUsingAgentAdapter(AgentAdapter):
            def __init__(self, agent, name, model):
                super().__init__(agent, name)
                self.model = model

            def _run_agent(self, query: str) -> str:
                # Make some model calls
                self.model.generate("prompt 1")
                self.model.generate("prompt 2")
                return "result"

        class TestBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                from conftest import DummyAgent

                agent = DummyAgent()
                agent_adapter = ModelUsingAgentAdapter(agent, "test_agent", model)
                # Manually register the model
                self.register("models", "test_model", model)
                return [agent_adapter], {"test_agent": agent_adapter}  # type: ignore[return-value]

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = TestBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Verify model traces
        assert "models" in traces
        assert "test_model" in traces["models"]
        model_trace = traces["models"]["test_model"]

        assert "model_id" in model_trace
        assert "total_calls" in model_trace
        assert model_trace["total_calls"] == 2
        assert "logs" in model_trace
        assert len(model_trace["logs"]) == 2

    def test_environment_traces_tool_invocations(self):
        """Test that Environment traces include tool information."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Verify environment traces include tools
        env_trace = traces["environment"]
        assert "tool_count" in env_trace
        assert "tools" in env_trace

    def test_simulator_traces_retry_attempts(self):
        """Test that LLMSimulator traces include retry attempt history."""
        from maseval.core.simulator import ToolLLMSimulator
        from conftest import DummyModelAdapter

        model = DummyModelAdapter(responses=['{"result": "success"}'])
        simulator = ToolLLMSimulator(
            model=model,
            tool_name="test_tool",
            tool_description="A test tool",
            tool_inputs={"param": {"type": "string", "description": "A parameter"}},
            max_try=3,
        )

        # Call simulator with actual_inputs keyword
        _ = simulator(actual_inputs={"param": "test_value"})

        # Gather traces
        traces = simulator.gather_traces()

        # Verify structure
        assert "simulator_type" in traces
        assert "total_calls" in traces
        assert "successful_calls" in traces
        assert "failed_calls" in traces
        assert "logs" in traces

        # Verify log entries
        assert len(traces["logs"]) > 0
        entry = traces["logs"][0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "status" in entry

    def test_callback_traces_custom_data(self):
        """Test that callbacks can provide custom trace data."""
        from conftest import DummyBenchmark
        from maseval import BenchmarkCallback

        class CustomCallback(BenchmarkCallback):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def on_task_repeat_start(self, benchmark, task, repeat_idx):
                self.call_count += 1

            def gather_traces(self):
                return {
                    **super().gather_traces(),
                    "call_count": self.call_count,
                    "custom_data": "test_value",
                }

        callback = CustomCallback()
        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark(callbacks=[callback])

        # Register callback for tracing
        benchmark.register("callbacks", "custom_callback", callback)

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Verify callback traces
        assert "callbacks" in traces
        assert "custom_callback" in traces["callbacks"]
        callback_trace = traces["callbacks"]["custom_callback"]
        assert callback_trace["call_count"] == 1
        assert callback_trace["custom_data"] == "test_value"

    def test_traces_json_serializable(self):
        """Test that all traces are JSON serializable."""
        import json
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {"key": "value"}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Should be able to serialize to JSON
        try:
            json_str = json.dumps(traces)
            assert len(json_str) > 0

            # Should be able to deserialize back
            traces_restored = json.loads(json_str)
            assert traces_restored["metadata"]["total_components"] == traces["metadata"]["total_components"]
        except (TypeError, ValueError) as e:
            pytest.fail(f"Traces are not JSON serializable: {e}")

    def test_traces_contain_timestamps(self):
        """Test that all trace components include timestamps."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Check metadata timestamp
        assert "timestamp" in traces["metadata"]

        # Check agent timestamp
        agent_trace = traces["agents"]["test_agent"]
        assert "gathered_at" in agent_trace

        # Check environment timestamp
        env_trace = traces["environment"]
        assert "gathered_at" in env_trace

    def test_traces_include_component_types(self):
        """Test that all traces include component type information."""
        from conftest import DummyBenchmark

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DummyBenchmark()

        reports = benchmark.run(tasks, agent_data={"model": "test"})
        traces = reports[0]["traces"]

        # Check agent type
        agent_trace = traces["agents"]["test_agent"]
        assert "type" in agent_trace
        assert agent_trace["type"] == "DummyAgentAdapter"

        # Check environment type
        env_trace = traces["environment"]
        assert "type" in env_trace
        assert env_trace["type"] == "DummyEnvironment"
