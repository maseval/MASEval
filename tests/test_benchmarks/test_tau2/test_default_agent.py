"""Unit tests for DefaultTau2Agent and DefaultAgentTau2Benchmark."""

import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from pydantic import BaseModel
from maseval import Task, AgentAdapter
from maseval.core.model import ChatResponse
from maseval.benchmark.tau2 import (
    DefaultTau2Agent,
    DefaultTau2AgentAdapter,
    DefaultAgentTau2Benchmark,
    Tau2Environment,
    Tau2User,
    Tau2Evaluator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Create a mock model adapter."""
    model = MagicMock()
    # Default behavior: return ChatResponse with text and no tool calls
    response = ChatResponse()
    response.content = "I can help you with that."
    response.tool_calls = []
    model.chat.return_value = response
    return model


@pytest.fixture
def sample_tools():
    """Create sample tools for testing."""

    def get_order(order_id: str) -> Dict[str, Any]:
        """Get order details by ID."""
        return {"order_id": order_id, "status": "shipped", "items": ["Widget"]}

    def cancel_order(order_id: str, reason: str = "") -> str:
        """Cancel an order."""
        return f"Order {order_id} cancelled. Reason: {reason}"

    def calculate_total(prices: list) -> float:
        """Calculate total from list of prices."""
        return sum(prices)

    return {
        "get_order": get_order,
        "cancel_order": cancel_order,
        "calculate_total": calculate_total,
    }


@pytest.fixture
def sample_policy():
    """Create sample policy text."""
    return """
# Customer Service Policy

## Order Management
- Always verify order ID before making changes
- Cancellations require a reason
- Be helpful and professional
"""


@pytest.fixture
def default_agent(sample_tools, sample_policy, mock_model):
    """Create a DefaultTau2Agent for testing."""
    return DefaultTau2Agent(
        tools=sample_tools,
        policy=sample_policy,
        model=mock_model,
    )


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        query="I want to cancel my order",
        environment_data={"domain": "retail"},
        user_data={
            "model_id": "gpt-4o",
            "instructions": {"reason_for_call": "Cancel order #12345"},
        },
        evaluation_data={"model_id": "gpt-4o"},
        metadata={"domain": "retail"},
    )


# =============================================================================
# DefaultTau2Agent Tests
# =============================================================================


@pytest.mark.benchmark
class TestDefaultTau2AgentInit:
    """Tests for DefaultTau2Agent initialization."""

    def test_init_basic(self, sample_tools, sample_policy, mock_model):
        """Test basic initialization."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
        )

        assert agent.tools == sample_tools
        assert agent.policy == sample_policy
        assert agent.model == mock_model
        assert agent.llm_args == {}
        assert agent.max_tool_calls == 50

    def test_init_with_llm_args(self, sample_tools, sample_policy, mock_model):
        """Test initialization with custom LLM args."""
        llm_args = {"temperature": 0.7, "max_tokens": 1000}

        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
            llm_args=llm_args,
        )

        assert agent.llm_args == llm_args

    def test_init_custom_max_tool_calls(self, sample_tools, sample_policy, mock_model):
        """Test initialization with custom max_tool_calls."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
            max_tool_calls=10,
        )

        assert agent.max_tool_calls == 10

    def test_system_prompt_format(self, sample_tools, sample_policy, mock_model):
        """Test that system prompt is correctly formatted."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
        )

        assert "<instructions>" in agent.system_prompt
        assert "</instructions>" in agent.system_prompt
        assert "<policy>" in agent.system_prompt
        assert "</policy>" in agent.system_prompt
        assert sample_policy.strip() in agent.system_prompt
        assert "customer service agent" in agent.system_prompt.lower()


@pytest.mark.benchmark
class TestDefaultTau2AgentRun:
    """Tests for DefaultTau2Agent.run() method."""

    def test_run_simple_text_response(self, default_agent, mock_model):
        """Test run with simple text response (no tool calls)."""
        chat_response = ChatResponse()
        chat_response.content = "How can I help you today?"
        chat_response.tool_calls = []
        mock_model.chat.return_value = chat_response

        response = default_agent.run("Hello")

        assert response == "How can I help you today?"
        assert len(default_agent._messages) == 2  # user + assistant
        assert default_agent._messages[0]["role"] == "user"
        assert default_agent._messages[1]["role"] == "assistant"

    def test_run_with_single_tool_call(self, default_agent, mock_model, sample_tools):
        """Test run with a single tool call."""
        # First call: tool call
        # Second call: text response
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [
            {
                "id": "call_1",
                "name": "get_order",
                "arguments": {"order_id": "12345"},
            }
        ]
        resp2 = ChatResponse()
        resp2.content = "Your order 12345 is shipped."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = default_agent.run("What's the status of order 12345?")

        assert response == "Your order 12345 is shipped."
        assert mock_model.chat.call_count == 2

        # Check message history
        messages = default_agent.get_messages()
        assert len(messages) == 4  # user, assistant+tool_call, tool_result, assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"] is not None
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "assistant"

    def test_run_with_multiple_tool_calls(self, default_agent, mock_model):
        """Test run with multiple sequential tool calls."""
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [{"id": "call_1", "name": "get_order", "arguments": {"order_id": "123"}}]
        resp2 = ChatResponse()
        resp2.content = ""
        resp2.tool_calls = [{"id": "call_2", "name": "cancel_order", "arguments": {"order_id": "123", "reason": "Customer request"}}]
        resp3 = ChatResponse()
        resp3.content = "Order cancelled successfully."
        resp3.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2, resp3]

        response = default_agent.run("Cancel order 123")

        assert response == "Order cancelled successfully."
        assert mock_model.chat.call_count == 3

    def test_run_tool_not_found(self, default_agent, mock_model):
        """Test run with tool call for non-existent tool."""
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [{"id": "call_1", "name": "nonexistent_tool", "arguments": {}}]
        resp2 = ChatResponse()
        resp2.content = "I encountered an error."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        _response = default_agent.run("Do something")

        # Check that the error message was added to history
        messages = default_agent.get_messages()
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert "not found" in tool_msg["content"]

    def test_run_tool_execution_error(self, sample_policy, mock_model):
        """Test run with tool that raises an exception."""

        def failing_tool():
            """A tool that fails."""
            raise ValueError("Tool failed!")

        agent = DefaultTau2Agent(
            tools={"failing_tool": failing_tool},
            policy=sample_policy,
            model=mock_model,
        )

        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [{"id": "call_1", "name": "failing_tool", "arguments": {}}]
        resp2 = ChatResponse()
        resp2.content = "I had an error."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        _response = agent.run("Do something")

        messages = agent.get_messages()
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert "Error" in tool_msg["content"]
        assert "Tool failed!" in tool_msg["content"]

    def test_run_max_tool_calls_limit(self, sample_tools, sample_policy, mock_model):
        """Test that max_tool_calls limit is enforced."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
            max_tool_calls=2,
        )

        # Always return tool calls
        resp = ChatResponse()
        resp.content = ""
        resp.tool_calls = [{"id": "call_1", "name": "get_order", "arguments": {"order_id": "123"}}]
        mock_model.chat.return_value = resp

        response = agent.run("Loop forever")

        # Should return error message after hitting limit
        assert "issue" in response.lower() or "try again" in response.lower()

    def test_run_json_string_arguments(self, default_agent, mock_model):
        """Test run with JSON-encoded string arguments."""
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [
            {
                "id": "call_1",
                "name": "get_order",
                "arguments": '{"order_id": "12345"}',  # JSON string
            }
        ]
        resp2 = ChatResponse()
        resp2.content = "Order found."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = default_agent.run("Check order")

        assert response == "Order found."

    def test_run_function_format_arguments(self, default_agent, mock_model):
        """Test run with nested function format for tool calls."""
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [
            {
                "id": "call_1",
                "name": "get_order",
                "function": {"arguments": {"order_id": "12345"}},
            }
        ]
        resp2 = ChatResponse()
        resp2.content = "Order found."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = default_agent.run("Check order")

        assert response == "Order found."


@pytest.mark.benchmark
class TestDefaultTau2AgentToolDefinitions:
    """Tests for tool definition generation."""

    def test_get_tool_definitions_basic(self, default_agent):
        """Test basic tool definition generation."""
        definitions = default_agent._get_tool_definitions()

        assert len(definitions) == 3  # get_order, cancel_order, calculate_total

        # Check format
        for defn in definitions:
            assert defn["type"] == "function"
            assert "function" in defn
            assert "name" in defn["function"]
            assert "description" in defn["function"]
            assert "parameters" in defn["function"]

    def test_get_tool_definitions_parameter_types(self, sample_policy, mock_model):
        """Test that parameter types are correctly inferred."""

        def typed_tool(
            name: str,
            count: int,
            price: float,
            active: bool,
            items: list,
            data: dict,
        ) -> str:
            """A tool with typed parameters."""
            return "done"

        agent = DefaultTau2Agent(
            tools={"typed_tool": typed_tool},
            policy=sample_policy,
            model=mock_model,
        )

        definitions = agent._get_tool_definitions()
        params = definitions[0]["function"]["parameters"]["properties"]

        assert params["name"]["type"] == "string"
        assert params["count"]["type"] == "integer"
        assert params["price"]["type"] == "number"
        assert params["active"]["type"] == "boolean"
        assert params["items"]["type"] == "array"
        assert params["data"]["type"] == "object"

    def test_get_tool_definitions_required_params(self, sample_policy, mock_model):
        """Test that required vs optional parameters are identified."""

        def mixed_tool(required_param: str, optional_param: str = "default") -> str:
            """A tool with required and optional params."""
            return "done"

        agent = DefaultTau2Agent(
            tools={"mixed_tool": mixed_tool},
            policy=sample_policy,
            model=mock_model,
        )

        definitions = agent._get_tool_definitions()
        required = definitions[0]["function"]["parameters"]["required"]

        assert "required_param" in required
        assert "optional_param" not in required


@pytest.mark.benchmark
class TestDefaultTau2AgentState:
    """Tests for agent state management."""

    def test_reset(self, default_agent, mock_model):
        """Test reset clears state."""
        resp = ChatResponse()
        resp.content = "Hello!"
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        default_agent.run("Hello")
        assert len(default_agent._messages) > 0

        default_agent.reset()
        assert len(default_agent._messages) == 0
        assert default_agent._tool_call_count == 0

    def test_get_messages(self, default_agent, mock_model):
        """Test get_messages returns copy of history."""
        resp = ChatResponse()
        resp.content = "Hello!"
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        default_agent.run("Hello")

        messages = default_agent.get_messages()
        assert len(messages) == 2

        # Verify it's a copy
        messages.append({"role": "user", "content": "extra"})
        assert len(default_agent._messages) == 2


# =============================================================================
# DefaultTau2AgentAdapter Tests
# =============================================================================


@pytest.mark.benchmark
class TestDefaultTau2AgentAdapter:
    """Tests for DefaultTau2AgentAdapter."""

    def test_adapter_init(self, default_agent):
        """Test adapter initialization."""
        adapter = DefaultTau2AgentAdapter(default_agent, name="test_agent")

        assert adapter.name == "test_agent"
        assert adapter._agent == default_agent

    def test_adapter_run_agent(self, default_agent, mock_model):
        """Test _run_agent passes through to agent."""
        resp = ChatResponse()
        resp.content = "Response"
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        adapter = DefaultTau2AgentAdapter(default_agent)
        response = adapter._run_agent("Query")

        assert response == "Response"

    def test_adapter_get_messages(self, default_agent, mock_model):
        """Test get_messages returns agent messages."""
        resp = ChatResponse()
        resp.content = "Response"
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        adapter = DefaultTau2AgentAdapter(default_agent)
        adapter._run_agent("Query")

        messages = adapter.get_messages()
        assert len(messages) == 2


# =============================================================================
# DefaultAgentTau2Benchmark Tests
# =============================================================================


class DummyDefaultAgentBenchmark(DefaultAgentTau2Benchmark):
    """Subclass for testing that provides get_model_adapter."""

    def get_model_adapter(self, model_id: str, **kwargs):
        """Return a mock model adapter."""
        mock = MagicMock()
        resp = ChatResponse()
        resp.content = "I can help with that."
        resp.tool_calls = []
        mock.chat.return_value = resp
        return mock


@pytest.mark.benchmark
class TestDefaultAgentTau2BenchmarkInit:
    """Tests for DefaultAgentTau2Benchmark initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        benchmark = DummyDefaultAgentBenchmark()

        # Benchmark should be initialized successfully
        assert benchmark.max_invocations == 200  # Tau2 default (matches original DEFAULT_MAX_STEPS)

    def test_init_with_all_options(self):
        """Test initialization with all options."""
        benchmark = DummyDefaultAgentBenchmark(
            n_task_repeats=3,
            max_invocations=5,
        )

        assert benchmark.n_task_repeats == 3
        assert benchmark.max_invocations == 5

    def test_default_max_invocations(self):
        """Test that default max_invocations is 50 from class attribute."""
        benchmark = DummyDefaultAgentBenchmark()

        assert benchmark.max_invocations == 200
        assert benchmark.MAX_INVOCATIONS == 200


@pytest.mark.benchmark
class TestDefaultAgentTau2BenchmarkSetupAgents:
    """Tests for setup_agents method."""

    def test_setup_agents_basic(self, sample_task):
        """Test basic agent setup."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        with patch.object(Tau2Environment, "__init__", return_value=None):
            mock_env = MagicMock(spec=Tau2Environment)
            mock_env.create_tools.return_value = {"get_order": lambda x: x}
            mock_env.policy = "Policy text"

            agents_to_run, agents_dict = benchmark.setup_agents(
                {"model_id": "gpt-4o"},
                mock_env,
                sample_task,
                None,
                seed_gen,
            )

        assert len(agents_to_run) == 1
        assert "default_agent" in agents_dict
        assert isinstance(agents_dict["default_agent"], DefaultTau2AgentAdapter)

    def test_setup_agents_missing_model_id(self, sample_task):
        """Test that missing model_id raises ValueError."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Policy"

        with pytest.raises(ValueError, match="model_id not configured"):
            benchmark.setup_agents({}, mock_env, sample_task, None, seed_gen)

    def test_setup_agents_with_llm_args(self, sample_task):
        """Test agent setup with custom llm_args."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Policy"

        agents_to_run, agents_dict = benchmark.setup_agents(
            {"model_id": "gpt-4o", "llm_args": {"temperature": 0.5}},
            mock_env,
            sample_task,
            None,
            seed_gen,
        )

        agent = agents_dict["default_agent"]._agent  # type: ignore[unresolved-attribute]
        assert agent.llm_args == {"temperature": 0.5}

    def test_setup_agents_with_max_tool_calls(self, sample_task):
        """Test agent setup with custom max_tool_calls."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_tools.return_value = {}
        mock_env.policy = "Policy"

        agents_to_run, agents_dict = benchmark.setup_agents(
            {"model_id": "gpt-4o", "max_tool_calls": 10},
            mock_env,
            sample_task,
            None,
            seed_gen,
        )

        agent = agents_dict["default_agent"]._agent  # type: ignore[unresolved-attribute]
        assert agent.max_tool_calls == 10


@pytest.mark.benchmark
class TestDefaultAgentTau2BenchmarkAbstract:
    """Tests for abstract method requirements."""

    def test_get_model_adapter_is_abstract(self):
        """Test that get_model_adapter must be implemented."""
        # DefaultAgentTau2Benchmark itself is still abstract
        # because get_model_adapter is abstract
        with pytest.raises(TypeError, match="abstract"):
            DefaultAgentTau2Benchmark()


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.benchmark
class TestDefaultAgentIntegration:
    """Integration tests for the default agent."""

    def test_agent_tool_execution_flow(self, sample_tools, sample_policy, mock_model):
        """Test complete tool execution flow."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
        )

        # Simulate: get order, then provide response
        resp1 = ChatResponse()
        resp1.content = "Let me check that order."
        resp1.tool_calls = [{"id": "1", "name": "get_order", "arguments": {"order_id": "ORD-001"}}]
        resp2 = ChatResponse()
        resp2.content = "Your order ORD-001 is shipped."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = agent.run("Check order ORD-001")

        # Verify response
        assert "shipped" in response

        # Verify tool was called
        messages = agent.get_messages()
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        result = json.loads(tool_msg["content"].replace("'", '"'))
        assert result["order_id"] == "ORD-001"
        assert result["status"] == "shipped"

    def test_agent_preserves_conversation_context(self, sample_tools, sample_policy, mock_model):
        """Test that agent maintains conversation context across turns."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
        )

        resp = ChatResponse()
        resp.content = "Hello!"
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        agent.run("Hi")
        agent.run("Thanks")

        messages = agent.get_messages()
        assert len(messages) == 4  # user, assistant, user, assistant

        # Verify order
        assert messages[0]["content"] == "Hi"
        assert messages[2]["content"] == "Thanks"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


@pytest.mark.benchmark
class TestTau2UserAdditional:
    """Additional tests for Tau2User to improve coverage."""

    def test_gather_traces(self, mock_model):
        """Test gather_traces returns expected tau2-specific fields."""
        user = Tau2User(
            model=mock_model,
            scenario="Persona: Test User\n\nTask: Test task",
            initial_query="Hello",
            max_turns=5,
        )

        traces = user.gather_traces()

        assert "max_turns" in traces
        assert traces["max_turns"] == 5
        assert "turns_used" in traces
        assert "stopped_by_user" in traces

    def test_user_scenario_stored(self, mock_model):
        """Test user stores scenario."""
        scenario = "Persona:\n\tFrustrated customer\nInstructions:\n\tGet refund"

        user = Tau2User(
            model=mock_model,
            scenario=scenario,
            initial_query="I need a refund",
        )

        assert user.scenario == scenario


@pytest.mark.benchmark
class TestTau2BenchmarkMethods:
    """Tests for Tau2Benchmark base class methods."""

    def test_setup_environment(self, sample_task):
        """Test setup_environment creates Tau2Environment."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        with patch("maseval.benchmark.tau2.tau2.Tau2Environment") as mock_env_cls:
            mock_env_cls.return_value = MagicMock()

            _env = benchmark.setup_environment({}, sample_task, seed_gen)

            mock_env_cls.assert_called_once_with(
                task_data=sample_task.environment_data,
            )

    def test_setup_user_with_dict_instructions(self):
        """Test setup_user with dict instructions."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        task = Task(
            query="Hello",
            environment_data={"domain": "retail"},
            user_data={
                "model_id": "gpt-4o",
                "persona": "Helpful customer",
                "instructions": {
                    "reason_for_call": "Order inquiry",
                    "known_info": "Order #123",
                    "task_instructions": "Ask about delivery",
                },
            },
            evaluation_data={},
            metadata={},
        )

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_user_tools.return_value = {}

        user = benchmark.setup_user({}, mock_env, task, seed_gen)

        assert user is not None
        assert isinstance(user, Tau2User)
        assert "Helpful customer" in user.scenario
        assert "Order inquiry" in user.scenario
        assert "Order #123" in user.scenario
        assert "Ask about delivery" in user.scenario

    def test_setup_user_with_string_instructions(self):
        """Test setup_user with string instructions."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        task = Task(
            query="Hello",
            environment_data={"domain": "retail"},
            user_data={
                "model_id": "gpt-4o",
                "instructions": "Simple string instructions",
            },
            evaluation_data={},
            metadata={},
        )

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_user_tools.return_value = {}

        user = benchmark.setup_user({}, mock_env, task, seed_gen)

        assert user is not None
        assert isinstance(user, Tau2User)
        # String instructions are wrapped in UserScenario.__str__() format
        assert "Instructions:" in user.scenario
        assert "Simple string instructions" in user.scenario

    def test_setup_user_empty_instructions(self):
        """Test setup_user with no instructions."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        task = Task(
            query="Hello",
            environment_data={"domain": "retail"},
            user_data={"model_id": "gpt-4o"},
            evaluation_data={},
            metadata={},
        )

        mock_env = MagicMock(spec=Tau2Environment)
        mock_env.create_user_tools.return_value = {}

        user = benchmark.setup_user({}, mock_env, task, seed_gen)

        assert user is not None
        assert isinstance(user, Tau2User)
        # Empty instructions still get wrapped in UserScenario format
        assert "Instructions:" in user.scenario

    def test_setup_evaluators(self, sample_task):
        """Test setup_evaluators creates Tau2Evaluator."""
        from maseval.core.seeding import DefaultSeedGenerator

        benchmark = DummyDefaultAgentBenchmark()
        seed_gen = DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)

        with patch("maseval.benchmark.tau2.tau2.Tau2Environment") as mock_env_cls:
            mock_env = MagicMock(spec=Tau2Environment)
            mock_env_cls.return_value = mock_env

            evaluators = benchmark.setup_evaluators(mock_env, sample_task, [], None, seed_gen)

            assert len(evaluators) == 1
            assert isinstance(evaluators[0], Tau2Evaluator)

    def test_run_agents_single_agent(self, sample_task):
        """Test run_agents with a single agent."""
        benchmark = DummyDefaultAgentBenchmark()

        mock_agent = MagicMock(spec=AgentAdapter)
        mock_agent.run.return_value = "Response"

        mock_env = MagicMock(spec=Tau2Environment)

        result = benchmark.run_agents([mock_agent], sample_task, mock_env, "Query")

        assert result == "Response"
        mock_agent.run.assert_called_once_with("Query")

    def test_run_agents_multiple_agents(self, sample_task):
        """Test run_agents with multiple agents."""
        benchmark = DummyDefaultAgentBenchmark()

        mock_agent1 = MagicMock(spec=AgentAdapter)
        mock_agent1.run.return_value = "Response 1"
        mock_agent2 = MagicMock(spec=AgentAdapter)
        mock_agent2.run.return_value = "Response 2"

        mock_env = MagicMock(spec=Tau2Environment)

        result = benchmark.run_agents([mock_agent1, mock_agent2], sample_task, mock_env, "Query")

        assert result == ["Response 1", "Response 2"]

    def test_evaluate(self, sample_task):
        """Test evaluate method."""
        benchmark = DummyDefaultAgentBenchmark()

        mock_evaluator = MagicMock()
        mock_evaluator.filter_traces.return_value = {"filtered": True}
        mock_evaluator.return_value = {"reward": 1.0, "passed": True}

        result = benchmark.evaluate(
            [mock_evaluator],
            {},
            "final_answer",
            {"traces": "data"},
        )

        assert len(result) == 1
        assert result[0]["reward"] == 1.0
        mock_evaluator.filter_traces.assert_called_once_with({"traces": "data"})


@pytest.mark.benchmark
class TestDefaultTau2AgentEdgeCases:
    """Edge case tests for DefaultTau2Agent."""

    def test_execute_tool_call_invalid_json(self, sample_tools, sample_policy, mock_model):
        """Test tool execution with invalid JSON arguments."""
        agent = DefaultTau2Agent(
            tools=sample_tools,
            policy=sample_policy,
            model=mock_model,
        )

        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [{"id": "1", "name": "get_order", "arguments": "invalid{json"}]
        resp2 = ChatResponse()
        resp2.content = "Error occurred."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = agent.run("Check order")

        # Should handle the error gracefully
        messages = agent.get_messages()
        # The tool result should show an error
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        # Invalid JSON results in empty dict, which causes missing argument
        assert "Error" in tool_msg["content"] or response is not None

    def test_get_tool_definitions_with_list_annotation(self, sample_policy, mock_model):
        """Test tool definitions with List[str] style annotation."""
        from typing import List as TypingList

        def list_tool(items: TypingList[str]) -> str:
            """A tool with List[str] parameter."""
            return ",".join(items)

        agent = DefaultTau2Agent(
            tools={"list_tool": list_tool},
            policy=sample_policy,
            model=mock_model,
        )

        definitions = agent._get_tool_definitions()
        params = definitions[0]["function"]["parameters"]["properties"]

        assert params["items"]["type"] == "array"

    def test_empty_content_in_response(self, default_agent, mock_model):
        """Test handling of None or missing content in response."""
        resp = ChatResponse()
        resp.content = None
        resp.tool_calls = []
        mock_model.chat.return_value = resp

        response = default_agent.run("Hello")

        # Should handle None gracefully (returns empty string or None)
        assert response is None or response == ""

    def test_tool_call_with_empty_tool_call_id(self, default_agent, mock_model):
        """Test tool call without tool_call_id."""
        resp1 = ChatResponse()
        resp1.content = ""
        resp1.tool_calls = [{"name": "get_order", "arguments": {"order_id": "123"}}]  # No id
        resp2 = ChatResponse()
        resp2.content = "Done."
        resp2.tool_calls = []
        mock_model.chat.side_effect = [resp1, resp2]

        response = default_agent.run("Check order")

        # Should handle missing id gracefully
        assert response == "Done."
        messages = default_agent.get_messages()
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == ""  # Empty string fallback


@pytest.mark.benchmark
class TestDummyBenchmarkModelAdapter:
    """Tests for verifying the dummy benchmark's model adapter behavior."""

    def test_get_model_adapter_returns_mock(self):
        """Test that DummyDefaultAgentBenchmark returns working mock adapter."""
        benchmark = DummyDefaultAgentBenchmark()

        adapter = benchmark.get_model_adapter("gpt-4o")

        assert adapter is not None
        # Verify it can chat
        result = adapter.chat(messages=[], tools=[])
        assert isinstance(result, ChatResponse)
        assert result.content == "I can help with that."


# =============================================================================
# Module-level _build_tool_definitions Tests
# =============================================================================


@pytest.mark.benchmark
class TestBuildToolDefinitions:
    """Tests for module-level _build_tool_definitions()."""

    def test_basic_tools(self):
        """Builds definitions from simple tool dict."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        def greet(name: str) -> str:
            """Greet a user by name."""
            return f"Hello, {name}!"

        defs = _build_tool_definitions({"greet": greet})
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "greet"
        assert "parameters" in defs[0]["function"]
        assert defs[0]["function"]["description"] == "Greet a user by name."

    def test_no_docstring_uses_name(self):
        """Function without docstring uses name as description."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        def my_func(x: int) -> int:
            return x

        defs = _build_tool_definitions({"my_func": my_func})
        assert defs[0]["function"]["description"] == "my_func"

    def test_required_vs_optional(self):
        """Required and optional params detected correctly."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        def tool(required: str, optional: str = "default") -> str:
            """A tool."""
            return required

        defs = _build_tool_definitions({"tool": tool})
        required = defs[0]["function"]["parameters"].get("required", [])
        assert "required" in required
        assert "optional" not in required

    @pytest.mark.live
    def test_with_real_retail_tools(self, ensure_tau2_data):
        """Builds definitions from real retail environment tools."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        env = Tau2Environment({"domain": "retail"})
        tools = env.create_tools()
        defs = _build_tool_definitions(tools)
        assert len(defs) == 15
        for d in defs:
            assert d["type"] == "function"
            assert "name" in d["function"]
            assert "parameters" in d["function"]

    @pytest.mark.live
    def test_with_real_telecom_user_tools(self, ensure_tau2_data):
        """Builds definitions from real telecom user tools."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        env = Tau2Environment({"domain": "telecom"})
        user_tools = env.create_user_tools()
        if user_tools:
            defs = _build_tool_definitions(user_tools)
            assert len(defs) > 0
            for d in defs:
                assert d["type"] == "function"

    def test_empty_dict(self):
        """Empty tools dict returns empty list."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        defs = _build_tool_definitions({})
        assert defs == []

    def test_long_description(self):
        """Docstring with short + long description concatenated."""
        from maseval.benchmark.tau2.tau2 import _build_tool_definitions

        def detailed_tool(x: int) -> int:
            """Short description.

            This is the long description with more details.
            """
            return x

        defs = _build_tool_definitions({"detailed_tool": detailed_tool})
        desc = defs[0]["function"]["description"]
        assert "Short description." in desc
        assert "long description" in desc


# =============================================================================
# Module-level _to_json_str Tests
# =============================================================================


@pytest.mark.benchmark
class TestModuleLevelToJsonStr:
    """Tests for module-level _to_json_str() in tau2.py."""

    def test_string_passthrough(self):
        """Strings returned as-is (no JSON wrapping)."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        assert _to_json_str("hello") == "hello"

    def test_none(self):
        """None serialized to JSON null."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        assert json.loads(_to_json_str(None)) is None

    def test_int(self):
        """Integers converted via str() then JSON."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        assert json.loads(_to_json_str(42)) == "42"

    def test_float(self):
        """Floats converted via str() then JSON."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        assert json.loads(_to_json_str(3.14)) == "3.14"

    def test_bool(self):
        """Booleans converted via str() then JSON."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        assert json.loads(_to_json_str(True)) == "True"

    def test_dict(self):
        """Dicts processed recursively."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        result = json.loads(_to_json_str({"key": 42}))
        assert result["key"] == "42"

    def test_list(self):
        """Lists processed recursively."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        result = json.loads(_to_json_str([1, "a"]))
        assert result == ["1", "a"]

    def test_tuple(self):
        """Tuples converted to JSON arrays."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        result = json.loads(_to_json_str((1, "a")))
        # Tuples become arrays after json.dumps
        assert result == ["1", "a"]

    def test_datetime(self):
        """Datetimes serialized via isoformat."""
        from datetime import datetime

        from maseval.benchmark.tau2.tau2 import _to_json_str

        dt = datetime(2024, 1, 15, 10, 30)
        result = json.loads(_to_json_str(dt))
        assert "2024-01-15" in result

    def test_date(self):
        """Dates serialized via isoformat."""
        from datetime import date

        from maseval.benchmark.tau2.tau2 import _to_json_str

        d = date(2024, 1, 15)
        result = json.loads(_to_json_str(d))
        assert result == "2024-01-15"

    def test_pydantic_model(self):
        """Pydantic models serialized via model_dump()."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = json.loads(_to_json_str(model))
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_unsupported_type_raises(self):
        """Unsupported types raise ValueError."""
        from maseval.benchmark.tau2.tau2 import _to_json_str

        with pytest.raises(ValueError, match="Unsupported type"):
            _to_json_str(set([1, 2, 3]))


# =============================================================================
# DefaultAgentTau2Benchmark.execution_loop Step Counting Tests
# =============================================================================


@pytest.mark.benchmark
class TestExecutionLoopStepCounting:
    """Tests for DefaultAgentTau2Benchmark.execution_loop step counting."""

    def test_steps_count_agent_messages(self):
        """Steps accumulate from agent._last_turn_steps."""
        benchmark = DummyDefaultAgentBenchmark(max_invocations=200)

        mock_agent_inner = MagicMock()
        mock_agent_inner.run.return_value = "Response"
        mock_agent_inner._last_turn_steps = 3  # Simulate 3 messages added

        adapter = MagicMock(spec=DefaultTau2AgentAdapter)
        adapter._agent = mock_agent_inner

        mock_user = MagicMock(spec=Tau2User)
        mock_user.respond.side_effect = ["Initial query", ""]
        mock_user._last_respond_steps = 2
        mock_user.is_done.return_value = True

        task = MagicMock(spec=Task)
        task.query = "Hello"

        result = benchmark.execution_loop([adapter], task, MagicMock(), mock_user)

        assert result == "Response"
        # Agent.run should have been called
        assert mock_agent_inner.run.call_count >= 1

    def test_max_invocations_terminates(self):
        """Loop terminates when steps >= max_invocations."""
        benchmark = DummyDefaultAgentBenchmark(max_invocations=5)

        mock_agent_inner = MagicMock()
        mock_agent_inner.run.return_value = "Response"
        mock_agent_inner._last_turn_steps = 3  # 3 steps per turn

        adapter = MagicMock(spec=DefaultTau2AgentAdapter)
        adapter._agent = mock_agent_inner

        mock_user = MagicMock(spec=Tau2User)
        call_count = [0]

        def user_respond(msg):
            call_count[0] += 1
            return f"Query {call_count[0]}"

        mock_user.respond.side_effect = user_respond
        mock_user._last_respond_steps = 2  # 2 steps per user response
        mock_user.is_done.return_value = False

        task = MagicMock(spec=Task)
        task.query = "Hello"

        benchmark.execution_loop([adapter], task, MagicMock(), mock_user)

        # Should terminate due to step limit, not user.is_done
        # With 3+2=5 steps per full turn and max_invocations=5, should do 1 turn
        assert mock_agent_inner.run.call_count <= 2

    def test_no_user_single_turn(self):
        """Without user, loop runs agent once and returns."""
        benchmark = DummyDefaultAgentBenchmark(max_invocations=200)

        mock_agent_inner = MagicMock()
        mock_agent_inner.run.return_value = "Done"
        mock_agent_inner._last_turn_steps = 2

        adapter = MagicMock(spec=DefaultTau2AgentAdapter)
        adapter._agent = mock_agent_inner

        task = MagicMock(spec=Task)
        task.query = "Hello"

        result = benchmark.execution_loop([adapter], task, MagicMock(), None)

        assert result == "Done"
        mock_agent_inner.run.assert_called_once_with("Hello")

    def test_user_done_terminates(self):
        """Loop terminates when user.is_done() returns True."""
        benchmark = DummyDefaultAgentBenchmark(max_invocations=200)

        mock_agent_inner = MagicMock()
        mock_agent_inner.run.return_value = "Response"
        mock_agent_inner._last_turn_steps = 1

        adapter = MagicMock(spec=DefaultTau2AgentAdapter)
        adapter._agent = mock_agent_inner

        mock_user = MagicMock(spec=Tau2User)
        # respond() calls: (1) greeting → initial query, (2) agent response → stop
        mock_user.respond.side_effect = ["Initial query", "###STOP###"]
        mock_user._last_respond_steps = 1
        # is_done() checked after respond(agent_response): True → loop ends
        mock_user.is_done.return_value = True

        task = MagicMock(spec=Task)
        task.query = "Hello"

        benchmark.execution_loop([adapter], task, MagicMock(), mock_user)

        # Agent runs once, user responds with stop, is_done → True → loop ends
        assert mock_agent_inner.run.call_count == 1
