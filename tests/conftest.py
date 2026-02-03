"""Shared fixtures for MASEval tests."""

import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from maseval import (
    Benchmark,
    AgentAdapter,
    Environment,
    LLMUser,
    User,
    Task,
    TaskQueue,
    Evaluator,
    MessageHistory,
    SeedGenerator,
)
from maseval.core.model import ModelAdapter, ChatResponse


# ==================== Dummy Components ====================


class DummyModelAdapter(ModelAdapter):
    """Minimal model adapter for testing.

    Simulates model responses without making actual API calls. Useful for
    unit tests and integration tests that don't require real LLM inference.

    Supports both chat() and generate() methods, returning responses from
    a predefined list in round-robin fashion.
    """

    def __init__(
        self,
        model_id: str = "test-model",
        responses: Optional[List[Optional[str]]] = None,
        tool_calls: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
        usage: Optional[Dict[str, int]] = None,
        stop_reason: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize DummyModelAdapter.

        Args:
            model_id: Identifier for this model instance.
            responses: List of text responses to return. Cycles through the list.
                Can include None for tool-only responses.
            tool_calls: Optional list of tool call lists. If provided, each call
                returns the corresponding tool_calls (cycling through the list).
                Can include None for text-only responses.
            usage: Optional usage dict to include in all responses. Should have
                input_tokens, output_tokens, total_tokens.
            stop_reason: Optional stop_reason to include in all responses.
            seed: Seed for deterministic generation.
        """
        super().__init__(seed=seed)
        self._model_id = model_id
        self._responses: List[Optional[str]] = responses or ["test response"]
        self._tool_calls = tool_calls
        self._usage = usage
        self._stop_reason = stop_reason
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def _chat_impl(
        self,
        messages: List[Dict[str, Any]],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Return a mock response.

        Args:
            messages: Input messages (ignored for mock).
            generation_params: Generation parameters (ignored for mock).
            tools: Tool definitions (ignored for mock).
            tool_choice: Tool choice (ignored for mock).
            **kwargs: Additional arguments (ignored for mock).

        Returns:
            ChatResponse with mock content and optional tool_calls.
        """
        response = self._responses[self._call_count % len(self._responses)]

        # Get tool_calls for this response if provided
        response_tool_calls = None
        if self._tool_calls:
            response_tool_calls = self._tool_calls[self._call_count % len(self._tool_calls)]

        self._call_count += 1

        return ChatResponse(
            content=response,
            tool_calls=response_tool_calls,
            role="assistant",
            model=self._model_id,
            usage=self._usage,
            stop_reason=self._stop_reason,
        )


class DummyAgent:
    """Minimal agent for testing."""

    def __init__(self, name: str = "dummy_agent"):
        self.name = name
        self.calls = []

    def run(self, query: str) -> str:
        self.calls.append(query)
        return f"Response to: {query}"


class FakeSmolagentsModel:
    """Fake model compatible with smolagents' expectations.

    This model exposes the interface that smolagents' CodeAgent expects:
    - .model_id property
    - .generate() method that returns an object with .content and .token_usage

    This is used for testing framework adapters without requiring real LLM calls.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        self.model_id = "mock-model"
        self._responses = responses or ["mock response"]
        self._call_index = 0

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate a response compatible with smolagents."""
        text = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1

        class TokenUsage:
            def __init__(self):
                self.input_tokens = 0
                self.output_tokens = 0
                self.total_tokens = 0

        class GenerationResult:
            def __init__(self, text):
                # smolagents expects a ChatMessage-like object with `.content`
                # and a `.token_usage` attribute.
                self.content = text
                self.token_usage = TokenUsage()

            def __str__(self):
                return self.content

        return GenerationResult(text)


# ==================== CAMEL Mock Helpers ====================


class MockCamelMemory:
    """Mock CAMEL memory that returns messages in OpenAI format.

    Used for testing CAMEL integrations without requiring actual CAMEL agents.
    Compatible with both contract tests and integration tests.
    """

    def __init__(self, messages=None):
        self._messages = messages or []

    def get_context(self):
        """Return (messages, token_count) tuple like real CAMEL memory."""
        return self._messages, len(self._messages) * 10

    def add_message(self, msg):
        self._messages.append(msg)


class MockCamelResponse:
    """Mock CAMEL ChatAgentResponse.

    Simulates the response structure returned by CAMEL ChatAgent.step().
    Supports both msgs list and singular msg attribute patterns.
    """

    def __init__(
        self,
        content="Test response",
        terminated=False,
        info=None,
        use_msg=False,
    ):
        """Create a mock response.

        Args:
            content: Response content string
            terminated: Whether the conversation should end
            info: Optional info dict (can include usage, etc.)
            use_msg: If True, use singular 'msg' instead of 'msgs' list
        """
        from unittest.mock import Mock

        mock_msg = Mock()
        mock_msg.content = content

        if use_msg:
            self.msgs = []
            self.msg = mock_msg
        else:
            self.msgs = [mock_msg]
            self.msg = mock_msg

        self.terminated = terminated
        self.info = info or {}


class MockCamelAgent:
    """Mock CAMEL ChatAgent for contract testing.

    This class properly tracks messages in memory, which is required for
    contract tests that verify message history behavior. Use this for
    contract tests; use create_mock_camel_agent() for simpler unit tests.
    """

    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.memory = MockCamelMemory()
        self.system_message = None
        self.model = None
        self.tools = None

    def step(self, user_msg):
        """Process a message and return a response."""
        # Record user message in memory
        if hasattr(user_msg, "content"):
            content = user_msg.content
        else:
            content = str(user_msg)

        self.memory.add_message({"role": "user", "content": content})

        # Get response
        response_content = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1

        # Record assistant message in memory
        self.memory.add_message({"role": "assistant", "content": response_content})

        return MockCamelResponse(content=response_content)


def create_mock_camel_agent(
    responses=None,
    memory_messages=None,
    system_message=None,
    model_type=None,
    tools=None,
    raise_on_step=None,
):
    """Create a mock CAMEL ChatAgent for testing.

    Factory function that creates a Mock object simulating CAMEL ChatAgent
    behavior. Unlike MockCamelAgent class, this does NOT track messages in
    memory automatically - use MockCamelAgent for contract tests that need
    message history tracking.

    Args:
        responses: List of response contents (cycles through)
        memory_messages: Initial messages in memory
        system_message: Optional system message content
        model_type: Optional model type string
        tools: Optional list of mock tools
        raise_on_step: Exception to raise when step() is called

    Returns:
        Mock object simulating CAMEL ChatAgent
    """
    from unittest.mock import Mock

    responses = responses or ["Test response"]
    call_count = [0]  # Use list for mutable closure

    mock_agent = Mock()

    # Memory
    mock_agent.memory = MockCamelMemory(memory_messages) if memory_messages else None

    # System message
    if system_message:
        mock_sys_msg = Mock()
        mock_sys_msg.content = system_message
        mock_agent.system_message = mock_sys_msg
    else:
        mock_agent.system_message = None

    # Model - set both model and model_backend
    if model_type:
        mock_model = Mock()
        mock_model.model_type = model_type
        mock_agent.model = mock_model
        mock_agent.model_backend = mock_model  # CAMEL uses model_backend in newer versions
    else:
        mock_agent.model = None
        mock_agent.model_backend = None

    # Tools - set both tools and tool_list
    if tools:
        mock_agent.tools = tools
        mock_agent.tool_list = tools  # CAMEL uses tool_list in newer versions
    else:
        mock_agent.tools = None
        mock_agent.tool_list = None

    # Step method
    def step_impl(user_msg):
        if raise_on_step:
            raise raise_on_step
        response_content = responses[call_count[0] % len(responses)]
        call_count[0] += 1
        return MockCamelResponse(content=response_content)

    mock_agent.step = Mock(side_effect=step_impl)

    return mock_agent


class DummyAgentAdapter(AgentAdapter):
    """Test agent adapter that populates message history."""

    def _run_agent(self, query: str) -> str:
        import time

        # Track timing
        start_time = time.time()

        # Run underlying agent
        response = self.agent.run(query)

        # Store history directly
        if self.messages is None:
            self.messages = MessageHistory()
        self.messages.add_message(role="user", content=query)
        self.messages.add_message(role="assistant", content=response)

        # Populate logs to fulfill contract
        duration = time.time() - start_time
        self.logs.append(
            {
                "query": query,
                "duration_seconds": duration,
                "status": "success",
                "response": response,
            }
        )

        # Return final answer (not the history)
        return response


class DummyEnvironment(Environment):
    """Minimal environment for testing."""

    def setup_state(self, task_data: dict) -> Any:
        return task_data.copy()

    def create_tools(self) -> dict:
        return {}


class DummyUser(LLMUser):
    """Minimal user simulator for testing.

    Properly inherits from LLMUser base class, allowing tests to verify base class
    behavior. The simulator is replaced with a mock to avoid LLM calls.

    Supports all base class features:
    - max_turns / stop_token for multi-turn interaction
    - is_done() / respond() / get_initial_query()
    - messages (MessageHistory) for conversation tracking
    """

    def __init__(self, name: str, model: ModelAdapter, **kwargs):
        """Initialize DummyUser with proper base class inheritance.

        Args:
            name: User name
            model: ModelAdapter instance
            **kwargs: Forwarded to LLMUser base class:
                - user_profile: Dict of user attributes
                - scenario: Scenario description
                - initial_query: Optional initial message
                - max_turns: Max interaction turns (default: 1)
                - stop_tokens: Early termination tokens (default: None)
                - early_stopping_condition: Description of when to stop (default: None)
        """
        super().__init__(
            name=name,
            model=model,
            user_profile=kwargs.get("user_profile", {}),
            scenario=kwargs.get("scenario", "test scenario"),
            initial_query=kwargs.get("initial_query"),
            max_turns=kwargs.get("max_turns", 1),
            stop_tokens=kwargs.get("stop_tokens"),
            early_stopping_condition=kwargs.get("early_stopping_condition"),
        )
        # Replace simulator with a mock to avoid LLM calls
        # Tests can set simulator.return_value or side_effect as needed
        from unittest.mock import MagicMock

        self.simulator = MagicMock(return_value="Mock user response")

    def get_tool(self) -> Any:
        """Return a dummy tool for testing."""
        return None


class DummyEvaluator(Evaluator):
    """Minimal evaluator for testing."""

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.task = task
        self.environment = environment
        self.user = user

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Return traces as-is for testing."""
        return traces

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        return {"score": 1.0, "passed": True}


class DummyBenchmark(Benchmark):
    """Minimal benchmark for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_environment_calls = []
        self.setup_user_calls = []
        self.setup_agents_calls = []
        self.setup_evaluators_calls = []
        self.run_agents_calls = []
        self.evaluate_calls = []

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a dummy model adapter for testing."""
        return DummyModelAdapter(model_id=model_id)

    def setup_environment(self, agent_data: Dict[str, Any], task: Task, seed_generator: Optional[SeedGenerator] = None) -> Environment:
        self.setup_environment_calls.append((agent_data, task))
        return DummyEnvironment(task.environment_data)

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        seed_generator: Optional[SeedGenerator] = None,
    ) -> Optional[User]:
        self.setup_user_calls.append((agent_data, environment, task))
        return None

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: Optional[SeedGenerator] = None,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        self.setup_agents_calls.append((agent_data, environment, task, user))
        agent = DummyAgent()
        agent_adapter = DummyAgentAdapter(agent, "test_agent")
        return [agent_adapter], {"test_agent": agent_adapter}

    def setup_evaluators(
        self,
        environment: Environment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[User],
        seed_generator: Optional[SeedGenerator] = None,
    ) -> Sequence[Evaluator]:
        self.setup_evaluators_calls.append((environment, task, agents, user))
        return [DummyEvaluator(task, environment, user)]

    def run_agents(self, agents: Sequence[AgentAdapter], task: Task, environment: Environment, query: str) -> Any:
        self.run_agents_calls.append((agents, task, environment, query))
        # Run the first agent and return final answer
        return agents[0].run(query)

    def evaluate(
        self, evaluators: Sequence[Evaluator], agents: Dict[str, AgentAdapter], final_answer: Any, traces: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        self.evaluate_calls.append((evaluators, agents, final_answer, traces))
        # Call evaluators with new API signature: filter_traces first, then call with filtered traces and final_answer
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results


# ==================== Fixtures ====================


@pytest.fixture
def dummy_model():
    """Create a dummy model adapter."""
    return DummyModelAdapter()


@pytest.fixture
def dummy_agent():
    """Create a dummy agent."""
    return DummyAgent()


@pytest.fixture
def dummy_agent_adapter(dummy_agent):
    """Create a dummy agent adapter."""
    return DummyAgentAdapter(dummy_agent, "test_agent")


@pytest.fixture
def dummy_environment():
    """Create a dummy environment."""
    return DummyEnvironment({"test_key": "test_value"})


@pytest.fixture
def dummy_user(dummy_model):
    """Create a dummy user with an initial query."""
    return DummyUser(
        name="test_user",
        model=dummy_model,
        user_profile={"role": "tester"},
        scenario="test scenario",
        initial_query="Hello",
        max_turns=2,  # Allow at least one respond() call after initial query
    )


@pytest.fixture
def dummy_task():
    """Create a single dummy task."""
    return Task(
        query="Test query",
        environment_data={"key": "value"},
        evaluation_data={"expected": "result"},
        metadata={"difficulty": "easy"},
    )


@pytest.fixture
def dummy_task_queue():
    """Create a collection of dummy tasks."""
    return TaskQueue.from_list(
        [
            {"query": "Query 1", "environment_data": {"task": 1}},
            {"query": "Query 2", "environment_data": {"task": 2}},
            {"query": "Query 3", "environment_data": {"task": 3}},
        ]
    )


@pytest.fixture
def simple_benchmark(dummy_task_queue):
    """Create a simple benchmark instance with tasks and agent_data.

    Returns:
        tuple: (benchmark, tasks, agent_data) - Call as benchmark.run(tasks, agent_data=agent_data)
    """
    benchmark = DummyBenchmark()
    agent_data = {"model": "test"}
    return benchmark, dummy_task_queue, agent_data


@pytest.fixture
def agent_data():
    """Create sample agent configuration data."""
    return {"model": "test-model", "temperature": 0.7, "max_tokens": 100}
