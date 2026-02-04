"""Shared fixtures for GAIA2 benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, etc.)
  These are automatically available via pytest's conftest inheritance.
- tests/test_benchmarks/test_gaia2/conftest.py: GAIA2-specific fixtures (this file)

GAIA2-Specific Components
-------------------------
- MockAREEnvironment: Mocks ARE's Environment for testing without ARE installed.
- MockARETool: Mocks ARE AppTool for testing tool wrappers.
- Gaia2AgentAdapter: Agent adapter for testing benchmark execution.
- ConcreteGaia2Benchmark: Concrete implementation for testing.

Since GAIA2 requires the ARE package (meta-agents-research-environments), tests mock
ARE components to allow testing without the dependency installed.
"""

import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
from unittest.mock import MagicMock

from conftest import DummyModelAdapter
from maseval import AgentAdapter, Task, TaskQueue, ModelAdapter
from maseval.core.seeding import SeedGenerator


# =============================================================================
# Mock ARE Components
# =============================================================================


class MockARETool:
    """Mock for ARE's AppTool class.

    Simulates an ARE tool for testing AREToolWrapper and Gaia2Environment.
    """

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        inputs: Optional[Dict[str, Any]] = None,
        return_value: Any = "mock result",
    ):
        self.name = name
        self.description = description
        self.inputs = inputs or {
            "properties": {"arg1": {"type": "string", "description": "First argument"}},
            "required": ["arg1"],
        }
        self._return_value = return_value
        self._calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs) -> Any:
        self._calls.append(kwargs)
        if callable(self._return_value):
            return self._return_value(**kwargs)
        return self._return_value

    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._calls


class MockAREApp:
    """Mock for ARE's App class that provides tools."""

    def __init__(self, tools: List[MockARETool]):
        self._tools = tools

    def get_tools(self) -> List[MockARETool]:
        """Get all tools from this app."""
        return self._tools


class MockAREEnvironment:
    """Mock for ARE's simulation Environment.

    Simulates ARE environment behavior for testing Gaia2Environment.
    Matches the real ARE interface where tools are accessed via apps.values().
    """

    def __init__(
        self,
        tools: Optional[List[MockARETool]] = None,
        completed_events: Optional[List[Any]] = None,
        current_time: float = 0.0,
    ):
        default_tools = tools or [
            MockARETool("Calendar__get_events", "Get calendar events"),
            MockARETool("Email__send", "Send an email"),
            MockARETool("SystemApp__get_current_time", "Get current time", return_value="2024-01-15T10:00:00"),
            MockARETool("SystemApp__wait_for_notification", "Wait for notification", return_value="No notifications"),
            MockARETool("AgentUserInterface__send_message_to_user", "Send message to user"),
        ]
        # Group tools by app name (part before __) to match real ARE structure
        apps_dict: Dict[str, List[MockARETool]] = {}
        for tool in default_tools:
            app_name = tool.name.split("__")[0] if "__" in tool.name else "default"
            if app_name not in apps_dict:
                apps_dict[app_name] = []
            apps_dict[app_name].append(tool)

        self.apps = {name: MockAREApp(tool_list) for name, tool_list in apps_dict.items()}
        self._completed_events = completed_events or []
        self._current_time = current_time
        self._initialized = False
        self._stopped = False

        # Also expose time_manager for compatibility
        self.time_manager = MagicMock()
        self.time_manager.current_time = current_time

    def initialize_scenario(self, scenario: Any) -> None:
        """Initialize scenario."""
        self._initialized = True

    def get_completed_events(self) -> List[Any]:
        """Get completed events for evaluation."""
        return self._completed_events

    def stop(self) -> None:
        """Stop the environment."""
        self._stopped = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_stopped(self) -> bool:
        return self._stopped


class MockJudgeResult:
    """Mock for ARE's judge evaluation result."""

    def __init__(self, passed: bool = True, partial_score: float = 1.0, event_results: Optional[List] = None):
        self.passed = passed
        self.partial_score = partial_score
        self.event_results = event_results or []


class MockGraphPerEventJudge:
    """Mock for ARE's GraphPerEventJudge."""

    def __init__(self, result: Optional[MockJudgeResult] = None):
        self._result = result or MockJudgeResult()

    def evaluate(self, oracle_events: Any, completed_events: Any, scenario: Any) -> MockJudgeResult:
        return self._result


# =============================================================================
# GAIA2-Specific Agent Adapter
# =============================================================================


class Gaia2AgentAdapter(AgentAdapter):
    """Agent adapter for testing GAIA2 benchmark execution.

    Provides controllable responses without needing a real agent implementation.
    """

    def __init__(self, name: str = "gaia2_test_agent"):
        super().__init__(agent_instance=MagicMock(), name=name)
        self._responses: List[str] = []
        self._call_count = 0
        self.run_calls: List[str] = []

    def set_responses(self, responses: List[str]) -> None:
        """Set canned responses for the agent."""
        self._responses = responses

    def _run_agent(self, query: str) -> str:
        self.run_calls.append(query)
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
        else:
            response = f"Task completed: {query[:50]}..."
        return response


# =============================================================================
# Concrete Benchmark Implementation
# =============================================================================


class ConcreteGaia2Benchmark:
    """Concrete Gaia2Benchmark implementation for testing.

    This is created dynamically to avoid importing Gaia2Benchmark at module load
    time (which would fail if ARE is not installed for the evaluator).
    """

    _instance = None

    @classmethod
    def create(cls, model_factory: Optional[Callable] = None, **kwargs):
        """Create a concrete benchmark instance.

        Args:
            model_factory: Callable that returns a ModelAdapter given a model name.
            **kwargs: Additional arguments passed to Gaia2Benchmark.

        Returns:
            ConcreteGaia2Benchmark instance.
        """
        from maseval.benchmark.gaia2 import Gaia2Benchmark, Gaia2Environment

        if model_factory is None:

            def model_factory(name):
                return DummyModelAdapter(
                    model_id=f"test-{name}",
                    responses=[
                        'Thought: I need to complete this task.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Done"}}<end_action>'
                    ],
                )

        class _ConcreteGaia2Benchmark(Gaia2Benchmark):
            def __init__(self, _model_factory, **kw):
                self._model_factory = _model_factory
                super().__init__(**kw)

            def get_model_adapter(self, model_id: str, **kw) -> ModelAdapter:
                register_name = kw.get("register_name", model_id)
                adapter = self._model_factory(register_name)
                if "register_name" in kw:
                    try:
                        self.register("models", kw["register_name"], adapter)
                    except ValueError:
                        pass
                return adapter

            def setup_agents(  # type: ignore[override]
                self,
                agent_data: Dict[str, Any],
                environment: Gaia2Environment,
                task: Task,
                user: Optional[Any],
                seed_generator: SeedGenerator,
            ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
                adapter = Gaia2AgentAdapter("test_agent")
                adapter.set_responses(["Task completed successfully."])
                return [adapter], {"test_agent": adapter}

        return _ConcreteGaia2Benchmark(model_factory, **kwargs)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_are_tool() -> MockARETool:
    """Create a single mock ARE tool."""
    return MockARETool(
        name="TestTool__do_something",
        description="A test tool that does something",
        inputs={
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
            "required": ["param1"],
        },
        return_value="Tool executed successfully",
    )


@pytest.fixture
def mock_are_tools() -> List[MockARETool]:
    """Create a set of mock ARE tools matching GAIA2 apps."""
    return [
        MockARETool("Calendar__get_events", "Get calendar events", return_value=[]),
        MockARETool("Calendar__create_event", "Create calendar event", return_value={"id": "evt_123"}),
        MockARETool("Email__send", "Send an email", return_value={"status": "sent"}),
        MockARETool("Email__read", "Read emails", return_value=[]),
        MockARETool("Messaging__send", "Send a message", return_value={"status": "sent"}),
        MockARETool("Contacts__search", "Search contacts", return_value=[]),
        MockARETool("SystemApp__get_current_time", "Get current time", return_value="2024-01-15T10:00:00Z"),
        MockARETool("SystemApp__wait_for_notification", "Wait for notification", return_value="No notifications"),
        MockARETool("AgentUserInterface__send_message_to_user", "Send message to user", return_value="Message sent"),
    ]


@pytest.fixture
def mock_are_environment(mock_are_tools) -> MockAREEnvironment:
    """Create a mock ARE environment."""
    return MockAREEnvironment(tools=mock_are_tools)


@pytest.fixture
def mock_judge_passed() -> MockGraphPerEventJudge:
    """Create a mock judge that returns passed=True."""
    return MockGraphPerEventJudge(MockJudgeResult(passed=True, partial_score=1.0))


@pytest.fixture
def mock_judge_failed() -> MockGraphPerEventJudge:
    """Create a mock judge that returns passed=False."""
    return MockGraphPerEventJudge(MockJudgeResult(passed=False, partial_score=0.0))


@pytest.fixture
def sample_gaia2_task() -> Task:
    """Create a sample GAIA2 task for testing."""
    return Task(
        id="gaia2_test_001",
        query="Schedule a meeting with John tomorrow at 2pm and send him an email confirmation.",
        environment_data={
            "scenario_json": {
                "id": "test_scenario",
                "initial_state": {},
                "oracle_events": [
                    {"type": "calendar_event_created", "data": {"title": "Meeting with John"}},
                    {"type": "email_sent", "data": {"recipient": "john@example.com"}},
                ],
            },
        },
        evaluation_data={
            "oracle_events": [
                {"type": "calendar_event_created"},
                {"type": "email_sent"},
            ],
        },
        user_data={},
        metadata={
            "capability": "execution",
            "difficulty": "medium",
        },
    )


@pytest.fixture
def sample_gaia2_task_queue(sample_gaia2_task) -> TaskQueue:
    """Create a TaskQueue with sample GAIA2 tasks."""
    tasks = [
        sample_gaia2_task,
        Task(
            id="gaia2_test_002",
            query="Check my calendar for today's events.",
            environment_data={"scenario_json": {"id": "test_2", "initial_state": {}, "oracle_events": []}},
            evaluation_data={"oracle_events": []},
            user_data={},
            metadata={"capability": "search"},
        ),
        Task(
            id="gaia2_test_003",
            query="Wait for a notification from the system.",
            environment_data={"scenario_json": {"id": "test_3", "initial_state": {}, "oracle_events": []}},
            evaluation_data={"oracle_events": []},
            user_data={},
            metadata={"capability": "time"},
        ),
    ]
    return TaskQueue(tasks)


@pytest.fixture
def gaia2_model_react() -> DummyModelAdapter:
    """Create a model that produces ReAct-style responses."""
    return DummyModelAdapter(
        model_id="test-react-model",
        responses=[
            # First call: use a tool
            'Thought: I need to check the calendar first.\n\nAction:\n{"action": "Calendar__get_events", "action_input": {}}<end_action>',
            # Second call: send final message
            'Thought: I have the information, now I should tell the user.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Your calendar is empty."}}<end_action>',
        ],
    )


@pytest.fixture
def gaia2_model_invalid_format() -> DummyModelAdapter:
    """Create a model that produces invalid format responses."""
    return DummyModelAdapter(
        model_id="test-invalid-model",
        responses=[
            "This response has no Thought or Action format.",
            "Still invalid format without proper JSON.",
        ],
    )


@pytest.fixture
def gaia2_model_termination() -> DummyModelAdapter:
    """Create a model that immediately terminates."""
    return DummyModelAdapter(
        model_id="test-termination-model",
        responses=[
            'Thought: Task is simple, I can respond directly.\n\nAction:\n{"action": "AgentUserInterface__send_message_to_user", "action_input": {"content": "Hello, I am ready to help!"}}<end_action>',
        ],
    )


@pytest.fixture
def gaia2_model_wait_notification() -> DummyModelAdapter:
    """Create a model that waits for notification."""
    return DummyModelAdapter(
        model_id="test-wait-model",
        responses=[
            'Thought: I need to wait for a notification.\n\nAction:\n{"action": "SystemApp__wait_for_notification", "action_input": {"timeout_seconds": 30}}<end_action>',
        ],
    )


@pytest.fixture
def sample_tools_dict() -> Dict[str, Callable]:
    """Create a sample tools dictionary for agent testing."""

    def mock_calendar_get_events(**kwargs):
        return []

    def mock_send_message(**kwargs):
        return "Message sent"

    def mock_get_time(**kwargs):
        return "2024-01-15T10:00:00Z"

    def mock_wait_notification(**kwargs):
        return "No notifications"

    return {
        "Calendar__get_events": mock_calendar_get_events,
        "AgentUserInterface__send_message_to_user": mock_send_message,
        "SystemApp__get_current_time": mock_get_time,
        "SystemApp__wait_for_notification": mock_wait_notification,
    }


@pytest.fixture
def sample_execution_traces() -> Dict[str, Any]:
    """Create sample execution traces for evaluator testing."""
    return {
        "agents": {
            "gaia2_agent": {
                "messages": [
                    {"role": "user", "content": "Do the task"},
                    {"role": "assistant", "content": "Thought: ...\n\nAction: ..."},
                ],
                "iteration_count": 3,
                "terminated": True,
            }
        },
        "tools": {
            "Calendar__get_events": {
                "invocations": [
                    {"inputs": {}, "outputs": [], "status": "success"},
                ]
            }
        },
        "environment": {
            "final_simulation_time": 120.5,
        },
    }


@pytest.fixture
def seed_gen():
    """Seed generator fixture for direct setup method calls.

    When seeding is disabled (global_seed=None), derive_seed() returns None.
    This fixture is for tests that call setup methods directly outside of run().
    """
    from maseval.core.seeding import DefaultSeedGenerator

    return DefaultSeedGenerator(global_seed=None).for_task("test").for_repetition(0)
