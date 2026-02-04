"""Shared fixtures for MACS benchmark tests.

Fixture Hierarchy
-----------------
- tests/conftest.py: Generic fixtures (DummyModelAdapter, dummy_model, dummy_agent_adapter, etc.)
  These are automatically available via pytest's conftest inheritance.
- tests/test_benchmarks/test_macs/conftest.py: MACS-specific fixtures (this file)

MACS tests can use fixtures from both levels - pytest handles this automatically.

MACS-Specific Components
------------------------
- MACSAgentAdapter: Returns MessageHistory (not strings) matching the AgentAdapter contract.
  Used for testing MACSBenchmark.run_agents() without a real agent implementation.
- ConcreteMACSBenchmark: Concrete implementation of MACSBenchmark for testing.

For model adapters, MACS tests use DummyModelAdapter from tests/conftest.py with JSON
responses passed explicitly (e.g., DummyModelAdapter(responses=['{"text": "..."}'])).
"""

import pytest
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock

from conftest import DummyModelAdapter
from maseval import AgentAdapter, Task, User, MessageHistory, TaskQueue
from maseval.benchmark.macs import MACSBenchmark, MACSEnvironment


# =============================================================================
# Session-Scoped Setup
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def ensure_macs_templates():
    """Download MACS prompt templates before running any benchmark tests.

    This fixture runs once per test session and ensures that the prompt template
    files (user.txt, system.txt, user_simulator.txt) exist locally. These templates
    are required by MACSEvaluator and MACSUser.
    """
    from maseval.benchmark.macs.data_loader import download_prompt_templates

    download_prompt_templates()


# =============================================================================
# MACS-Specific Mock Components
# =============================================================================


class MACSAgentAdapter(AgentAdapter):
    """Agent adapter for testing MACS benchmark execution.

    Unlike DummyAgentAdapter (which wraps a real agent object), this adapter
    provides controllable responses without needing a real agent implementation.

    Used for testing MACSBenchmark.run_agents() and integration scenarios.

    Attributes:
        run_calls: List of queries passed to _run_agent (for verification).
    """

    def __init__(self, name: str = "macs_test_agent"):
        super().__init__(agent_instance=MagicMock(), name=name)
        self._responses: List[str] = []
        self._call_count = 0
        self.run_calls: List[str] = []

    def set_responses(self, responses: List[str]) -> None:
        """Set canned responses for the agent."""
        self._responses = responses

    def _run_agent(self, query: str) -> MessageHistory:
        self.run_calls.append(query)
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
        else:
            response = f"Response to: {query}"
        return MessageHistory([{"role": "assistant", "content": response}])


# =============================================================================
# MACS-Specific Benchmark Implementation
# =============================================================================


class ConcreteMACSBenchmark(MACSBenchmark):
    """Concrete MACSBenchmark implementation for testing.

    MACSBenchmark is abstract (setup_agents and get_model_adapter must be implemented).
    This provides a minimal implementation using MACSAgentAdapter and a configurable
    model factory.
    """

    def __init__(
        self,
        model_factory: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initialize with optional model factory.

        Args:
            model_factory: Either a callable that takes a model name and returns a ModelAdapter,
                          or a single ModelAdapter instance (for convenience in simple tests).
                          If not provided, creates DummyModelAdapter instances.
            **kwargs: Additional arguments passed to MACSBenchmark
        """
        # Handle both callable factory and single model instance
        if model_factory is None:
            self._model_factory = lambda model_name: DummyModelAdapter(
                model_id=f"test-model-{model_name}",
                responses=['{"text": "Default response", "details": {}}'],
            )
        elif callable(model_factory):
            self._model_factory = model_factory
        else:
            # Single model instance - create a factory that always returns it
            self._model_factory = lambda model_name: model_factory
        super().__init__(**kwargs)

    def get_model_adapter(self, model_id: str, **kwargs):
        """Create a model adapter for the given component.

        For testing, the factory is called with register_name (if provided) instead
        of model_id, allowing test factories to return different responses based on
        the component type (e.g., "user_simulator", "evaluator_user_gsr").
        """
        # Use register_name for factory lookup if provided, else fall back to model_id
        factory_key = kwargs.get("register_name", model_id)
        adapter = self._model_factory(factory_key)

        # Register for tracing if registration info provided
        # Skip registration if no explicit register_name - tests often share model instances
        register_name = kwargs.get("register_name")
        if register_name:
            # Check if this adapter is already registered (same factory can return same instance)
            try:
                self.register("models", register_name, adapter)
            except ValueError:
                pass  # Already registered - that's fine for tests

        return adapter

    def setup_agents(  # type: ignore[invalid-method-override]
        self,
        agent_data: Dict[str, Any],
        environment: MACSEnvironment,
        task: Task,
        user: Optional[User],
        seed_generator=None,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create test agents using MACSAgentAdapter."""
        adapter = MACSAgentAdapter("macs_test_agent")
        return [adapter], {"macs_test_agent": adapter}


# =============================================================================
# Model Fixtures
#
# These use DummyModelAdapter from tests/conftest.py with JSON responses.
# =============================================================================


@pytest.fixture
def macs_model():
    """Model adapter with default JSON responses for MACS tests.

    Returns responses in ToolLLMSimulator format: {"text": "...", "details": {...}}
    """
    return DummyModelAdapter(responses=['{"text": "Default response", "details": {}}'])


@pytest.fixture
def macs_model_factory(macs_model):
    """Model factory that always returns the same macs_model.

    Used for MACSEnvironment which requires a callable factory.
    """
    return lambda model_name: macs_model


@pytest.fixture
def macs_model_evaluator():
    """Model configured for MACSEvaluator tests.

    Returns JSON array format expected by MACSEvaluator._parse_evaluation_response().
    """
    return DummyModelAdapter(responses=['[{"assertion": "Test", "answer": "TRUE", "evidence": "OK"}]'])


@pytest.fixture
def macs_model_tool():
    """Model configured for ToolLLMSimulator tests."""
    return DummyModelAdapter(responses=['{"text": "Tool executed successfully", "details": {}}'])


@pytest.fixture
def macs_model_user():
    """Model configured for UserLLMSimulator tests."""
    return DummyModelAdapter(responses=['{"text": "Yes, that works for me.", "details": {}}'])


# =============================================================================
# MACS Tool Specification Fixtures
# =============================================================================


@pytest.fixture
def simple_tool_spec():
    """Simple tool specification for basic tests."""
    return {
        "name": "search_flights",
        "description": "Search for available flights",
        "input_schema": {
            "properties": {
                "origin": {"type": "string", "description": "Origin airport code"},
                "destination": {"type": "string", "description": "Destination airport code"},
            }
        },
    }


@pytest.fixture
def complex_tool_spec():
    """Tool specification with various input types."""
    return {
        "name": "book_hotel",
        "description": "Book a hotel room",
        "input_schema": {
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "check_in": {"data_type": "date", "description": "Check-in date"},
                "guests": {"type": "integer", "description": "Number of guests"},
                "amenities": {"type": "array", "description": "Requested amenities"},
            }
        },
    }


@pytest.fixture
def minimal_tool_spec():
    """Minimal tool specification with only name."""
    return {"name": "simple_action"}


@pytest.fixture
def sample_tool_specs():
    """Sample tool specifications in MACS format (tool groups with actions)."""
    return [
        {
            "tool_name": "flight_tools",
            "actions": [
                {
                    "name": "search_flights",
                    "description": "Search for available flights",
                    "input_schema": {
                        "properties": {
                            "origin": {"type": "string", "description": "Origin airport"},
                            "destination": {"type": "string", "description": "Destination airport"},
                        }
                    },
                },
                {
                    "name": "book_flight",
                    "description": "Book a flight",
                    "input_schema": {
                        "properties": {
                            "flight_id": {"type": "string", "description": "Flight ID to book"},
                        }
                    },
                },
            ],
        },
        {
            "tool_name": "hotel_tools",
            "actions": [
                {
                    "name": "search_hotels",
                    "description": "Search for hotels",
                    "input_schema": {
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        }
                    },
                },
            ],
        },
    ]


# =============================================================================
# MACS Task Fixtures
# =============================================================================


@pytest.fixture
def sample_task():
    """Sample MACS task with typical structure.

    Includes model_id in environment_data, user_data, and evaluation_data
    as required by MACSBenchmark.
    """
    return Task(
        query="Book a flight to NYC",
        environment_data={
            "model_id": "test-model",
            "tools": [
                {
                    "tool_name": "flight_tools",
                    "actions": [
                        {"name": "search_flights", "description": "Search flights"},
                    ],
                }
            ],
        },
        user_data={
            "model_id": "test-model",
        },
        evaluation_data={
            "model_id": "test-model",
            "assertions": [
                "user: Booking confirmed",
                "agent: Database updated",
            ],
        },
        metadata={"scenario": "Business trip to NYC"},
    )


@pytest.fixture
def sample_task_no_scenario():
    """Task without scenario in metadata."""
    return Task(
        query="Test query",
        environment_data={"model_id": "test-model", "tools": []},
        user_data={"model_id": "test-model"},
        evaluation_data={"model_id": "test-model", "assertions": []},
        metadata={},
    )


@pytest.fixture
def sample_task_no_assertions():
    """Task with no assertions."""
    return Task(
        query="Simple query",
        environment_data={"model_id": "test-model"},
        user_data={"model_id": "test-model"},
        evaluation_data={"model_id": "test-model", "assertions": []},
        metadata={"scenario": "Simple scenario"},
    )


@pytest.fixture
def travel_task():
    """Detailed travel domain task for integration tests."""
    return Task(
        query="I need to book a flight from San Francisco to New York for next Monday.",
        environment_data={
            "model_id": "test-model",
            "tools": [
                {
                    "tool_name": "travel_tools",
                    "actions": [
                        {
                            "name": "search_flights",
                            "description": "Search for available flights between cities",
                            "input_schema": {
                                "properties": {
                                    "origin": {"type": "string", "description": "Origin city or airport code"},
                                    "destination": {"type": "string", "description": "Destination city or airport code"},
                                    "date": {"type": "string", "description": "Travel date"},
                                }
                            },
                        },
                        {
                            "name": "book_flight",
                            "description": "Book a specific flight",
                            "input_schema": {
                                "properties": {
                                    "flight_id": {"type": "string", "description": "Flight identifier"},
                                    "passenger_name": {"type": "string", "description": "Passenger name"},
                                }
                            },
                        },
                    ],
                }
            ],
        },
        user_data={"model_id": "test-model"},
        evaluation_data={
            "model_id": "test-model",
            "assertions": [
                "user: The user's flight booking request was acknowledged",
                "user: The user received flight options or a confirmation",
                "agent: The search_flights tool was called with correct parameters",
            ],
        },
        metadata={
            "scenario": """Goal: The user wants to book a flight from San Francisco to New York.

Background:
* User's name is Alice Johnson
* User is a business traveler
* User prefers morning flights
* User has Delta SkyMiles membership""",
            "category": "travel",
            "complexity": "simple",
        },
    )


@pytest.fixture
def macs_task_queue(sample_task, travel_task):
    """Collection of MACS tasks for benchmark.run() tests."""
    return TaskQueue.from_list([sample_task, travel_task])


# =============================================================================
# MACS Task Data Fixtures (for Environment creation)
# =============================================================================


@pytest.fixture
def sample_task_data(sample_tool_specs):
    """Sample task data dict for Environment creation."""
    return {
        "environment_data": {
            "tools": sample_tool_specs,
        }
    }


# =============================================================================
# MACS Agent Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_data():
    """Sample MACS agent configuration."""
    return {
        "agents": [
            {
                "agent_id": "supervisor",
                "agent_name": "Supervisor",
                "agent_instruction": "Coordinate agents",
                "tools": ["flight_tools"],
            }
        ],
        "primary_agent_id": "supervisor",
    }


@pytest.fixture
def sample_agent_spec_flight():
    """Agent spec with only flight tools."""
    return {
        "agent_id": "flight_agent",
        "agent_name": "Flight Agent",
        "tools": ["flight_tools"],
    }


@pytest.fixture
def sample_agent_spec_all():
    """Agent spec with all tools."""
    return {
        "agent_id": "supervisor",
        "agent_name": "Supervisor Agent",
        "tools": ["flight_tools", "hotel_tools"],
    }


@pytest.fixture
def sample_agent_spec_none():
    """Agent spec with no matching tools."""
    return {
        "agent_id": "router",
        "agent_name": "Router Agent",
        "tools": ["unknown_tools"],
    }


# =============================================================================
# MACS Trace and History Fixtures
# =============================================================================


@pytest.fixture
def sample_trace():
    """Sample conversation trace."""
    return MessageHistory(
        [
            {"role": "user", "content": "I need to book a flight to New York"},
            {"role": "assistant", "content": "Sure! When would you like to travel?"},
            {"role": "user", "content": "Next Monday"},
            {"role": "assistant", "content": "I found a flight. Your confirmation number is ABC123."},
        ]
    )


@pytest.fixture
def sample_tool_traces():
    """Sample tool invocation traces."""
    return {
        "search_flights": {
            "invocations": [
                {
                    "inputs": {"origin": "LAX", "destination": "JFK"},
                    "outputs": "Found 3 flights",
                    "status": "success",
                }
            ]
        },
        "book_flight": {
            "invocations": [
                {
                    "inputs": {"flight_id": "AA123"},
                    "outputs": "Booking confirmed",
                    "status": "success",
                }
            ]
        },
    }


@pytest.fixture
def sample_conversation():
    """Sample multi-turn conversation."""
    return MessageHistory(
        [
            {"role": "user", "content": "I need to book a flight from San Francisco to New York for next Monday."},
            {"role": "assistant", "content": "I'll search for flights for you. What time do you prefer to depart?"},
            {"role": "user", "content": "Morning, preferably around 8am."},
            {
                "role": "assistant",
                "content": "I found a Delta flight departing at 8:15am. The fare is $450. Would you like to book this?",
            },
            {"role": "user", "content": "Yes, please book it."},
            {
                "role": "assistant",
                "content": "Your flight has been booked. Confirmation number: DL123456.",
            },
        ]
    )


# =============================================================================
# MACS Scenario Fixtures
# =============================================================================


@pytest.fixture
def sample_scenario():
    """Sample MACS scenario with background."""
    return """Goal: The user wants to book a flight to New York for a business meeting.

Background:
* User's name is John Smith
* User is a frequent business traveler
* User has preferred airline status with Delta
* User prefers aisle seats"""


@pytest.fixture
def minimal_scenario():
    """Minimal scenario without background section."""
    return "User wants to order food for delivery."


@pytest.fixture
def initial_query():
    """Sample initial user query."""
    return "I need to book a flight to New York for Monday."


# =============================================================================
# MACS Benchmark Fixtures
# =============================================================================


@pytest.fixture
def macs_benchmark(sample_agent_data, dummy_model):
    """Create a MACS benchmark with dummy model for testing.

    Uses dummy_model from parent conftest.py.
    Returns tuple (benchmark, agent_data) for use with run().
    """
    return ConcreteMACSBenchmark(dummy_model), sample_agent_data
