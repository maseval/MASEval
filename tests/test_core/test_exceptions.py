"""Tests for exception classification in benchmark execution.

These tests verify that different exception types (AgentError, EnvironmentError,
UserError) are correctly classified into their respective TaskExecutionStatus
values, enabling fair scoring by distinguishing agent faults from infrastructure
failures.
"""

import pytest
from maseval import (
    TaskQueue,
    TaskExecutionStatus,
    AgentError,
    EnvironmentError,
    UserError,
    validate_argument_type,
    validate_required_arguments,
    validate_no_extra_arguments,
    validate_arguments_from_schema,
)


class TestExceptionClassification:
    """Tests for exception classification in benchmark execution."""

    def test_agent_error_classified_correctly(self):
        """Test that AgentError is classified as AGENT_ERROR."""
        from conftest import DummyBenchmark, DummyAgentAdapter

        class AgentErrorRaisingAgent:
            def run(self, query: str) -> str:
                raise AgentError("Invalid tool argument", component="test_tool")

        class AgentErrorAdapter(DummyAgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class AgentErrorBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = AgentErrorRaisingAgent()
                adapter = AgentErrorAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = AgentErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        assert reports[0]["status"] == TaskExecutionStatus.AGENT_ERROR.value
        assert reports[0]["error"]["error_type"] == "AgentError"
        assert reports[0]["error"]["component"] == "test_tool"

    def test_environment_error_classified_correctly(self):
        """Test that EnvironmentError is classified as ENVIRONMENT_ERROR."""
        from conftest import DummyBenchmark, DummyAgentAdapter

        class EnvironmentErrorRaisingAgent:
            def run(self, query: str) -> str:
                raise EnvironmentError("Database connection failed", component="db_tool")

        class EnvironmentErrorAdapter(DummyAgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class EnvironmentErrorBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = EnvironmentErrorRaisingAgent()
                adapter = EnvironmentErrorAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = EnvironmentErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        assert reports[0]["status"] == TaskExecutionStatus.ENVIRONMENT_ERROR.value
        assert reports[0]["error"]["error_type"] == "EnvironmentError"
        assert reports[0]["error"]["component"] == "db_tool"

    def test_user_error_classified_correctly(self):
        """Test that UserError is classified as USER_ERROR."""
        from conftest import DummyBenchmark, DummyAgentAdapter

        class UserErrorRaisingAgent:
            def run(self, query: str) -> str:
                raise UserError("User simulator API failed", component="user_simulator")

        class UserErrorAdapter(DummyAgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class UserErrorBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = UserErrorRaisingAgent()
                adapter = UserErrorAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = UserErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        assert reports[0]["status"] == TaskExecutionStatus.USER_ERROR.value
        assert reports[0]["error"]["error_type"] == "UserError"
        assert reports[0]["error"]["component"] == "user_simulator"

    def test_generic_exception_classified_as_unknown(self):
        """Test that generic exceptions are classified as UNKNOWN_EXECUTION_ERROR."""
        from conftest import DummyBenchmark, DummyAgentAdapter

        class GenericErrorRaisingAgent:
            def run(self, query: str) -> str:
                raise ValueError("Some internal error")

        class GenericErrorAdapter(DummyAgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class GenericErrorBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = GenericErrorRaisingAgent()
                adapter = GenericErrorAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = GenericErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        assert reports[0]["status"] == TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value
        assert reports[0]["error"]["error_type"] == "ValueError"

    def test_error_details_included_in_report(self):
        """Test that error details are included in the report."""
        from conftest import DummyBenchmark, DummyAgentAdapter

        class DetailedAgentErrorRaisingAgent:
            def run(self, query: str) -> str:
                raise AgentError(
                    "Invalid argument type",
                    component="my_tool",
                    details={"expected": "int", "actual": "str", "argument": "count"},
                )

        class DetailedAgentErrorAdapter(DummyAgentAdapter):
            def _run_agent(self, query: str) -> str:
                return self.agent.run(query)

        class DetailedAgentErrorBenchmark(DummyBenchmark):
            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                agent = DetailedAgentErrorRaisingAgent()
                adapter = DetailedAgentErrorAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list([{"query": "Test", "environment_data": {}}])
        benchmark = DetailedAgentErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        assert len(reports) == 1
        error = reports[0]["error"]
        assert error["component"] == "my_tool"
        assert error["details"]["expected"] == "int"
        assert error["details"]["actual"] == "str"


@pytest.mark.core
class TestTaskTimeoutError:
    """Tests for TaskTimeoutError exception."""

    def test_timeout_error_attributes(self):
        """TaskTimeoutError should have elapsed, timeout, partial_traces attributes."""
        from maseval import TaskTimeoutError

        error = TaskTimeoutError(
            "Task exceeded 60s deadline",
            component="execution_loop",
            elapsed=62.5,
            timeout=60.0,
            partial_traces={"agents": {"agent1": {"steps": 3}}},
        )

        assert error.elapsed == 62.5
        assert error.timeout == 60.0
        assert error.partial_traces == {"agents": {"agent1": {"steps": 3}}}

    def test_timeout_error_message(self):
        """TaskTimeoutError message should include timing info."""
        from maseval import TaskTimeoutError

        error = TaskTimeoutError(
            "Task exceeded 60s deadline after 62.5s",
            component="timeout_check",
            elapsed=62.5,
            timeout=60.0,
        )

        assert "60s" in str(error)
        assert "62.5s" in str(error)

    def test_timeout_error_inherits_from_maseval_error(self):
        """TaskTimeoutError should inherit from MASEvalError."""
        from maseval import TaskTimeoutError
        from maseval.core.exceptions import MASEvalError

        error = TaskTimeoutError("timeout", elapsed=1.0, timeout=0.5)

        assert isinstance(error, MASEvalError)
        assert isinstance(error, Exception)

    def test_timeout_error_defaults(self):
        """TaskTimeoutError should have sensible defaults."""
        from maseval import TaskTimeoutError

        error = TaskTimeoutError("timeout")

        assert error.elapsed == 0.0
        assert error.timeout == 0.0
        assert error.partial_traces == {}


class TestAgentErrorSuggestion:
    """Tests for AgentError suggestion field."""

    def test_agent_error_with_suggestion(self):
        """AgentError can include a suggestion for self-correction."""
        error = AgentError(
            "Expected int for 'count', got str",
            component="search_tool",
            suggestion="Provide count as an integer, e.g., count=10",
        )
        assert error.suggestion == "Provide count as an integer, e.g., count=10"
        assert "Suggestion:" in str(error)
        assert "count=10" in str(error)

    def test_agent_error_without_suggestion(self):
        """AgentError works without suggestion."""
        error = AgentError("Simple error")
        assert error.suggestion is None
        assert "Suggestion:" not in str(error)

    def test_validation_helpers_include_suggestions(self):
        """Validation helpers include helpful suggestions in errors."""
        # Type validation
        try:
            validate_argument_type("hello", "integer", "count")
        except AgentError as e:
            assert e.suggestion is not None
            assert "count" in e.suggestion
            assert "integer" in e.suggestion

        # Required arguments
        try:
            validate_required_arguments({}, ["query"])
        except AgentError as e:
            assert e.suggestion is not None
            assert "query" in e.suggestion

        # Extra arguments
        try:
            validate_no_extra_arguments({"extra": 1}, ["allowed"])
        except AgentError as e:
            assert e.suggestion is not None
            assert "extra" in e.suggestion


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_argument_type_valid(self):
        """Test that valid arguments pass validation."""
        # These should not raise
        validate_argument_type("hello", "string", "name")
        validate_argument_type(42, "integer", "count")
        validate_argument_type(3.14, "number", "value")
        validate_argument_type(True, "boolean", "flag")
        validate_argument_type([1, 2, 3], "array", "items")
        validate_argument_type({"key": "value"}, "object", "config")

    def test_validate_argument_type_invalid(self):
        """Test that invalid arguments raise AgentError."""
        with pytest.raises(AgentError, match="expected string"):
            validate_argument_type(42, "string", "name")

        with pytest.raises(AgentError, match="expected integer"):
            validate_argument_type("not an int", "integer", "count")

        with pytest.raises(AgentError, match="expected boolean"):
            validate_argument_type(1, "boolean", "flag")  # int != bool

    def test_validate_argument_type_bool_not_int(self):
        """Test that bool is not accepted as integer."""
        with pytest.raises(AgentError, match="expected integer, got boolean"):
            validate_argument_type(True, "integer", "count")

    def test_validate_required_arguments_present(self):
        """Test that present required arguments pass validation."""
        # Should not raise
        validate_required_arguments({"a": 1, "b": 2}, ["a", "b"])
        validate_required_arguments({"a": 1, "b": 2, "c": 3}, ["a"])

    def test_validate_required_arguments_missing(self):
        """Test that missing required arguments raise AgentError."""
        with pytest.raises(AgentError, match="Missing required argument"):
            validate_required_arguments({"a": 1}, ["a", "b"])

    def test_validate_no_extra_arguments_valid(self):
        """Test that no extra arguments pass validation."""
        # Should not raise
        validate_no_extra_arguments({"a": 1, "b": 2}, ["a", "b", "c"])
        validate_no_extra_arguments({}, ["a", "b"])

    def test_validate_no_extra_arguments_invalid(self):
        """Test that extra arguments raise AgentError."""
        with pytest.raises(AgentError, match="Unexpected argument"):
            validate_no_extra_arguments({"a": 1, "extra": 2}, ["a"])

    def test_validate_arguments_from_schema_valid(self):
        """Test that valid arguments pass schema validation."""
        schema = {
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }

        # Should not raise
        validate_arguments_from_schema({"name": "test", "count": 5}, schema)
        validate_arguments_from_schema({"name": "test"}, schema)  # count optional

    def test_validate_arguments_from_schema_missing_required(self):
        """Test that missing required arguments raise AgentError."""
        schema = {
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        with pytest.raises(AgentError, match="Missing required"):
            validate_arguments_from_schema({}, schema)

    def test_validate_arguments_from_schema_wrong_type(self):
        """Test that wrong types raise AgentError."""
        schema = {
            "properties": {
                "count": {"type": "integer"},
            },
        }

        with pytest.raises(AgentError, match="expected integer"):
            validate_arguments_from_schema({"count": "not an int"}, schema)

    def test_validate_arguments_from_schema_strict_mode(self):
        """Test that strict mode rejects extra arguments."""
        schema = {
            "properties": {
                "name": {"type": "string"},
            },
        }

        # Non-strict (default) allows extra args
        validate_arguments_from_schema({"name": "test", "extra": 1}, schema, strict=False)

        # Strict mode rejects extra args
        with pytest.raises(AgentError, match="Unexpected argument"):
            validate_arguments_from_schema({"name": "test", "extra": 1}, schema, strict=True)


class TestFilteringByErrorType:
    """Tests for filtering failed tasks by error type."""

    def test_filter_agent_errors_only(self):
        """Test filtering to get only agent errors."""
        from conftest import DummyBenchmark, DummyAgentAdapter, DummyAgent

        class MixedErrorBenchmark(DummyBenchmark):
            task_counter = 0

            def setup_agents(self, agent_data, environment, task, user, seed_generator):
                self.task_counter += 1

                class DynamicAgent:
                    def __init__(self, error_type):
                        self.error_type = error_type

                    def run(self, query: str) -> str:
                        if self.error_type == "agent":
                            raise AgentError("Agent fault")
                        elif self.error_type == "env":
                            raise EnvironmentError("Env fault")
                        return "success"

                if self.task_counter == 1:
                    agent = DynamicAgent("agent")
                elif self.task_counter == 2:
                    agent = DynamicAgent("env")
                else:
                    agent = DummyAgent()

                adapter = DummyAgentAdapter(agent, "agent")
                return [adapter], {"agent": adapter}

        tasks = TaskQueue.from_list(
            [
                {"query": "Task 1", "environment_data": {}},
                {"query": "Task 2", "environment_data": {}},
                {"query": "Task 3", "environment_data": {}},
            ]
        )

        benchmark = MixedErrorBenchmark()
        reports = benchmark.run(tasks, agent_data={})

        # Should have 1 success, 1 agent error, 1 env error
        statuses = [r["status"] for r in reports]
        assert TaskExecutionStatus.AGENT_ERROR.value in statuses
        assert TaskExecutionStatus.ENVIRONMENT_ERROR.value in statuses
        assert TaskExecutionStatus.SUCCESS.value in statuses

        # Filter only agent errors
        agent_errors = benchmark.get_failed_tasks(TaskExecutionStatus.AGENT_ERROR)
        assert len(agent_errors) == 1

        # Filter only environment errors
        env_errors = benchmark.get_failed_tasks(TaskExecutionStatus.ENVIRONMENT_ERROR)
        assert len(env_errors) == 1

        # Filter multiple types
        all_failures = benchmark.get_failed_tasks(
            [
                TaskExecutionStatus.AGENT_ERROR,
                TaskExecutionStatus.ENVIRONMENT_ERROR,
            ]
        )
        assert len(all_failures) == 2

    def test_scoring_guidance(self):
        """Test the scoring guidance from TaskExecutionStatus docstring."""
        # Include in agent score: SUCCESS, AGENT_ERROR
        scoreable_statuses = {
            TaskExecutionStatus.SUCCESS.value,
            TaskExecutionStatus.AGENT_ERROR.value,
        }

        # Exclude from agent score: ENVIRONMENT_ERROR, USER_ERROR, UNKNOWN_EXECUTION_ERROR
        exclude_statuses = {
            TaskExecutionStatus.ENVIRONMENT_ERROR.value,
            TaskExecutionStatus.USER_ERROR.value,
            TaskExecutionStatus.UNKNOWN_EXECUTION_ERROR.value,
        }

        # Verify no overlap
        assert scoreable_statuses.isdisjoint(exclude_statuses)

        # Verify all execution statuses are accounted for (excluding setup/eval)
        all_exec_statuses = scoreable_statuses | exclude_statuses
        assert TaskExecutionStatus.SETUP_FAILED.value not in all_exec_statuses
        assert TaskExecutionStatus.EVALUATION_FAILED.value not in all_exec_statuses
