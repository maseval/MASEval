"""Exception hierarchy for MASEval error classification.

MASEval distinguishes between errors caused by the agent (agent's fault) and errors
caused by the evaluation infrastructure (environment, user simulator, etc.). This
distinction enables fair scoring by excluding infrastructure failures from agent
performance metrics.

Exception Hierarchy:
    MASEvalError (base)
    ├── AgentError           - Agent violated contract at boundary (agent's fault)
    ├── EnvironmentError     - Environment/tool infrastructure failed (not agent's fault)
    └── UserError            - User simulator failed (not agent's fault)

Usage in Tools:
    The boundary between agent responsibility and environment responsibility is
    INPUT VALIDATION. If validation passes but execution fails, it's an environment error.

    ```python
    def my_tool(a: int, b: int) -> int:
        # 1. Validate inputs - agent's responsibility
        if not isinstance(a, int):
            raise AgentError(f"Expected int for 'a', got {type(a).__name__}")

        # 2. Execute - our responsibility (if this fails, it's on us)
        try:
            return some_external_api_call(a, b)
        except ExternalAPIError as e:
            raise EnvironmentError(f"API call failed: {e}") from e
    ```

Usage in Benchmark Results:
    After running a benchmark, filter results by error type for fair scoring:

    ```python
    results = benchmark.run(tasks)

    # Tasks where agent is accountable
    scoreable = [r for r in results if r["status"] in ("success", "agent_error")]

    # Infrastructure failures to investigate separately
    infra_failures = [r for r in results if r["status"] in ("environment_error", "user_error")]
    ```
"""

from typing import Any, Dict, List, Optional


class MASEvalError(Exception):
    """Base exception for all MASEval-controlled component failures.

    This is the base class for exceptions that occur at boundaries we control
    (tools, environment, user simulator). Errors from agent framework internals
    should NOT use this hierarchy - they remain as generic exceptions and are
    classified as UNKNOWN_EXECUTION_ERROR.
    """

    def __init__(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MASEvalError.

        Args:
            message: Human-readable error description.
            component: Name of the component that raised the error (e.g., tool name).
            details: Additional structured information about the error.
        """
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.component:
            return f"[{self.component}] {self.message}"
        return self.message


class AgentError(MASEvalError):
    """Agent violated the contract at a boundary we control.

    Raised when the agent provides invalid inputs to components we control.
    This is the agent's fault - these tasks count against their score.

    The `suggestion` field provides agent-friendly hints for self-correction
    that some agent frameworks may use for automatic recovery.

    When to raise:
        - Agent passed wrong argument types to a tool
        - Agent passed arguments that violate documented constraints
        - Agent is missing required arguments
        - Agent called a tool with semantically invalid input
        - Agent exceeded documented limits (max retries, rate limits, etc.)

    Examples:
        ```python
        # Wrong type with suggestion
        raise AgentError(
            "Expected int for 'count', got str",
            component="search_tool",
            suggestion="Provide count as a number, e.g., count=10"
        )

        # Missing required argument
        raise AgentError(
            "Missing required argument 'query'",
            component="search_tool",
            suggestion="Include query='your search terms'"
        )

        # Constraint violation
        raise AgentError(
            "Argument 'limit' must be positive, got -5",
            component="fetch_tool",
            suggestion="Use a positive value, e.g., limit=10"
        )
        ```
    """

    def __init__(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        """Initialize AgentError.

        Args:
            message: Human-readable error description explaining what went wrong.
            component: Name of the component that raised the error (e.g., tool name).
            details: Additional structured information about the error.
            suggestion: Agent-friendly hint for correcting the error. Some agent
                frameworks use this for automatic retry with corrected inputs.
        """
        super().__init__(message, component=component, details=details)
        self.suggestion = suggestion

    def __str__(self) -> str:
        base = super().__str__()
        if self.suggestion:
            return f"{base}. Suggestion: {self.suggestion}"
        return base


class EnvironmentError(MASEvalError):
    """Environment or tool infrastructure failed.

    Raised when our code fails AFTER validating agent inputs. This indicates
    a problem with the evaluation infrastructure, not the agent's behavior.
    These tasks should be excluded from agent scoring.

    When to raise:
        - Tool implementation has a bug
        - External API/database our tool depends on failed
        - ToolLLMSimulator failed to parse model output
        - Model adapter for tool simulation failed
        - Resource exhaustion in environment components
        - File I/O errors in environment setup

    Examples:
        ```python
        # Tool bug
        raise EnvironmentError("Internal error in calculation", component="calc_tool")

        # External dependency failed
        raise EnvironmentError("Database connection failed", component="db_tool")

        # Simulator failed
        raise EnvironmentError(
            "Failed to parse LLM response after 3 attempts",
            component="flight_search",
            details={"attempts": 3, "last_error": "Invalid JSON"}
        )
        ```

    Note:
        Python has a built-in `EnvironmentError` (alias for `OSError`), but it's
        rarely used directly. This class shadows it intentionally for clean semantics.
        If you need the built-in, use `OSError` explicitly.
    """

    pass


class UserError(MASEvalError):
    """User simulator failed.

    Raised when the user simulation infrastructure fails. This is NOT the
    agent's fault - these tasks should be excluded from agent scoring.

    When to raise:
        - UserLLMSimulator couldn't reach the LLM API
        - User model returned unparseable response after retries
        - User simulator configuration error
        - User profile data is malformed

    Examples:
        ```python
        # API failure
        raise UserError("OpenAI API unreachable", component="user_simulator")

        # Parse failure
        raise UserError(
            "Failed to parse user response after 3 attempts",
            component="user_simulator",
            details={"attempts": 3, "last_error": "Missing 'text' field"}
        )
        ```
    """

    pass


class TaskTimeoutError(MASEvalError):
    """Task execution exceeded configured timeout.

    This is classified as TASK_TIMEOUT in benchmark results, separate from
    other error types. Timeout is neither agent's fault nor infrastructure's
    fault—it's a resource constraint.

    When to raise:
        - Task execution time exceeds TaskProtocol.timeout_seconds
        - Cooperative timeout check detects deadline has passed
        - Hard timeout backstop triggers

    Attributes:
        elapsed: Time elapsed before timeout was detected.
        timeout: The configured timeout value in seconds.
        partial_traces: Any traces collected before timeout occurred.

    Examples:
        ```python
        # Cooperative timeout at checkpoint
        raise TaskTimeoutError(
            "Task exceeded 60s deadline",
            component="execution_loop",
            elapsed=62.5,
            timeout=60.0
        )

        # Hard timeout with partial traces
        raise TaskTimeoutError(
            "Task exceeded 120s hard deadline",
            component="timeout_backstop",
            elapsed=125.0,
            timeout=120.0,
            partial_traces={"agents": {"main": {"messages": [...]}}}
        )
        ```
    """

    def __init__(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        elapsed: float = 0.0,
        timeout: float = 0.0,
        partial_traces: Optional[Dict[str, Any]] = None,
    ):
        """Initialize TaskTimeoutError.

        Args:
            message: Human-readable error description.
            component: Name of the component that raised the error.
            details: Additional structured information about the error.
            elapsed: Time elapsed before timeout was detected.
            timeout: The configured timeout value in seconds.
            partial_traces: Any traces collected before timeout occurred.
        """
        super().__init__(message, component=component, details=details)
        self.elapsed = elapsed
        self.timeout = timeout
        self.partial_traces = partial_traces or {}


# =============================================================================
# Convenience functions for tool implementers
# =============================================================================


def validate_argument_type(
    value: Any,
    expected_type: str,
    arg_name: str,
    component: Optional[str] = None,
) -> None:
    """Validate that a value matches an expected JSON schema type.

    Raises AgentError if validation fails.

    Args:
        value: The value to validate.
        expected_type: JSON schema type ("string", "integer", "number", "boolean", "array", "object").
        arg_name: Name of the argument (for error message).
        component: Optional component name for error context.

    Raises:
        AgentError: If value doesn't match expected type.

    Example:
        ```python
        def my_tool(count: int, name: str):
            validate_argument_type(count, "integer", "count", "my_tool")
            validate_argument_type(name, "string", "name", "my_tool")
            # ... tool logic
        ```
    """
    type_map = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }

    # Special case: integer should not accept bool (bool is subclass of int in Python)
    if expected_type == "integer" and isinstance(value, bool):
        raise AgentError(
            f"Argument '{arg_name}' expected integer, got boolean",
            component=component,
            details={"argument": arg_name, "expected": expected_type, "actual": type(value).__name__},
            suggestion=f"Provide {arg_name} as an integer, e.g., 10 (not true/false)",
        )

    expected_types = type_map.get(expected_type)
    if expected_types is None:
        # Unknown type - accept anything
        return

    if not isinstance(value, expected_types):
        # Build a suggestion based on expected type
        type_hints = {
            "string": 'a string, e.g., "example"',
            "integer": "an integer, e.g., 10",
            "number": "a number, e.g., 3.14",
            "boolean": "a boolean, e.g., true or false",
            "array": "a list, e.g., [1, 2, 3]",
            "object": "an object, e.g., {}",
        }
        hint = type_hints.get(expected_type, f"a {expected_type}")
        raise AgentError(
            f"Argument '{arg_name}' expected {expected_type}, got {type(value).__name__}",
            component=component,
            details={"argument": arg_name, "expected": expected_type, "actual": type(value).__name__},
            suggestion=f"Provide {arg_name} as {hint}",
        )


def validate_required_arguments(
    kwargs: Dict[str, Any],
    required: List[str],
    component: Optional[str] = None,
) -> None:
    """Validate that all required arguments are present.

    Raises AgentError if any required argument is missing.

    Args:
        kwargs: The keyword arguments dict to validate.
        required: List of required argument names.
        component: Optional component name for error context.

    Raises:
        AgentError: If any required argument is missing.

    Example:
        ```python
        def my_tool(**kwargs):
            validate_required_arguments(kwargs, ["query", "limit"], "my_tool")
            # ... tool logic
        ```
    """
    missing = [arg for arg in required if arg not in kwargs]
    if missing:
        raise AgentError(
            f"Missing required argument(s): {', '.join(missing)}",
            component=component,
            details={"missing": missing, "required": required},
            suggestion=f"Include the following argument(s): {', '.join(missing)}",
        )


def validate_no_extra_arguments(
    kwargs: Dict[str, Any],
    allowed: List[str],
    component: Optional[str] = None,
) -> None:
    """Validate that no unexpected arguments are present.

    Raises AgentError if any argument is not in the allowed list.

    Args:
        kwargs: The keyword arguments dict to validate.
        allowed: List of allowed argument names.
        component: Optional component name for error context.

    Raises:
        AgentError: If any unexpected argument is present.

    Example:
        ```python
        def my_tool(**kwargs):
            validate_no_extra_arguments(kwargs, ["query", "limit"], "my_tool")
            # ... tool logic
        ```
    """
    extra = [arg for arg in kwargs if arg not in allowed]
    if extra:
        raise AgentError(
            f"Unexpected argument(s): {', '.join(extra)}",
            component=component,
            details={"unexpected": extra, "allowed": allowed},
            suggestion=f"Remove the unexpected argument(s): {', '.join(extra)}. Valid arguments: {', '.join(allowed)}",
        )


def validate_arguments_from_schema(
    kwargs: Dict[str, Any],
    schema: Dict[str, Any],
    component: Optional[str] = None,
    *,
    strict: bool = False,
) -> None:
    """Validate arguments against a JSON schema.

    This is the main validation function for tool implementers. It validates:
    - Required arguments are present
    - Argument types match the schema
    - No extra arguments (if strict=True)

    Args:
        kwargs: The keyword arguments dict to validate.
        schema: JSON schema with 'properties' and optionally 'required'.
        component: Optional component name for error context.
        strict: If True, reject arguments not in schema. Default False.

    Raises:
        AgentError: If validation fails.

    Example:
        ```python
        SCHEMA = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

        def my_tool(**kwargs):
            validate_arguments_from_schema(kwargs, SCHEMA, "my_tool")
            # ... tool logic
        ```
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required arguments
    validate_required_arguments(kwargs, required, component)

    # Check for extra arguments if strict
    if strict:
        validate_no_extra_arguments(kwargs, list(properties.keys()), component)

    # Validate types for provided arguments
    for arg_name, value in kwargs.items():
        if arg_name in properties:
            expected_type = properties[arg_name].get("type")
            if expected_type:
                validate_argument_type(value, expected_type, arg_name, component)
