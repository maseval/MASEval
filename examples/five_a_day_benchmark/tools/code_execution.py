"""Code execution tool collection using RestrictedPython.

Tools:
- python_executor_execute: Run Python code and see output
- python_executor_test: Run code against test cases
"""

from typing import Any

from RestrictedPython import compile_restricted

from .base import BaseTool, ToolResult


from RestrictedPython import safe_globals, limited_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence


def get_safe_builtins() -> dict:
    """Get the set of allowed built-in functions for code execution.

    Returns a permissive set of builtins that covers common Python operations
    needed for algorithm problems while maintaining safety.
    """
    return {
        **limited_builtins,
        # Type constructors
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        # Common operations
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        # Aggregation functions
        "max": max,
        "min": min,
        "sum": sum,
        "abs": abs,
        "all": all,
        "any": any,
        # Sorting and ordering
        "sorted": sorted,
        "reversed": reversed,
        # Functional programming
        "map": map,
        "filter": filter,
        # I/O (for debugging)
        "print": print,
    }


def get_safe_guards() -> dict:
    """Get the set of RestrictedPython guards for safe code execution.

    Returns a permissive set of guards that allows most common Python operations
    including iteration, indexing, attribute access, and assignments.
    """
    return {
        "_getitem_": lambda obj, index: obj[index],
        "_setitem_": lambda obj, index, value: obj.__setitem__(index, value),
        "_getattr_": getattr,
        "_setattr_": setattr,
        "_write_": lambda obj: obj,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    }


def get_safe_python_exec_environment() -> dict:
    """Get a complete safe execution environment for RestrictedPython.

    Always includes PrintCollector for capturing print output. After exec(),
    retrieve captured output via: env.get('_print', lambda: '')().

    Returns:
        A dictionary suitable for use as globals in exec() with RestrictedPython.
    """
    from RestrictedPython.PrintCollector import PrintCollector

    return {
        **safe_globals,
        "__builtins__": get_safe_builtins(),
        **get_safe_guards(),
        "_print_": PrintCollector,
    }


class CodeExecutionState:
    """Shared state for code execution tools.

    Maintains test cases and safe execution environment.
    """

    def __init__(self, test_cases: list[dict[str, Any]] | None = None):
        self.test_cases = test_cases or []
        # Get shared safe execution environment with print collector for capturing output
        self.safe_env = get_safe_python_exec_environment()


class PythonExecutorExecuteTool(BaseTool):
    """Execute Python code and return output."""

    def __init__(self, code_state: CodeExecutionState):
        super().__init__(
            "python_executor_execute",
            "Execute Python code safely and see the output (use print() to display results)",
            tool_args=["code"],
        )
        self.state = code_state

    def execute(self, **kwargs) -> ToolResult:
        """Execute Python code and return output."""
        code = kwargs.get("code")

        if not code:
            return ToolResult(success=False, data=None, error="code is required")

        try:
            # Compile the code with RestrictedPython
            compile_result = compile_restricted(code, "<string>", "exec")

            if hasattr(compile_result, "errors") and compile_result.errors:
                # Format errors properly
                error_msgs = []
                for err in compile_result.errors:
                    if isinstance(err, tuple):
                        error_msgs.append(str(err[0]) if err else "Unknown error")
                    else:
                        error_msgs.append(str(err))
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {'; '.join(error_msgs)}",
                )

            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute with RestrictedPython environment
            exec_globals = self.state.safe_env.copy()
            exec(code_obj, exec_globals)

            # Get output from PrintCollector
            output = exec_globals.get("printed", "")

            return ToolResult(
                success=True,
                data={
                    "output": output.strip() if output else None,
                    "status": "completed",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Execution error: {str(e)}",
            )


class PythonExecutorTestTool(BaseTool):
    """Run code against test cases."""

    def __init__(self, code_state: CodeExecutionState):
        super().__init__(
            "python_executor_test",
            "Execute Python code and run it against test cases to verify correctness",
            tool_args=["code"],
        )
        self.state = code_state

    def execute(self, **kwargs) -> ToolResult:
        """Execute code and run test cases."""
        code = kwargs.get("code")

        if not code:
            return ToolResult(success=False, data=None, error="code is required")

        if not self.state.test_cases:
            return ToolResult(
                success=False,
                data=None,
                error="No test cases configured for this task",
            )

        try:
            # Compile the code with RestrictedPython
            compile_result = compile_restricted(code, "<string>", "exec")

            if hasattr(compile_result, "errors") and compile_result.errors:
                error_msgs = []
                for err in compile_result.errors:
                    if isinstance(err, tuple):
                        error_msgs.append(str(err[0]) if err else "Unknown error")
                    else:
                        error_msgs.append(str(err))
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {'; '.join(error_msgs)}",
                )

            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute code to define functions
            exec_globals = self.state.safe_env.copy()
            exec(code_obj, exec_globals)

            # Find the main function (assume it's the first non-builtin function)
            func_name = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith("_"):
                    func_name = name
                    break

            if not func_name:
                return ToolResult(
                    success=False,
                    data=None,
                    error="No function found in code",
                )

            func = exec_globals[func_name]

            # Run test cases
            test_results = []
            all_passed = True

            for i, test_case in enumerate(self.state.test_cases):
                test_input = test_case["input"]
                expected_output = test_case["expected_output"]

                try:
                    actual_output = func(test_input)
                    passed = actual_output == expected_output

                    test_results.append(
                        {
                            "test_case": i + 1,
                            "input": test_input,
                            "expected": expected_output,
                            "actual": actual_output,
                            "passed": passed,
                        }
                    )

                    if not passed:
                        all_passed = False

                except Exception as e:
                    test_results.append(
                        {
                            "test_case": i + 1,
                            "input": test_input,
                            "expected": expected_output,
                            "actual": None,
                            "passed": False,
                            "error": str(e),
                        }
                    )
                    all_passed = False

            return ToolResult(
                success=True,
                data={
                    "all_passed": all_passed,
                    "test_results": test_results,
                    "total_tests": len(test_results),
                    "passed_tests": sum(1 for r in test_results if r["passed"]),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Test execution error: {str(e)}",
            )


class CodeExecutionToolCollection:
    """Code execution tool collection factory.

    Creates a shared state with test cases and returns code execution sub-tools.

    Usage:
        code_state = CodeExecutionState(test_cases)
        collection = CodeExecutionToolCollection(code_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, code_execution_state: CodeExecutionState):
        self.state = code_execution_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all code execution sub-tools."""
        return [
            PythonExecutorExecuteTool(self.state),
            PythonExecutorTestTool(self.state),
        ]
