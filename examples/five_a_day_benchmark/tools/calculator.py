"""Calculator tool - single purpose mathematical evaluation."""

import math

from RestrictedPython import compile_restricted, safe_globals

from .base import BaseTool, ToolResult


class CalculatorCalculateTool(BaseTool):
    """Perform safe mathematical calculations."""

    def __init__(self):
        super().__init__(
            "calculator_calculate",
            "Evaluate mathematical expressions with functions like sqrt, sin, cos, log, etc.",
            tool_args=["expression"],
        )

        # Safe globals with math functions
        self.safe_env = {
            **safe_globals,
            "__builtins__": {
                # Basic types
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                # Math operations
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "pow": pow,
                # Constants
                "True": True,
                "False": False,
                "None": None,
            },
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

    def execute(self, **kwargs) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        expression = kwargs.get("expression")

        if not expression:
            return ToolResult(success=False, data=None, error="expression is required")

        try:
            # Compile the expression
            compile_result = compile_restricted(expression, "<calculator>", "eval")

            # Check if it's a CompileResult object with errors attribute
            if hasattr(compile_result, "errors") and compile_result.errors:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {', '.join(compile_result.errors)}",
                )

            # Get the code object
            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute with restricted globals
            result = eval(code_obj, self.safe_env, {})

            return ToolResult(
                success=True,
                data={"result": result, "expression": expression},
            )

        except ZeroDivisionError:
            return ToolResult(success=False, data=None, error="Division by zero")
        except OverflowError:
            return ToolResult(success=False, data=None, error="Numerical overflow")
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Calculation error: {str(e)}")


class CalculatorToolCollection:
    """Calculator tool collection factory.

    Currently only contains one tool, but structured as a collection
    for consistency with other tool patterns.
    """

    def __init__(self):
        pass

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all calculator sub-tools."""
        return [
            CalculatorCalculateTool(),
        ]
