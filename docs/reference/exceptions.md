# Exceptions

Exception classes for error classification in benchmark execution.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/exceptions.py){ .md-source-file }

## Exception Hierarchy

```
MASEvalError (base)
├── AgentError           - Agent violated contract (agent's fault)
├── EnvironmentError     - Environment/tool failed (not agent's fault)
├── UserError            - User simulator failed (not agent's fault)
└── TaskTimeoutError     - Task exceeded configured timeout

SimulatorError (base for simulators)
├── ToolSimulatorError   - Also inherits EnvironmentError
└── UserSimulatorError   - Also inherits UserError
```

## Core Exceptions

::: maseval.core.exceptions.MASEvalError

::: maseval.core.exceptions.AgentError

::: maseval.core.exceptions.EnvironmentError

::: maseval.core.exceptions.UserError

::: maseval.core.exceptions.TaskTimeoutError

## Simulator Exceptions

::: maseval.core.simulator.SimulatorError

::: maseval.core.simulator.ToolSimulatorError

::: maseval.core.simulator.UserSimulatorError

## Validation Helpers

These functions simplify input validation and raise `AgentError` with helpful suggestions:

::: maseval.core.exceptions.validate_argument_type

::: maseval.core.exceptions.validate_required_arguments

::: maseval.core.exceptions.validate_no_extra_arguments

::: maseval.core.exceptions.validate_arguments_from_schema
