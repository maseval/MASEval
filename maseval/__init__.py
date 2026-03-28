"""Top-level package exports for convenience.

Expose a small, stable surface area for users to import core abstractions directly from `maseval`,
for example: `from maseval import Task, Benchmark`.

Core library sits in the top namespace for easy access.
Interfaces sit in the `maseval.interface` submodule.
Benchmarks sit in the `maseval.benchmark` submodule.
"""

from .core.task import (
    Task,
    TaskProtocol,
    TimeoutAction,
    # Task queue classes
    BaseTaskQueue,
    TaskQueue,
    SequentialTaskQueue,
    InformativeSubsetQueue,
    DISCOQueue,
    PriorityTaskQueue,
    AdaptiveTaskQueue,
)
from .core.environment import Environment
from .core.agent import AgentAdapter
from .core.benchmark import Benchmark, TaskExecutionStatus
from .core.callback_handler import CallbackHandler
from .core.callback import BenchmarkCallback, EnvironmentCallback, AgentCallback
from .core.callbacks import MessageTracingAgentCallback
from .core.simulator import (
    ToolLLMSimulator,
    UserLLMSimulator,
    SimulatorError,
    ToolSimulatorError,
    UserSimulatorError,
)
from .core.model import ModelAdapter, ChatResponse
from .core.scorer import ModelScorer
from .core.user import User, LLMUser, AgenticLLMUser, TerminationReason
from .core.evaluator import Evaluator
from .core.history import MessageHistory, ToolInvocationHistory
from .core.tracing import TraceableMixin
from .core.usage import Usage, TokenUsage, UsageTrackableMixin
from .core.usage import CostCalculator, StaticPricingCalculator, UsageReporter
from .core.registry import ComponentRegistry
from .core.context import TaskContext
from .core.exceptions import (
    MASEvalError,
    AgentError,
    EnvironmentError,
    UserError,
    UserExhaustedError,
    TaskFrozenError,
    TaskTimeoutError,
    validate_argument_type,
    validate_required_arguments,
    validate_no_extra_arguments,
    validate_arguments_from_schema,
)
from .core.seeding import SeedGenerator, DefaultSeedGenerator, SeedingError

__all__ = [
    # Tasks
    "Task",
    "TaskProtocol",
    "TimeoutAction",
    # Core abstractions
    "Environment",
    "AgentAdapter",
    "Benchmark",
    "TaskExecutionStatus",
    # Callbacks
    "CallbackHandler",
    "BenchmarkCallback",
    "EnvironmentCallback",
    "AgentCallback",
    "MessageTracingAgentCallback",
    # Simulators
    "ToolLLMSimulator",
    "UserLLMSimulator",
    "SimulatorError",
    "ToolSimulatorError",
    "UserSimulatorError",
    # User simulation
    "User",
    "LLMUser",
    "AgenticLLMUser",
    "TerminationReason",
    # Evaluation
    "Evaluator",
    # History and tracing
    "MessageHistory",
    "ToolInvocationHistory",
    "TraceableMixin",
    # Usage tracking
    "Usage",
    "TokenUsage",
    "UsageTrackableMixin",
    "CostCalculator",
    "StaticPricingCalculator",
    "UsageReporter",
    # Registry and execution context
    "ComponentRegistry",
    "TaskContext",
    # Task queues
    "BaseTaskQueue",
    "TaskQueue",
    "SequentialTaskQueue",
    "InformativeSubsetQueue",
    "DISCOQueue",
    "PriorityTaskQueue",
    "AdaptiveTaskQueue",
    # Model adapters and scorers
    "ModelAdapter",
    "ChatResponse",
    "ModelScorer",
    # Exceptions and validation
    "MASEvalError",
    "AgentError",
    "EnvironmentError",
    "UserError",
    "UserExhaustedError",
    "TaskFrozenError",
    "TaskTimeoutError",
    "validate_argument_type",
    "validate_required_arguments",
    "validate_no_extra_arguments",
    "validate_arguments_from_schema",
    # Seeding
    "SeedGenerator",
    "DefaultSeedGenerator",
    "SeedingError",
]
