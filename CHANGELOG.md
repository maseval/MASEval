# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Benchmarks**

- GAIA2 Benchmark: Integration with Meta's ARE (Agent Research Environments) platform for evaluating LLM-based agents on dynamic, multi-step scenarios (PR: #26)
- `Gaia2Benchmark`, `Gaia2Environment`, `Gaia2Evaluator` components for framework-agnostic evaluation with ARE simulation (PR: #26)
- `DefaultAgentGaia2Benchmark` with ReAct-style agent for direct comparison with ARE reference implementation (PR: #26)
- Tool wrapper (`AREToolWrapper`) for MASEval tracing of ARE tools with simulation time tracking (PR: #26)
- Data loading utilities: `load_tasks()`, `configure_model_ids()` for loading scenarios from HuggingFace (PR: #26)
- Metrics: `compute_gaia2_metrics()` for GSR (Goal Success Rate) computation by capability type (PR: #26)
- Support for 7 capability dimensions: execution, search, adaptability, time, ambiguity, agent2agent, noise (PR: #26)
- Added `gaia2` optional dependency: `pip install maseval[gaia2]` (PR: #26)

**Examples**

- Gaia2 benchmark example with Google GenAI and OpenAI model support (PR: #26)

**Seeding System**

- Added `SeedGenerator` abstract base class and `DefaultSeedGenerator` implementation for reproducible benchmark runs via SHA-256-based seed derivation (PR: #24)
- Added `seed` and `seed_generator` parameters to `Benchmark.__init__` for enabling reproducibility (PR: #24)
- Added `seed_generator` parameter to all benchmark setup methods (`setup_environment`, `setup_user`, `setup_agents`, `setup_evaluators`) (PR: #24)
- Added `seed` parameter to `ModelAdapter.__init__` for deterministic model inference (PR: #24)
- Added `SeedingError` exception for providers that don't support seeding (Anthropic models raise this if seed is provided) (PR: #24)
- Added seed support to interface adapters: `OpenAIModelAdapter`, `GoogleGenAIModelAdapter`, `LiteLLMModelAdapter`, `HuggingFaceModelAdapter` pass seeds to underlying APIs (PR: #24)

**Interface**

- CAMEL-AI integration: `CamelAgentAdapter` and `CamelLLMUser` for evaluating CAMEL-AI ChatAgent-based systems (PR: #22)
- Added `CamelAgentUser` for using a CAMEL ChatAgent as the user in agent-to-agent evaluation (PR: #22)
- Added `camel_role_playing_execution_loop()` for benchmarks using CAMEL's RolePlaying semantics (PR: #22)
- Added `CamelRolePlayingTracer` and `CamelWorkforceTracer` for capturing orchestration-level traces from CAMEL's multi-agent systems (PR: #22)

### Changed

**Benchmarks**

- `MACSBenchmark` and `Tau2Benchmark` benchmarks now actively use the seeding system by deriving seeds for model adapters. Seeds are passed to agents, user simulators, tool simulators, and LLM-based evaluators for reproducible runs. (PR: #26)
  - `Gaia2Benchmark`: Seeds `agents/gaia2_agent`, `evaluators/judge`
  - `MACSBenchmark`: Seeds `environment/tools/tool_{name}`, `simulators/user`, `evaluators/user_gsr`, `evaluators/system_gsr`
  - `Tau2Benchmark`: Seeds `simulators/user`, `agents/default_agent`

**User**

- Refactored `User` class into abstract base class defining the interface (`get_initial_query()`, `respond()`, `is_done()`) with `LLMUser` as the concrete LLM-driven implementation. This enables non-LLM user implementations (scripted, human-in-the-loop, agent-based). (PR: #22)
- Renamed `AgenticUser` → `AgenticLLMUser` for consistency with the new hierarchy (PR: #22)

**Interface**

- Renamed framework-specific user classes to reflect the new `LLMUser` base (PR: #22):
  - `SmolAgentUser` → `SmolAgentLLMUser`
  - `LangGraphUser` → `LangGraphLLMUser`
  - `LlamaIndexUser` → `LlamaIndexLLMUser`

### Fixed

### Removed

## [0.3.0] - 2025-01-18

### Added

**Parallel Execution**

- Added parallel task execution with `num_workers` parameter in `Benchmark.run()` using `ThreadPoolExecutor` (PR: #14)
- Added `ComponentRegistry` class for thread-safe component registration with thread-local storage (PR: #14)
- Added `TaskContext` for cooperative timeout checking with `check_timeout()`, `elapsed`, `remaining`, and `is_expired` properties (PR: #14)
- Added `TaskProtocol` dataclass with `timeout_seconds`, `timeout_action`, `max_retries`, `priority`, and `tags` fields for task-level execution control (PR: #14)
- Added `TimeoutAction` enum (`SKIP`, `RETRY`, `RAISE`) for configurable timeout behavior (PR: #14)
- Added `TaskTimeoutError` exception with `elapsed`, `timeout`, and `partial_traces` attributes (PR: #14)
- Added `TASK_TIMEOUT` to `TaskExecutionStatus` enum for timeout classification (PR: #14)

**Task Queue Abstraction**

- Added `TaskQueue` abstract base class with iterator interface for flexible task scheduling (PR: #14)
- Added `SequentialQueue` for simple FIFO task ordering (PR: #14)
- Added `PriorityQueue` for priority-based task scheduling using `TaskProtocol.priority` (PR: #14)
- Added `AdaptiveTaskQueue` abstract base class for feedback-based adaptive scheduling with `initial_state()`, `select_next_task(remaining, state)`, and `update_state(task, report, state)` methods (PR: #14)

**ModelAdapter Chat Interface**

- Added `chat()` method to `ModelAdapter` as the primary interface for LLM inference, accepting a list of messages in OpenAI format and returning a `ChatResponse` object and accepting tools
- Added `ChatResponse` dataclass containing `content`, `tool_calls`, `role`, `usage`, `model`, and `stop_reason` fields for structured response handling

**AnthropicModelAdapter**

- New `AnthropicModelAdapter` for direct integration with Anthropic Claude models via the official Anthropic SDK
- Handles Anthropic-specific message format conversion (system messages, tool_use/tool_result blocks) internally while accepting OpenAI-compatible input
- Added `anthropic` optional dependency: `pip install maseval[anthropic]`

**Benchmarks**

- Tau2 Benchmark: Full implementation of the tau2-bench benchmark for evaluating LLM-based agents on customer service tasks across airline, retail, and telecom domains (PR: #16)
- `Tau2Benchmark`, `Tau2Environment`, `Tau2User`, `Tau2Evaluator` components for framework-agnostic evaluation (PR: #16)
- `DefaultAgentTau2Benchmark` using an agent setup closely resembeling to the original tau2-bench implementation (PR: #16)
- Data loading utilities: `load_tasks()`, `ensure_data_exists()`, `configure_model_ids()` (PR: #16)
- Metrics: `compute_benchmark_metrics()`, `compute_pass_at_k()`, `compute_pass_hat_k()` for tau2-style scoring (PR: #16)
- Domain implementations with tool kits: `AirlineTools`, `RetailTools`, `TelecomTools` with full database simulation (PR: #16)

**User**

- `AgenticUser` class for users that can use tools during conversations (PR: #16)
- Multiple stop token support: `User` now accepts `stop_tokens` (list) instead of single `stop_token`, enabling different termination reasons (PR: #16)
- Stop reason tracking: `User` traces now include `stop_reason`, `max_turns`, `turns_used`, and `stopped_by_user` for detailed termination analysis (PR: #16)

**Simulator**

- `AgenticUserLLMSimulator` for LLM-based user simulation with tool use capabilities (PR: #16)

**Examples**

- Tau2 benchmark example with default agent implementation and result comparison scripts (PR: #16)

### Changed

**Benchmark**

- `Benchmark.agent_data` parameter is now optional (defaults to empty dict) (PR: #16)
- Refactored `Benchmark` to delegate registry operations to `ComponentRegistry` class (PR: #)
- `Benchmark.run()` now accepts optional `queue` parameter (`BaseTaskQueue`) for custom task scheduling (PR: #14)

**Task**

- `Task.id` is now `str` type instead of `UUID`. Benchmarks can provide human-readable IDs directly (e.g., `Task(id="retail_001", ...)`). Auto-generates UUID string if not provided. (PR: #16)

### Fixed

- Task reports now use `task.id` directly instead of `metadata["task_id"]` (PR: #16)

### Removed

## [0.2.0] - 2025-12-05

### Added

**Exceptions and Error Classification**

- Added `AgentError`, `EnvironmentError`, `UserError` exception hierarchy in `maseval.core.exceptions` for classifying execution failures by responsibility (PR: #13)
- Added `TaskExecutionStatus.AGENT_ERROR`, `ENVIRONMENT_ERROR`, `USER_ERROR`, `UNKNOWN_EXECUTION_ERROR` for fine-grained error classification enabling fair scoring (PR: #13)
- Added validation helpers: `validate_argument_type()`, `validate_required_arguments()`, `validate_no_extra_arguments()`, `validate_arguments_from_schema()` for tool implementers (PR: #13)
- Added `ToolSimulatorError` and `UserSimulatorError` exception subclasses for simulator-specific context while inheriting proper classification (PR: #13)

**Documentation**

- Added Exception Handling guide explaining error classification, fair scoring, and rerunning failed tasks (PR: #13)

**Benchmarks**

- MACS Benchmark: Multi-Agent Collaboration Scenarios benchmark (PR: #13)

**Benchmark**

- Added `execution_loop()` method to `Benchmark` base class enabling iterative agent-user interaction (PR: #13)
- Added `max_invocations` constructor parameter to `Benchmark` (default: 1 for backwards compatibility) (PR: #13)
- Added abstract `get_model_adapter(model_id, **kwargs)` method to `Benchmark` base class as universal model factory to be used throughout the benchmarks. (PR: #13)

**User**

- Added `max_turns` and `stop_token` parameters to `User` base class for multi-turn support with early stopping. Same applied to `UserLLMSimulator`. (PR: #13)
- Added `is_done()`, `_check_stop_token()`, and `increment_turn()` methods to `User` base class (PR: #13)
- Added `get_initial_query()` method to `User` base class for LLM-generated initial messages (PR: #13)
- Added `initial_query` parameter in `User` base class to trigger the agentic system. (PR: #13)

**Environment**

- Added `Environment.get_tool(name)` method for single-tool lookup (PR: #13)

**Interface**

- [LlamaIndex](https://github.com/run-llama/llama_index) integration: `LlamaIndexAgentAdapter` and `LlamaIndexLLMUser` for evaluating LlamaIndex workflow-based agents (PR: #7)
- The `logs` property inside `SmolAgentAdapter` and `LanggraphAgentAdapter` are now properly filled. (PR: #3)

**Examples**

- Added a new example: The `5_a_day_benchmark` (PR: #10)

### Changed

**Exception Handling**

- Benchmark now classifies execution errors into `AGENT_ERROR` (agent's fault), `ENVIRONMENT_ERROR` (tool/infra failure), `USER_ERROR` (user simulator failure), or `UNKNOWN_EXECUTION_ERROR` (unclassified) instead of generic `TASK_EXECUTION_FAILED` (PR: #13)
- `ToolLLMSimulator` now raises `ToolSimulatorError` (classified as `ENVIRONMENT_ERROR`) on failure (PR: #13)
- `UserLLMSimulator` now raises `UserSimulatorError` (classified as `USER_ERROR`) on failure (PR: #13)

**Environment**

- `Environment.create_tools()` now returns `Dict[str, Any]` instead of `list` (PR: #13)

**Benchmark**

- `Benchmark.run_agents()` signature changed: added `query: str` parameter (PR: #13)
- `Benchmark.run()` now uses `execution_loop()` internally to handle agent-user interaction cycles (PR: #13)
- `Benchmark` class now has a `fail_on_setup_error` flag that raises errors observed during setup of task (PR: #10)

**Callback**

- `FileResultLogger` now accepts `pathlib.Path` for argument `output_dir` and has an `overwrite` argument to prevent overwriting of existing logs files.

**Evaluator**

- The `Evaluator` class now has a `filter_traces` base method to conveniently adapt the same evaluator to different entities in the traces (PR: #10).

**Simulator**

- The `LLMSimulator` now throws an exception when json cannot be decoded instead of returning the error message as text to the agent (PR: #13).

**Other**

- Documentation formatting improved. Added darkmode and links to `Github` (PR: #11).
- Improved Quick Start Guide in `docs/getting-started/quickstart.md`. (PR: #10)
- `maseval.interface.agents` structure changed. Tools requiring framework imports (beyond just typing) now in `<framework>_optional.py` and imported dynamically from `<framework>.py`. (PR: #12)
- Various formatting improvements in the documentation (PR: #12)
- Added documentation for View Source Code pattern in `CONTRIBUTING.md` and `_optional.py` pattern in interface README (PR: #12)

### Fixed

**Interface**

- `LlamaIndexAgentAdapter` now supports multiple LlamaIndex agent types including `ReActAgent` (workflow-based), `FunctionAgent`, and legacy agents by checking for `.chat()`, `.query()`, and `.run()` methods in priority order (PR: #10)

**Other**

- Consistent naming of agent `adapter` over `wrapper` (PR: #3)
- Fixed an issue that `LiteLLM` interface and `Mixin`s were not shown in documentation properly (#PR: 12)

### Removed

- Removed `set_message_history`, `append_message_history` and `clear_message_history` for `AgentAdapter` and subclasses. (PR: #3)

## [0.1.2] - 2025-11-18

### Added

- Automated release workflow with version verification
- Documentation for release process

### Changed

- Improved project documentation structure

## [0.1.1] - [Previous release date]

<!-- Previous changes here -->

[Unreleased]: https://github.com/parameterlab/maseval/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/parameterlab/maseval/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/parameterlab/maseval/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/parameterlab/maseval/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/parameterlab/maseval/releases/tag/v0.1.1
