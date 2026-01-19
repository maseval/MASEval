# Gaia2 Benchmark - Provenance Documentation

This document tracks the source mapping between MASEval's Gaia2 implementation and Meta's ARE (Agent Research Environments) platform.

## Base Version

**ARE v1.2.0** (January 2025)

Repository: https://github.com/facebookresearch/agent-research-environments
Package: `meta-agents-research-environments` (PyPI)
License: MIT License
Copyright: (c) Meta Platforms, Inc. and affiliates

**Dataset**: `meta-agents-research-environments/gaia2` (HuggingFace)

## Integration Approach

Unlike Tau2 which ports tool implementations, Gaia2 **integrates** with ARE as a runtime dependency. ARE provides:

- Simulation environment with time management
- App implementations (Calendar, Email, Messaging, etc.)
- Scenario parsing and initialization
- Judge (GraphPerEventJudge) for evaluation

MASEval provides:

- Orchestration and task management
- Tracing and observability
- Agent framework flexibility
- Metrics aggregation

## Component Mapping

### Core Infrastructure

| MASEval Component   | ARE Component                                         | Notes                                      |
| ------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `data_loader.py`    | HuggingFace `datasets` + `JsonScenarioImporter`       | Loads from HF, converts to MASEval Tasks   |
| `tool_wrapper.py`   | `are.simulation.apps.*.AppTool`                       | Wraps ARE tools for MASEval tracing        |
| `environment.py`    | `are.simulation.environment.Environment`              | Wraps ARE Environment in MASEval interface |
| `evaluator.py`      | `are.simulation.validation.GraphPerEventJudge`        | Uses ARE's judge, returns MASEval format   |

### Environment Integration

| MASEval Method                   | ARE Method/Component                  | Notes                                |
| -------------------------------- | ------------------------------------- | ------------------------------------ |
| `Gaia2Environment.setup_state()` | `Environment.initialize_scenario()`   | Initializes ARE simulation           |
| `Gaia2Environment.create_tools()`| `App.get_tools()` for all apps        | Wraps all app tools with tracing     |
| `Gaia2Environment.cleanup()`     | `Environment.stop()`                  | Ensures proper resource cleanup      |
| `get_simulation_time()`          | `TimeManager.current_time`            | Exposes simulation time for tracing  |

### Evaluator Integration

| MASEval Method                  | ARE Component                           | Notes                                |
| ------------------------------- | --------------------------------------- | ------------------------------------ |
| `Gaia2Evaluator.__call__()`     | `GraphPerEventJudge.evaluate()`         | Delegates to ARE's deterministic judge |
| `filter_traces()`               | N/A                                     | MASEval-specific trace extraction    |
| `compute_gaia2_metrics()`       | N/A                                     | MASEval-specific metrics aggregation |

### Tool Wrapper

| MASEval Feature                 | ARE Feature                             | Notes                                |
| ------------------------------- | --------------------------------------- | ------------------------------------ |
| `AREToolWrapper.__call__()`     | `AppTool.__call__()`                    | Delegates execution to ARE           |
| `history` (ToolInvocationHistory) | N/A                                   | MASEval tracing addition             |
| Simulation time tracking        | `TimeManager.current_time`              | Records time before/after each call  |

## Data Source

Scenarios are loaded from HuggingFace:

```
https://huggingface.co/datasets/meta-agents-research-environments/gaia2
```

| Config      | Description                                | Split      |
| ----------- | ------------------------------------------ | ---------- |
| `validation`| Full validation set (all capabilities)     | validation |
| `execution` | Execution capability only                  | validation |
| `search`    | Search capability only                     | validation |
| `adaptability` | Adaptability capability only            | validation |
| `time`      | Temporal reasoning only                    | validation |
| `ambiguity` | Ambiguity handling only                    | validation |
| `agent2agent` | Multi-agent collaboration only           | validation |
| `noise`     | Noise handling only                        | validation |

## MASEval-Specific Additions

These components are new implementations that don't have direct ARE equivalents:

| Component                       | Description                                        |
| ------------------------------- | -------------------------------------------------- |
| `Gaia2Benchmark`                | Abstract benchmark base following MASEval patterns |
| `DefaultGaia2Agent`             | ReAct-style reference agent implementation         |
| `DefaultGaia2AgentAdapter`      | AgentAdapter wrapper for default agent             |
| `AREToolWrapper`                | Tool wrapper with MASEval tracing integration      |
| `compute_gaia2_metrics()`       | GSR aggregation by capability type                 |
| `configure_model_ids()`         | Optional LLM-based judge configuration             |
| TraceableMixin integration      | Execution tracing for all components               |
| ConfigurableMixin integration   | Configuration gathering                            |

## Key Characteristics

| Aspect          | Description                                       |
| --------------- | ------------------------------------------------- |
| Tools           | ARE app tools (real implementations)              |
| State           | ARE simulation state with time management         |
| Time Control    | Agent controls time via `wait_for_notification()` |
| Evaluation      | Deterministic event-based (GraphPerEventJudge)    |
| Interaction     | Event-driven (not turn-based user simulation)     |

## ARE Apps Available

The following ARE apps are exposed as tools to agents:

| App          | Description                              |
| ------------ | ---------------------------------------- |
| Calendar     | Event scheduling and management          |
| Email        | Email sending and reading                |
| Messaging    | Instant messaging                        |
| Contacts     | Contact management                       |
| Shopping     | E-commerce operations                    |
| Cab          | Ride booking                             |
| City         | Location and venue information           |
| FileSystem   | File operations                          |
| Browser      | Web browsing simulation                  |
| ChatsApp     | Group chat functionality                 |
| SystemApp    | Time control (`get_current_time`, `wait_for_notification`) |
| Timer        | Timer management                         |

## Validation Strategy

1. **Event-based evaluation**: ARE's GraphPerEventJudge compares completed events against oracle events
2. **GSR (Goal Success Rate)**: Binary success metric per scenario
3. **Partial GSR**: Partial credit for partially completed scenarios
4. **Capability breakdown**: Metrics computed per capability dimension

## Intentional Implementation Choices

### No User Simulator

Unlike Tau2, Gaia2 does not use a turn-based user simulator. User interactions happen through scheduled events in the ARE simulation (e.g., "user sends message at t=30s"). This is handled automatically by ARE's event system.

### Single-Turn Agent Execution

The default `max_invocations=1` means agents run once and complete the task. Time advancement happens through tool calls (`wait_for_notification`), not through multiple benchmark invocations.

### Tool Tracing with Simulation Time

`AREToolWrapper` records both wall-clock time and simulation time for each tool invocation, enabling analysis of agent temporal reasoning behavior.

## License

ARE is used under the MIT License. See the original repository for full license text.

The Gaia2 dataset on HuggingFace is subject to Meta's data usage terms.
