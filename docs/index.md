# MASEval

Welcome to the MASEval documentation.

MASEval is an evaluation library that provides a unified interface for benchmarking (multi-)agent systems. It offers standardized abstractions for running any agent implementation—whether built with AutoGen, LangChain, custom frameworks, or direct API calls—against established benchmarks like GAIA and AgentBench, or your own custom evaluation tasks.

Analogous to pytest for testing or MLflow for ML experimentation, MASEval focuses exclusively on evaluation infrastructure. It does not implement agents, define multi-agent communication protocols, or turn LLMs into agents. Instead, it wraps existing agent systems via simple adapters, orchestrates the evaluation lifecycle (setup, execution, measurement, teardown), and provides lifecycle hooks for tracing, logging, and metrics collection. This separation allows researchers to compare different agent architectures apples-to-apples across frameworks, while maintaining full control over their agent implementations.

## Install

Install the package from PyPI:

```bash
pip install maseval
```

More details in the [Quickstart](getting-started/quickstart.md)

## Why MASEval?

Compare multi-agent evaluation frameworks across key capabilities.

| Library           | Multi-Agent | System Evaluation | Agent-Agnostic | Benchmarks | Multi-turn User | No Lock-In | BYO | State-Action Eval | Error Attr | Lightweight | Project Maturity | Sandboxed Environment |
| ----------------- | :---------: | :---------------: | :------------: | :--------: | :-------------: | :--------: | :-: | :---------------: | :--------: | :---------: | :--------------: | :-------------------: |
| **MASEval**       |     ✅      |        ✅         |       ✅       |     ✅     |       ✅        |     ✅     | 🟢  |        ✅         |     ✅     |     ✅      |        ✅        |          🟢           |
| **HAL Harness**   |     🟡      |        ✅         |       ✅       |     ✅     |       🟡        |     ✅     | 🟡  |        🟡         |     ❌     |     ✅      |        🟡        |          ✅           |
| **AnyAgent**      |     🟡      |        ✅         |       ✅       |     ❌     |       🟡        |     ✅     | 🟢  |        🟡         |     ❌     |     ✅      |        ✅        |          ❌           |
| **Inspect-AI**    |     🟡      |        ✅         |       🟡       |     ✅     |       🟡        |     ✅     | 🟡  |        🟡         |     ❌     |     🟡      |        ✅        |          ✅           |
| **MLflow GenAI**  |     🟡      |        🟡         |       🟢       |     ❌     |       🟡        |     ✅     | 🟢  |        ✅         |     ❌     |     🟡      |        ✅        |          🟡           |
| **LangSmith**     |     🟡      |        🟡         |       🟡       |     ❌     |       ✅        |     ❌     | 🟡  |        ✅         |     ❌     |     ✅      |        ✅        |          ❌           |
| **OpenCompass**   |     ❌      |        🟡         |       ❌       |     ✅     |       🟡        |     ✅     | 🟡  |        🟡         |     ❌     |     ❌      |        ✅        |          🟡           |
| **AgentGym**      |     ❌      |        ❌         |       ❌       |     ✅     |       🟡        |     ✅     | 🟢  |        🟡         |     ❌     |     ❌      |        🟡        |          🟡           |
| **Arize Phoenix** |     🟡      |        ❌         |       🟡       |     ❌     |       ❌        |     🟡     | 🟢  |        ✅         |     ❌     |     🟡      |        ✅        |          ❌           |
| **MARBLE**        |     ✅      |        ❌         |       ❌       |     ✅     |       ❌        |     ✅     | ❌  |        🟡         |     ?      |     🟡      |        🟡        |          🟡           |
| **TruLens**       |     🟡      |        ❌         |       🟡       |     ❌     |       ❌        |     ✅     | 🟡  |        🟢         |     ❌     |     🟡      |        ✅        |          ❌           |
| **AgentBeats**    |     🟡      |        ❌         |       🟡       |     ❌     |       ❌        |     🟡     | 🟡  |        🟡         |     ?      |     ✅      |        🟡        |          🟡           |
| **DeepEval**      |     🟡      |        ❌         |       🟡       |     ❌     |       🟡        |     🟡     | 🟡  |        🟡         |     ❌     |     🟡      |        ✅        |          ❌           |
| **MCPEval**       |     ❌      |        ❌         |       ❌       |     ✅     |       ❌        |     ✅     | 🟡  |        🟡         |     ❌     |     🟡      |        🟡        |          ❌           |
| **Galileo**       |     🟡      |        ❌         |       🟡       |     ❌     |       ❌        |     ❌     | 🟡  |        🟡         |     ❌     |     🟡      |        ✅        |          ❌           |

**✅** Full/Native · **🟢** Flexible for BYO · **🟡** Partial/Limited · **❌** Not possible

??? info "Column Explanation"

    | Column                | Feature                      | One-Liner                                                                                                          |
    | --------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
    | **Multi-Agent**       | Multi-Agent Native           | Native orchestration with per-agent tracing, independent message histories, and explicit coordination patterns.    |
    | **System Evaluation** | System-Level Comparison      | Compare different framework implementations on the same benchmark (not just swapping LLMs).                        |
    | **Agent Agnostic**    | Agent Framework Agnostic     | Evaluate agents from any framework via thin adapters without requiring protocol adoption or code recreation.       |
    | **Benchmarks**        | Pre-Implemented Benchmarks   | Ships complete, ready-to-run benchmarks with environments, tools, and evaluators (not just templates).             |
    | **Multi-turn User**   | User-Agent Multi-turn        | First-class user simulation with personas, stop tokens, and tool access for realistic multi-turn conversations.    |
    | **No Lock-In**        | No Vendor Lock-In            | Fully open-source, works offline, permissive license (MIT/Apache), no mandatory cloud services or telemetry.       |
    | **BYO**               | BYO Philosophy               | Bring your own logging, agents, environments, and tools — flexibility over opinionated defaults.                   |
    | **State-Action Eval** | Trace-First Evaluation       | Evaluate intermediate steps and tool usage patterns via trace filtering, not just final output scoring.            |
    | **Error Attr**        | Structured Error Attribution | Structured exceptions distinguish between different failure for fair scoring (`AgentError` vs `EnvironmentError`). |
    | **Lightweight**       | Lightweight                  | Minimal dependencies, small codebase (~20k LOC), quick time to first evaluation (~5-15 min).                       |
    | **Project Maturity**  | Professional Tooling         | Published on PyPI, CI/CD, good test coverage, structured logging, active maintenance, excellent docs.              |
    | **Sandbox**           | Sandboxed Execution          | Built-in Docker/K8s/VM isolation for safe code execution (or BYO sandbox via abstract Environment).                |

## Core Principles

- **Evaluation, Not Implementation:** MASEval provides the evaluation infrastructure—you bring your agent implementation. Whether you've built agents with AutoGen, LangChain, custom code, or direct LLM calls, MASEval wraps them via simple adapters and runs them through standardized benchmarks.

- **System-Level Benchmarking:** The fundamental unit of evaluation is the complete system—the full configuration of agents, prompts, tools, and their interaction patterns. This allows meaningful comparison between entirely different architectural approaches.

- **Task-Specific Configurations:** Each benchmark task is a self-contained evaluation unit with its own instructions, environment state, success criteria, and custom evaluation logic. One task might measure success by environment state changes, another by programmatic output validation.

- **Framework Agnostic by Design:** MASEval is intentionally unopinionated about agent frameworks, model providers, and system architectures. Simple, standardized interfaces and adapters enable any agent system to be evaluated without modification to the core library.

- **Lifecycle Hooks via Callbacks:** Inject custom logic at any point in the evaluation lifecycle (e.g., `on_run_start`, `on_task_start`, `on_agent_step_end`) through a callback system. This enables extensibility without modifying core evaluation logic.

- **Pluggable Backends:** Tracing, logging, metrics, and data storage are implemented as callbacks. Easily add new backends or combine existing ones—log to WandB and Langfuse simultaneously, or implement custom metrics collectors.

- **Extensible Benchmark Suite:** Researchers can implement new benchmarks by inheriting from base classes and focusing on task construction and evaluation logic, while leveraging built-in evaluation infrastructure.

- **Abstract Base Classes:** The library provides abstract base classes for core components (Task, Benchmark, Environment, Evaluator) with optional default implementations, giving users flexibility to customize while maintaining interface consistency.

## API

See the automatic API reference under `Reference`.
