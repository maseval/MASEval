<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/parameterlab/MASEval/refs/heads/main/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/parameterlab/MASEval/refs/heads/main/assets/logo-light.svg">
    <img src="https://raw.githubusercontent.com/parameterlab/MASEval/refs/heads/main/assets/logo-light.svg" alt="MASEval logo" width="240" />
  </picture>
</p>

# LLM-based Multi-Agent Evaluation & Benchmark Framework

[![ParameterLab](https://img.shields.io/badge/Parameter-Lab-black.svg)](https://www.parameterlab.de)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/maseval.svg)](https://badge.fury.io/py/maseval)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://maseval.readthedocs.io/en/stable/)
[![Tests](https://github.com/parameterlab/MASEval/actions/workflows/test.yml/badge.svg)](https://github.com/parameterlab/MASEval/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/parameterlab/MASEval/graph/badge.svg?token=HMFU71QVB2)](https://codecov.io/gh/parameterlab/MASEval)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

MASEval is an evaluation library that provides a unified interface for benchmarking (multi-)agent systems. It offers standardized abstractions for running any agent implementation—whether built with AutoGen, LangChain, custom frameworks, or direct API calls—against established benchmarks like GAIA and AgentBench, or your own custom evaluation tasks.

Analogous to pytest for testing or MLflow for ML experimentation, MASEval focuses exclusively on evaluation infrastructure. It does not implement agents, define multi-agent communication protocols, or turn LLMs into agents. Instead, it wraps existing agent systems via simple adapters, orchestrates the evaluation lifecycle (setup, execution, measurement, teardown), and provides lifecycle hooks for tracing, logging, and metrics collection. This separation allows researchers to compare different agent architectures apples-to-apples across frameworks, while maintaining full control over their agent implementations.

## Why MASEval?

Compare multi-agent evaluation frameworks across key capabilities.

| Library           | Multi-Agent | System Eval | Agent-Agnostic | Benchmarks | Flexible Interaction | BYO | Trace-First | Mature |
| ----------------- | :---------: | :---------: | :------------: | :--------: | :------------------: | :-: | :---------: | :----: |
| **MASEval**       |     ✅      |     ✅      |       ✅       |     ✅     |          ✅          | ✅  |     ✅      |   ✅   |
| **AnyAgent**      |     🟡      |     ✅      |       ✅       |     ❌     |          🟡          | ✅  |     🟡      |   ✅   |
| **MLflow GenAI**  |     🟡      |     🟡      |       ✅       |     ❌     |          🟡          | ✅  |     ✅      |   ✅   |
| **HAL Harness**   |     🟡      |     ✅      |       ✅       |     ✅     |          🟡          | 🟡  |     🟡      |   🟡   |
| **Inspect-AI**    |     🟡      |     ✅      |       🟡       |     ✅     |          🟡          | 🟡  |     🟡      |   ✅   |
| **OpenCompass**   |     ❌      |     🟡      |       ❌       |     ✅     |          🟡          | 🟡  |     🟡      |   ✅   |
| **AgentGym**      |     ❌      |     ❌      |       ❌       |     ✅     |          🟡          | ✅  |     🟡      |   🟡   |
| **Arize Phoenix** |     🟡      |     ❌      |       🟡       |     ❌     |          ❌          | 🟡  |     ✅      |   ✅   |
| **TruLens**       |     🟡      |     ❌      |       🟡       |     ❌     |          ❌          | 🟡  |     ✅      |   ✅   |
| **MARBLE**        |     ✅      |     ❌      |       ❌       |     ✅     |          ❌          | ❌  |     🟡      |   🟡   |
| **DeepEval**      |     🟡      |     ❌      |       🟡       |     ❌     |          🟡          | 🟡  |     🟡      |   ✅   |
| **MCPEval**       |     ❌      |     ❌      |       ❌       |     ✅     |          ❌          | 🟡  |     🟡      |   🟡   |

**✅** Full/Native · **🟡** Partial/Limited · **❌** Not supported

<details>
<summary>Expand for Column Explanation</summary>

| Column                   | Feature                         | One-Liner                                                                                                         |
| ------------------------ | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Multi-Agent**          | Multi-Agent Native              | Native orchestration with per-agent tracing, independent message histories, and explicit coordination patterns.   |
| **System Eval**          | System-Level Comparison         | Compare different framework implementations on the same benchmark (not just swapping LLMs).                       |
| **Agent-Agnostic**       | Agent Framework Agnostic        | Evaluate agents from any framework via thin adapters without requiring protocol adoption or code recreation.      |
| **Benchmarks**           | Pre-Implemented Benchmarks      | Ships complete, ready-to-run benchmarks with environments, tools, and evaluators (not just templates).            |
| **Flexible Interaction** | Flexible Agent-Environment-User | First-class user simulation with personas and tool access for realistic multi-turn conversations.                 |
| **BYO**                  | BYO Philosophy                  | Bring your own logging, agents, environments, and tools. Open-source, works offline, no mandatory cloud services. |
| **Trace-First**          | Trace-First Evaluation          | Evaluate intermediate steps across environment and agents via first-class traces, not post-hoc fixes.             |
| **Mature**               | Professional Tooling            | Published on PyPI, CI/CD, good test coverage, active maintenance.                                                 |

</details>

## Core Principles:

- **Evaluation, Not Implementation:** MASEval provides the evaluation infrastructure—you bring your agent implementation. Whether you've built agents with AutoGen, LangChain, custom code, or direct LLM calls, MASEval wraps them via simple adapters and runs them through standardized benchmarks.

- **System-Level Benchmarking:** The fundamental unit of evaluation is the complete system—the full configuration of agents, prompts, tools, and their interaction patterns. This allows meaningful comparison between entirely different architectural approaches.

- **Task-Specific Configurations:** Each benchmark task is a self-contained evaluation unit with its own instructions, environment state, success criteria, and custom evaluation logic. One task might measure success by environment state changes, another by programmatic output validation.

- **Framework Agnostic by Design:** MASEval is intentionally unopinionated about agent frameworks, model providers, and system architectures. Simple, standardized interfaces and adapters enable any agent system to be evaluated without modification to the core library.

- **Lifecycle Hooks via Callbacks:** Inject custom logic at any point in the evaluation lifecycle (e.g., on_run_start, on_task_start, on_agent_step_end) through a callback system. This enables extensibility without modifying core evaluation logic.

- **Pluggable Backends:** Tracing, logging, metrics, and data storage are implemented as callbacks. Easily add new backends or combine existing ones—log to WandB and Langfuse simultaneously, or implement custom metrics collectors.

- **Extensible Benchmark Suite:** Researchers can implement new benchmarks by inheriting from base classes and focusing on task construction and evaluation logic, while leveraging built-in evaluation infrastructure.

- **Abstract Base Classes:** The library provides abstract base classes for core components (Task, Benchmark, Environment, Evaluator) with optional default implementations, giving users flexibility to customize while maintaining interface consistency.

## Install

The package is published on PyPI as `maseval`. To install the stable release for general use, run:

```bash
pip install maseval
```

If you want the optional integrations used by the examples (smolagents, langgraph, llamaindex, etc.), install the examples extras:

```bash
pip install "maseval[examples]"
```

Or install specific framework integrations:

```bash
# Smolagents
pip install "maseval[smolagents]"

# LangGraph
pip install "maseval[langgraph]"

# LlamaIndex
pip install "maseval[llamaindex]"
```

Or install benchmark-specific dependencies:

```bash
# MMLU (HuggingFace models)
pip install "maseval[mmlu]"
```

## Example

Examples are available in the [Documentation](https://maseval.readthedocs.io/en/stable/).

## Contribute

We welcome any contributions. Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file to learn more!

## Benchmarks

This library includes implementations for several benchmarks to evaluate a variety of multi-agent scenarios. Each benchmark is designed to test specific collaboration and problem-solving skills.

➡️ **[See here for a full list and description of all available benchmarks including licenses.](./BENCHMARKS.md)**

## Citation

Please consider citing the MASEval library.

```
@misc{emde2026maseval,
      title={MASEval: Extending Multi-Agent Evaluation from Models to Systems},
      author={Cornelius Emde and Alexander Rubinstein and Anmol Goel and Ahmed Heakl and Sangdoo Yun and Seong Joon Oh and Martin Gubri},
      year={2026},
      eprint={2603.08835},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.08835},
      note={Alexander Rubinstein, Anmol Goel, and Ahmed Heakl contributed equally.},
}
```
