# Five-A-Day Benchmark

Like the health advice to eat "5 a day" (fruits and vegetables), this benchmark provides **5 tasks a day** to keep your agent evaluation healthy. An educational example demonstrating **maseval** through 5 diverse tasks across smolagents, LangGraph, and LlamaIndex.

**What it shows:** Framework-agnostic tools, diverse evaluation types (assertions, LLM judges, unit tests, optimization), single vs multi-agent patterns, and reproducible seeding.

## Quick Start

```bash
export GOOGLE_API_KEY="your-api-key-here"

# Run all 5 tasks
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py

# Try different frameworks
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework langgraph
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --framework llamaindex

# Single task, multi-agent, with seed
uv run python examples/five_a_day_benchmark/five_a_day_benchmark.py --task 0 --config-type multi --seed 42
```

**Options:** `--framework {smolagents,langgraph,llamaindex}`, `--config-type {single,multi}`, `--task N`, `--seed N`, `--limit N`, `--temperature N`

## The 5 Tasks

| Task                   | Description                 | Tools                                      | Evaluation Focus                           |
| ---------------------- | --------------------------- | ------------------------------------------ | ------------------------------------------ |
| **0. Email & Banking** | Verify payment, draft email | `email`, `banking`                         | Financial accuracy, email quality, privacy |
| **1. Finance Calc**    | Stock inheritance split     | `family_info`, `stock_price`, `calculator` | Arithmetic, data retrieval                 |
| **2. Code Gen**        | House robber algorithm      | `python_executor`                          | Unit tests, complexity, code quality       |
| **3. Calendar**        | Find meeting slots (MCP)    | `my_calendar_mcp`, `other_calendar_mcp`    | Slot matching, MCP integration             |
| **4. Hotel**           | Optimize hotel selection    | `hotel_search`, `calculator`               | Multi-criteria optimization                |

## Architecture

**Environment:** `FiveADayEnvironment` loads task data from JSON and creates framework-agnostic tools using a collection pattern.

**Tools:** `BaseTool` subclasses with `.to_smolagents()`, `.to_langgraph()`, `.to_llamaindex()` methods. Tool adapters support tracing.

**Tasks:** Defined in `data/tasks.json` with query, environment data, evaluation criteria, and metadata.

**Evaluators:** 13 evaluators across 5 tasks using assertions, LLM judges, unit tests, pattern matching, static analysis, and optimization comparisons.

**Agents:** Single-agent (one agent, all tools) or multi-agent (orchestrator + specialists). Framework-specific adapters provide unified interface.

**Seeding:** Uses MASEval's `Benchmark(seed=...)` parameter with `DefaultSeedGenerator` for reproducible per-agent seeds via SHA256 hashing.

## Project Structure

```
five_a_day_benchmark/
├── five_a_day_benchmark.py       # Main benchmark (FiveADayBenchmark class)
├── utils.py                       # sanitize_name() helper
├── tools/                         # BaseTool subclasses with framework converters
├── evaluators/                    # Task-specific Evaluator subclasses
└── data/
    ├── tasks.json                # Task definitions
    ├── singleagent.json          # Single-agent configs
    └── multiagent.json           # Multi-agent configs
```

Results saved to `results/*.jsonl` with full message traces.

## Dependencies

`maseval`, `litellm`, `smolagents`, `langgraph`, `langchain-litellm`, `llama-index-core`, `llama-index-llms-litellm`, `RestrictedPython`

Set `GOOGLE_API_KEY` for Gemini models.
