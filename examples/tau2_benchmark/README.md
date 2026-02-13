# Tau 2 Benchmark Examples

This directory contains examples for running the Tau 2 benchmark with MASEval.

## Overview

The Tau 2 benchmark evaluates single-agent customer service tasks across three domains:

| Domain | Tasks | Description |
|--------|-------|-------------|
| airline | 50 | Flight reservation management |
| retail | 114 | E-commerce order management |
| telecom | 114 | Telecom customer service |

## Key Features

- **Real tool implementations** - Tools execute actual business logic and modify database state
- **Deterministic evaluation** - Success is measured by database state comparison
- **Pass@k metrics** - Recommended for robust evaluation

## Usage

### Running with smolagents

```bash
# Run on retail domain with 5 tasks
uv run python examples/tau2_benchmark/tau2_benchmark.py \
    --framework smolagents \
    --domain retail \
    --limit 5
```

### Running with langgraph

```bash
# Run on airline domain with 5 tasks
uv run python examples/tau2_benchmark/tau2_benchmark.py \
    --framework langgraph \
    --domain airline \
    --limit 5
```

### Computing Pass@k Metrics

Run each task multiple times for more robust Pass@k metrics:

```bash
# Run 4 times per task for Pass@1,2,3,4
uv run python examples/tau2_benchmark/tau2_benchmark.py \
    --framework smolagents \
    --domain telecom \
    --repeats 4
```

## Requirements

- Google API key: Set `GOOGLE_API_KEY` environment variable
- Dependencies: `smolagents`, `langgraph`, `langchain-google-genai`

## Reference

- Paper: [Tau-Bench: A Benchmark for Tool-Agent-User Interaction](https://arxiv.org/abs/2506.07982)
- Data: [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench)
