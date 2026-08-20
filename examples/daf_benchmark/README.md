# DAF Benchmark Example

This example demonstrates how to evaluate agent systems using MASEval, with support for:
- **Ollama** (local LLMs via OpenAI-compatible API)
- **DAF Backend** (Declarative Agentic Framework)
- **Other LLM providers** (OpenAI, Anthropic, Google, etc. via LiteLLM)

## What This Example Shows

### Metrics Tracked

1. **Latency** - Time taken per task and aggregate
   - Per-task duration
   - Average duration by category
   - Total wall time

2. **Cost** - Token-based cost estimation
   - Input/output tokens
   - Total cost (when provider reports it)

3. **Accuracy/Completeness** - Evaluation against expected answers
   - Exact match (factual questions)
   - Numeric match (math problems)
   - Keyword match (explanations)
   - Code presence (coding tasks)

## Quick Start

### 1. Set Up Environment

Copy the example `.env` file and configure:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Install Dependencies

```bash
# From the MASEval root
uv sync --all-extras

# Or install just what you need
pip install "maseval[litellm]"
pip install python-dotenv
```

### 3. Run with Ollama (Local LLM)

First, make sure Ollama is running:

```bash
# Start Ollama (if not already running)
ollama serve

# Pull a model
ollama pull llama3.2
```

Then run the benchmark:

```bash
uv run python examples/daf_benchmark/daf_benchmark.py \
    --model "ollama/llama3.2" \
    --api-base "http://localhost:11434/v1"
```

### 4. Run with OpenAI (for comparison)

```bash
export OPENAI_API_KEY="sk-your-key"

uv run python examples/daf_benchmark/daf_benchmark.py \
    --model "gpt-4o-mini"
```

### 5. Run with DAF Backend

If you have a DAF backend running:

```bash
export DAF_API_KEY="daf_your_key"

uv run python examples/daf_benchmark/daf_benchmark.py \
    --use-daf \
    --daf-url "http://localhost:8012" \
    --daf-skill "general"
```

## Sample Output

```
================================================================================
Results Summary
================================================================================
Total tasks: 5
Overall accuracy: 80.00%
Average duration per task: 2.34s
Total duration: 11.70s
Wall time: 11.85s
Total tokens: 1250
  Input: 750
  Output: 500

Metrics by Category:
  factual:
    Accuracy: 100.00%
    Avg duration: 1.23s
    Tasks: 1
  reasoning:
    Accuracy: 75.00%
    Avg duration: 3.45s
    Tasks: 2
  coding:
    Accuracy: 100.00%
    Avg duration: 2.10s
    Tasks: 1
  explanation:
    Accuracy: 66.67%
    Avg duration: 2.80s
    Tasks: 1
```

## Architecture

```
SimpleBenchmark (MASEval Benchmark)
├── SimpleLLMAgent (wrapped in SimpleAgentAdapter)
│   └── LiteLLMModelAdapter (connects to Ollama/OpenAI/etc.)
├── SimpleEnvironment (tracks task state)
├── AccuracyEvaluator (scores responses)
└── Tasks (sample evaluation questions)
```

## Using DAF Adapter

The `DAFAgentAdapter` wraps DAF's orchestration API:

```python
from maseval.interface.agents.daf import DAFAgentAdapter
from daf_sdk import DAF

# Create DAF client
client = DAF(base_url="http://localhost:8012", api_key="daf_key")

# Create adapter
adapter = DAFAgentAdapter(
    agent_instance=client,
    skill_id="my_skill",
    name="daf_agent",
)

# Use in benchmark
result = adapter.run("Your task")
traces = adapter.gather_traces()
```

## Running Tests

```bash
# Run DAF integration tests
uv run pytest tests/test_interface/test_daf/ -v

# Run with coverage
uv run pytest tests/test_interface/test_daf/ --cov=maseval.interface.agents.daf
```

## Extending This Example

### Add Custom Tasks

```python
from maseval import Task

task = Task(
    query="Your custom question",
    metadata={
        "task_id": "custom_001",
        "category": "your_category",
        "expected": "expected_answer",
    },
)
```

### Add Custom Evaluator

```python
from maseval import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(self, environment, agents, user):
        # Your evaluation logic
        return {"score": 0.95, "details": "..."}
```

### Compare Multiple Models

```python
models = ["ollama/llama3.2", "gpt-4o-mini", "claude-3-haiku"]

for model_id in models:
    model = LiteLLMModelAdapter(model_id=model_id, ...)
    benchmark = SimpleBenchmark(model=model, ...)
    results = benchmark.run(tasks=tasks, agent_data={})
    metrics = compute_benchmark_metrics(results)
    print(f"{model_id}: {metrics['accuracy']:.2%}")
```

## Files

- `daf_benchmark.py` - Main benchmark script
- `.env.example` - Environment configuration template
- `__init__.py` - Package initialization

## Related

- [MASEval Quickstart](../../docs/getting-started/quickstart.md)
- [DAF Adapter Source](../../maseval/interface/agents/daf.py)
- [Tau2 Benchmark Example](../tau2_benchmark/) - More complex multi-agent evaluation
