# ColBench: Collaborative Agent Benchmark

ColBench evaluates LLM agents on **collaborative backend-programming tasks** where the agent must interact with a simulated human user to gather requirements before producing a solution. It tests an agent's ability to ask clarifying questions, extract hidden constraints from user responses, and generate correct Python code, all within a limited number of dialogue turns.

Originally introduced in [Collaborative Agent Bench (sweet_rl)](https://github.com/facebookresearch/collaborative-llm-agent), this integration adapts ColBench to the MASEval framework.

## How It Works

Each task follows a multi-turn dialogue loop between the **agent** (LLM under test) and a **human simulator** (another LLM acting as the user):

```
┌─────────────────────────────────────────────────────┐
│                  ColBench Task Loop                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. User presents a problem description             │
│     "Write a function that calculates tip totals"   │
│                                                     │
│  2. Agent asks clarifying questions                 │
│     "What format is the input? Are tips per-person?"│
│                                                     │
│  3. Simulator responds with hidden information      │
│     "Tips are a dict mapping provider to amount..." │
│                                                     │
│  4. Repeat for up to max_steps turns                │
│                                                     │
│  5. Agent signals: "I WANT TO ANSWER:" + code       │
│                                                     │
│  6. Code is evaluated against hidden unit tests     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

The human simulator has access to **hidden information** (ground-truth code and constraints) that the agent must extract through dialogue. The agent signals completion by prefixing its response with `"I WANT TO ANSWER:"` followed by the Python code.

## Architecture

ColBench maps to MASEval's component model:

| MASEval Component | ColBench Implementation | Role |
|---|---|---|
| `Benchmark` | `ColBenchBenchmark` | Orchestrates the task loop |
| `User` | `ColBenchUser` | Human simulator (LLM-backed) |
| `AgentAdapter` | `ColBenchAgentAdapter` | Agent under test |
| `Environment` | `ColBenchEnvironment` | Holds task artifacts |
| `Evaluator` | `ColBenchCodeEvaluator` | Unit-test scoring |
| `ModelAdapter` | `OpenAIModelAdapter` | OpenAI-compatible API client |

## Quick Start

### 1. Start a vLLM Server

ColBench uses OpenAI-compatible API servers for both the agent and the human simulator. Start a vLLM server:

```bash
vllm serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu_memory_utilization 0.9 \
    --dtype bfloat16 \
    --max_model_len 23000
```

For production runs, use a larger model (e.g., `Llama-3.1-70B-Instruct`) for the human simulator and a separate server.

### 2. Run the Benchmark

```bash
python examples/colbench_benchmark/colbench.py \
    --agent_model meta-llama/Llama-3.1-8B-Instruct \
    --hostname localhost \
    --port 8001 \
    --env_model meta-llama/Llama-3.1-8B-Instruct \
    --input_path examples/colbench_benchmark/results/test.jsonl \
    --output_path examples/colbench_benchmark/results/temp_test.jsonl \
    --num_tasks 1000
```

### 3. Evaluate Existing Trajectories

To re-evaluate previously saved trajectories without re-running interactions:

```bash
python examples/colbench_benchmark/colbench.py \
    --evaluate_only \
    --output_path examples/colbench_benchmark/results/temp_test.jsonl
```

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--input_path` | *(required)* | Path to JSONL task file |
| `--output_path` | `colbench_results.jsonl` | Path to save trajectory results |
| `--num_tasks` | `1000` | Number of tasks to run |
| `--agent_model` | `Llama-3.1-8B-Instruct` | Model ID for the agent under test |
| `--env_model` | `auto` | Model ID for the human simulator (`auto` → `Llama-3.1-70B-Instruct`) |
| `--hostname` | `localhost` | Hostname of the vLLM server (human simulator) |
| `--port` | `8000` | Port of the vLLM server (human simulator) |
| `--agent_hostname` | same as `--hostname` | Hostname for the agent vLLM server |
| `--agent_port` | same as `--port` | Port for the agent vLLM server |
| `--max_steps` | `10` | Maximum dialogue turns per task |
| `--best_of_n` | `1` | Independent runs per task |
| `--temperature` | `1.0` | Agent sampling temperature |
| `--num_workers` | `1` | Parallel task workers |
| `--user_prompt_path` | *(built-in)* | Custom human simulator prompt file |
| `--agent_prompt_path` | *(built-in)* | Custom agent system prompt file |
| `--evaluate_only` | `False` | Skip interaction; evaluate existing output file |
| `--debug` | `False` | Fail fast on any error |

## Multi-Server Setup

For best results, use separate models for the agent and human simulator:

```bash
# Terminal 1: Human simulator (larger model)
vllm serve \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 8

# Terminal 2: Agent under test
vllm serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8001 \
    --tensor-parallel-size 1

# Terminal 3: Run benchmark
python examples/colbench_benchmark/colbench.py \
    --agent_model meta-llama/Llama-3.1-8B-Instruct \
    --env_model meta-llama/Llama-3.1-70B-Instruct \
    --hostname localhost --port 8000 \
    --agent_hostname localhost --agent_port 8001 \
    --input_path examples/colbench_benchmark/results/test.jsonl \
    --output_path examples/colbench_benchmark/results/results.jsonl
```

## Task Format

Input tasks are stored in JSONL format. Each line contains:

```json
{
  "problem_description": "Write a Python function that ...",
  "ground_truth": "def my_function(x):\n    return x * 2",
  "test_cases": {
    "test1": "my_function(1)",
    "test2": "my_function(5)"
  }
}
```

| Field | Description |
|---|---|
| `problem_description` | The task shown to the agent as the opening message |
| `ground_truth` | Reference Python code (hidden from agent, visible to simulator) |
| `test_cases` | Dict of Python expressions used to evaluate correctness |

## Output Format

Trajectories are saved in JSONL format, backward-compatible with the original sweet_rl evaluation scripts:

```json
{
  "task": {
    "problem_description": "Write a Python function ...",
    "ground_truth": "def my_function(x): ...",
    "test_cases": {"test1": "my_function(1)", ...}
  },
  "dialogue_history": [
    {"role": "user", "content": "Write a Python function ..."},
    {"role": "assistant", "content": "Can you clarify ..."},
    {"role": "user", "content": "The input is a list ..."},
    {"role": "assistant", "content": "I WANT TO ANSWER:\ndef my_function(...): ..."}
  ],
  "answer": "\ndef my_function(...): ...",
  "reward": 0.8
}
```

## Evaluation

The evaluator runs each hidden test case against both the ground-truth and agent-generated code, comparing outputs for equality:

- **`correctness`**: Fraction of test cases passed (0.0–1.0)
- **`success`**: Boolean — `True` only if all tests pass
- **`num_tests`**: Total number of test cases
- **`num_passed`**: Number of tests matching ground truth

Safety measures include blocked patterns (`import os`, `open(`, `exit(`, etc.) and 1-second execution timeouts.

## Programmatic Usage

```python
from openai import OpenAI
from maseval.benchmark.colbench import ColBenchBenchmark, OpenAIModelAdapter

client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")

def model_factory(model_id, **kwargs):
    return OpenAIModelAdapter(client, model_id=model_id)

tasks = ColBenchBenchmark.load_tasks("examples/colbench_benchmark/results/test.jsonl", num_tasks=100)

benchmark = ColBenchBenchmark(
    model_factory=model_factory,
    human_simulator_model_id="meta-llama/Llama-3.1-8B-Instruct",
    agent_model_id="meta-llama/Llama-3.1-8B-Instruct",
    max_steps=10,
)

reports = benchmark.run(
    tasks=tasks,
    agent_data={"model": "meta-llama/Llama-3.1-8B-Instruct"},
)

# Access results
for report in reports:
    eval_result = report["eval"][0]
    print(f"Task {report['task_id']}: correctness={eval_result['correctness']:.1%}")
```

## Citation

```bibtex
@article{zhou2025colbench,
  title={Sweet-RL: Training Multi-Turn LLM Agents on Collaborative Tasks with Reinforcement Learning},
  author={Zhou, Yifei and Yan, An and Jansen, Peter and Peng, Hao and Choi, Yejin},
  year={2025}
}
```