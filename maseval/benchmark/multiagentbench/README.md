# MultiAgentBench Integration

Framework-agnostic implementation of the MultiAgentBench benchmark suite from MARBLE
(Multi-Agent Coordination Backbone with LLM Engine) for evaluating multi-agent collaboration.

**Original Paper**: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents"
(arXiv:2503.01935)

**Original Repository**: https://github.com/ulab-uiuc/MARBLE

## Setup

This benchmark requires the MARBLE source code. You can set it up automatically or manually.

### Option 1: Automatic Setup (Recommended)

MARBLE will be automatically downloaded when you first use it:

```python
from maseval.benchmark.multiagentbench import ensure_marble_exists, load_tasks

# This downloads MARBLE if not present (about 50MB)
ensure_marble_exists()

# Now load tasks
tasks = load_tasks("research", limit=1)
print(f"Loaded {len(tasks)} task(s)")
```

### Option 2: Manual Clone

If you prefer to clone manually:

```bash
cd maseval/benchmark/multiagentbench
git clone https://github.com/ulab-uiuc/MARBLE.git marble
cd marble
# Pin to tested version (recommended)
git checkout <pinned-commit-hash>
```

### Install MARBLE Dependencies

MARBLE requires additional dependencies. Add them to your environment:

```bash
# If using uv (recommended)
uv add litellm ruamel.yaml

# Or with pip
pip install litellm ruamel.yaml
```

### Verify Setup

```python
from maseval.benchmark.multiagentbench import load_tasks

# Should load without error
tasks = load_tasks("research", limit=1)
print(f"Loaded {len(tasks)} task(s)")
```

## Usage

### Basic Usage (Abstract Base)

The abstract `MultiAgentBenchBenchmark` provides task loading, environment setup,
and evaluation infrastructure. You implement `setup_agents()` with your framework:

```python
from maseval.benchmark.multiagentbench import (
    MultiAgentBenchBenchmark,
    MultiAgentBenchEnvironment,
    load_tasks,
    configure_model_ids,
)

class MyMultiAgentBenchmark(MultiAgentBenchBenchmark):
    def setup_agents(self, agent_data, environment, task, user, seed_generator):
        # Your framework-specific agent creation
        ...

    def get_model_adapter(self, model_id, **kwargs):
        adapter = MyModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

# Load and configure tasks
tasks = load_tasks("research", limit=5)
configure_model_ids(tasks, agent_model_id="gpt-4o")

# Run
benchmark = MyMultiAgentBenchmark()
results = benchmark.run(tasks)
```

### MARBLE Reproduction

Use `MarbleMultiAgentBenchBenchmark` for exact reproduction of MARBLE's published results:

```python
from maseval.benchmark.multiagentbench import (
    MarbleMultiAgentBenchBenchmark,
    load_tasks,
    configure_model_ids,
)

# Load tasks from a simple domain (no Docker required)
tasks = load_tasks("research", limit=5)
configure_model_ids(tasks, agent_model_id="gpt-4o")

# Create benchmark with model adapter implementation
class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
    def get_model_adapter(self, model_id, **kwargs):
        from maseval.interface.openai import OpenAIModelAdapter
        adapter = OpenAIModelAdapter(model_id)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter

benchmark = MyMarbleBenchmark()
results = benchmark.run(tasks)

# Print results
for result in results:
    print(f"Task: {result['task_id']}")
    print(f"Status: {result['status']}")
    if result['eval']:
        print(f"Passed: {result['eval'][0]['passed']}")
```

## Domains

MultiAgentBench includes 7 domains with different requirements:

| Domain          | External Dependencies | Initial Support |
| --------------- | --------------------- | --------------- |
| Research        | None                  | Yes             |
| Bargaining      | None                  | Yes             |
| Coding          | Filesystem access     | Yes             |
| Web             | Network access        | Yes             |
| WorldSimulation | None                  | Yes             |
| Database        | Docker + PostgreSQL   | Optional        |
| Minecraft       | External game server  | Deferred        |

### Domain-Specific Notes

- **Research/Bargaining**: Recommended for initial testing - no infrastructure required
- **Coding**: Creates files in a workspace directory
- **Database**: Requires Docker with PostgreSQL image
- **Minecraft**: Not currently supported (requires external game server)

## Known Limitations

1. **Chain coordination mode bug**: MARBLE's `engine.py` references `get_agent_profiles_linked()`
   which doesn't exist in `AgentGraph`. Tasks using chain coordination may fail.

2. **SharedMemory is per-agent**: Despite the name, each MARBLE agent creates its own
   `SharedMemory` instance. Use `msg_box` for inter-agent communication.

3. **Requires manual MARBLE clone**: MARBLE must be cloned manually into the `marble/`
   subdirectory (gitignored by default).

## File Structure

```
multiagentbench/
├── __init__.py              # Public API exports
├── README.md                # This file
├── PROVENANCE.md            # MARBLE version and license info
├── .gitignore               # Ignores marble/ directory
├── multiagentbench.py       # Benchmark classes
├── environment.py           # MultiAgentBenchEnvironment
├── data_loader.py           # Task loading utilities
├── adapters/
│   ├── __init__.py
│   └── marble_adapter.py    # MarbleAgentAdapter
└── marble/                  # ← Vendored MARBLE (gitignored)
    └── ...
```
