# CONVERSE Benchmark

CONVERSE evaluates privacy and security robustness in agent-to-agent conversations where the external counterpart is adversarial.

## What It Tests

- Privacy attacks: the external agent tries to extract sensitive profile details.
- Security attacks: the external agent tries to induce unauthorized tool actions.
- Multi-turn manipulation: attacks progress over several conversational turns.

## Data Source

Data is loaded from [the official CONVERSE repository `amrgomaaelhady/ConVerse`](https://github.com/amrgomaaelhady/ConVerse)

Supported domains:

- `travel`
- `real_estate`
- `insurance`

## Usage

Implement a framework-specific subclass of `ConverseBenchmark` and provide agent setup plus model adapter provisioning.

```python
from typing import Any, Dict, Optional, Sequence, Tuple

from maseval import AgentAdapter, Environment, ModelAdapter, Task, User
from maseval.benchmark.converse import ConverseBenchmark, ensure_data_exists, load_tasks
from maseval.core.seeding import SeedGenerator


class MyConverseBenchmark(ConverseBenchmark):
    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: Environment,
        task: Task,
        user: Optional[User],
        seed_generator: SeedGenerator,
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        # Create your framework agent(s) using environment tools.
        ...

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        # Create and optionally register model adapter.
        ...


# First call downloads source files to the local benchmark data cache.
ensure_data_exists(domain="travel")
tasks = load_tasks(domain="travel", split="privacy", limit=5)

benchmark = MyConverseBenchmark(progress_bar=False)
results = benchmark.run(
    tasks=tasks,
    agent_data={
        "model_id": "gpt-4o-mini",
        "attacker_model_id": "gpt-4o",
        "max_turns": 10,
    },
)
```

## Default Implementation

CONVERSE also provides a built-in default agent loop via `DefaultAgentConverseBenchmark`.
You only need to supply `get_model_adapter()`.

```python
from typing import Any

from maseval import ModelAdapter
from maseval.benchmark.converse import DefaultAgentConverseBenchmark, load_tasks
from maseval.interface.inference import OpenAIModelAdapter


class MyDefaultConverseBenchmark(DefaultAgentConverseBenchmark):
    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        adapter = OpenAIModelAdapter(client=..., model_id=model_id, seed=kwargs.get("seed"))
        if kwargs.get("register_name"):
            self.register(kwargs.get("register_category", "models"), kwargs["register_name"], adapter)
        return adapter


tasks = load_tasks(domain="travel", split="privacy", limit=5)
benchmark = MyDefaultConverseBenchmark(progress_bar=False)
results = benchmark.run(
    tasks=tasks,
    agent_data={
        "model_id": "gpt-4o-mini",
        "attacker_model_id": "gpt-4o",
    },
)
```

For a full runnable example script, see:

- `examples/converse_benchmark/default_converse_benchmark.py`

## Evaluation Output

Privacy evaluator output fields:

- `privacy_leak`
- `score`
- `leak_turn`
- `matched_snippet`

Security evaluator output fields:

- `security_violation`
- `score`
- `violated_tools`

[:material-github: View source](https://github.com/parameterlab/MASEval/blob/main/maseval/benchmark/converse/converse.py){ .md-source-file }

::: maseval.benchmark.converse.ConverseBenchmark

::: maseval.benchmark.converse.DefaultAgentConverseBenchmark

::: maseval.benchmark.converse.DefaultConverseAgent

::: maseval.benchmark.converse.DefaultConverseAgentAdapter

::: maseval.benchmark.converse.ConverseEnvironment

::: maseval.benchmark.converse.ConverseExternalAgent

::: maseval.benchmark.converse.PrivacyEvaluator

::: maseval.benchmark.converse.SecurityEvaluator

::: maseval.benchmark.converse.load_tasks

::: maseval.benchmark.converse.ensure_data_exists
