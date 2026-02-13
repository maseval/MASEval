"""MultiAgentBench - Multi-Agent Coordination Benchmark from MARBLE.

Framework-agnostic implementation of the MultiAgentBench benchmark suite for
evaluating multi-agent collaboration and competition in LLM-based systems.

Original Repository: https://github.com/ulab-uiuc/MARBLE
Paper: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents"
       (arXiv:2503.01935)

Domains:
    - research: Research idea generation and collaboration
    - bargaining: Negotiation and bargaining scenarios
    - coding: Software development collaboration
    - database: Database manipulation and querying (requires Docker)
    - minecraft: Collaborative building (requires external server)
    - web: Web-based task completion
    - worldsimulation: World simulation and interaction

Setup:
    This benchmark requires MARBLE source code. It will be automatically
    downloaded when you first use `load_tasks()` or you can set it up manually:

    ```python
    # Option 1: Automatic download (recommended)
    from maseval.benchmark.multiagentbench import ensure_marble_exists
    ensure_marble_exists()  # Downloads MARBLE if not present

    # Option 2: Manual clone
    # cd maseval/benchmark/multiagentbench
    # git clone https://github.com/ulab-uiuc/MARBLE.git marble
    ```

    See README.md in this directory for detailed setup instructions.

Usage:
    from maseval.benchmark.multiagentbench import (
        MultiAgentBenchBenchmark,
        MarbleMultiAgentBenchBenchmark,
        MultiAgentBenchEnvironment,
        MultiAgentBenchEvaluator,
        MarbleAgentAdapter,
        load_tasks,
        configure_model_ids,
        ensure_marble_exists,
        get_domain_info,
        VALID_DOMAINS,
    )

    # Ensure MARBLE is installed (auto-downloads if needed)
    ensure_marble_exists()

    # Load and configure tasks
    tasks = load_tasks("research", limit=5)
    configure_model_ids(tasks, agent_model_id="gpt-4o")

    # Create your framework-specific benchmark subclass
    class MyMultiAgentBenchmark(MultiAgentBenchBenchmark):
        def setup_agents(self, agent_data, environment, task, user):
            # Create your agents
            ...

        def get_model_adapter(self, model_id, **kwargs):
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run benchmark
    benchmark = MyMultiAgentBenchmark()
    results = benchmark.run(tasks, agent_data={})

MARBLE Reproduction Mode:
    For exact reproduction of MARBLE's published results, use
    MarbleMultiAgentBenchBenchmark which wraps MARBLE's native agents:

    ```python
    class MyMarbleBenchmark(MarbleMultiAgentBenchBenchmark):
        def get_model_adapter(self, model_id, **kwargs):
            from maseval.interface.openai import OpenAIModelAdapter
            adapter = OpenAIModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    benchmark = MyMarbleBenchmark()
    results = benchmark.run(tasks, agent_data={})
    ```
"""

# Core benchmark classes
from maseval.benchmark.multiagentbench.multiagentbench import (
    MultiAgentBenchBenchmark,
    MarbleMultiAgentBenchBenchmark,
)

# Environment
from maseval.benchmark.multiagentbench.environment import (
    MultiAgentBenchEnvironment,
    INFRASTRUCTURE_DOMAINS,
)

# Evaluator
from maseval.benchmark.multiagentbench.evaluator import (
    MultiAgentBenchEvaluator,
    MultiAgentBenchMetrics,
)

# Agent adapters
from maseval.benchmark.multiagentbench.adapters import (
    MarbleAgentAdapter,
)
from maseval.benchmark.multiagentbench.adapters.marble_adapter import (
    create_marble_agents,
)

# Data loading and setup
from maseval.benchmark.multiagentbench.data_loader import (
    load_tasks,
    configure_model_ids,
    get_domain_info,
    ensure_marble_exists,
    download_marble,
    VALID_DOMAINS,
    INFRASTRUCTURE_DOMAINS as INFRASTRUCTURE_REQUIRED_DOMAINS,
)


__all__ = [
    # Core benchmark classes
    "MultiAgentBenchBenchmark",
    "MarbleMultiAgentBenchBenchmark",
    # Environment
    "MultiAgentBenchEnvironment",
    "INFRASTRUCTURE_DOMAINS",
    # Evaluator
    "MultiAgentBenchEvaluator",
    "MultiAgentBenchMetrics",
    # Agent adapters
    "MarbleAgentAdapter",
    "create_marble_agents",
    # Data loading and setup
    "load_tasks",
    "configure_model_ids",
    "get_domain_info",
    "ensure_marble_exists",
    "download_marble",
    "VALID_DOMAINS",
    "INFRASTRUCTURE_REQUIRED_DOMAINS",
]
