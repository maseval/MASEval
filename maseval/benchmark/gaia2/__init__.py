"""Gaia2/ARE Benchmark for MASEval.

Framework-agnostic implementation of the Gaia2 benchmark for evaluating
LLM-based agents on dynamic, multi-step scenarios using Meta's ARE
(Agent Research Environments) platform.

Reference Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

Capabilities:
    - execution: Basic task execution
    - search: Information retrieval tasks
    - adaptability: Adapting to changing requirements
    - time: Temporal reasoning tasks
    - ambiguity: Handling ambiguous instructions
    - agent2agent: Multi-agent collaboration
    - noise: Handling noisy inputs

Usage:
    from maseval.benchmark.gaia2 import (
        Gaia2Benchmark, Gaia2Environment, Gaia2Evaluator,
        DefaultAgentGaia2Benchmark,
        load_tasks, configure_model_ids,
    )

    # Load data
    tasks = load_tasks(capability="execution", limit=5)

    # Create your framework-specific benchmark subclass
    class MyGaia2Benchmark(Gaia2Benchmark):
        def setup_agents(self, agent_data, environment, task, user, seed_generator=None):
            tools = environment.create_tools()
            # Create your agent with these tools
            ...

        def get_model_adapter(self, model_id, **kwargs):
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run
    benchmark = MyGaia2Benchmark()
    results = benchmark.run(tasks)

    # Compute metrics
    from maseval.benchmark.gaia2 import compute_gaia2_metrics
    metrics = compute_gaia2_metrics(results)
    print(f"GSR: {metrics['gsr']:.2%}")
"""

# Main benchmark components
from maseval.benchmark.gaia2.gaia2 import (
    Gaia2Benchmark,
    DefaultGaia2Agent,
    DefaultGaia2AgentAdapter,
    DefaultAgentGaia2Benchmark,
)

# Environment
from maseval.benchmark.gaia2.environment import Gaia2Environment

# Evaluator
from maseval.benchmark.gaia2.evaluator import (
    Gaia2Evaluator,
    compute_gaia2_metrics,
)

# Tool wrapper
from maseval.benchmark.gaia2.tool_wrapper import (
    AREToolWrapper,
    wrap_are_tools,
)

# Data loading
from maseval.benchmark.gaia2.data_loader import (
    load_tasks,
    configure_model_ids,
    VALID_CAPABILITIES,
    VALID_SPLITS,
    HF_DATASET_ID,
)


__all__ = [
    # Benchmark
    "Gaia2Benchmark",
    # Default agent implementation
    "DefaultGaia2Agent",
    "DefaultGaia2AgentAdapter",
    "DefaultAgentGaia2Benchmark",
    # Environment
    "Gaia2Environment",
    # Evaluator
    "Gaia2Evaluator",
    "compute_gaia2_metrics",
    # Tool wrapper
    "AREToolWrapper",
    "wrap_are_tools",
    # Data loading
    "load_tasks",
    "configure_model_ids",
    "VALID_CAPABILITIES",
    "VALID_SPLITS",
    "HF_DATASET_ID",
]
