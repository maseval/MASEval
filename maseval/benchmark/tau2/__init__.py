"""Tau 2 Benchmark - Tool-Agent-User Interaction in Real-World Domains.

Framework-agnostic implementation of the tau2-bench benchmark for evaluating
LLM-based agents on customer service tasks across multiple domains.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Reference Paper: "Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"
https://arxiv.org/abs/2406.12045

Domains:
    - airline: Flight reservation management (50 base tasks)
    - retail: E-commerce order management (114 base tasks)
    - telecom: Telecom customer service (114 base tasks)

Usage:
    from maseval.benchmark.tau2 import (
        Tau2Benchmark, Tau2Environment, Tau2Evaluator, Tau2User,
        load_tasks, configure_model_ids, ensure_data_exists,
        compute_benchmark_metrics, compute_pass_at_k,
    )

    # Ensure domain data is downloaded
    ensure_data_exists(domain="retail")

    # Load data and configure model IDs
    tasks = load_tasks("retail", split="base", limit=5)
    configure_model_ids(
        tasks,
        user_model_id="gpt-4o",
        evaluator_model_id="gpt-4o",  # Optional - only for NL assertions
    )

    # Create your framework-specific benchmark subclass
    class MyTau2Benchmark(Tau2Benchmark):
        def setup_agents(self, agent_data, environment, task, user, seed_generator):
            # Get real tools from environment
            tools = environment.tools
            # Create your agent with these tools
            ...

        def get_model_adapter(self, model_id, **kwargs):
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    # Run with multiple repeats for Pass@k / Pass^k
    benchmark = MyTau2Benchmark(agent_data={}, n_task_repeats=4)
    results = benchmark.run(tasks)

    # Compute metrics
    metrics = compute_benchmark_metrics(results)
    pass_k = compute_pass_at_k(results, k_values=[1, 2, 3, 4])  # At least 1 success in k
    pass_hat_k = compute_pass_hat_k(results, k_values=[1, 2, 3, 4])  # Combinatorial (paper metric)
"""

# Main benchmark components
from maseval.benchmark.tau2.tau2 import (
    Tau2Benchmark,
    Tau2User,
    DefaultTau2Agent,
    DefaultTau2AgentAdapter,
    DefaultAgentTau2Benchmark,
)

# Environment
from maseval.benchmark.tau2.environment import Tau2Environment, get_environment_constructor

# Evaluator
from maseval.benchmark.tau2.evaluator import (
    Tau2Evaluator,
    RewardType,
    TerminationReason,
    compute_benchmark_metrics,
    compute_pass_at_k,
    compute_pass_hat_k,
    pass_hat_k,
)

# Data loading
from maseval.benchmark.tau2.data_loader import (
    load_tasks,
    load_domain_config,
    configure_model_ids,
    download_domain_data,
    ensure_data_exists,
    VALID_DOMAINS,
    TASK_SPLITS,
)

# Domain base classes
from maseval.benchmark.tau2.domains import DB, ToolKitBase, ToolType, is_tool

# Retail domain
from maseval.benchmark.tau2.domains.retail import RetailDB, RetailTools

# Airline domain
from maseval.benchmark.tau2.domains.airline import AirlineDB, AirlineTools

# Telecom domain
from maseval.benchmark.tau2.domains.telecom import TelecomDB, TelecomTools


__all__ = [
    # Benchmark
    "Tau2Benchmark",
    "Tau2User",
    # Default agent implementation
    "DefaultTau2Agent",
    "DefaultTau2AgentAdapter",
    "DefaultAgentTau2Benchmark",
    # Environment
    "Tau2Environment",
    "get_environment_constructor",
    # Evaluator
    "Tau2Evaluator",
    "RewardType",
    "TerminationReason",
    "compute_benchmark_metrics",
    "compute_pass_at_k",
    "compute_pass_hat_k",
    "pass_hat_k",
    # Data loading
    "load_tasks",
    "load_domain_config",
    "configure_model_ids",
    "download_domain_data",
    "ensure_data_exists",
    "VALID_DOMAINS",
    "TASK_SPLITS",
    # Domain base classes
    "DB",
    "ToolKitBase",
    "ToolType",
    "is_tool",
    # Retail domain
    "RetailDB",
    "RetailTools",
    # Airline domain
    "AirlineDB",
    "AirlineTools",
    # Telecom domain
    "TelecomDB",
    "TelecomTools",
]
