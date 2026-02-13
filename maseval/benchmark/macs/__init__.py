"""MACS Benchmark - Multi-Agent Collaboration Scenarios.

This module provides the AWS MACS benchmark for evaluating multi-agent
collaboration in enterprise applications.

Reference:
    Paper: https://arxiv.org/abs/2412.05449
    Data: https://github.com/aws-samples/multiagent-collab-scenario-benchmark
"""

from .macs import (
    MACSBenchmark,
    MACSEnvironment,
    MACSEvaluator,
    MACSGenericTool,
    MACSUser,
    compute_benchmark_metrics,
)
from .data_loader import (
    load_tasks,
    load_agent_config,
    configure_model_ids,
    ensure_data_exists,
    process_data,
    download_original_data,
    download_prompt_templates,
    restructure_data,
    VALID_DOMAINS,
    DEFAULT_DATA_DIR,
)

__all__ = [
    # Core classes
    "MACSBenchmark",
    "MACSEnvironment",
    "MACSEvaluator",
    "MACSGenericTool",
    "MACSUser",
    # Data loading
    "load_tasks",
    "load_agent_config",
    "configure_model_ids",
    "ensure_data_exists",
    "process_data",
    "download_original_data",
    "download_prompt_templates",
    "restructure_data",
    "VALID_DOMAINS",
    "DEFAULT_DATA_DIR",
    # Utilities
    "compute_benchmark_metrics",
]
