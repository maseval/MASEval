"""MMLU Benchmark for MASEval.

Implements MMLU evaluation with anchor point-based task selection for DISCO prediction.

Usage:
    from maseval.benchmark.mmlu import (
        DefaultMMLUBenchmark,
        load_tasks,
    )
    from maseval.core.task import DISCOQueue, InformativeSubsetQueue

    # Load tasks and anchor points
    tasks = load_tasks(
        data_path="path/to/mmlu_prompts_examples.json",
        anchor_points_path="path/to/anchor_points.pkl",  # Optional
    )

    # Run benchmark
    benchmark = DefaultMMLUBenchmark(model_id="meta-llama/Llama-2-7b-hf")
    results = benchmark.run(tasks=tasks, agent_data={"model_id": "meta-llama/Llama-2-7b-hf"})
"""

from maseval.core.task import DISCOQueue, InformativeSubsetQueue

from .mmlu import (
    DEFAULT_AGENT_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHOICES,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_REGISTER_NAME,
    MMLU_TASK_NAME,
    STATUS_SUCCESS,
    TARGET_DELIMITER,
    TASK_TYPE_MMLU,
    MMLUBenchmark,
    DefaultMMLUBenchmark,
    MMLUEnvironment,
    MMLUEvaluator,
    load_tasks,
    compute_benchmark_metrics,
)

__all__ = [
    "DEFAULT_AGENT_NAME",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHOICES",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_REGISTER_NAME",
    "MMLU_TASK_NAME",
    "STATUS_SUCCESS",
    "TARGET_DELIMITER",
    "TASK_TYPE_MMLU",
    "MMLUBenchmark",
    "DefaultMMLUBenchmark",
    "MMLUEnvironment",
    "MMLUEvaluator",
    "InformativeSubsetQueue",
    "DISCOQueue",
    "load_tasks",
    "compute_benchmark_metrics",
]
