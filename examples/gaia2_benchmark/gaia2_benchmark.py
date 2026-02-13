"""Gaia2 Benchmark Example.

This example demonstrates running the Gaia2 benchmark with the default agent
using Meta's ARE (Agent Research Environments) platform.

The Gaia2 benchmark evaluates agents on dynamic, multi-step scenarios across
7 capability dimensions:
- execution: Basic task execution
- search: Information retrieval tasks
- adaptability: Adapting to changing requirements
- time: Temporal reasoning tasks
- ambiguity: Handling ambiguous instructions
- agent2agent: Multi-agent collaboration
- noise: Handling noisy inputs

Key Features:
- ARE-based simulation environment with real-time dynamics
- Tool-based time control (wait_for_notification)
- Deterministic evaluation via GraphPerEventJudge
- Multi-app environment (Calendar, Email, Messaging, etc.)

Reference:
    Paper: "GAIA-2: A Controllable Multi-Turn Conversational Benchmark for Agents"
    Data: https://huggingface.co/datasets/meta-agents-research-environments/gaia2

Usage:
    # Run on execution capability
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --capability execution --limit 5

    # Run on time capability with specific model
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --capability time --model gpt-4o --limit 5

    # Run all capabilities
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --limit 10
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from google.genai import Client as GoogleGenAIClient
from openai import OpenAI as OpenAIClient

from maseval.benchmark.gaia2 import (
    DefaultAgentGaia2Benchmark,
    compute_gaia2_metrics,
    configure_model_ids,
    load_tasks,
    VALID_CAPABILITIES,
)
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.interface.inference import OpenAIModelAdapter
from maseval.interface.inference.google_genai import GoogleGenAIModelAdapter

load_dotenv()


# =============================================================================
# Model Setup
# =============================================================================

_google_client: Optional[GoogleGenAIClient] = None


def get_google_client() -> GoogleGenAIClient:
    """Get or create the shared Google GenAI client."""
    global _google_client
    if _google_client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        _google_client = GoogleGenAIClient(api_key=api_key)
    return _google_client


_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create the shared OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAIClient(api_key=api_key)
    return _openai_client


def get_provider_from_model(model_id: str) -> Literal["openai", "google", "anthropic"]:
    """Determine the provider from a model ID."""
    model_lower = model_id.lower()

    if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "chatgpt"]):
        return "openai"
    if any(x in model_lower for x in ["gemini", "palm", "bard"]):
        return "google"
    if any(x in model_lower for x in ["claude"]):
        return "anthropic"

    return "openai"


# =============================================================================
# Benchmark Implementations
# =============================================================================


class GoogleGenAIGaia2Benchmark(DefaultAgentGaia2Benchmark):
    """Gaia2 Benchmark using Google GenAI for the default agent."""

    def __init__(self, model_id: str = "gemini-2.5-flash", **kwargs: Any):
        agent_data = kwargs.pop("agent_data", {})
        agent_data["model_id"] = model_id
        super().__init__(agent_data=agent_data, **kwargs)

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> GoogleGenAIModelAdapter:
        """Create a Google GenAI model adapter."""
        seed = kwargs.get("seed")
        adapter = GoogleGenAIModelAdapter(get_google_client(), model_id=model_id, seed=seed)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter


class OpenAIGaia2Benchmark(DefaultAgentGaia2Benchmark):
    """Gaia2 Benchmark using OpenAI for the default agent."""

    def __init__(self, model_id: str = "gpt-4o", **kwargs: Any):
        agent_data = kwargs.pop("agent_data", {})
        agent_data["model_id"] = model_id
        super().__init__(agent_data=agent_data, **kwargs)

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> OpenAIModelAdapter:
        """Create an OpenAI model adapter."""
        seed = kwargs.get("seed")
        adapter = OpenAIModelAdapter(get_openai_client(), model_id=model_id, seed=seed)
        if "register_name" in kwargs:
            self.register("models", kwargs["register_name"], adapter)
        return adapter


# =============================================================================
# Benchmark Selection
# =============================================================================


def get_benchmark_class(model_id: str = "gemini-2.5-flash") -> type:
    """Get the benchmark class for the specified model."""
    provider = get_provider_from_model(model_id)
    if provider == "google":
        return GoogleGenAIGaia2Benchmark
    elif provider == "openai":
        return OpenAIGaia2Benchmark
    else:
        raise ValueError(f"Provider '{provider}' is not yet supported. Supported: google, openai")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_benchmark(
    capability: Optional[str] = None,
    model_id: str = "gemini-2.5-flash",
    limit: Optional[int] = None,
    n_task_repeats: int = 1,
    output_dir: Optional[Path] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run the Gaia2 benchmark.

    Args:
        capability: Filter by capability type (execution, search, etc.). None for all.
        model_id: Model ID to use for the agent.
        limit: Maximum number of tasks to run.
        n_task_repeats: Number of times to repeat each task.
        output_dir: Output directory for results.
        temperature: LLM temperature.

    Returns:
        Summary metrics dict.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap_str = capability or "all"
    print(f"Loading Gaia2 tasks (capability={cap_str})...")

    try:
        tasks = load_tasks(capability=capability, limit=limit)
        print(f"Loaded {len(tasks)} tasks")
    except ImportError as e:
        print(f"Error: {e}")
        print("\nTo run this example, install the required dependencies:")
        print("  pip install datasets meta-agents-research-environments")
        return {}

    # Configure model IDs (optional for LLM-based judge)
    configure_model_ids(tasks, evaluator_model_id=model_id)

    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"gaia2_{cap_str}_{{timestamp}}.jsonl",
    )

    BenchmarkClass = get_benchmark_class(model_id)

    benchmark = BenchmarkClass(
        model_id=model_id,
        callbacks=[logger],
        n_task_repeats=n_task_repeats,
        fail_on_setup_error=True,
        fail_on_task_error=True,
        fail_on_evaluation_error=True,
        agent_data={
            "verbose": 1,
            "llm_args": {"temperature": temperature},
        },
    )

    print("\nRunning Gaia2 benchmark...")
    print(f"Capability: {cap_str}")
    print(f"Model: {model_id}")

    results = benchmark.run(tasks=tasks)

    # Compute and print summary
    summary = compute_gaia2_metrics(results)

    print("\n" + "=" * 60)
    print("GAIA2 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Capability: {cap_str}")
    print(f"Model: {model_id}")
    print(f"Total Tasks: {summary.get('total_tasks', len(results))}")
    print(f"Scored Tasks: {summary.get('scored_tasks', 0)}")
    print(f"GSR: {summary.get('gsr', 0):.2%}")
    print(f"Partial GSR: {summary.get('partial_gsr', 0):.2%}")

    if summary.get("by_capability"):
        print("\nBy Capability:")
        for cap, data in summary["by_capability"].items():
            print(f"  {cap}: GSR={data.get('gsr', 0):.2%}, Count={data.get('count', 0)}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)

    return summary


def main():
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the Gaia2 benchmark with ARE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on execution capability
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --capability execution --limit 5

    # Run on time capability with specific model
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --capability time --model gpt-4o --limit 5

    # Run all capabilities
    uv run python examples/gaia2_benchmark/gaia2_benchmark.py --limit 10
        """,
    )

    parser.add_argument(
        "--capability",
        type=str,
        default=None,
        choices=list(VALID_CAPABILITIES),
        help="Capability to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model ID to use (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each task (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_benchmark(
        capability=args.capability,
        model_id=args.model,
        limit=args.limit,
        n_task_repeats=args.repeats,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
