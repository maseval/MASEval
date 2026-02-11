"""Run CONVERSE using the built-in default agent implementation.

Usage:
    uv run python examples/converse_benchmark/default_converse_benchmark.py \
        --domain travel \
        --split privacy \
        --model gpt-4o-mini \
        --attacker-model gpt-4o \
        --limit 5
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from maseval import ModelAdapter
from maseval.benchmark.converse import DefaultAgentConverseBenchmark, load_tasks
from maseval.core.callbacks.result_logger import FileResultLogger
from maseval.interface.inference import OpenAIModelAdapter


def create_openai_adapter(model_id: str, seed: Optional[int] = None) -> ModelAdapter:
    """Create an OpenAI model adapter for CONVERSE components."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    return OpenAIModelAdapter(client=client, model_id=model_id, seed=seed)


class OpenAIDefaultConverseBenchmark(DefaultAgentConverseBenchmark):
    """Default CONVERSE benchmark backed by OpenAI-compatible inference."""

    def get_model_adapter(self, model_id: str, **kwargs: Any) -> ModelAdapter:
        adapter = create_openai_adapter(model_id=model_id, seed=kwargs.get("seed"))

        register_name = kwargs.get("register_name")
        if register_name:
            self.register(kwargs.get("register_category", "models"), register_name, adapter)

        return adapter


def summarize_results(results: List[Dict[str, Any]]) -> None:
    """Print a compact summary from benchmark reports."""
    privacy_total = 0
    privacy_passed = 0
    security_total = 0
    security_passed = 0

    for report in results:
        for eval_result in report.get("eval") or []:
            if "privacy_leak" in eval_result:
                privacy_total += 1
                if not eval_result["privacy_leak"]:
                    privacy_passed += 1
            if "security_violation" in eval_result:
                security_total += 1
                if not eval_result["security_violation"]:
                    security_passed += 1

    print(f"Completed {len(results)} run(s).")
    if privacy_total > 0:
        print(f"Privacy robustness: {privacy_passed}/{privacy_total} ({(privacy_passed / privacy_total) * 100:.1f}%)")
    if security_total > 0:
        print(f"Security robustness: {security_passed}/{security_total} ({(security_passed / security_total) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CONVERSE benchmark with built-in default agent")
    parser.add_argument("--domain", choices=["travel", "real_estate", "insurance"], required=True)
    parser.add_argument("--split", choices=["privacy", "security", "all"], default="privacy")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model ID for the default assistant agent")
    parser.add_argument("--attacker-model", default="gpt-4o", help="Model ID for the adversarial external agent")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to run")
    parser.add_argument("--output-dir", default="results", help="Directory for JSONL benchmark logs")
    args = parser.parse_args()

    tasks = load_tasks(domain=args.domain, split=args.split, limit=args.limit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = FileResultLogger(
        output_dir=output_dir,
        filename_pattern=f"converse_default_{args.domain}_{{timestamp}}.jsonl",
    )

    benchmark = OpenAIDefaultConverseBenchmark(callbacks=[logger], fail_on_setup_error=True)
    results = benchmark.run(
        tasks=tasks,
        agent_data={
            "model_id": args.model,
            "attacker_model_id": args.attacker_model,
            "max_turns": 10,
        },
    )

    summarize_results(results)
    print(f"Detailed logs: {output_dir}")


if __name__ == "__main__":
    main()
