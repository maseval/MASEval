#!/usr/bin/env python3
"""
ColBench runner — drop-in replacement for the original two-step workflow.

    python -m maseval.benchmark.colbench.run \\
        --agent_model meta-llama/Llama-3.1-8B-Instruct \\
        --hostname localhost --port 8001 \\
        --env_model meta-llama/Llama-3.1-8B-Instruct \\
        --input_path examples/colbench_benchmark/results/test.jsonl \\
        --output_path examples/colbench_benchmark/results/temp_test.jsonl

    # Evaluate-only:
    python -m maseval.benchmark.colbench.run \\
        --evaluate_only \\
        --output_path examples/colbench_benchmark/results/temp_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ColBench benchmark runner (MASEval integration)",
    )

    # Task input
    p.add_argument("--input_path", type=str, required=False)
    p.add_argument("--output_path", type=str, default="colbench_results.jsonl")
    p.add_argument("--num_tasks", type=int, default=1000)

    # Models
    p.add_argument("--agent_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--env_model", type=str, default="auto")

    # Server
    p.add_argument("--hostname", type=str, default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--agent_hostname", type=str, default=None)
    p.add_argument("--agent_port", type=int, default=None)

    # Benchmark params
    p.add_argument("--task_type", type=str, default="code", choices=["code"])
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--best_of_n", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=1)

    # Prompts
    p.add_argument("--user_prompt_path", type=str, default=None)
    p.add_argument("--agent_prompt_path", type=str, default=None)

    # Modes
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument(
        "--debug", action="store_true",
        help="Fail fast on any error (sets fail_on_*_error=True)",
    )

    return p


def resolve_env_model(env_model: str, task_type: str) -> str:
    if env_model != "auto":
        return env_model
    if task_type == "html":
        return "Qwen/Qwen2-VL-72B-Instruct"
    return "meta-llama/Llama-3.1-70B-Instruct"


def load_prompt(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    with open(path) as f:
        return f.read()


def evaluate_only(output_path: str, k: int = 1) -> None:
    """Evaluate existing trajectories (replaces evaluate_code.py)."""
    from maseval.benchmark.colbench.evaluator import check_correctness

    with open(output_path) as f:
        trajectories = [json.loads(line) for line in f]

    print(f"Number of trajectories: {len(trajectories)}")

    all_correctness = []
    for traj in trajectories:
        gt = traj["task"]["ground_truth"]
        answer = traj.get("answer", "No answer")
        test_cases = traj["task"]["test_cases"]
        correctness = check_correctness(gt, answer, test_cases)
        traj["reward"] = correctness
        all_correctness.append(correctness)

    raw = np.array(all_correctness).reshape(k, -1)
    best = np.max(raw, axis=0)

    print(f"Average correctness: {np.mean(all_correctness):.4f}")
    print(f"Success rate: {np.mean(np.array(all_correctness) == 1.0):.4f}")
    print(f"Best-of-{k} average correctness: {np.mean(best):.4f}")
    print(f"Best-of-{k} success rate: {np.mean(best == 1.0):.4f}")

    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")
    print(f"Saved to {output_path}")


def run_benchmark(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from maseval.benchmark.colbench.colbench import ColBenchBenchmark
    from maseval.benchmark.colbench.openai_model_adapter import OpenAIModelAdapter

    env_model_id = resolve_env_model(args.env_model, args.task_type)
    agent_hostname = args.agent_hostname or args.hostname
    agent_port = args.agent_port or args.port

    # Create OpenAI clients
    env_client = OpenAI(
        base_url=f"http://{args.hostname}:{args.port}/v1",
        api_key="EMPTY",
    )
    if agent_hostname == args.hostname and agent_port == args.port:
        agent_client = env_client
    else:
        agent_client = OpenAI(
            base_url=f"http://{agent_hostname}:{agent_port}/v1",
            api_key="EMPTY",
        )

    # Model factory — routes model_id to the correct client
    def model_factory(model_id: str, **kwargs) -> OpenAIModelAdapter:
        if model_id == env_model_id:
            return OpenAIModelAdapter(
                env_client, model_id=model_id,
                default_temperature=0.0,
            )
        else:
            return OpenAIModelAdapter(
                agent_client, model_id=model_id,
                default_temperature=args.temperature,
                default_max_tokens=1024,
            )

    human_prompt = load_prompt(args.user_prompt_path)
    agent_prompt = load_prompt(args.agent_prompt_path)

    # Fail fast by default so errors surface immediately
    benchmark = ColBenchBenchmark(
        model_factory=model_factory,
        human_simulator_model_id=env_model_id,
        agent_model_id=args.agent_model,
        human_prompt=human_prompt,
        agent_prompt=agent_prompt,
        max_steps=args.max_steps,
        agent_temperature=args.temperature,
        n_task_repeats=args.best_of_n,
        num_workers=args.num_workers,
        fail_on_setup_error=True,
        fail_on_task_error=True,
        fail_on_evaluation_error=True,
    )

    # ── Load tasks ───────────────────────────────────────────────────────
    tasks = ColBenchBenchmark.load_tasks(args.input_path, num_tasks=args.num_tasks)

    # Also load raw task data for saving trajectories later
    # (SequentialTaskQueue is consumed by benchmark.run())
    with open(args.input_path) as f:
        raw_tasks = [json.loads(line) for i, line in enumerate(f)
                     if args.num_tasks is None or i < args.num_tasks]

    print(f"Loaded {len(raw_tasks)} tasks from {args.input_path}")

    agent_data = {
        "model": args.agent_model,
        "env_model": env_model_id,
    }

    print(f"Running ColBench benchmark:")
    print(f"  Agent model:     {args.agent_model}")
    print(f"  Simulator model: {env_model_id}")
    print(f"  Agent server:    http://{agent_hostname}:{agent_port}/v1")
    print(f"  Env server:      http://{args.hostname}:{args.port}/v1")
    print(f"  Max steps:       {args.max_steps}")
    print(f"  Best-of-n:       {args.best_of_n}")
    print(f"  Num workers:     {args.num_workers}")
    print()

    reports = benchmark.run(tasks=tasks, agent_data=agent_data)

    # ── Save results in original trajectory format ───────────────────────
    trajectories = _reports_to_trajectories(reports, raw_tasks)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")

    _print_summary(reports, args.best_of_n)
    print(f"\nTrajectories saved to {output_path}")


def _reports_to_trajectories(
    reports: List[Dict[str, Any]],
    raw_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert MASEval reports to original sweet_rl trajectory format.

    Reports come in the same order as tasks (sequential execution),
    so we zip them directly with the raw task dicts from the JSONL.
    """
    trajectories = []
    for i, report in enumerate(reports):
        # Get original task data by index
        raw_task = raw_tasks[i % len(raw_tasks)] if raw_tasks else {}

        # Extract answer from user traces
        user_traces = report.get("traces", {}).get("user", {})
        answer = "No answer"
        if isinstance(user_traces, dict):
            answer = user_traces.get("answer", "No answer") or "No answer"

        # Extract dialogue from user traces
        dialogue_history = []
        if isinstance(user_traces, dict):
            dialogue_history = user_traces.get("messages", [])

        # Build trajectory in original format
        traj: Dict[str, Any] = {
            "task": {
                "problem_description": raw_task.get("problem_description", ""),
                "ground_truth": raw_task.get("ground_truth", ""),
                "test_cases": raw_task.get("test_cases", {}),
            },
            "dialogue_history": dialogue_history,
            "answer": answer,
        }

        # Add evaluation results if available
        eval_results = report.get("eval")
        if eval_results and len(eval_results) > 0:
            traj["reward"] = eval_results[0].get("correctness", 0.0)

        trajectories.append(traj)

    return trajectories


def _print_summary(reports: List[Dict[str, Any]], k: int = 1) -> None:
    eval_results = []
    for report in reports:
        evals = report.get("eval") or [{}]
        correctness = evals[0].get("correctness", 0.0) if evals else 0.0
        eval_results.append(correctness)

    if not eval_results:
        print("No results to report.")
        return

    arr = np.array(eval_results)
    print(f"\n{'='*60}")
    print(f"ColBench Evaluation Results")
    print(f"{'='*60}")
    print(f"Number of trajectories: {len(arr)}")
    print(f"Average correctness: {np.mean(arr):.4f}")
    print(f"Success rate: {np.mean(arr == 1.0):.4f}")

    if k > 1:
        num_tasks = len(arr) // k
        raw = arr.reshape(k, num_tasks)
        best = np.max(raw, axis=0)
        print(f"Best-of-{k} average correctness: {np.mean(best):.4f}")
        print(f"Best-of-{k} success rate: {np.mean(best == 1.0):.4f}")
    print(f"{'='*60}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_only(args.output_path, k=args.best_of_n)
    else:
        if args.input_path is None:
            parser.error("--input_path is required for interaction mode")
        run_benchmark(args)


if __name__ == "__main__":
    main()