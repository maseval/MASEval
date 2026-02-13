"""Tau 2 Benchmark - Evaluator.

Evaluator for tau2 benchmark tasks using multiple evaluation strategies:
- Environment assertions (database state checks) - DETERMINISTIC
- Action assertions (correct tool usage) - DETERMINISTIC
- Communication assertions (appropriate responses) - DETERMINISTIC
- NL assertions (natural language goal satisfaction) - LLM-based

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from:
- src/tau2/evaluator/evaluator.py
- src/tau2/evaluator/evaluator_env.py
- src/tau2/evaluator/evaluator_action.py
- src/tau2/evaluator/evaluator_communicate.py

Uses deterministic database state comparison for reproducible evaluation,
with optional LLM-based natural language assertion checking.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from maseval import Evaluator, Task, TaskExecutionStatus

from maseval.benchmark.tau2.environment import Tau2Environment, get_environment_constructor
from maseval.benchmark.tau2.utils import compare_tool_calls


# Statuses where agent is accountable (included in scoring)
# Note: task_timeout is included - timeouts count as failures in tau2
SCOREABLE_STATUSES = frozenset(
    {
        TaskExecutionStatus.SUCCESS.value,
        TaskExecutionStatus.AGENT_ERROR.value,
        TaskExecutionStatus.TASK_TIMEOUT.value,
    }
)


class RewardType(str, Enum):
    """Types of rewards that can be computed.

    Adapted from: tau2-bench src/tau2/data_model/tasks.py:RewardType
    """

    DB = "DB"  # Database state match
    ENV_ASSERTION = "ENV_ASSERTION"  # Environment assertions
    NL_ASSERTION = "NL_ASSERTION"  # Natural language assertions
    ACTION = "ACTION"  # Action verification
    COMMUNICATE = "COMMUNICATE"  # Communication verification


class TerminationReason(str, Enum):
    """Reasons for simulation termination.

    Adapted from: tau2-bench src/tau2/data_model/simulation.py
    """

    AGENT_STOP = "agent_stop"  # Agent signaled completion
    USER_STOP = "user_stop"  # User signaled satisfaction
    MAX_STEPS = "max_steps"  # Hit maximum interaction limit
    TOO_MANY_ERRORS = "too_many_errors"  # Too many tool errors


class Tau2Evaluator(Evaluator):
    """Evaluator for tau2 benchmark tasks.

    Combines multiple evaluation strategies:
    - Environment assertions (database state checks)
    - Action assertions (correct tool usage)
    - Communication assertions (appropriate responses)

    Uses DETERMINISTIC evaluation based on actual database state comparison.

    Adapted from: tau2-bench src/tau2/evaluator/
    """

    def __init__(
        self,
        task: Task,
        environment: Tau2Environment,
    ):
        """Initialize the evaluator.

        Args:
            task: Task being evaluated
            environment: Tau2Environment instance
        """
        self.task = task
        self.environment = environment

        # Extract evaluation criteria from task
        eval_data = task.evaluation_data
        self.actions = eval_data.get("actions")
        self.env_assertions = eval_data.get("env_assertions")
        self.communicate_info = eval_data.get("communicate_info")
        self.nl_assertions = eval_data.get("nl_assertions")

        # Parse reward basis
        reward_basis = eval_data.get("reward_basis", ["DB", "COMMUNICATE"])
        self.reward_basis = [RewardType(r) for r in reward_basis]

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for evaluation.

        Args:
            traces: Full execution traces

        Returns:
            Filtered traces with:
                - messages: Agent messages
                - tool_calls: List of tool invocations
                - environment: Environment traces
                - termination_reason: Why execution ended
        """
        # Extract messages from agents
        messages = []
        agents_trace = traces.get("agents", {})
        for agent_name, agent_data in agents_trace.items():
            agent_messages = agent_data.get("messages", [])
            messages.extend(agent_messages)

        # Extract tool calls from tools traces
        tool_calls = []
        tools_trace = traces.get("tools", {})
        for tool_name, tool_data in tools_trace.items():
            invocations = tool_data.get("invocations", [])
            for inv in invocations:
                tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": inv.get("inputs", {}),
                        "result": inv.get("outputs"),
                        "status": inv.get("status"),
                    }
                )

        # Get environment traces
        env_trace = traces.get("environment", {})

        return {
            "messages": messages,
            "tool_calls": tool_calls,
            "environment": env_trace,
            "termination_reason": traces.get("termination_reason"),
        }

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate task completion.

        Args:
            traces: Filtered execution traces
            final_answer: Final answer from agent

        Returns:
            Dict with:
                - reward: Float [0.0, 1.0] - overall success metric
                - reward_breakdown: Dict with per-evaluator scores
                - passed: Boolean - did task pass all criteria?
                - env_check: Environment evaluation results
                - action_check: Action evaluation results
                - communicate_check: Communication evaluation results
        """
        # Check for premature termination
        termination_reason = traces.get("termination_reason")
        if termination_reason in {TerminationReason.TOO_MANY_ERRORS.value, TerminationReason.MAX_STEPS.value}:
            return {
                "reward": 0.0,
                "passed": False,
                "reward_breakdown": {},
                "note": f"Simulation terminated prematurely: {termination_reason}",
            }

        # Run evaluations
        env_result = self._evaluate_environment(traces)
        action_result = self._evaluate_actions(traces)
        communicate_result = self._evaluate_communication(traces)

        # Combine rewards based on reward_basis
        reward = 1.0
        reward_breakdown: Dict[str, float] = {}

        env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
        action_bases = {RewardType.ACTION}
        comm_bases = {RewardType.COMMUNICATE}

        if set(self.reward_basis) & env_bases:
            reward_breakdown.update(env_result.get("breakdown", {}))
            reward *= env_result.get("reward", 1.0)

        if set(self.reward_basis) & action_bases:
            reward_breakdown.update(action_result.get("breakdown", {}))
            reward *= action_result.get("reward", 1.0)

        if set(self.reward_basis) & comm_bases:
            reward_breakdown.update(communicate_result.get("breakdown", {}))
            reward *= communicate_result.get("reward", 1.0)

        passed = reward == 1.0

        return {
            "reward": reward,
            "passed": passed,
            "reward_breakdown": reward_breakdown,
            "env_check": env_result,
            "action_check": action_result,
            "communicate_check": communicate_result,
        }

    def _evaluate_environment(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate environment state assertions.

        Compares predicted database state against expected state
        after replaying golden actions.

        Adapted from: tau2-bench src/tau2/evaluator/evaluator_env.py

        Args:
            traces: Filtered execution traces

        Returns:
            Dict with db_match, db_reward, env_assertions results
        """
        if self.actions is None and self.env_assertions is None:
            return {"reward": 1.0, "note": "No environment criteria"}

        # Get current (predicted) database state hash
        env_trace = traces.get("environment", {})
        predicted_db_hash = env_trace.get("final_db_hash") or self.environment.get_db_hash()

        # Create gold environment and replay expected actions
        gold_env_constructor = get_environment_constructor(self.task.environment_data)
        gold_env = gold_env_constructor()

        # Apply initial state if present
        initial_state = self.task.environment_data.get("initial_state")
        if initial_state:
            init_data = initial_state.get("initialization_data")
            if init_data and init_data.get("agent_data"):
                gold_env.toolkit.update_db(init_data["agent_data"])

        # Replay expected actions on gold environment
        golden_actions = self.actions or []
        action_errors = []
        for action in golden_actions:
            try:
                requestor = action.get("requestor", "assistant")
                if requestor == "assistant":
                    gold_env.make_tool_call(
                        tool_name=action["name"],
                        **action.get("arguments", {}),
                    )
            except Exception as e:
                action_errors.append({"action": action, "error": str(e)})

        # Compare database states
        gold_db_hash = gold_env.get_db_hash()
        db_match = predicted_db_hash == gold_db_hash
        db_reward = 1.0 if db_match else 0.0

        # Run environment assertions (if any)
        env_assertion_checks = []
        env_assertion_reward = 1.0

        if self.env_assertions:
            for assertion in self.env_assertions:
                # Run assertion on predicted environment
                success = self._run_env_assertion(self.environment, assertion)
                env_assertion_checks.append(
                    {
                        "assertion": assertion,
                        "passed": success,
                        "reward": 1.0 if success else 0.0,
                    }
                )
                if not success:
                    env_assertion_reward = 0.0

        # Combine rewards based on reward_basis
        reward = 1.0
        breakdown = {}

        if RewardType.DB in self.reward_basis:
            breakdown["db"] = db_reward
            reward *= db_reward

        if RewardType.ENV_ASSERTION in self.reward_basis:
            breakdown["env_assertion"] = env_assertion_reward
            reward *= env_assertion_reward

        return {
            "reward": reward,
            "breakdown": breakdown,
            "db_match": db_match,
            "db_reward": db_reward,
            "predicted_hash": predicted_db_hash,
            "expected_hash": gold_db_hash,
            "env_assertions": env_assertion_checks,
            "action_errors": action_errors,
        }

    def _run_env_assertion(self, env: Tau2Environment, assertion: Dict[str, Any]) -> bool:
        """Run an environment assertion.

        Routes assertion to the appropriate toolkit based on env_type:
        - "user": Routes to user_toolkit (for user device assertions)
        - "assistant" or default: Routes to agent toolkit

        Args:
            env: Environment to check
            assertion: Assertion spec with func_name, arguments, assert_value, env_type

        Returns:
            True if assertion passes, False otherwise
        """
        try:
            func_name = assertion.get("func_name")
            if func_name is None:
                return False

            arguments = assertion.get("arguments", {})
            expected_value = assertion.get("assert_value", True)

            # Determine which toolkit to use based on env_type
            env_type = assertion.get("env_type", "assistant")
            if env_type == "user":
                toolkit = env.user_toolkit
            else:
                toolkit = env.toolkit

            if toolkit is None:
                return False

            # Call the assertion function on the appropriate toolkit
            if toolkit.has_tool(func_name):
                result = toolkit.use_tool(func_name, **arguments)
                return bool(result) == expected_value
            else:
                return False
        except Exception:
            return False

    def _evaluate_actions(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate tool usage patterns.

        Checks if expected actions were called with correct arguments.

        Adapted from: tau2-bench src/tau2/evaluator/evaluator_action.py

        Args:
            traces: Filtered execution traces

        Returns:
            Dict with action verification results
        """
        if self.actions is None or RewardType.ACTION not in self.reward_basis:
            return {"reward": 1.0, "note": "No action criteria"}

        actual_tool_calls = traces.get("tool_calls", [])

        # Check each expected action
        action_checks = []
        all_matched = True

        for action in self.actions:
            expected_name = action.get("name")
            expected_args = action.get("arguments", {})
            compare_args = action.get("compare_args")

            # Find matching tool call
            matched = False
            for call in actual_tool_calls:
                if compare_tool_calls(
                    expected_name,
                    expected_args,
                    call.get("name", ""),
                    call.get("arguments", {}),
                    compare_args,
                ):
                    matched = True
                    break

            action_checks.append(
                {
                    "action_id": action.get("action_id"),
                    "name": expected_name,
                    "matched": matched,
                }
            )
            if not matched:
                all_matched = False

        reward = 1.0 if all_matched else 0.0

        return {
            "reward": reward,
            "breakdown": {"action": reward},
            "action_checks": action_checks,
            "all_matched": all_matched,
        }

    def _evaluate_communication(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent-user communication quality.

        Checks if required information was communicated to the user.

        Adapted from: tau2-bench src/tau2/evaluator/evaluator_communicate.py

        Args:
            traces: Filtered execution traces

        Returns:
            Dict with communication verification results
        """
        if self.communicate_info is None or RewardType.COMMUNICATE not in self.reward_basis:
            return {"reward": 1.0, "note": "No communication criteria"}

        messages = traces.get("messages", [])

        # Combine all message content
        all_content = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
            all_content += " " + content

        all_content = all_content.lower()

        # Check each required info
        comm_checks = []
        all_found = True

        for info in self.communicate_info:
            found = info.lower() in all_content
            comm_checks.append({"info": info, "found": found})
            if not found:
                all_found = False

        reward = 1.0 if all_found else 0.0

        return {
            "reward": reward,
            "breakdown": {"communicate": reward},
            "comm_checks": comm_checks,
            "all_found": all_found,
        }


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    Infrastructure errors are excluded from scoring metrics.
    Uses SCOREABLE_STATUSES to determine which results count toward agent score.

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with success_rate, mean_reward, pass_at_k, status_counts
    """
    if not results:
        return {
            "total_tasks": 0,
            "scored_tasks": 0,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "status_counts": {},
        }

    total_tasks = len(results)
    scored_tasks = 0
    successful_tasks = 0
    total_reward = 0.0
    status_counts: Dict[str, int] = {}

    for res in results:
        status = res.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        if status not in SCOREABLE_STATUSES:
            continue  # Skip infrastructure errors

        scored_tasks += 1
        evals = res.get("eval") or []

        for entry in evals:
            reward = entry.get("reward", 0.0)
            total_reward += reward
            if entry.get("passed", False):
                successful_tasks += 1
                break

    success_rate = successful_tasks / scored_tasks if scored_tasks > 0 else 0.0
    mean_reward = total_reward / scored_tasks if scored_tasks > 0 else 0.0

    return {
        "total_tasks": total_tasks,
        "scored_tasks": scored_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "status_counts": status_counts,
    }


def compute_pass_at_k(
    results: List[Dict[str, Any]],
    k_values: List[int] = [1, 2, 3, 4],
) -> Dict[str, float]:
    """Compute Pass@k metrics from benchmark results.

    Pass@k: Probability that at least 1 of k attempts succeeds.

    Requires running benchmark with n_task_repeats >= max(k_values).

    Args:
        results: List of result dicts from benchmark.run()
        k_values: k values to compute (default: 1, 2, 3, 4 per tau2 paper)

    Returns:
        Dict with pass@1, pass@2, etc. scores
    """
    # Group results by task_id
    task_results: Dict[str, List[bool]] = {}
    for res in results:
        task_id = res.get("task_id", "")
        if res.get("status") not in SCOREABLE_STATUSES:
            continue  # Skip infrastructure errors

        evals = res.get("eval") or []
        passed = any(entry.get("passed", False) for entry in evals)

        if task_id not in task_results:
            task_results[task_id] = []
        task_results[task_id].append(passed)

    # Compute pass@k for each k
    pass_at_k: Dict[str, float] = {}

    for k in k_values:
        successes = 0
        total = 0

        for task_id, attempts in task_results.items():
            if len(attempts) < k:
                continue  # Not enough attempts for this k

            total += 1
            # Pass@k: at least 1 success in first k attempts
            if any(attempts[:k]):
                successes += 1

        pass_at_k[f"pass@{k}"] = successes / total if total > 0 else 0.0

    return pass_at_k


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Compute the pass^k metric for a single task.

    Pass^k is a combinatorial metric from https://arxiv.org/pdf/2406.12045
    that estimates the probability of getting k successes in k draws
    without replacement from a pool of num_trials attempts.

    Formula: C(success_count, k) / C(num_trials, k)

    Args:
        num_trials: Total number of attempts for the task
        success_count: Number of successful attempts
        k: Number of draws to consider

    Returns:
        Pass^k probability (0.0 to 1.0)

    Raises:
        ValueError: If num_trials < k
    """
    from math import comb

    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")

    if success_count < k:
        return 0.0

    return comb(success_count, k) / comb(num_trials, k)


def compute_pass_hat_k(
    results: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute Pass^k metrics from benchmark results.

    Pass^k is the combinatorial metric from the tau2 paper that estimates
    the probability of k successes in k draws without replacement.

    This differs from Pass@k which only checks if at least 1 of k attempts succeeds.

    Requires running benchmark with n_task_repeats >= max(k_values).

    Args:
        results: List of result dicts from benchmark.run()
        k_values: k values to compute. If None, uses 1 to max trials.

    Returns:
        Dict with pass^1, pass^2, etc. scores (averaged across all tasks)
    """
    # Group results by task_id
    task_results: Dict[str, List[bool]] = {}
    for res in results:
        task_id = res.get("task_id", "")
        if res.get("status") not in SCOREABLE_STATUSES:
            continue  # Skip infrastructure errors

        evals = res.get("eval") or []
        passed = any(entry.get("passed", False) for entry in evals)

        if task_id not in task_results:
            task_results[task_id] = []
        task_results[task_id].append(passed)

    if not task_results:
        return {}

    # Determine max k based on minimum trials across tasks
    min_trials = min(len(attempts) for attempts in task_results.values())

    if k_values is None:
        k_values = list(range(1, min_trials + 1))

    # Compute pass^k for each k
    pass_hat_k_results: Dict[str, float] = {}

    for k in k_values:
        task_scores = []

        for task_id, attempts in task_results.items():
            num_trials = len(attempts)
            if num_trials < k:
                continue  # Not enough attempts for this k

            success_count = sum(attempts)
            score = pass_hat_k(num_trials, success_count, k)
            task_scores.append(score)

        if task_scores:
            pass_hat_k_results[f"pass^{k}"] = sum(task_scores) / len(task_scores)

    return pass_hat_k_results
