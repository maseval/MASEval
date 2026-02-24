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

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from maseval import Evaluator, ModelAdapter, Task

from maseval.benchmark.tau2.environment import Tau2Environment, get_environment_constructor
from maseval.benchmark.tau2.utils import compare_tool_calls

logger = logging.getLogger(__name__)


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
        nl_model: Optional[ModelAdapter] = None,
    ):
        """Initialize the evaluator.

        Args:
            task: Task being evaluated
            environment: Tau2Environment instance
            nl_model: Optional model for NL assertion evaluation
        """
        self.task = task
        self.environment = environment
        self.nl_model = nl_model

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
        """Build full message trajectory from agent and user traces.

        Matches original tau2-bench where evaluate_simulation receives
        simulation.messages — a flat ordered list of ALL messages.

        Args:
            traces: Full execution traces

        Returns:
            Dict with full_trajectory, environment traces, termination_reason
        """
        full_trajectory = self._build_full_trajectory(traces)
        return {
            "full_trajectory": full_trajectory,
            "environment": traces.get("environment", {}),
            "termination_reason": traces.get("termination_reason"),
        }

    @staticmethod
    def _build_full_trajectory(traces: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build full ordered message trajectory from agent and user traces.

        Merges agent messages (which contain agent tool calls) with user messages
        (which contain user tool calls and the initial greeting). Matches the
        original tau2-bench simulation.messages structure.

        Args:
            traces: Full execution traces

        Returns:
            Ordered list of all messages in the conversation
        """
        agent_msgs = next((d.get("messages", []) for d in traces.get("agents", {}).values()), [])
        user_msgs = next((d.get("messages", []) for d in traces.get("users", {}).values()), [])

        if not agent_msgs:
            return []
        if not user_msgs:
            return list(agent_msgs)

        # Extract greeting and user-side tool call sequences from user messages.
        # User msgs structure: [greeting, user_q1, asst_text_1, (user_tc, tool_r)*, user_q2, ...]
        # We pair each user tool sequence with the agent text content that preceded it.
        greeting = None
        user_tool_seqs: List[tuple] = []  # (preceding_asst_content, [tool_call_msgs])

        i = 0
        if user_msgs[0].get("role") == "assistant":
            greeting = user_msgs[0]
            i = 1

        last_asst_content = greeting.get("content") if greeting else None
        while i < len(user_msgs):
            msg = user_msgs[i]
            if msg.get("role") == "assistant":
                last_asst_content = msg.get("content")
                i += 1
            elif msg.get("role") == "user" and msg.get("tool_calls"):
                seq = [msg]
                i += 1
                while i < len(user_msgs) and user_msgs[i].get("role") == "tool":
                    seq.append(user_msgs[i])
                    i += 1
                user_tool_seqs.append((last_asst_content, seq))
            else:
                i += 1

        # Build trajectory: greeting + agent messages with user tool sequences
        # inserted after the matching agent text response.
        trajectory: List[Dict[str, Any]] = []
        if greeting:
            trajectory.append(greeting)

        seq_idx = 0
        for msg in agent_msgs:
            trajectory.append(msg)
            # After agent text response, insert matching user tool calls
            if (
                msg.get("role") == "assistant"
                and not msg.get("tool_calls")
                and seq_idx < len(user_tool_seqs)
                and user_tool_seqs[seq_idx][0] == msg.get("content")
            ):
                trajectory.extend(user_tool_seqs[seq_idx][1])
                seq_idx += 1

        return trajectory

    def __call__(
        self,
        traces: Dict[str, Any],
        final_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate task completion.

        Matches original tau2-bench evaluate_simulation():
        - Premature termination → reward=0.0
        - Always runs ALL evaluators (M7: not gated by reward_basis)
        - Only uses reward_basis when COMBINING scores

        Args:
            traces: Filtered execution traces (from filter_traces)
            final_answer: Final answer from agent

        Returns:
            Dict with reward, passed, reward_breakdown, and per-evaluator results
        """
        # Premature termination → reward=0.0 (matching original)
        termination_reason = traces.get("termination_reason")
        if termination_reason in {TerminationReason.TOO_MANY_ERRORS.value, TerminationReason.MAX_STEPS.value}:
            return {
                "reward": 0.0,
                "passed": False,
                "reward_breakdown": {},
                "note": f"Simulation terminated prematurely: {termination_reason}",
            }

        full_trajectory = traces.get("full_trajectory", [])

        # M7: Always run ALL evaluators regardless of reward_basis
        env_result = self._evaluate_environment(full_trajectory)
        action_result = self._evaluate_actions(full_trajectory)
        communicate_result = self._evaluate_communication(full_trajectory)
        nl_result = self._evaluate_nl_assertions(full_trajectory)

        # Combine rewards based on reward_basis only
        reward = 1.0
        reward_breakdown: Dict[str, float] = {}
        task_reward_basis = set(self.reward_basis)

        if task_reward_basis & {RewardType.DB, RewardType.ENV_ASSERTION}:
            reward_breakdown.update(env_result.get("breakdown", {}))
            reward *= env_result.get("reward", 1.0)

        if task_reward_basis & {RewardType.ACTION}:
            reward_breakdown.update(action_result.get("breakdown", {}))
            reward *= action_result.get("reward", 1.0)

        if task_reward_basis & {RewardType.COMMUNICATE}:
            reward_breakdown.update(communicate_result.get("breakdown", {}))
            reward *= communicate_result.get("reward", 1.0)

        if task_reward_basis & {RewardType.NL_ASSERTION}:
            reward_breakdown.update(nl_result.get("breakdown", {}))
            reward *= nl_result.get("reward", 1.0)

        # D22: Use epsilon comparison matching original agent_metrics.py
        passed = (1 - 1e-6) <= reward <= (1 + 1e-6)

        return {
            "reward": reward,
            "passed": passed,
            "reward_breakdown": reward_breakdown,
            "env_check": env_result,
            "action_check": action_result,
            "communicate_check": communicate_result,
            "nl_check": nl_result,
        }

    def _evaluate_environment(self, full_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate environment state by reconstructing predicted and gold environments.

        Matches original tau2-bench evaluator_env.py:
        - Creates fresh predicted_environment, replays full_trajectory via set_state
        - Creates fresh gold_environment, replays initial message_history + golden actions
        - Compares DB hashes between predicted and gold
        - Runs env_assertions on predicted_environment using getattr (C3)

        Args:
            full_trajectory: Full ordered message trajectory

        Returns:
            Dict with db_match, db_reward, env_assertions results
        """
        if self.actions is None and self.env_assertions is None:
            return {"reward": 1.0, "note": "No environment criteria"}

        env_data = self.task.environment_data
        initial_state = env_data.get("initial_state") or {}
        initialization_data = initial_state.get("initialization_data")
        initialization_actions = initial_state.get("initialization_actions")
        message_history = initial_state.get("message_history") or []

        env_constructor = get_environment_constructor(self.task.environment_data)

        # C1/C10: Create fresh predicted environment, replay full trajectory
        predicted_env = env_constructor()
        predicted_env.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=full_trajectory,
        )

        # Create fresh gold environment, replay only initial message_history
        gold_env = env_constructor()
        gold_env.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Execute golden actions on gold environment
        golden_actions = self.actions or []
        for action in golden_actions:
            try:
                gold_env.make_tool_call(
                    tool_name=action["name"],
                    requestor=action.get("requestor", "assistant"),
                    **action.get("arguments", {}),
                )
            except Exception as e:
                logger.warning(f"Error in golden actions {action['name']}({action.get('arguments', {})}): {e}")

        # Compare DB hashes
        agent_db_match = gold_env.get_db_hash() == predicted_env.get_db_hash()
        user_db_match = gold_env.get_user_db_hash() == predicted_env.get_user_db_hash()
        db_match = agent_db_match and user_db_match
        db_reward = 1.0 if db_match else 0.0

        # C3: Run env_assertions on predicted_environment using getattr (not use_tool)
        env_assertion_checks = []
        env_assertion_reward = 1.0
        for assertion in self.env_assertions or []:
            success = predicted_env.run_env_assertion(assertion, raise_assertion_error=False)
            env_assertion_checks.append(
                {
                    "assertion": assertion,
                    "passed": success,
                    "reward": 1.0 if success else 0.0,
                }
            )
            env_assertion_reward *= 1.0 if success else 0.0

        # Build breakdown (always include, reward_basis gating is in __call__)
        reward = 1.0
        breakdown: Dict[str, float] = {}
        if RewardType.DB in self.reward_basis:
            breakdown[RewardType.DB.value] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in self.reward_basis:
            breakdown[RewardType.ENV_ASSERTION.value] = env_assertion_reward
            reward *= env_assertion_reward

        return {
            "reward": reward,
            "breakdown": breakdown,
            "db_match": db_match,
            "agent_db_match": agent_db_match,
            "user_db_match": user_db_match,
            "db_reward": db_reward,
            "env_assertions": env_assertion_checks,
        }

    def _evaluate_actions(self, full_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate tool usage patterns.

        H7: Extracts tool calls from the full message trajectory (both assistant
        and user messages), matching original evaluator_action.py which iterates
        full_trajectory for AssistantMessage/UserMessage with is_tool_call().

        Args:
            full_trajectory: Full ordered message trajectory

        Returns:
            Dict with action verification results
        """
        if not self.actions:
            return {"reward": 1.0, "note": "No action criteria"}

        # H7: Extract tool calls from trajectory messages (both assistant and user)
        predicted_tool_calls = []
        for msg in full_trajectory:
            role = msg.get("role", "")
            tool_calls = msg.get("tool_calls")
            if tool_calls and role in ("assistant", "user"):
                for tc in tool_calls:
                    if "function" in tc:
                        name = tc["function"].get("name", "")
                        args = tc["function"].get("arguments", {})
                    else:
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    predicted_tool_calls.append({"name": name, "arguments": args})

        # Check each expected action against predicted tool calls
        action_checks = []
        all_matched = True
        for action in self.actions:
            matched = any(
                compare_tool_calls(
                    action.get("name", ""),
                    action.get("arguments", {}),
                    tc["name"],
                    tc["arguments"],
                    action.get("compare_args"),
                )
                for tc in predicted_tool_calls
            )
            action_checks.append(
                {
                    "action_id": action.get("action_id"),
                    "name": action.get("name"),
                    "matched": matched,
                }
            )
            if not matched:
                all_matched = False

        reward = 1.0 if all_matched else 0.0
        return {
            "reward": reward,
            "breakdown": {RewardType.ACTION.value: reward},
            "action_checks": action_checks,
            "all_matched": all_matched,
        }

    def _evaluate_communication(self, full_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate agent-user communication quality.

        H8: Searches the full message trajectory for assistant messages,
        matching original evaluator_communicate.py which iterates full_trajectory
        for AssistantMessage with has_text_content().

        Args:
            full_trajectory: Full ordered message trajectory

        Returns:
            Dict with communication verification results
        """
        if not self.communicate_info:
            return {"reward": 1.0, "note": "No communication criteria"}

        # H8: Search full trajectory assistant messages (not just agent traces)
        comm_checks = []
        all_found = True
        for info in self.communicate_info:
            found = False
            for msg in full_trajectory:
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
                if not content:
                    continue
                # D4: Strip commas before matching (matching original)
                if info.lower() in content.lower().replace(",", ""):
                    found = True
                    break
            comm_checks.append({"info": info, "found": found})
            if not found:
                all_found = False

        reward = 1.0 if all_found else 0.0
        return {
            "reward": reward,
            "breakdown": {RewardType.COMMUNICATE.value: reward},
            "comm_checks": comm_checks,
            "all_found": all_found,
        }

    def _evaluate_nl_assertions(self, full_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate natural language assertions using LLM judge.

        C6: Matches original tau2-bench evaluator_nl_assertions.py:
        - Builds trajectory string: "role: content" for each message
        - System prompt: exact copy of original's NL assertion judge prompt
        - User prompt: trajectory + expected outcomes
        - Calls model.chat(), parses JSON response
        - Reward: 1.0 if all met, 0.0 otherwise

        Args:
            full_trajectory: Full ordered message trajectory

        Returns:
            Dict with NL assertion evaluation results
        """
        if not self.nl_assertions:
            return {"reward": 1.0, "note": "No NL assertions"}

        if self.nl_model is None:
            return {
                "reward": 1.0,
                "note": "NL assertions present but no nl_model configured — skipped",
                "breakdown": {},
            }

        # Build trajectory string matching original
        trajectory_str = "\n".join(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in full_trajectory)

        # System prompt matching original evaluator_nl_assertions.py exactly
        system_prompt = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading

        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """

        user_prompt = f"""
        conversation:
        {trajectory_str}

        expectedOutcomes:
        {self.nl_assertions}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.nl_model.chat(messages=messages, temperature=0.0)
            content = response.content
            assert content is not None, "NL assertion model returned None content"
            result_data = json.loads(content)
            nl_checks = []
            for result in result_data.get("results", []):
                nl_checks.append(
                    {
                        "nl_assertion": result.get("expectedOutcome", ""),
                        "met": result.get("metExpectation", False),
                        "justification": result.get("reasoning", ""),
                    }
                )
            all_met = all(c["met"] for c in nl_checks)
            reward = 1.0 if all_met else 0.0
        except Exception as e:
            logger.warning(f"NL assertion evaluation failed: {e}")
            nl_checks = []
            reward = 0.0

        return {
            "reward": reward,
            "breakdown": {RewardType.NL_ASSERTION.value: reward},
            "nl_checks": nl_checks,
        }


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    H9: ALL simulations count in the denominator (matching original).
    Terminated simulations get reward=0.0 (handled by evaluator).

    Args:
        results: List of result dicts from benchmark.run()

    Returns:
        Dict with success_rate, mean_reward, status_counts
    """
    if not results:
        return {
            "total_tasks": 0,
            "successful_tasks": 0,
            "success_rate": 0.0,
            "mean_reward": 0.0,
            "status_counts": {},
        }

    total_tasks = len(results)
    successful_tasks = 0
    total_reward = 0.0
    status_counts: Dict[str, int] = {}

    for res in results:
        status = res.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        # H9: ALL simulations count — terminated ones already have reward=0.0
        evals = res.get("eval") or []
        for entry in evals:
            total_reward += entry.get("reward", 0.0)
            if entry.get("passed", False):
                successful_tasks += 1
                break

    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": successful_tasks / total_tasks,
        "mean_reward": total_reward / total_tasks,
        "status_counts": status_counts,
    }


def compute_pass_at_k(
    results: List[Dict[str, Any]],
    k_values: List[int] = [1, 2, 3, 4],
) -> Dict[str, float]:
    """Compute Pass@k metrics from benchmark results.

    Pass@k: Probability that at least 1 of k attempts succeeds.
    H9: ALL simulations count (terminated ones are failures).

    Args:
        results: List of result dicts from benchmark.run()
        k_values: k values to compute (default: 1, 2, 3, 4 per tau2 paper)

    Returns:
        Dict with pass@1, pass@2, etc. scores
    """
    task_results: Dict[str, List[bool]] = {}
    for res in results:
        task_id = res.get("task_id", "")
        evals = res.get("eval") or []
        passed = any(entry.get("passed", False) for entry in evals)
        task_results.setdefault(task_id, []).append(passed)

    pass_at_k: Dict[str, float] = {}
    for k in k_values:
        successes = 0
        total = 0
        for attempts in task_results.values():
            if len(attempts) < k:
                continue
            total += 1
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
    # H9: ALL simulations count (terminated ones are failures)
    task_results: Dict[str, List[bool]] = {}
    for res in results:
        task_id = res.get("task_id", "")
        evals = res.get("eval") or []
        passed = any(entry.get("passed", False) for entry in evals)
        task_results.setdefault(task_id, []).append(passed)

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
