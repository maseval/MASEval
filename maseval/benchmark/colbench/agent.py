"""
ColBench Agent — the agent-under-test for ColBench tasks.

Replaces ``VLLMAgent`` from sweet_rl for per-task execution.  The original
``VLLMAgent`` did batched local inference via vLLM's ``LLM`` class.  In
MASEval, each task runs independently, so we call the agent model via the
OpenAI-compatible API served by vLLM.

This class is wrapped by ``AgentAdapter`` in ``setup_agents()`` — it does
NOT subclass ``AgentAdapter`` directly.

The agent receives:
    - A system prompt (ColBench agent instructions)
    - The growing dialogue history

And returns raw text, which may contain ``"I WANT TO ANSWER:"`` to signal
the final code submission.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from maseval.core.model import ModelAdapter
from maseval.core.agent import AgentAdapter

logger = logging.getLogger(__name__)

DEFAULT_AGENT_CODE_PROMPT = (
    "You are a helpful LLM agent. \n"
    "Your task is to help a human user to resolve their problem, in particular "
    "python programming.\n"
    "1) Note that the problem is highly personalized so you need to explicitly "
    "gather information \nby asking questions to the human user about some "
    "hidden information and implicit constraints.\n"
    "YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS.\n"
    "2) Note that you should not ask human users complicated questions as they "
    "will only answer questions briefly in two sentences.\n"
    '3) When you have gathered enough information to answer, say "I WANT TO '
    'ANSWER:" in the beginning of your response and provide your final answer.\n'
    "4) Note that you can only interact with the human users WITHIN 10 "
    "back-and-forth rounds and you have to provide your final answer before "
    "the conversation ends.\n"
    "5) You should be as concise as possible in your response to human.\n\n\n"
    '"I WANT TO ANSWER:" should be included in your response to human if you '
    "think that you have gathered enough information for addressing this problem.\n"
    'Directly output the raw python code after "I WANT TO ANSWER:".\n\n'
)


class ColBenchAgentInner:
    """Inner agent logic for ColBench.

    This is a simple stateful agent that accumulates dialogue history and
    calls the model via ``ModelAdapter.chat()``.  It is wrapped by MASEval's
    ``AgentAdapter`` for tracing.

    Attributes:
        model: The ``ModelAdapter`` for the agent LLM.
        system_prompt: Agent instructions.
        temperature: Sampling temperature.
        max_tokens: Max tokens per response.
        dialogue_history: Accumulated messages.
    """

    def __init__(
        self,
        model: ModelAdapter,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_AGENT_CODE_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dialogue_history: List[Dict[str, str]] = []

    def run(self, query: str) -> str:
        """Execute one agent turn.

        Called by ``ColBenchBenchmark.run_agents()`` via ``execution_loop()``.

        Args:
            query: The current user message (problem description or
                human simulator reply).

        Returns:
            The agent's raw response text.
        """
        self.dialogue_history.append({"role": "user", "content": query})

        api_messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.dialogue_history

        response = self.model.chat(
            messages=api_messages,
            generation_params={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        text = response.content or ""

        self.dialogue_history.append({"role": "assistant", "content": text})
        return text


class ColBenchAgentAdapter(AgentAdapter):
    """Concrete AgentAdapter that delegates _run_agent to ColBenchAgentInner.run()."""

    def __init__(self, inner_agent: ColBenchAgentInner, name: str = "colbench_agent"):
        super().__init__(agent_instance=inner_agent, name=name)
        self.inner_agent = inner_agent

    def _run_agent(self, query: str) -> str:
        return self.inner_agent.run(query)