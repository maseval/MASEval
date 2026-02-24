"""
ColBench Human Simulator — MASEval User adapter.

Converts the sweet_rl ``HumanInteractionEnv`` into MASEval's ``User`` interface.

Execution flow (driven by Benchmark.execution_loop):
    1. execution_loop calls ``get_initial_query()`` → problem_description
    2. Loop:  ``run_agents(query)`` → agent_text
              ``respond(agent_text)`` → checks termination, invokes simulator
              ``is_done()`` → True if agent answered or max_steps exhausted
    3. When the agent emits ``"I WANT TO ANSWER:"``, ``respond()`` extracts
       the answer, marks done, and returns ``""``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from maseval.core.user import User
from maseval.core.model import ModelAdapter
from maseval.core.history import MessageHistory

logger = logging.getLogger(__name__)

HUMAN_RESPONSE_CHARACTER_LIMIT = 400

DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT = (
    "Your task is to simulate a human user that interacts with an LLM agent "
    "in a dialogue.\n"
    "You would like the LLM agent to help you with the following problem:\n"
    "{problem_description}\n\n"
    "Your goal is to engage in the conversation with the LLM agent so that "
    "it can get to a personalized answer.\n"
    "You should make use of the following hidden information to answer the "
    "LLM agent.\n"
    'YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.\n'
    "You SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE "
    'HIDDEN INFORMATION, AND SAY YOU DON"T KNOW IF THE ANSWER CAN NOT BE '
    "FOUND IN THE HIDDEN INFORMATION.\n\n"
    "{hidden_information}\n\n"
    "Here is the dialogue so far:\n"
    "{dialogue_history}\n\n\n"
    "Now directly output your answer to the LLM agent IN TWO SENTENCES. "
    "DO NOT SAY ANYTHING ELSE."
)


class ColBenchUser(User):
    """MASEval User that replicates the ColBench human-simulator behaviour.

    Faithfully converts ``HumanInteractionEnv`` from sweet_rl:

    * The initial query is the task's ``problem_description``.
    * On each ``respond()`` call the agent's message is inspected for
      ``"I WANT TO ANSWER:"``; if found the user marks done and does NOT
      call the simulator LLM.
    * Otherwise, the human-simulator LLM is invoked with the full dialogue
      history formatted identically to the original prompt template.
    * Responses are truncated to ``HUMAN_RESPONSE_CHARACTER_LIMIT`` (400).
    * Interaction terminates when the agent answers or ``max_steps`` exhausted.
    """

    def __init__(
        self,
        problem_description: str,
        hidden_information: str,
        model: ModelAdapter,
        human_prompt: Optional[str] = None,
        max_steps: int = 10,
        response_char_limit: int = HUMAN_RESPONSE_CHARACTER_LIMIT,
    ):
        super().__init__()
        self.problem_description = str(problem_description)
        self.hidden_information = str(hidden_information)
        self.model = model
        self.human_prompt = human_prompt or DEFAULT_HUMAN_SIMULATOR_CODE_PROMPT
        self.max_steps = max_steps
        self.response_char_limit = response_char_limit

        self.messages = MessageHistory()
        self.logs: List[Dict[str, Any]] = []
        self._turn_count: int = 0
        self._done: bool = False
        self.answer: Optional[str] = None

    # ── User protocol ────────────────────────────────────────────────────

    def get_initial_query(self) -> str:
        self.messages.add_message("user", self.problem_description)
        return self.problem_description

    def respond(self, agent_message: str) -> str:
        if self._done:
            return ""

        self._turn_count += 1
        raw_response = agent_message

        # Strip OUTPUT: prefix (matches original step() logic)
        if "OUTPUT:" in agent_message:
            agent_message = agent_message.split("OUTPUT:")[1]
            raw_response = "OUTPUT:".join(raw_response.split("OUTPUT:")[:2])

        # Termination check
        if "I WANT TO ANSWER:" in agent_message:
            self._done = True
            self.answer = agent_message.split("I WANT TO ANSWER:")[1]
        elif self._turn_count >= self.max_steps:
            self._done = True
            self.answer = agent_message

        # Record agent turn
        self.messages.add_message("assistant", agent_message)

        # Invoke human simulator if interaction continues
        if self._done:
            return ""

        start_time = time.time()
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "turn": self._turn_count,
            "agent_message_preview": agent_message[:200],
            "status": "success",
        }

        try:
            simulator_reply = self._invoke_simulator()
        except Exception as exc:
            log_entry["duration_seconds"] = time.time() - start_time
            log_entry["status"] = "error"
            log_entry["error"] = str(exc)
            log_entry["error_type"] = type(exc).__name__
            self.logs.append(log_entry)
            raise

        simulator_reply = simulator_reply[: self.response_char_limit]

        log_entry["duration_seconds"] = time.time() - start_time
        log_entry["response_preview"] = simulator_reply[:200]
        self.logs.append(log_entry)

        self.messages.add_message("user", simulator_reply)
        return simulator_reply

    def is_done(self) -> bool:
        return self._done

    # ── Simulator invocation ─────────────────────────────────────────────

    def _format_dialogue_history(self) -> str:
        """Format dialogue as original ``str_dialogue_history()``."""
        result = ""
        for msg in self.messages.to_list():
            role = msg["role"] if msg["role"] != "assistant" else "agent"
            result += f"{role}:{msg['content']}\n\n\n\n"
        return result + "agent:"

    def _invoke_simulator(self) -> str:
        """Call the human-simulator LLM via ModelAdapter.chat().

        Replicates ``HumanInteractionEnv.invoke_model()`` with 3-retry logic.
        Uses ``model.chat()`` → ``ChatResponse.content``.
        """
        prompt_text = self.human_prompt.format(
            problem_description=self.problem_description,
            hidden_information=self.hidden_information,
            dialogue_history=self._format_dialogue_history(),
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]

        for attempt in range(3):
            try:
                response = self.model.chat(
                    messages=messages,
                    generation_params={
                        "max_tokens": 4096,
                        "temperature": 0,
                    },
                )
                return response.content or "No response."
            except Exception as e:
                logger.warning(
                    "Human simulator call failed (attempt %d/3): %s",
                    attempt + 1,
                    e,
                )
                if attempt == 2:
                    return "No response."
        return "No response."

    # ── Tracing / Config ─────────────────────────────────────────────────

    def gather_traces(self) -> Dict[str, Any]:
        return {
            **super().gather_traces(),
            "name": "colbench_human_simulator",
            "message_count": len(self.messages),
            "messages": self.messages.to_list(),
            "logs": self.logs,
            "max_steps": self.max_steps,
            "turns_used": self._turn_count,
            "answer": self.answer,
            "termination_reason": (
                "agent_answered"
                if self.answer is not None and self._turn_count < self.max_steps
                else "max_steps" if self._done else "not_terminated"
            ),
        }

    def gather_config(self) -> Dict[str, Any]:
        return {
            **super().gather_config(),
            "name": "colbench_human_simulator",
            "max_steps": self.max_steps,
            "response_char_limit": self.response_char_limit,
            "prompt_template_length": len(self.human_prompt),
        }