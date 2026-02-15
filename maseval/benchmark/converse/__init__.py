"""CONVERSE Benchmark — Contextual Safety in Agent-to-Agent Conversations.

Framework-agnostic implementation of the CONVERSE benchmark for evaluating
the resilience of LLM-based assistants against adversarial social-engineering
attacks from external service-provider agents.

Original benchmark: https://github.com/amrgomaaelhady/ConVerse
Reference Paper: "ConVerse: Contextual Safety in Agent-to-Agent Conversations"
https://arxiv.org/abs/2506.15753

Domains:

- travel: Travel planning scenarios (4 personas)
- real_estate: Real estate advisory scenarios (4 personas)
- insurance: Insurance consultation scenarios (4 personas)

Usage::

    from maseval.benchmark.converse import (
        ConverseBenchmark, DefaultAgentConverseBenchmark,
        ConverseEnvironment, ConverseExternalAgent,
        PrivacyEvaluator, SecurityEvaluator,
        load_tasks, ensure_data_exists,
    )

    # Ensure domain data is downloaded
    ensure_data_exists(domain="travel")

    # Load tasks for a domain
    tasks = load_tasks("travel", split="all", limit=5)

    # Create your framework-specific benchmark subclass
    class MyConverseBenchmark(ConverseBenchmark):
        def setup_agents(self, agent_data, environment, task, user, seed_generator):
            tools = environment.get_tools()
            # Create your agent with these tools
            ...

        def get_model_adapter(self, model_id, **kwargs):
            adapter = MyModelAdapter(model_id)
            if "register_name" in kwargs:
                self.register("models", kwargs["register_name"], adapter)
            return adapter

    benchmark = MyConverseBenchmark(agent_data={"attacker_model_id": "gpt-4o"})
    results = benchmark.run(tasks)
"""

from .converse import ConverseBenchmark, DefaultAgentConverseBenchmark, DefaultConverseAgent, DefaultConverseAgentAdapter
<<<<<<< Updated upstream
from .data_loader import ConverseDomain, ensure_data_exists, load_tasks
=======
from .data_loader import configure_model_ids, ensure_data_exists, load_tasks
>>>>>>> Stashed changes
from .environment import ConverseEnvironment
from .evaluator import LLMPrivacyEvaluator, LLMSecurityEvaluator, LLMUtilityEvaluator, PrivacyEvaluator, SecurityEvaluator
from .external_agent import ConverseExternalAgent

__all__ = [
    "ConverseBenchmark",
    "DefaultConverseAgent",
    "DefaultConverseAgentAdapter",
    "DefaultAgentConverseBenchmark",
    "ConverseEnvironment",
    "ConverseExternalAgent",
    "PrivacyEvaluator",
    "SecurityEvaluator",
<<<<<<< Updated upstream
    "ConverseDomain",
=======
    "LLMPrivacyEvaluator",
    "LLMSecurityEvaluator",
    "LLMUtilityEvaluator",
>>>>>>> Stashed changes
    "load_tasks",
    "ensure_data_exists",
    "configure_model_ids",
]
