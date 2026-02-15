"""CONVERSE Benchmark — Contextual Safety in Agent-to-Agent Conversations.

Adapted from ConVerse (https://github.com/amrgomaaelhady/ConVerse, commit d474f6a).
Original work licensed under the MIT License.

Citation:
    Gomaa, A., Salem, A., & Abdelnabi, S. (2025). ConVerse: Benchmarking Contextual
    Safety in Agent-to-Agent Conversations. arXiv:2511.05359.

Framework-agnostic implementation of the CONVERSE benchmark for evaluating
the resilience of LLM-based assistants against adversarial social-engineering
attacks from external service-provider agents.

Domains:

- travel_planning: Travel planning scenarios (4 personas)
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
    ensure_data_exists(domain="travel_planning")

    # Load tasks for a domain
    tasks = load_tasks("travel_planning", split="all", limit=5)

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
from .data_loader import ConverseDomain, configure_model_ids, ensure_data_exists, load_tasks
from .environment import ConverseEnvironment
from .evaluator import PrivacyEvaluator, SecurityEvaluator, UtilityEvaluator
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
    "UtilityEvaluator",
    "ConverseDomain",
    "configure_model_ids",
    "load_tasks",
    "ensure_data_exists",
]
