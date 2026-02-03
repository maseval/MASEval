"""Agent framework adapters.

This package contains adapters for different agent frameworks.
"""

__all__ = []

# Conditionally import smolagents
try:
    from .smolagents import SmolAgentAdapter, SmolAgentLLMUser  # noqa: F401

    __all__.extend(["SmolAgentAdapter", "SmolAgentLLMUser"])
except ImportError:
    pass

# Conditionally import langgraph
try:
    from .langgraph import LangGraphAgentAdapter, LangGraphLLMUser  # noqa: F401

    __all__.extend(["LangGraphAgentAdapter", "LangGraphLLMUser"])
except ImportError:
    pass

# Conditionally import llamaindex
try:
    from .llamaindex import LlamaIndexAgentAdapter, LlamaIndexLLMUser  # noqa: F401

    __all__.extend(["LlamaIndexAgentAdapter", "LlamaIndexLLMUser"])
except ImportError:
    pass

# Conditionally import camel
try:
    from .camel import (  # noqa: F401
        CamelAgentAdapter,
        CamelLLMUser,
        CamelAgentUser,
        camel_role_playing_execution_loop,
        CamelRolePlayingTracer,
        CamelWorkforceTracer,
    )

    __all__.extend(
        [
            "CamelAgentAdapter",
            "CamelLLMUser",
            "CamelAgentUser",
            "camel_role_playing_execution_loop",
            "CamelRolePlayingTracer",
            "CamelWorkforceTracer",
        ]
    )
except ImportError:
    pass
