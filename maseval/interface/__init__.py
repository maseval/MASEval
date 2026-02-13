"""maseval.interface

This package contains adapters and thin shims that integrate external libraries and services
with MASEval. Each integration is optional and requires installing the corresponding extra.

Organization:
- inference/: Model inference adapters (OpenAI, Google, HuggingFace, etc.)
- agents/: Agent framework adapters (smolagents, langgraph, etc.)
- logging/: Logging platform adapters (wandb, langfuse, etc.)

Canonical rules:
- Keep adapters thin: translate between MASEval internal abstractions and the external API.
- Avoid heavy imports at module import time; import lazily inside functions/classes.

See `maseval/interface/README.md` for more details and conventions for optional dependencies,
packaging extras, and testing.
"""

# Import subpackages
from . import inference, agents
from . import logging as logging_  # Rename to avoid conflict with stdlib

__all__ = ["inference", "agents", "logging_"]
