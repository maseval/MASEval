"""Inference model adapters for various providers.

This package contains concrete implementations of ModelAdapter for different
inference providers. Each adapter requires the corresponding optional dependency.

Available adapters:
    - AnthropicModelAdapter: Anthropic Claude models (requires anthropic)
    - GoogleGenAIModelAdapter: Google Gemini models (requires google-genai)
    - HuggingFaceModelAdapter: HuggingFace transformers (requires transformers)
    - LiteLLMModelAdapter: 100+ providers via LiteLLM (requires litellm)
    - OpenAIModelAdapter: OpenAI and compatible APIs (requires openai)

Example:
    ```python
    from maseval.interface.inference import LiteLLMModelAdapter

    # Use any supported provider
    model = LiteLLMModelAdapter(model_id="gpt-4")
    response = model.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)
    ```
"""

__all__ = []

# Conditionally import Anthropic adapter
try:
    from .anthropic import AnthropicModelAdapter  # noqa: F401

    __all__.append("AnthropicModelAdapter")
except ImportError:
    pass

# Conditionally import google-genai adapter
try:
    from .google_genai import GoogleGenAIModelAdapter  # noqa: F401

    __all__.append("GoogleGenAIModelAdapter")
except ImportError:
    pass

# Conditionally import OpenAI adapter
try:
    from .openai import OpenAIModelAdapter  # noqa: F401

    __all__.append("OpenAIModelAdapter")
except ImportError:
    pass

# Conditionally import HuggingFace adapter
try:
    from .huggingface import HuggingFaceModelAdapter, ToolCallingNotSupportedError  # noqa: F401

    __all__.append("HuggingFaceModelAdapter")
    __all__.append("ToolCallingNotSupportedError")
except ImportError:
    pass

# Conditionally import LiteLLM adapter
try:
    from .litellm import LiteLLMModelAdapter  # noqa: F401

    __all__.append("LiteLLMModelAdapter")
except ImportError:
    pass
