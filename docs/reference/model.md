# Model Adapters

Model Adapters provide a uniform runtime interface to heterogeneous model providers (HuggingFace, OpenAI, Google GenAI, or simple callables). Each adapter exposes a stable API so that `maseval` can call models without handling provider-specific shapes.

!!! note

    `Benchmark` expects `AgentAdapter` instances; it does not consume model adapters directly. ModelAdapters are used by agents, simulators, others that directly invoke models.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/model.py){ .md-source-file }

::: maseval.core.model.ModelAdapter

## Interfaces

The following adapter classes implement the ModelAdapter interface for specific providers. Each requires their own dependencies.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/openai.py){ .md-source-file }

::: maseval.interface.inference.openai.OpenAIModelAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/huggingface.py){ .md-source-file }

::: maseval.interface.inference.huggingface.HuggingFaceModelAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/google_genai.py){ .md-source-file }

::: maseval.interface.inference.google_genai.GoogleGenAIModelAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/litellm.py){ .md-source-file }

::: maseval.interface.inference.litellm.LiteLLMModelAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/anthropic.py){ .md-source-file }

::: maseval.interface.inference.anthropic.AnthropicModelAdapter
