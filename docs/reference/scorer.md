# Model Scorers

Model Scorers provide a uniform interface for log-likelihood computation across model providers. Unlike `ModelAdapter` (which handles text generation and chat), scorers evaluate how likely a model considers a given continuation given some context.

!!! note

    `ModelScorer` is the scoring counterpart to `ModelAdapter`. Use it when you need log-likelihood evaluation (e.g., multiple-choice benchmarks) rather than text generation.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/scorer.py){ .md-source-file }

::: maseval.core.scorer.ModelScorer

## Interfaces

The following scorer classes implement the ModelScorer interface for specific providers.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/inference/huggingface_scorer.py){ .md-source-file }

::: maseval.interface.inference.huggingface_scorer.HuggingFaceModelScorer
