# Callback

Callbacks allow you to hook into benchmark execution at various points. Use them for logging, monitoring, tracing, or custom side effects during agent runs.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/callback.py){ .md-source-file }

::: maseval.core.callback.BenchmarkCallback

::: maseval.core.callback.EnvironmentCallback

::: maseval.core.callback.AgentCallback

## Built-in Callbacks

MASEval provides built-in callback implementations:

### Message Tracing

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/callbacks/message_tracing.py){ .md-source-file }

::: maseval.core.callbacks.message_tracing.MessageTracingAgentCallback

### Result Logging

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/callbacks/result_logger.py){ .md-source-file }

::: maseval.core.callbacks.result_logger.ResultLogger

::: maseval.core.callbacks.result_logger.FileResultLogger

### Progress Bars

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/callbacks/progress_bar.py){ .md-source-file }

::: maseval.core.callbacks.progress_bar.ProgressBarCallback

::: maseval.core.callbacks.progress_bar.TqdmProgressBarCallback

::: maseval.core.callbacks.progress_bar.RichProgressBarCallback
