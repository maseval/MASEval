# Tasks

Tasks define individual benchmark scenarios including inputs, expected outputs, and metadata for evaluation. Task queues control execution order and scheduling strategy.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L55){ .md-source-file }

::: maseval.core.task.Task

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L27){ .md-source-file }

::: maseval.core.task.TaskProtocol

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L18){ .md-source-file }

::: maseval.core.task.TimeoutAction

## Task Queues

Task queues determine the order in which tasks are executed. Pass a queue to `Benchmark.run(queue=...)` to customize scheduling.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L86){ .md-source-file }

::: maseval.core.task.BaseTaskQueue

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L256){ .md-source-file }

::: maseval.core.task.SequentialTaskQueue

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L276){ .md-source-file }

::: maseval.core.task.PriorityTaskQueue

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/task.py#L322){ .md-source-file }

::: maseval.core.task.AdaptiveTaskQueue
