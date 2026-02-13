# Benchmark

The `Benchmark` class is the core orchestrator for running multi-agent system experiments. It manages the complete execution lifecycle—from setting up environments and agents to running tasks and evaluating results. By implementing a few abstract methods, you define your benchmark's specific logic while the framework handles task iteration, data collection, and reproducibility.

Think of it as the experiment controller: you describe _what_ to set up and _how_ agents should run, and the framework handles _when_ and _how many times_ everything executes.

Each method below is documented starting with the formal method definition and a following `How to use` section that is more educational and should help getting started.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/benchmark.py){ .md-source-file }

::: maseval.core.benchmark.Benchmark

::: maseval.core.benchmark.TaskExecutionStatus
