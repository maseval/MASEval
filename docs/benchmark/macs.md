# MACS: Multi-Agent Collaboration Scenarios

The **Multi-Agent Collaboration Scenarios (MACS)** benchmark evaluates how well multi-agent systems collaborate to solve complex enterprise tasks across multiple domains.

## Overview

[Multi-Agent Collaboration Scenarios (MACS)](https://arxiv.org/abs/2412.05449) is designed to test collaborative problem-solving in realistic enterprise scenarios. The benchmark includes tasks spanning multiple domains such as travel planning, retail, and more. Each task involves multiple agents that must coordinate their actions to achieve user goals.

Check out the [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md) file for more information including licenses.

## Quick Start

```python
from maseval.benchmark.macs import (
    MACSBenchmark, MACSEnvironment, MACSEvaluator, MACSGenericTool,
    load_tasks, load_agent_config,
)

# Load data
tasks = load_tasks("travel", limit=5)
agent_config = load_agent_config("travel")

# Create your framework-specific benchmark subclass
class MyMACSBenchmark(MACSBenchmark):
    def setup_agents(self, agent_data, environment, task, user):
        # Your framework-specific agent creation
        ...

# Run
benchmark = MyMACSBenchmark(agent_data=agent_config, model=my_model)
results = benchmark.run(tasks)
```

::: maseval.benchmark.macs.MACSBenchmark

::: maseval.benchmark.macs.MACSUser

::: maseval.benchmark.macs.MACSEnvironment

::: maseval.benchmark.macs.MACSEvaluator

::: maseval.benchmark.macs.MACSGenericTool
