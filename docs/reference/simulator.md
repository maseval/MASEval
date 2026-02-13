# Simulators

Simulators in MASEval are used to create reproducible and scalable testing environments for multi-agent systems. Their primary function is to mock the behavior of external dependencies such as APIs, databases, or human users. By using simulators, developers can test agent interactions in a controlled, isolated environment. This approach eliminates variability from external factors like network latency or API availability, ensuring that benchmark results are consistent and deterministic. This allows for a more precise evaluation of agent performance and reliable comparison between different implementations.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/simulator.py){ .md-source-file }

::: maseval.core.simulator.LLMSimulator

::: maseval.core.simulator.ToolLLMSimulator

::: maseval.core.simulator.UserLLMSimulator

::: maseval.core.simulator.AgenticUserLLMSimulator

::: maseval.core.simulator.SimulatorCallStatus
