# Environment

Environments define the execution context for agents, including available tools, state, and any external resources needed during task execution.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/environment.py){ .md-source-file }

::: maseval.core.environment.Environment

## Tools and agent-provided helpers

Some agent adapters expose helper tools or user-simulation tools that can be used by the Environment. For example:

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/smolagents.py){ .md-source-file }

::: maseval.interface.agents.smolagents.SmolAgentAdapter

::: maseval.interface.agents.smolagents.SmolAgentLLMUser
