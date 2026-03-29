# Environment

Environments define the execution context for agents, including available tools, state, and any external resources needed during task execution.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/environment.py){ .md-source-file }

::: maseval.core.environment.Environment

## Tools and agent-provided helpers

Some agent adapters expose helper tools or user-simulation tools that can be used by the Environment. See the framework-specific interface pages for details:

- [SmolAgents](../interface/agents/smolagents.md) — `SmolAgentAdapter`, `SmolAgentLLMUser`
- [LangGraph](../interface/agents/langgraph.md) — `LangGraphAgentAdapter`
- [LlamaIndex](../interface/agents/llamaindex.md) — `LlamaIndexAgentAdapter`
