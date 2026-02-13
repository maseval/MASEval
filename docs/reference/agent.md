# Agents

Agent adapters wrap agents from any framework to provide a unified interface for benchmarking. They handle execution, message history tracking, and callback hooks.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/agent.py){ .md-source-file align=right }

::: maseval.core.agent.AgentAdapter
options:
show_source: false

## Interfaces

Adapters that integrate MASEval with agent frameworks:

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/smolagents.py){ .md-source-file align=right }

::: maseval.interface.agents.smolagents.SmolAgentAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/langgraph.py){ .md-source-file align=right }

::: maseval.interface.agents.langgraph.LangGraphAgentAdapter

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/llamaindex.py){ .md-source-file align=right }

::: maseval.interface.agents.llamaindex.LlamaIndexAgentAdapter
