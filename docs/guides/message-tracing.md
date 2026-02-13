# Message Tracing

## Overview

MASEval provides message tracing to capture agent conversations during benchmark execution. This is useful for:

- **Debugging**: Inspect what agents actually said and which tools they called
- **Analysis**: Understand agent behavior patterns across tasks
- **Dataset Creation**: Extract conversations for further analysis or training

!!! info "Tracing vs Logging"

    **Tracing** = Collecting execution data during task runs (messages, tool calls, metrics)

    **Logging** = Persisting traces and evaluation results to disk/databases after benchmarks complete

    This guide covers tracing. Logging functionality for persisting results is planned for future releases.

## Core Concepts

**`MessageHistory`**: OpenAI-compatible message storage that all agent adapters use internally.

**`AgentAdapter.get_messages()`**: Standard method to retrieve conversation history from any wrapped agent.

**`MessageTracingAgentCallback`**: Optional callback that automatically collects all agent conversations in memory.

## Basic Usage

### Accessing Message History

Every agent adapter exposes message history through `get_messages()`:

```python
from maseval.interface.agents import SmolAgentAdapter

# Create and run your agent
agent_adapter = SmolAgentAdapter(agent, name="researcher")
result = agent_adapter.run("What's the capital of France?")

# Get the conversation
messages = agent_adapter.get_messages()

# Inspect messages
for msg in messages:
    print(f"{msg['role']}: {msg.get('content', '')}")
    if 'tool_calls' in msg:
        print(f"  Tools called: {[tc['function']['name'] for tc in msg['tool_calls']]}")
```

### Fresh Conversations for Multiple Tasks

In benchmarks, you typically want a fresh agent instance for each task:

```python
# In your benchmark loop
for task in benchmark.tasks:
    # Create a new adapter instance for each task
    agent_adapter = YourAgentAdapter(agent_instance=agent, name="task_agent")
    result = agent_adapter.run(task.query)
    evaluate(result, task.ground_truth)
```

This ensures each task starts with a clean slate and avoids conversation history contamination.

## Using the Tracing Callback

For multi-agent systems or when you need to collect conversations from many runs, use `MessageTracingAgentCallback`:

```python
from maseval.core.callbacks import MessageTracingAgentCallback

# Create tracer
tracer = MessageTracingAgentCallback()

# Attach to your agent(s)
agent_adapter = SmolAgentAdapter(agent, name="assistant", callbacks=[tracer])

# Run tasks
agent_adapter.run("Task 1")
agent_adapter.run("Task 2")
agent_adapter.run("Task 3")

# Get all conversations
conversations = tracer.get_all_conversations()

# Each conversation contains:
# - agent_name: which agent ran
# - query: the input query
# - messages: full conversation history
# - message_count: number of messages
```

### Multi-Agent Tracing

Share one tracer across multiple agents to collect all conversations:

```python
tracer = MessageTracingAgentCallback()

# Attach to multiple agents
agent1 = SmolAgentAdapter(agent1, name="researcher", callbacks=[tracer])
agent2 = SmolAgentAdapter(agent2, name="writer", callbacks=[tracer])

# Run both agents
agent1.run("Research topic X")
agent2.run("Write about topic X")

# Get conversations by agent
research_convs = tracer.get_conversations_by_agent("researcher")
writer_convs = tracer.get_conversations_by_agent("writer")

# Or get statistics
stats = tracer.get_statistics()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Total messages: {stats['total_messages']}")
```

### Memory Management

For long-running benchmarks, periodically clear the tracer's memory:

```python
tracer = MessageTracingAgentCallback()

for batch in task_batches:
    for task in batch:
        agent_adapter.run(task.query)

    # Process this batch
    conversations = tracer.get_all_conversations()
    save_to_disk(conversations)

    # Clear memory for next batch
    tracer.clear()
```

## Configuration

### Tracer Options

```python
MessageTracingAgentCallback(
    include_metadata=True,   # Include timestamps and metadata (default: True)
    verbose=False            # Print trace info to console (default: False)
)
```

**Typical configurations:**

```python
# Debugging - see what's happening
tracer = MessageTracingAgentCallback(verbose=True)

# Production - minimal overhead
tracer = MessageTracingAgentCallback(include_metadata=False, verbose=False)
```

## Message Format

Messages use OpenAI's chat completion format:

```python
{
    "role": "user" | "assistant" | "system" | "tool",
    "content": str,
    "tool_calls": [...],      # Present when assistant calls tools
    "tool_call_id": str,      # Present in tool responses
    "name": str,              # Tool name (for tool role)
}
```

### Tool Call Example

```python
# Assistant calling a tool
{
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "NYC"}'
        }
    }]
}

# Tool response
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "name": "get_weather",
    "content": "72°F, Sunny"
}
```

## Custom Agent Adapters

If you're implementing a custom adapter, the framework handles message storage automatically via `get_messages()`. Just ensure your `_run_agent()` method returns a `MessageHistory`:

```python
from maseval import AgentAdapter, MessageHistory

class MyAgentAdapter(AgentAdapter):
    def _run_agent(self, query: str) -> MessageHistory:
        # Run your agent
        result = self.agent.run(query)

        # Convert to MessageHistory
        history = MessageHistory()
        history.add_message(role="user", content=query)
        history.add_message(role="assistant", content=result.text)

        # Framework automatically stores this
        return history
```

See the [AgentAdapter guide](../reference/agent.md) for details on implementing custom adapters.

## Tips

**For debugging**: Use `verbose=True` to see traces in real-time.

**For benchmarks**: Create a new adapter instance for each task to ensure clean conversation history.

**For multi-agent systems**: Use a shared tracer and `get_conversations_by_agent()` to analyze each agent separately.

**For memory efficiency**: Periodically `clear()` the tracer and save conversations to disk.
