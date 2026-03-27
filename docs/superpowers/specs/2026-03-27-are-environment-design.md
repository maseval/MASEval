# AREEnvironment Integration Design

**Date:** 2026-03-27
**Status:** Draft
**Scope:** Add generic ARE (Meta Agents Research Environments) integration to maseval as a reusable environment building block.

## Context

[ARE](https://github.com/facebookresearch/meta-agents-research-environments) is Meta's platform for evaluating AI agents in dynamic, time-evolving scenarios. It provides apps (Email, Calendar, Messaging, etc.), events, time management, and validation infrastructure.

maseval already has a Gaia2 benchmark that wraps ARE, but that integration is tightly coupled to Gaia2-specific logic (oracle preprocessing, judge config, scenario format). This design adds a generic `AREEnvironment` that any maseval benchmark can use to build interactive ARE-based environments.

## Goals

1. Generic ARE integration usable by any maseval benchmark
2. Support both loading ARE `Scenario` objects and programmatic composition (apps + events)
3. Framework-agnostic tool wrapping (layer 1->2 only; agent adapters handle 2->3)
4. Built-in notification polling for event-driven scenarios
5. Optional oracle mode with traces for evaluation
6. Do not modify existing Gaia2 code

## Non-Goals

- Framework-specific tool conversion (smolagents, LangGraph, etc.) -- that's the agent adapter's job
- Replacing or refactoring `Gaia2Environment`
- Building a fluent builder API (can be added later)
- ARE scenario authoring tools

## Module Structure

```
maseval/interface/environments/
    __init__.py
    are.py              # AREEnvironment class
    are_tool_wrapper.py # AREToolWrapper class
```

ARE is an optional dependency, imported lazily inside methods (matching Gaia2's pattern).

## AREEnvironment

### Class Definition

```python
class AREEnvironment(Environment):
    """Generic maseval Environment wrapping ARE's simulation infrastructure.

    Supports two construction paths via task_data (= task.environment_data):

    1. Scenario path: task_data = {"scenario": <ARE Scenario>}
    2. Shorthand path: task_data = {"apps": [...], "events": [...], "duration": 1800, ...}

    The shorthand path internally constructs an ARE Scenario from the provided
    apps, events, and config, then follows the same initialization as the
    scenario path.

    Lifecycle is user-controlled: call start() before run_agents(), stop()
    after. pause()/resume_with_offset() control simulation time.
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[EnvironmentCallback]] = None,
        run_oracle: bool = False,
        notification_verbosity: str = "medium",
    ):
        """Initialize AREEnvironment.

        Args:
            task_data: task.environment_data dict. Must contain either:
                - "scenario": ARE Scenario object, OR
                - "apps": list of ARE App instances, plus optional "events",
                  "duration", "seed", "start_time", "time_increment_in_seconds"
            callbacks: Optional maseval EnvironmentCallbacks.
            run_oracle: If True, run ARE oracle mode during setup to generate
                expected event log. Stored in traces for evaluation.
            notification_verbosity: ARE notification verbosity level.
                "low" = no environment notifications,
                "medium" = standard notifications (email, calendar, etc.),
                "high" = all notifications.
        """
```

### setup_state(task_data) -> Dict[str, Any]

1. **Detect input mode**: check for `"scenario"` key vs `"apps"` key
2. **Shorthand -> Scenario**: if apps/events provided, construct an ARE `Scenario`:
   - Instantiate Scenario with provided apps, events, duration, seed
   - Call `scenario.initialize()` to populate app state and event graph
3. **Oracle mode** (if `run_oracle=True`):
   - Create ARE Environment in oracle mode
   - Run scenario to completion (no agent)
   - Capture oracle event log (expected actions)
   - Soft-reset scenario for agent run
4. **Create ARE Environment and register apps**:
   - Build `EnvironmentConfig` from scenario params
   - Create `are.simulation.Environment(config, notification_system=...)`
   - Register apps from scenario onto the ARE env (`env.register_apps(scenario.apps)`)
   - Store scenario for later use by `start()` (event scheduling happens at start)
   - Store as `self._are_env` (but do NOT start the event loop yet -- user calls `start()`)
5. **Return state dict**:
   ```python
   {
       "scenario_id": scenario.scenario_id,
       "duration": scenario.duration,
       "seed": scenario.seed,
       "start_time": scenario.start_time,
       "app_names": [app.name for app in scenario.apps],
       "oracle_traces": oracle_event_log,  # None if oracle not run
   }
   ```

### create_tools() -> Dict[str, AREToolWrapper]

Iterates all apps in the ARE environment, wraps each app's tools:

```python
tools = {}
for app in self._are_env.apps.values():
    for are_tool in app.get_tools():
        wrapper = AREToolWrapper(are_tool, self)
        tools[are_tool.name] = wrapper
return tools
```

No tool filtering by default (unlike Gaia2 which removes AUI tools). Subclasses or config can filter if needed (e.g., `tool_filter` callable parameter).

**Note on shorthand apps:** The `"apps"` list in the shorthand path must contain instantiated ARE App objects (not classes), since apps hold mutable state (inbox contents, calendar entries, etc.) that defines the initial environment.

### Lifecycle Methods

```python
def start(self) -> None:
    """Start the ARE simulation event loop.

    Call this after environment setup and before running agents.
    Runs the scenario with wait_for_end=False so control returns
    immediately for agent interaction.
    """

def stop(self) -> None:
    """Stop the ARE simulation event loop."""

def pause(self) -> None:
    """Pause simulation time progression.

    Call during LLM generation to prevent simulation time from
    advancing while the agent is "thinking".
    """

def resume_with_offset(self, offset: float) -> None:
    """Resume simulation with a time offset.

    Args:
        offset: Seconds to advance simulation clock before resuming.
    """
```

### Notification Polling

```python
def poll_notifications(self) -> Tuple[List[str], List[str], bool]:
    """Drain pending notifications from ARE's notification queue.

    Returns:
        Tuple of (user_messages, env_notifications, has_stop_signal).
        user_messages: Messages from simulated users.
        env_notifications: System events (new email, calendar reminder, etc.).
        has_stop_signal: True when simulation has ended.

    Agent adapters should call this between agent steps and inject
    the messages into the agent's context.
    """
```

### Data Access

```python
def get_simulation_time(self) -> float:
    """Current simulation time in seconds since scenario start."""

def get_are_environment(self) -> Any:
    """Underlying ARE Environment instance for advanced use."""

def get_oracle_traces(self) -> Optional[Dict[str, Any]]:
    """Oracle event log if oracle mode was enabled, else None."""
```

### Tracing

```python
def gather_traces(self) -> Dict[str, Any]:
    """Collect traces from environment and all tools.

    Returns dict with:
    - Standard TraceableMixin fields (type, gathered_at)
    - scenario_id, duration, seed
    - app_names
    - oracle_traces (if oracle was run)
    - final_simulation_time
    - tool_count
    - tools: {name: tool.gather_traces() for each tool}
    """

def gather_config(self) -> Dict[str, Any]:
    """Environment configuration for reproducibility logging.

    Returns dict with:
    - Standard ConfigurableMixin fields (type, gathered_at)
    - scenario_id, duration, seed, start_time
    - notification_verbosity
    - run_oracle
    - tool_count, tool_names
    """
```

### Cleanup

```python
def cleanup(self) -> None:
    """Stop ARE simulation. Called by maseval after task completes."""
```

## AREToolWrapper

```python
class AREToolWrapper:
    """Wraps an ARE Tool into a maseval-compatible tool with tracing.

    This is the layer 1->2 wrapper:
    - Layer 1: ARE Tool (forward(), inputs, output_type)
    - Layer 2: maseval generic (callable, ToolInvocationHistory, metadata)
    - Layer 3: framework-specific (smolagents Tool, LangGraph tool, etc.)
      -- NOT handled here, that's the agent adapter's responsibility.

    Exposes ARE tool metadata (name, description, inputs schema, output_type)
    so that agent adapters can construct framework-native tools.
    """

    def __init__(self, are_tool: Any, environment: "AREEnvironment"):
        self.are_tool = are_tool
        self.environment = environment
        self.history = ToolInvocationHistory()

        # Metadata for framework adapters
        self.name: str = are_tool.name
        self.description: str = are_tool.description
        self.inputs: dict = are_tool.inputs
        self.output_type: str = are_tool.output_type

    def __call__(self, **kwargs) -> Any:
        """Call the ARE tool with tracing.

        Args:
            **kwargs: Tool arguments matching the inputs schema.

        Returns:
            Tool output (type varies per tool).

        Raises:
            Any exception from the underlying ARE tool.
        """
        try:
            result = self.are_tool(**kwargs)
            self.history.add_invocation(
                inputs=kwargs, outputs=result, status="success"
            )
            return result
        except Exception as exc:
            self.history.add_invocation(
                inputs=kwargs, outputs=str(exc), status="error"
            )
            raise

    def gather_traces(self) -> Dict[str, Any]:
        return {
            "type": "AREToolWrapper",
            "name": self.name,
            "invocations": self.history.to_list(),
            "total_invocations": len(self.history),
        }
```

## Usage Example: Custom Benchmark

```python
from maseval import Benchmark, Task, Environment
from maseval.interface.environments.are import AREEnvironment


class MyCustomBenchmark(Benchmark):

    def load_tasks(self):
        # Custom environment from apps + events
        return [
            Task(
                query="Schedule a meeting with Alice for tomorrow at 2pm",
                environment_data={
                    "apps": [EmailClient(), Calendar(), Contacts()],
                    "events": [
                        # Simulated user sends email at t=60s
                        SendEmailEvent(at=60, from_="alice", subject="Meeting?"),
                    ],
                    "duration": 600,
                    "seed": 42,
                },
                evaluation_data={"expected_action": "calendar_create_event"},
            )
        ]

    def setup_environment(self, agent_data, task, seed_generator):
        env = AREEnvironment(
            task_data=task.environment_data,
            run_oracle=True,
        )
        return env

    def run_agents(self, agents, task, environment, query):
        environment.start()  # Start ARE event loop
        try:
            # Agent adapter calls environment.poll_notifications()
            # between steps and injects messages into context
            result = agents[0].run(query)
        finally:
            environment.stop()
        return result
```

## Usage Example: Loading ARE Scenario

```python
def setup_environment(self, agent_data, task, seed_generator):
    # task.environment_data = {"scenario": <ARE Scenario object>}
    env = AREEnvironment(
        task_data=task.environment_data,
        run_oracle=True,
        notification_verbosity="medium",
    )
    return env
```

## Dependencies

- `meta-agents-research-environments` as optional dependency
- Lazy import pattern (matching Gaia2's approach)
- Added to `pyproject.toml` under an `[are]` optional extra

## Testing Strategy

- Unit tests for AREToolWrapper (mock ARE Tool, verify tracing)
- Unit tests for AREEnvironment construction (both paths)
- Integration test with a minimal ARE scenario (if ARE is installed)
- Verify gather_traces() and gather_config() output structure
