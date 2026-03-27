# AREEnvironment Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic AREEnvironment to maseval that wraps Meta's ARE simulation infrastructure as a reusable building block for interactive agent environments.

**Architecture:** `AREEnvironment` is a maseval `Environment` subclass in `maseval/interface/environments/` that wraps ARE's `Environment`, `Scenario`, and `Tool` classes. It accepts either a pre-built ARE `Scenario` or a shorthand dict of apps+events via `task_data` (= `task.environment_data`). Tools are wrapped in `AREToolWrapper` with `ToolInvocationHistory` tracing. The ARE event loop is user-controlled (start/stop/pause/resume).

**Tech Stack:** Python 3.10+, maseval core (`Environment`, `TraceableMixin`, `ConfigurableMixin`, `ToolInvocationHistory`), ARE (`meta-agents-research-environments>=1.2.0`) as optional dependency.

**Spec:** `docs/superpowers/specs/2026-03-27-are-environment-design.md`

---

### Task 1: Create `maseval/interface/environments/` Package

**Files:**
- Create: `maseval/interface/environments/__init__.py`
- Modify: `maseval/interface/__init__.py`

- [ ] **Step 1: Create the environments package init**

Create `maseval/interface/environments/__init__.py` with conditional ARE import (matching the pattern in `maseval/interface/agents/__init__.py`):

```python
"""maseval.interface.environments

Environment integrations for external simulation platforms.
"""

__all__: list[str] = []

try:
    from .are import AREEnvironment  # noqa: F401
    from .are_tool_wrapper import AREToolWrapper  # noqa: F401

    __all__.extend(["AREEnvironment", "AREToolWrapper"])
except ImportError:
    pass
```

- [ ] **Step 2: Register the environments subpackage in `maseval/interface/__init__.py`**

Add `environments` to the interface package. Modify `maseval/interface/__init__.py`:

```python
"""maseval.interface

This package contains adapters and thin shims that integrate external libraries and services
with MASEval. Each integration is optional and requires installing the corresponding extra.

Organization:
- inference/: Model inference adapters (OpenAI, Google, HuggingFace, etc.)
- agents/: Agent framework adapters (smolagents, langgraph, etc.)
- environments/: Environment integrations (ARE, etc.)
- logging/: Logging platform adapters (wandb, langfuse, etc.)

Canonical rules:
- Keep adapters thin: translate between MASEval internal abstractions and the external API.
- Avoid heavy imports at module import time; import lazily inside functions/classes.

See `maseval/interface/README.md` for more details and conventions for optional dependencies,
packaging extras, and testing.
"""

# Import subpackages
from . import inference, agents, environments
from . import logging as logging_  # Rename to avoid conflict with stdlib

__all__ = ["inference", "agents", "environments", "logging_"]
```

- [ ] **Step 3: Commit**

```bash
git add maseval/interface/environments/__init__.py maseval/interface/__init__.py
git commit -m "feat: add maseval/interface/environments/ package"
```

---

### Task 2: Implement `AREToolWrapper`

**Files:**
- Create: `maseval/interface/environments/are_tool_wrapper.py`
- Create: `tests/interface/environments/test_are_tool_wrapper.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/interface/environments/__init__.py` (empty) and `tests/interface/environments/test_are_tool_wrapper.py`:

```python
"""Tests for AREToolWrapper."""

from unittest.mock import MagicMock
import pytest

from maseval.interface.environments.are_tool_wrapper import AREToolWrapper


class TestAREToolWrapper:
    """Tests for AREToolWrapper."""

    def _make_mock_are_tool(self, name="Calendar__create_event", description="Create a calendar event",
                            inputs=None, output_type="string", return_value="Event created"):
        """Create a mock ARE tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputs = inputs or {"title": {"type": "string", "description": "Event title"}}
        tool.output_type = output_type
        tool.return_value = return_value
        tool.__call__ = MagicMock(return_value=return_value)
        return tool

    def test_metadata_from_are_tool(self):
        """Wrapper exposes ARE tool metadata."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        assert wrapper.name == "Calendar__create_event"
        assert wrapper.description == "Create a calendar event"
        assert wrapper.inputs == {"title": {"type": "string", "description": "Event title"}}
        assert wrapper.output_type == "string"

    def test_call_delegates_to_are_tool(self):
        """Calling wrapper delegates to underlying ARE tool."""
        are_tool = self._make_mock_are_tool(return_value="Event created")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        result = wrapper(title="Standup")

        are_tool.assert_called_once_with(title="Standup")
        assert result == "Event created"

    def test_call_records_success_in_history(self):
        """Successful calls are recorded in invocation history."""
        are_tool = self._make_mock_are_tool(return_value="OK")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        wrapper(title="Test")

        assert len(wrapper.history) == 1
        record = wrapper.history.to_list()[0]
        assert record["inputs"] == {"title": "Test"}
        assert record["outputs"] == "OK"
        assert record["status"] == "success"

    def test_call_records_error_in_history(self):
        """Failed calls are recorded in invocation history and re-raised."""
        are_tool = self._make_mock_are_tool()
        are_tool.side_effect = ValueError("Invalid title")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        with pytest.raises(ValueError, match="Invalid title"):
            wrapper(title="")

        assert len(wrapper.history) == 1
        record = wrapper.history.to_list()[0]
        assert record["status"] == "error"
        assert "Invalid title" in record["outputs"]

    def test_gather_traces(self):
        """gather_traces returns structured trace data."""
        are_tool = self._make_mock_are_tool(return_value="Done")
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        wrapper(title="Test1")
        wrapper(title="Test2")

        traces = wrapper.gather_traces()
        assert traces["type"] == "AREToolWrapper"
        assert traces["name"] == "Calendar__create_event"
        assert traces["total_invocations"] == 2
        assert len(traces["invocations"]) == 2

    def test_gather_config(self):
        """gather_config returns tool configuration."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        config = wrapper.gather_config()

        assert config["name"] == "Calendar__create_event"
        assert config["description"] == "Create a calendar event"
        assert "input_schema" in config

    def test_repr(self):
        """String representation shows tool signature."""
        are_tool = self._make_mock_are_tool()
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)
        r = repr(wrapper)

        assert "AREToolWrapper" in r
        assert "Calendar__create_event" in r
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/interface/environments/test_are_tool_wrapper.py -v
```

Expected: FAIL with `ModuleNotFoundError` (module doesn't exist yet).

- [ ] **Step 3: Implement AREToolWrapper**

Create `maseval/interface/environments/are_tool_wrapper.py`:

```python
"""ARE Tool Wrapper for MASEval.

Framework-agnostic wrapper for ARE Tool instances. Provides a callable
interface with ToolInvocationHistory tracing and metadata exposure for
framework adapters (smolagents, LangGraph, etc.) to build framework-native
tools from.

This is the layer 1->2 wrapper:
- Layer 1: ARE Tool (forward(), inputs, output_type)
- Layer 2: maseval generic (callable, ToolInvocationHistory, metadata)
- Layer 3: framework-specific -- NOT handled here.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin
from maseval.core.history import ToolInvocationHistory

if TYPE_CHECKING:
    from maseval.interface.environments.are import AREEnvironment


class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    """Framework-agnostic wrapper for ARE tools with maseval tracing.

    Wraps an ARE Tool and exposes its metadata (name, description, inputs,
    output_type) so that agent adapters can construct framework-native tools.

    Example for smolagents::

        class MySmolagentsTool(smolagents.Tool):
            skip_forward_signature_validation = True

            def __init__(self, wrapper: AREToolWrapper):
                self.wrapper = wrapper
                self.name = wrapper.name
                self.description = wrapper.description
                self.inputs = wrapper.inputs
                self.output_type = wrapper.output_type
                super().__init__()

            def forward(self, **kwargs) -> str:
                return self.wrapper(**kwargs)
    """

    def __init__(self, are_tool: Any, environment: "AREEnvironment"):
        """Initialize the tool wrapper.

        Args:
            are_tool: ARE Tool instance to wrap.
            environment: The AREEnvironment this tool belongs to.
        """
        super().__init__()
        self._are_tool = are_tool
        self._environment = environment
        self.history = ToolInvocationHistory()

        # Expose ARE tool metadata for framework adapters
        self.name: str = are_tool.name
        self.description: str = are_tool.description
        self.inputs: Dict[str, Any] = are_tool.inputs
        self.output_type: str = are_tool.output_type

        # Extract JSON schema from ARE tool args (if available)
        self.input_schema: Dict[str, Any] = self._extract_schema(are_tool)

    @staticmethod
    def _extract_schema(are_tool: Any) -> Dict[str, Any]:
        """Convert ARE's args list to JSON schema format.

        Args:
            are_tool: ARE Tool instance.

        Returns:
            JSON schema dict with properties and required fields.
        """
        args = getattr(are_tool, "args", None)
        if not args:
            return {}

        properties = {}
        required = []

        for arg in args:
            param_name = getattr(arg, "name", None)
            if not param_name:
                continue
            properties[param_name] = {
                "type": getattr(arg, "arg_type", "string"),
                "description": getattr(arg, "description", ""),
            }
            if not getattr(arg, "has_default", True):
                required.append(param_name)

        return {"properties": properties, "required": required}

    def __call__(self, **kwargs: Any) -> Any:
        """Execute the ARE tool with tracing.

        Args:
            **kwargs: Tool arguments matching the inputs schema.

        Returns:
            Tool output (type varies per tool).

        Raises:
            Any exception from the underlying ARE tool is re-raised.
        """
        start_time = datetime.now()
        status = "success"
        result = None
        error_message = None

        try:
            result = self._are_tool(**kwargs)
        except Exception as e:
            status = "error"
            error_message = str(e)
            raise
        finally:
            self.history.add_invocation(
                inputs=kwargs,
                outputs=result if status == "success" else error_message,
                status=status,
                timestamp=start_time.isoformat(),
            )

        return result

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this tool.

        Returns:
            Dictionary with tool name, invocation history, and counts.
        """
        return {
            **super().gather_traces(),
            "name": self.name,
            "invocations": self.history.to_list(),
            "total_invocations": len(self.history),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration from this tool.

        Returns:
            Dictionary with tool name, description, and schema.
        """
        return {
            **super().gather_config(),
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in self.inputs.items())
        return f"{self.__class__.__name__}({self.name}({args}) -> {self.output_type})"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/interface/environments/test_are_tool_wrapper.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are_tool_wrapper.py tests/interface/environments/__init__.py tests/interface/environments/test_are_tool_wrapper.py
git commit -m "feat: add AREToolWrapper with tracing and metadata"
```

---

### Task 3: Implement `AREEnvironment` — Core Structure and Scenario Path

**Files:**
- Create: `maseval/interface/environments/are.py`
- Create: `tests/interface/environments/test_are_environment.py`

This task implements the core class with the Scenario construction path. The shorthand path (apps+events) is added in Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/interface/environments/test_are_environment.py`:

```python
"""Tests for AREEnvironment."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

from maseval.interface.environments.are import AREEnvironment


def _make_mock_scenario(scenario_id="test-001", duration=600, seed=42, start_time=0):
    """Create a mock ARE Scenario."""
    scenario = MagicMock()
    scenario.scenario_id = scenario_id
    scenario.duration = duration
    scenario.seed = seed
    scenario.start_time = start_time
    scenario.time_increment_in_seconds = 1
    scenario.apps = [MagicMock(name="EmailClient"), MagicMock(name="Calendar")]
    return scenario


def _make_mock_are_env(apps=None):
    """Create a mock ARE Environment."""
    env = MagicMock()
    if apps is None:
        # Create mock apps with mock tools
        email_app = MagicMock()
        email_tool = MagicMock()
        email_tool.name = "EmailClient__send_email"
        email_tool.description = "Send an email"
        email_tool.inputs = {"to": {"type": "string", "description": "Recipient"}}
        email_tool.output_type = "string"
        email_app.get_tools.return_value = [email_tool]
        email_app.name = "EmailClient"

        calendar_app = MagicMock()
        cal_tool = MagicMock()
        cal_tool.name = "Calendar__create_event"
        cal_tool.description = "Create event"
        cal_tool.inputs = {"title": {"type": "string", "description": "Title"}}
        cal_tool.output_type = "string"
        calendar_app.get_tools.return_value = [cal_tool]
        calendar_app.name = "Calendar"

        env.apps = {"EmailClient": email_app, "Calendar": calendar_app}
    else:
        env.apps = apps
    env.current_time = 0.0
    return env


class TestAREEnvironmentScenarioPath:
    """Tests for AREEnvironment with Scenario objects."""

    @patch("maseval.interface.environments.are._import_are")
    def test_setup_state_with_scenario(self, mock_import):
        """setup_state initialises from an ARE Scenario and returns state dict."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        task_data = {"scenario": scenario}

        env = AREEnvironment(task_data)

        assert env.state["scenario_id"] == "test-001"
        assert env.state["duration"] == 600
        assert env.state["seed"] == 42
        assert env._are_env is mock_are_env

    @patch("maseval.interface.environments.are._import_are")
    def test_setup_state_requires_scenario_or_apps(self, mock_import):
        """setup_state raises ValueError if neither scenario nor apps provided."""
        mock_import.return_value = MagicMock()

        with pytest.raises(ValueError, match="must contain either"):
            AREEnvironment(task_data={})

    @patch("maseval.interface.environments.are._import_are")
    def test_create_tools_wraps_are_tools(self, mock_import):
        """create_tools wraps all ARE app tools in AREToolWrapper."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        tools = env.get_tools()
        assert "EmailClient__send_email" in tools
        assert "Calendar__create_event" in tools
        assert len(tools) == 2

        # Check wrapper metadata
        email_tool = tools["EmailClient__send_email"]
        assert email_tool.name == "EmailClient__send_email"
        assert email_tool.description == "Send an email"

    @patch("maseval.interface.environments.are._import_are")
    def test_start_runs_scenario(self, mock_import):
        """start() calls are_env.run() with the scenario."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.start()

        mock_are_env.run.assert_called_once()
        call_kwargs = mock_are_env.run.call_args
        assert call_kwargs[1].get("wait_for_end") is False

    @patch("maseval.interface.environments.are._import_are")
    def test_stop_stops_env(self, mock_import):
        """stop() calls are_env.stop()."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.stop()

        mock_are_env.stop.assert_called_once()

    @patch("maseval.interface.environments.are._import_are")
    def test_pause_and_resume(self, mock_import):
        """pause() and resume_with_offset() delegate to ARE env."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.pause()
        mock_are_env.pause.assert_called_once()

        env.resume_with_offset(5.0)
        mock_are_env.resume_with_offset.assert_called_once_with(5.0)

    @patch("maseval.interface.environments.are._import_are")
    def test_get_simulation_time(self, mock_import):
        """get_simulation_time() returns ARE env's current_time."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_env.current_time = 42.5
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        assert env.get_simulation_time() == 42.5

    @patch("maseval.interface.environments.are._import_are")
    def test_cleanup_stops_env(self, mock_import):
        """cleanup() stops the ARE environment."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})
        env.cleanup()

        mock_are_env.stop.assert_called_once()

    @patch("maseval.interface.environments.are._import_are")
    def test_gather_traces(self, mock_import):
        """gather_traces returns structured trace data."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_env.current_time = 100.0
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        traces = env.gather_traces()
        assert traces["scenario_id"] == "test-001"
        assert traces["tool_count"] == 2
        assert "tools" in traces
        assert traces["final_simulation_time"] == 100.0

    @patch("maseval.interface.environments.are._import_are")
    def test_gather_config(self, mock_import):
        """gather_config returns environment configuration."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        config = env.gather_config()
        assert config["scenario_id"] == "test-001"
        assert config["duration"] == 600
        assert config["notification_verbosity"] == "medium"
        assert "tool_names" in config
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/interface/environments/test_are_environment.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement AREEnvironment**

Create `maseval/interface/environments/are.py`:

```python
"""AREEnvironment — generic maseval Environment wrapping ARE simulation.

Provides a reusable building block for interactive agent environments
using Meta's ARE (Agents Research Environments) infrastructure.

Original Repository: https://github.com/facebookresearch/meta-agents-research-environments
Code License: MIT
"""

from typing import Any, Dict, List, Optional, Tuple

from maseval.core.environment import Environment
from maseval.core.callback import EnvironmentCallback
from maseval.interface.environments.are_tool_wrapper import AREToolWrapper


def _check_are_installed() -> None:
    """Check if ARE is installed and raise a helpful error if not."""
    try:
        import are  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ARE (Agent Research Environments) is required for AREEnvironment.\n"
            "Install with: pip install maseval[are]\n"
            "Or: uv add meta-agents-research-environments"
        ) from e


def _import_are() -> Any:
    """Lazily import and return the ARE simulation module.

    Returns:
        The ``are.simulation`` module namespace with Environment,
        EnvironmentConfig, Scenario, etc.

    Raises:
        ImportError: If ARE is not installed.
    """
    _check_are_installed()
    from types import SimpleNamespace
    from are.simulation.environment import Environment as AREEnv  # type: ignore[import-not-found]
    from are.simulation.environment import EnvironmentConfig  # type: ignore[import-not-found]

    return SimpleNamespace(
        Environment=AREEnv,
        EnvironmentConfig=EnvironmentConfig,
    )


class AREEnvironment(Environment):
    """Generic maseval Environment wrapping ARE's simulation infrastructure.

    Supports two construction paths via ``task_data`` (= ``task.environment_data``):

    1. **Scenario path:** ``task_data = {"scenario": <ARE Scenario>}``
    2. **Shorthand path:** ``task_data = {"apps": [...], "events": [...], "duration": 1800, ...}``

    Lifecycle is user-controlled: call ``start()`` before ``run_agents()``,
    ``stop()`` after. ``pause()``/``resume_with_offset()`` control simulation time.
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
            task_data: ``task.environment_data`` dict. Must contain either:
                - ``"scenario"``: ARE Scenario object, OR
                - ``"apps"``: list of ARE App instances, plus optional ``"events"``,
                  ``"duration"``, ``"seed"``, ``"start_time"``, ``"time_increment_in_seconds"``
            callbacks: Optional maseval EnvironmentCallbacks.
            run_oracle: If True, run ARE oracle mode during setup to generate
                expected event log. Stored in traces for evaluation.
            notification_verbosity: ARE notification verbosity level.
                ``"low"`` = no environment notifications,
                ``"medium"`` = standard notifications,
                ``"high"`` = all notifications.
        """
        self._run_oracle = run_oracle
        self._notification_verbosity = notification_verbosity
        self._are_env: Any = None
        self._scenario: Any = None
        self._oracle_traces: Optional[Dict[str, Any]] = None
        self._tool_wrappers: Dict[str, AREToolWrapper] = {}

        super().__init__(task_data, callbacks)

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ARE environment from task data.

        Args:
            task_data: Dict with ``"scenario"`` or ``"apps"`` key.

        Returns:
            State dict with scenario metadata.

        Raises:
            ValueError: If task_data contains neither ``"scenario"`` nor ``"apps"``.
        """
        are_mod = _import_are()

        scenario = task_data.get("scenario")

        if scenario is None and "apps" not in task_data:
            raise ValueError(
                "task_data must contain either 'scenario' (ARE Scenario object) "
                "or 'apps' (list of ARE App instances)."
            )

        if scenario is None:
            scenario = self._build_scenario_from_shorthand(task_data)

        self._scenario = scenario

        # Run oracle mode if requested
        if self._run_oracle:
            self._oracle_traces = self._run_oracle_mode(are_mod, scenario)

        # Create ARE Environment (but don't start the event loop yet)
        config = are_mod.EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
            time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
        )
        if getattr(scenario, "start_time", None) and scenario.start_time > 0:
            config.start_time = scenario.start_time

        # Create notification system based on verbosity
        notification_system = self._create_notification_system()
        self._are_env = are_mod.Environment(config, notification_system=notification_system)

        # Register apps from scenario so tools are available before start()
        self._are_env.register_apps(scenario.apps)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": scenario.duration,
            "seed": getattr(scenario, "seed", None),
            "start_time": getattr(scenario, "start_time", None),
            "app_names": [getattr(app, "name", str(app)) for app in scenario.apps],
            "oracle_traces": self._oracle_traces,
        }

    def _build_scenario_from_shorthand(self, task_data: Dict[str, Any]) -> Any:
        """Build an ARE Scenario from shorthand task_data.

        Args:
            task_data: Dict with ``"apps"``, and optional ``"events"``,
                ``"duration"``, ``"seed"``, ``"start_time"``,
                ``"time_increment_in_seconds"``.

        Returns:
            ARE Scenario instance.
        """
        from are.simulation.scenarios.scenario import Scenario  # type: ignore[import-not-found]

        apps = task_data["apps"]
        events = task_data.get("events", [])
        duration = task_data.get("duration", 1800)
        seed = task_data.get("seed", 0)
        start_time = task_data.get("start_time", 0)
        time_increment = task_data.get("time_increment_in_seconds", 1)

        scenario = Scenario(
            scenario_id=task_data.get("scenario_id", "custom"),
            apps=apps,
            events=events,
            duration=duration,
            seed=seed,
            start_time=start_time,
            time_increment_in_seconds=time_increment,
        )
        scenario.initialize()
        return scenario

    def _run_oracle_mode(self, are_mod: Any, scenario: Any) -> Dict[str, Any]:
        """Run ARE oracle mode to generate expected event log.

        Args:
            are_mod: ARE module namespace.
            scenario: ARE Scenario instance.

        Returns:
            Dict with oracle event log.
        """
        oracle_config = are_mod.EnvironmentConfig(
            oracle_mode=True,
            duration=scenario.duration,
            time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
        )
        oracle_env = are_mod.Environment(oracle_config)
        oracle_env.run(scenario, wait_for_end=True, schedule_events=True)

        # Capture oracle state
        oracle_traces = {
            "apps_state": oracle_env.get_apps_state() if hasattr(oracle_env, "get_apps_state") else {},
            "world_logs": oracle_env.get_world_logs() if hasattr(oracle_env, "get_world_logs") else [],
        }

        # Soft-reset so app state is clean for agent run
        scenario.soft_reset()

        return oracle_traces

    def _create_notification_system(self) -> Any:
        """Create ARE notification system based on verbosity setting.

        Returns:
            ARE NotificationSystem instance.
        """
        try:
            from are.simulation.notification_system import (  # type: ignore[import-not-found]
                VerboseNotificationSystem,
                VerbosityLevel,
            )

            level_map = {
                "low": VerbosityLevel.LOW,
                "medium": VerbosityLevel.MEDIUM,
                "high": VerbosityLevel.HIGH,
            }
            level = level_map.get(self._notification_verbosity, VerbosityLevel.MEDIUM)
            return VerboseNotificationSystem(verbosity_level=level)
        except ImportError:
            return None

    def create_tools(self) -> Dict[str, AREToolWrapper]:
        """Wrap all ARE app tools in AREToolWrapper.

        Returns:
            Dict mapping tool names to AREToolWrapper instances.
        """
        tools: Dict[str, AREToolWrapper] = {}

        if self._are_env is None:
            return tools

        for app in self._are_env.apps.values():
            for are_tool in app.get_tools():
                wrapper = AREToolWrapper(are_tool, self)
                tools[are_tool.name] = wrapper
                self._tool_wrappers[are_tool.name] = wrapper

        return tools

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the ARE simulation event loop.

        Call this after environment setup and before running agents.
        Runs the scenario with ``wait_for_end=False`` so control returns
        immediately for agent interaction.
        """
        if self._are_env is not None and self._scenario is not None:
            self._are_env.run(self._scenario, wait_for_end=False, schedule_events=True)

    def stop(self) -> None:
        """Stop the ARE simulation event loop."""
        if self._are_env is not None:
            self._are_env.stop()

    def pause(self) -> None:
        """Pause simulation time progression."""
        if self._are_env is not None:
            self._are_env.pause()

    def resume_with_offset(self, offset: float) -> None:
        """Resume simulation with a time offset.

        Args:
            offset: Seconds to advance simulation clock before resuming.
        """
        if self._are_env is not None:
            self._are_env.resume_with_offset(offset)

    # ── Notification Polling ──────────────────────────────────────────

    def poll_notifications(self) -> Tuple[List[str], List[str], bool]:
        """Drain pending notifications from ARE's notification queue.

        Returns:
            Tuple of ``(user_messages, env_notifications, has_stop_signal)``.
            ``user_messages``: Messages from simulated users.
            ``env_notifications``: System events (new email, calendar reminder, etc.).
            ``has_stop_signal``: True when simulation has ended.

        Agent adapters should call this between agent steps and inject
        the messages into the agent's context.
        """
        if self._are_env is None:
            return [], [], False

        notification_system = getattr(self._are_env, "notification_system", None)
        if notification_system is None:
            return [], [], False

        try:
            from datetime import datetime, timezone
            from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

            sim_time = self.get_simulation_time()
            timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
            unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

            if not unhandled:
                return [], [], False

            user_messages: List[str] = []
            env_notifications: List[str] = []
            has_stop = False

            for notif in unhandled:
                msg_type = getattr(notif, "message_type", None)
                if msg_type == MessageType.USER_MESSAGE:
                    user_messages.append(notif.message)
                elif msg_type == MessageType.ENVIRONMENT_NOTIFICATION:
                    ts = notif.timestamp.strftime("%Y-%m-%d %H:%M:%S") if notif.timestamp else ""
                    env_notifications.append(f"[{ts}] {notif.message}")
                elif msg_type == MessageType.ENVIRONMENT_STOP:
                    has_stop = True

            return user_messages, env_notifications, has_stop

        except Exception:
            return [], [], False

    # ── Data Access ───────────────────────────────────────────────────

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds since scenario start."""
        if self._are_env is None:
            return 0.0
        try:
            return self._are_env.current_time
        except AttributeError:
            return 0.0

    def get_are_environment(self) -> Any:
        """Get the underlying ARE Environment instance."""
        return self._are_env

    def get_oracle_traces(self) -> Optional[Dict[str, Any]]:
        """Get oracle event log if oracle mode was enabled.

        Returns:
            Oracle traces dict, or None if oracle was not run.
        """
        return self._oracle_traces

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop ARE simulation. Called by maseval after task completes."""
        if self._are_env is not None:
            try:
                self._are_env.stop()
            except Exception:
                pass

    # ── Tracing & Config ──────────────────────────────────────────────

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces from environment and all tools."""
        tool_traces = {}
        for name, wrapper in self._tool_wrappers.items():
            tool_traces[name] = wrapper.gather_traces()

        return {
            **super().gather_traces(),
            "scenario_id": self.state.get("scenario_id"),
            "duration": self.state.get("duration"),
            "seed": self.state.get("seed"),
            "app_names": self.state.get("app_names", []),
            "oracle_traces": self._oracle_traces,
            "final_simulation_time": self.get_simulation_time(),
            "tool_count": len(self._tool_wrappers),
            "tools": tool_traces,
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather environment configuration for reproducibility."""
        return {
            **super().gather_config(),
            "scenario_id": self.state.get("scenario_id"),
            "duration": self.state.get("duration"),
            "seed": self.state.get("seed"),
            "start_time": self.state.get("start_time"),
            "notification_verbosity": self._notification_verbosity,
            "run_oracle": self._run_oracle,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/interface/environments/test_are_environment.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "feat: add AREEnvironment with scenario path and lifecycle control"
```

---

### Task 4: Implement Shorthand Construction Path

**Files:**
- Modify: `maseval/interface/environments/are.py` (already has `_build_scenario_from_shorthand` stub)
- Modify: `tests/interface/environments/test_are_environment.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/interface/environments/test_are_environment.py`:

```python
class TestAREEnvironmentShorthandPath:
    """Tests for AREEnvironment with apps+events shorthand."""

    @patch("maseval.interface.environments.are._import_are")
    @patch("maseval.interface.environments.are.AREEnvironment._build_scenario_from_shorthand")
    def test_shorthand_builds_scenario(self, mock_build, mock_import):
        """Shorthand task_data with 'apps' key triggers scenario construction."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_scenario = _make_mock_scenario()
        mock_build.return_value = mock_scenario

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        task_data = {
            "apps": [MagicMock(), MagicMock()],
            "events": [MagicMock()],
            "duration": 300,
            "seed": 99,
        }
        env = AREEnvironment(task_data=task_data)

        mock_build.assert_called_once_with(task_data)
        assert env._scenario is mock_scenario

    @patch("maseval.interface.environments.are._import_are")
    def test_shorthand_passes_config_to_scenario(self, mock_import):
        """Shorthand config values are passed through to Scenario construction."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        # Patch Scenario import inside _build_scenario_from_shorthand
        mock_scenario_cls = MagicMock()
        mock_scenario_instance = _make_mock_scenario()
        mock_scenario_cls.return_value = mock_scenario_instance

        with patch.dict("sys.modules", {
            "are": MagicMock(),
            "are.simulation": MagicMock(),
            "are.simulation.scenarios": MagicMock(),
            "are.simulation.scenarios.scenario": MagicMock(Scenario=mock_scenario_cls),
        }):
            apps = [MagicMock(), MagicMock()]
            task_data = {
                "apps": apps,
                "duration": 300,
                "seed": 99,
                "start_time": 100,
                "time_increment_in_seconds": 5,
            }
            env = AREEnvironment(task_data=task_data)

            mock_scenario_cls.assert_called_once()
            call_kwargs = mock_scenario_cls.call_args[1]
            assert call_kwargs["duration"] == 300
            assert call_kwargs["seed"] == 99
            assert call_kwargs["start_time"] == 100
            assert call_kwargs["time_increment_in_seconds"] == 5
            mock_scenario_instance.initialize.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentShorthandPath -v
```

Expected: Tests should be runnable now since the code exists; verify the shorthand path logic.

- [ ] **Step 3: Run full test suite to verify everything still passes**

```bash
uv run pytest tests/interface/environments/ -v
```

Expected: All tests PASS (shorthand path was already stubbed in Task 3).

- [ ] **Step 4: Commit**

```bash
git add tests/interface/environments/test_are_environment.py
git commit -m "test: add shorthand construction path tests for AREEnvironment"
```

---

### Task 5: Add Optional Dependency to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `are` optional extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add the `are` extra near the existing benchmark extras (near `gaia2`):

```toml
are = ["meta-agents-research-environments>=1.2.0"]
```

- [ ] **Step 2: Add pytest marker**

In `pyproject.toml` under `[tool.pytest.ini_options]` markers, add:

```toml
"are: Tests that specifically require ARE (Agent Research Environments)",
```

Note: check existing markers — if there is already an equivalent `gaia2` marker that covers ARE, this may be redundant. Add only if no existing marker covers the `are` extra specifically.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add 'are' optional dependency extra"
```

---

### Task 6: Integration Smoke Test (optional, requires ARE installed)

**Files:**
- Create: `tests/interface/environments/test_are_integration.py`

This test is marked with `@pytest.mark.are` so it only runs when ARE is installed.

- [ ] **Step 1: Write the integration test**

Create `tests/interface/environments/test_are_integration.py`:

```python
"""Integration tests for AREEnvironment (requires ARE installed)."""

import pytest

try:
    import are  # noqa: F401
    HAS_ARE = True
except ImportError:
    HAS_ARE = False


@pytest.mark.are
@pytest.mark.skipif(not HAS_ARE, reason="ARE not installed")
class TestAREEnvironmentIntegration:
    """Integration tests that exercise real ARE infrastructure."""

    def test_import_works(self):
        """AREEnvironment can be imported when ARE is installed."""
        from maseval.interface.environments.are import AREEnvironment
        assert AREEnvironment is not None

    def test_tool_wrapper_import(self):
        """AREToolWrapper can be imported when ARE is installed."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper
        assert AREToolWrapper is not None

    def test_package_init_exports(self):
        """Package __init__ exports AREEnvironment when ARE is installed."""
        from maseval.interface.environments import AREEnvironment, AREToolWrapper
        assert AREEnvironment is not None
        assert AREToolWrapper is not None
```

- [ ] **Step 2: Run integration tests (only if ARE is installed)**

```bash
uv run pytest tests/interface/environments/test_are_integration.py -v -m are
```

Expected: PASS if ARE is installed, SKIP otherwise.

- [ ] **Step 3: Run full test suite to verify nothing is broken**

```bash
uv run pytest tests/ -v --ignore=tests/interface/environments/test_are_integration.py
```

Expected: All existing tests PASS, new unit tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/interface/environments/test_are_integration.py
git commit -m "test: add ARE integration smoke tests"
```
