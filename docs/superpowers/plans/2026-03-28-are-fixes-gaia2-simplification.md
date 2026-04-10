# AREEnvironment Fixes & Gaia2 Simplification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix silent-failure bugs in AREEnvironment (issues from AREISSUES.md), then simplify Gaia2Environment by making it a subclass of AREEnvironment and eliminating Gaia2GenericTool.

**Architecture:** Phase 1 fixes AREEnvironment and AREToolWrapper to be correct and feature-complete (simulation time tracking, error propagation, AppToolAdapter, AUI filtering, turn notifications). Phase 2 makes Gaia2Environment subclass AREEnvironment, overriding only setup_state and create_tools, and deletes Gaia2GenericTool. All existing Gaia2 tests must continue to pass unchanged to confirm behavioral equivalence.

**Tech Stack:** Python, unittest.mock, pytest, ARE (optional dependency, mocked in tests)

---

## File Structure

### Phase 1 — AREEnvironment Fixes

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `maseval/interface/environments/are_tool_wrapper.py` | Add simulation time tracking, use AppToolAdapter, remove silent defaults |
| Modify | `maseval/interface/environments/are.py` | Fix oracle mode, add AUI filtering, add get_turn_notifications, fix error handling, add convenience accessors |
| Modify | `tests/interface/environments/test_are_environment.py` | Add tests for all new/changed behavior |

### Phase 2 — Gaia2 Simplification

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `maseval/benchmark/gaia2/environment.py` | Subclass AREEnvironment, keep only GAIA2-specific logic |
| Delete | `maseval/benchmark/gaia2/tool_wrapper.py` | Replaced by AREToolWrapper |
| Modify | `tests/test_benchmarks/test_gaia2/test_tool_wrapper.py` | Update imports to AREToolWrapper |
| Modify | `tests/test_benchmarks/test_gaia2/test_environment.py` | Update imports, verify all tests still pass |
| Modify | `tests/test_benchmarks/test_gaia2/conftest.py` | Update fixtures if they reference Gaia2GenericTool |

---

## Phase 1: AREEnvironment Fixes

### Task 1: Add simulation time tracking to AREToolWrapper

**Files:**
- Modify: `maseval/interface/environments/are_tool_wrapper.py`
- Test: `tests/interface/environments/test_are_environment.py`

**AREISSUES.md ref:** Issue #2 (simulation time tracking), Issue #3 (schema defaults), Issue #8 (AppToolAdapter)

- [ ] **Step 1: Write failing test for simulation time in invocation meta**

Add to `tests/interface/environments/test_are_environment.py`:

```python
class TestAREToolWrapper:
    """Tests for AREToolWrapper."""

    def test_invocation_records_simulation_time(self):
        """Tool invocations record simulation_time_before/after in meta."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        # Simulate time advancing during tool call
        mock_env.get_simulation_time.side_effect = [100.0, 105.0]

        mock_tool = MagicMock()
        mock_tool.name = "Email__send"
        mock_tool.description = "Send email"
        mock_tool.inputs = {"to": {"type": "string"}}
        mock_tool.output_type = "string"
        mock_tool.args = []
        mock_tool.return_value = "sent"

        wrapper = AREToolWrapper(mock_tool, mock_env)
        wrapper(to="alice@example.com")

        invocation = wrapper.history.to_list()[0]
        assert invocation["meta"]["simulation_time_before"] == 100.0
        assert invocation["meta"]["simulation_time_after"] == 105.0
        assert invocation["meta"]["simulation_time_elapsed"] == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREToolWrapper::test_invocation_records_simulation_time -v`

Expected: FAIL — `meta` will be `{}` (no simulation time recorded).

- [ ] **Step 3: Write failing test for simulation time when get_simulation_time raises**

```python
    def test_invocation_records_none_when_sim_time_unavailable(self):
        """If get_simulation_time() raises, meta records None without crashing."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

        mock_env = MagicMock()
        mock_env.get_simulation_time.side_effect = AttributeError("no current_time")

        mock_tool = MagicMock()
        mock_tool.name = "Email__send"
        mock_tool.description = "Send email"
        mock_tool.inputs = {}
        mock_tool.output_type = "string"
        mock_tool.args = []
        mock_tool.return_value = "sent"

        wrapper = AREToolWrapper(mock_tool, mock_env)
        wrapper()

        invocation = wrapper.history.to_list()[0]
        assert invocation["meta"]["simulation_time_before"] is None
        assert invocation["meta"]["simulation_time_after"] is None
        assert invocation["meta"]["simulation_time_elapsed"] is None
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREToolWrapper::test_invocation_records_none_when_sim_time_unavailable -v`

Expected: FAIL.

- [ ] **Step 5: Implement simulation time tracking in AREToolWrapper**

In `maseval/interface/environments/are_tool_wrapper.py`, add a `_get_simulation_time` helper and update `__call__`:

```python
from typing import TYPE_CHECKING, Any, Dict, Optional

# ... existing imports ...

class AREToolWrapper(TraceableMixin, ConfigurableMixin):
    # ... existing __init__ ...

    def _get_simulation_time(self) -> Optional[float]:
        """Get current simulation time from the parent AREEnvironment.

        Returns:
            Simulation time in seconds, or None if unavailable.
        """
        try:
            return self._environment.get_simulation_time()
        except Exception:
            return None

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
        sim_time_before = self._get_simulation_time()
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
            sim_time_after = self._get_simulation_time()
            self.history.add_invocation(
                inputs=kwargs,
                outputs=result if status == "success" else error_message,
                status=status,
                timestamp=start_time.isoformat(),
                meta={
                    "wall_time": start_time.isoformat(),
                    "simulation_time_before": sim_time_before,
                    "simulation_time_after": sim_time_after,
                    "simulation_time_elapsed": (
                        sim_time_after - sim_time_before
                        if sim_time_after is not None and sim_time_before is not None
                        else None
                    ),
                },
            )

        return result
```

Also update the import at the top of the file to include `Optional`:

```python
from typing import TYPE_CHECKING, Any, Dict, Optional
```

- [ ] **Step 6: Run both tests to verify they pass**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREToolWrapper -v`

Expected: PASS (both tests).

- [ ] **Step 7: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/interface/environments/are_tool_wrapper.py tests/interface/environments/test_are_environment.py
git commit -m "feat(are): add simulation time tracking to AREToolWrapper invocations

Record simulation_time_before, simulation_time_after, and
simulation_time_elapsed in invocation meta dict, matching
Gaia2GenericTool behavior. Gracefully returns None when
simulation time is unavailable."
```

### Task 2: Remove silent defaults from AREToolWrapper schema extraction and use AppToolAdapter

**Files:**
- Modify: `maseval/interface/environments/are_tool_wrapper.py`
- Test: `tests/interface/environments/test_are_environment.py`

**AREISSUES.md ref:** Issue #3 (schema defaults), Issue #8 (AppToolAdapter)

- [ ] **Step 1: Write failing test — schema extraction crashes on missing arg_type**

```python
    def test_schema_extraction_crashes_on_missing_arg_type(self):
        """_extract_schema raises AttributeError if arg lacks arg_type."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

        mock_tool = MagicMock()
        mock_arg = MagicMock(spec=[])  # empty spec — no attributes
        mock_arg.name = "param1"
        # Deliberately no arg_type or has_default
        mock_tool.args = [mock_arg]

        with pytest.raises(AttributeError):
            AREToolWrapper._extract_schema(mock_tool)
```

- [ ] **Step 2: Run test to verify it fails (currently getattr returns "string" instead of crashing)**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREToolWrapper::test_schema_extraction_crashes_on_missing_arg_type -v`

Expected: FAIL — test expects AttributeError, but getattr returns "string" silently.

- [ ] **Step 3: Write failing test — AppToolAdapter is used for metadata**

```python
    @patch("maseval.interface.environments.are_tool_wrapper.AppToolAdapter")
    def test_uses_app_tool_adapter_for_metadata(self, mock_adapter_cls):
        """AREToolWrapper delegates metadata extraction to AppToolAdapter."""
        mock_adapter = MagicMock()
        mock_adapter.name = "Calendar__create_event"
        mock_adapter.description = "Create a calendar event"
        mock_adapter.inputs = {"title": {"type": "string"}}
        mock_adapter.output_type = "string"
        mock_adapter.actual_return_type = "str"
        mock_adapter_cls.return_value = mock_adapter

        mock_tool = MagicMock()
        mock_tool.args = []

        mock_env = MagicMock()

        wrapper = AREToolWrapper(mock_tool, mock_env)

        mock_adapter_cls.assert_called_once_with(mock_tool)
        assert wrapper.name == "Calendar__create_event"
        assert wrapper.description == "Create a calendar event"
        assert wrapper.inputs == {"title": {"type": "string"}}
        assert wrapper.output_type == "string"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREToolWrapper::test_uses_app_tool_adapter_for_metadata -v`

Expected: FAIL — AppToolAdapter not imported or used.

- [ ] **Step 5: Implement — use AppToolAdapter, remove silent defaults**

Update `maseval/interface/environments/are_tool_wrapper.py`:

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
from typing import TYPE_CHECKING, Any, Dict, Optional

from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin
from maseval.core.history import ToolInvocationHistory

try:
    from are.simulation.tool_utils import AppToolAdapter  # type: ignore[import-not-found]
except ImportError:
    AppToolAdapter = None  # type: ignore[assignment,misc]

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

        Raises:
            ImportError: If ARE is not installed (AppToolAdapter unavailable).
        """
        super().__init__()
        self._are_tool = are_tool
        self._environment = environment
        self.history = ToolInvocationHistory()

        # Delegate metadata extraction to ARE's AppToolAdapter (tool_utils.py:544-584).
        # This is the source of truth for tool name, description, inputs, and output_type.
        if AppToolAdapter is None:
            raise ImportError(
                "ARE (Agent Research Environments) is required for AREToolWrapper.\n"
                "Install with: pip install maseval[are]"
            )
        adapter = AppToolAdapter(are_tool)
        self.name: str = adapter.name
        self.description: str = adapter.description
        self.inputs: Dict[str, Any] = adapter.inputs
        self.output_type: str = adapter.output_type
        self.actual_return_type: Optional[str] = adapter.actual_return_type

        # Extract JSON schema from ARE tool args (if available)
        self.input_schema: Dict[str, Any] = self._extract_schema(are_tool)

    @staticmethod
    def _extract_schema(are_tool: Any) -> Dict[str, Any]:
        """Convert ARE's args list to JSON schema format.

        Args:
            are_tool: ARE Tool instance.

        Returns:
            JSON schema dict with properties and required fields.

        Raises:
            AttributeError: If an arg lacks expected attributes (arg_type,
                has_default). This is intentional — a missing attribute means
                the ARE API changed and the schema would be wrong.
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
                "type": arg.arg_type,
                "description": getattr(arg, "description", ""),
            }
            if not arg.has_default:
                required.append(param_name)

        return {"properties": properties, "required": required}

    def _get_simulation_time(self) -> Optional[float]:
        """Get current simulation time from the parent AREEnvironment.

        Returns:
            Simulation time in seconds, or None if unavailable.
        """
        try:
            return self._environment.get_simulation_time()
        except Exception:
            return None

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
        sim_time_before = self._get_simulation_time()
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
            sim_time_after = self._get_simulation_time()
            self.history.add_invocation(
                inputs=kwargs,
                outputs=result if status == "success" else error_message,
                status=status,
                timestamp=start_time.isoformat(),
                meta={
                    "wall_time": start_time.isoformat(),
                    "simulation_time_before": sim_time_before,
                    "simulation_time_after": sim_time_after,
                    "simulation_time_elapsed": (
                        sim_time_after - sim_time_before
                        if sim_time_after is not None and sim_time_before is not None
                        else None
                    ),
                },
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

- [ ] **Step 6: Update existing tests to mock AppToolAdapter**

The existing tests in `test_are_environment.py` that create `AREEnvironment` instances need to mock `AppToolAdapter` since it's now imported at module level. Update the mock tool setup in `_make_mock_are_env` and add the patch:

In `_make_mock_are_env`, the mock tools already have `.name`, `.description`, `.inputs`, `.output_type` — but now AREToolWrapper reads these from `AppToolAdapter(are_tool)` instead of directly. Add a module-level patch for tests:

```python
@pytest.fixture(autouse=True)
def mock_app_tool_adapter():
    """Mock AppToolAdapter so AREToolWrapper can initialize without ARE installed."""
    def make_adapter(are_tool):
        adapter = MagicMock()
        adapter.name = are_tool.name
        adapter.description = are_tool.description
        adapter.inputs = are_tool.inputs
        adapter.output_type = are_tool.output_type
        adapter.actual_return_type = None
        return adapter

    with patch("maseval.interface.environments.are_tool_wrapper.AppToolAdapter", side_effect=make_adapter):
        yield
```

- [ ] **Step 7: Run all AREEnvironment tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py -v`

Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/interface/environments/are_tool_wrapper.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): use AppToolAdapter for metadata, remove silent schema defaults

AREToolWrapper now delegates to ARE's AppToolAdapter for tool
metadata (name, description, inputs, output_type), matching
Gaia2GenericTool. Schema extraction no longer silently fabricates
'string' type or 'optional' status when attributes are missing —
crashes immediately so ARE API changes are detected."
```

### Task 3: Fix oracle mode — remove hasattr fallbacks

**Files:**
- Modify: `maseval/interface/environments/are.py`
- Test: `tests/interface/environments/test_are_environment.py`

**AREISSUES.md ref:** Issue #1

- [ ] **Step 1: Write failing test — oracle mode calls methods directly**

```python
class TestAREEnvironmentOracleMode:
    """Tests for AREEnvironment oracle mode."""

    @patch("maseval.interface.environments.are._import_are")
    def test_oracle_mode_captures_traces(self, mock_import):
        """Oracle mode runs scenario and captures apps_state and world_logs."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_oracle_env = MagicMock()
        mock_oracle_env.get_apps_state.return_value = {"email": {"inbox": []}}
        mock_oracle_env.get_world_logs.return_value = [{"event": "email_sent"}]

        mock_agent_env = _make_mock_are_env()

        # First Environment() call = oracle env, second = agent env
        mock_are_mod.Environment.side_effect = [mock_oracle_env, mock_agent_env]

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario}, run_oracle=True)

        assert env._oracle_traces is not None
        assert env._oracle_traces["apps_state"] == {"email": {"inbox": []}}
        assert env._oracle_traces["world_logs"] == [{"event": "email_sent"}]
        mock_oracle_env.get_apps_state.assert_called_once()
        mock_oracle_env.get_world_logs.assert_called_once()
        scenario.soft_reset.assert_called_once()

    @patch("maseval.interface.environments.are._import_are")
    def test_oracle_mode_crashes_if_get_apps_state_missing(self, mock_import):
        """Oracle mode raises AttributeError if ARE env lacks get_apps_state."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_oracle_env = MagicMock(spec=[])  # no methods
        mock_oracle_env.run = MagicMock()  # only run() exists

        mock_agent_env = _make_mock_are_env()
        mock_are_mod.Environment.side_effect = [mock_oracle_env, mock_agent_env]

        scenario = _make_mock_scenario()
        with pytest.raises(AttributeError):
            AREEnvironment(task_data={"scenario": scenario}, run_oracle=True)
```

- [ ] **Step 2: Run tests to verify the second test fails**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentOracleMode -v`

Expected: `test_oracle_mode_captures_traces` PASS, `test_oracle_mode_crashes_if_get_apps_state_missing` FAIL (hasattr returns False, silently returns `{}`).

- [ ] **Step 3: Fix — remove hasattr fallbacks**

In `maseval/interface/environments/are.py`, replace `_run_oracle_mode`:

```python
    def _run_oracle_mode(self, are_mod: Any, scenario: Any) -> Dict[str, Any]:
        """Run ARE oracle mode to generate expected event log.

        Args:
            are_mod: ARE module namespace.
            scenario: ARE Scenario instance.

        Returns:
            Dict with oracle event log.

        Raises:
            AttributeError: If ARE environment lacks expected oracle methods.
        """
        oracle_config = are_mod.EnvironmentConfig(
            oracle_mode=True,
            duration=scenario.duration,
            time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
        )
        oracle_env = are_mod.Environment(oracle_config)
        oracle_env.run(scenario, wait_for_end=True, schedule_events=True)

        oracle_traces = {
            "apps_state": oracle_env.get_apps_state(),
            "world_logs": oracle_env.get_world_logs(),
        }

        # Soft-reset so app state is clean for agent run
        scenario.soft_reset()

        return oracle_traces
```

- [ ] **Step 4: Run oracle mode tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentOracleMode -v`

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): remove hasattr fallbacks from oracle mode

Oracle mode now calls get_apps_state() and get_world_logs()
directly. If the ARE API changed, this crashes immediately
instead of silently returning empty data that produces
meaningless evaluation scores."
```

### Task 4: Remove silent exception swallowing from poll_notifications and lifecycle methods

**Files:**
- Modify: `maseval/interface/environments/are.py`
- Test: `tests/interface/environments/test_are_environment.py`

**AREISSUES.md ref:** Issue #4 (poll_notifications), Issue #6 (pause/resume)

- [ ] **Step 1: Write failing test — poll_notifications propagates unexpected errors**

```python
class TestAREEnvironmentNotifications:
    """Tests for notification polling."""

    @patch("maseval.interface.environments.are._import_are")
    def test_poll_notifications_propagates_errors(self, mock_import):
        """poll_notifications does not swallow unexpected exceptions."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        # Set up notification system that raises on access
        mock_notif_sys = MagicMock()
        mock_notif_sys.message_queue.get_by_timestamp.side_effect = RuntimeError("corrupt queue")
        mock_are_env.notification_system = mock_notif_sys

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        with pytest.raises(RuntimeError, match="corrupt queue"):
            env.poll_notifications()
```

- [ ] **Step 2: Write failing test — pause propagates errors**

```python
    @patch("maseval.interface.environments.are._import_are")
    def test_pause_propagates_errors(self, mock_import):
        """pause() lets exceptions propagate for fail_on_task_error to catch."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_env.pause.side_effect = RuntimeError("pause failed")
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        with pytest.raises(RuntimeError, match="pause failed"):
            env.pause()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentNotifications -v`

Expected: FAIL — exceptions are currently swallowed.

Note: `test_pause_propagates_errors` will actually PASS already because AREEnvironment's `pause()` currently does NOT wrap in try/except. The test for `poll_notifications` is the one that will fail. Keep both tests — they document the intended contract.

- [ ] **Step 4: Fix — remove except Exception from poll_notifications**

In `maseval/interface/environments/are.py`, replace the `poll_notifications` method. Remove the outer `try/except Exception` block:

```python
    def poll_notifications(self) -> Tuple[List[str], List[str], bool]:
        """Drain pending notifications from ARE's notification queue.

        Returns:
            Tuple of ``(user_messages, env_notifications, has_stop_signal)``.
            ``user_messages``: Messages from simulated users.
            ``env_notifications``: System events (new email, calendar reminder, etc.).
            ``has_stop_signal``: True when simulation has ended.

        Raises:
            Any unexpected exception from the notification system propagates
            so that the benchmark runner can classify it via ``fail_on_*`` flags.

        Agent adapters should call this between agent steps and inject
        the messages into the agent's context.
        """
        if self._are_env is None:
            return [], [], False

        notification_system = getattr(self._are_env, "notification_system", None)
        if notification_system is None:
            return [], [], False

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
```

- [ ] **Step 5: Run all notification and lifecycle tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentNotifications tests/interface/environments/test_are_environment.py::TestAREEnvironmentScenarioPath::test_pause_and_resume -v`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): propagate errors from poll_notifications and lifecycle methods

Remove bare except Exception from poll_notifications — errors on
the data path must propagate so the benchmark runner can classify
them and respect fail_on_task_error / fail_on_setup_error settings.
pause() and resume_with_offset() already propagate (keep as-is).
cleanup() keeps try/except (teardown only)."
```

### Task 5: Add AUI tool filtering and get_turn_notifications to AREEnvironment

**Files:**
- Modify: `maseval/interface/environments/are.py`
- Test: `tests/interface/environments/test_are_environment.py`

**AREISSUES.md ref:** Issue #5 (AUI filtering), Issue #7 (get_turn_notifications)

- [ ] **Step 1: Write failing test — AUI tools filtered when opt-in enabled**

```python
class TestAREEnvironmentAUIFiltering:
    """Tests for AUI tool filtering."""

    @patch("maseval.interface.environments.are._import_are")
    def test_aui_tools_filtered_when_enabled(self, mock_import):
        """AUI message-retrieval tools are excluded when filter_aui_tools=True."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        # Create app with AUI tools and a normal tool
        aui_app = MagicMock()
        aui_app.name = "AgentUserInterface"
        aui_tool_get = MagicMock()
        aui_tool_get.name = "AgentUserInterface__get_last_message_from_user"
        aui_tool_send = MagicMock()
        aui_tool_send.name = "AgentUserInterface__send_message_to_user"
        aui_tool_send.description = "Send message"
        aui_tool_send.inputs = {}
        aui_tool_send.output_type = "string"
        aui_tool_send.args = []
        aui_app.get_tools.return_value = [aui_tool_get, aui_tool_send]

        email_app = MagicMock()
        email_app.name = "EmailClient"
        email_tool = MagicMock()
        email_tool.name = "EmailClient__send_email"
        email_tool.description = "Send email"
        email_tool.inputs = {"to": {"type": "string"}}
        email_tool.output_type = "string"
        email_tool.args = []
        email_app.get_tools.return_value = [email_tool]

        mock_are_env = MagicMock()
        mock_are_env.apps = {"AgentUserInterface": aui_app, "EmailClient": email_app}
        mock_are_env.current_time = 0.0
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario}, filter_aui_tools=True)

        tools = env.get_tools()
        assert "AgentUserInterface__get_last_message_from_user" not in tools
        assert "AgentUserInterface__send_message_to_user" in tools
        assert "EmailClient__send_email" in tools

    @patch("maseval.interface.environments.are._import_are")
    def test_aui_tools_not_filtered_by_default(self, mock_import):
        """AUI tools are included by default (filter_aui_tools=False)."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        aui_app = MagicMock()
        aui_app.name = "AgentUserInterface"
        aui_tool = MagicMock()
        aui_tool.name = "AgentUserInterface__get_last_message_from_user"
        aui_tool.description = "Get message"
        aui_tool.inputs = {}
        aui_tool.output_type = "string"
        aui_tool.args = []
        aui_app.get_tools.return_value = [aui_tool]

        mock_are_env = MagicMock()
        mock_are_env.apps = {"AgentUserInterface": aui_app}
        mock_are_env.current_time = 0.0
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(task_data={"scenario": scenario})

        tools = env.get_tools()
        assert "AgentUserInterface__get_last_message_from_user" in tools
```

- [ ] **Step 2: Write failing test — get_turn_notifications re-queues env notifications**

```python
class TestAREEnvironmentTurnNotifications:
    """Tests for get_turn_notifications."""

    @patch("maseval.interface.environments.are._import_are")
    def test_get_turn_notifications_requeues_env_notifications(self, mock_import):
        """get_turn_notifications separates user messages and re-queues env notifications."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        # Mock notification system with MessageType enum
        mock_message_type = MagicMock()
        mock_message_type.USER_MESSAGE = "user"
        mock_message_type.ENVIRONMENT_NOTIFICATION = "env"
        mock_message_type.ENVIRONMENT_STOP = "stop"

        user_notif = MagicMock()
        user_notif.message_type = mock_message_type.USER_MESSAGE
        user_notif.message = "Hello agent"

        env_notif = MagicMock()
        env_notif.message_type = mock_message_type.ENVIRONMENT_NOTIFICATION
        env_notif.message = "New email arrived"

        mock_notif_sys = MagicMock()
        mock_notif_sys.message_queue.get_by_timestamp.return_value = [user_notif, env_notif]
        mock_are_env.notification_system = mock_notif_sys

        scenario = _make_mock_scenario()

        with patch("maseval.interface.environments.are.MessageType", mock_message_type):
            env = AREEnvironment(task_data={"scenario": scenario})
            user_msgs, has_env, has_stop = env.get_turn_notifications()

        assert user_msgs == ["Hello agent"]
        assert has_env is True
        assert has_stop is False
        # Env notification was re-queued
        mock_notif_sys.message_queue.put.assert_called_once_with(env_notif)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentAUIFiltering tests/interface/environments/test_are_environment.py::TestAREEnvironmentTurnNotifications -v`

Expected: FAIL — `filter_aui_tools` parameter doesn't exist, `get_turn_notifications` doesn't exist.

- [ ] **Step 4: Implement AUI filtering and get_turn_notifications**

In `maseval/interface/environments/are.py`:

Add `filter_aui_tools` parameter to `__init__`:

```python
    # Tools removed by ARE's remove_aui_irrelevant_tools()
    # ARE agents/default_agent/are_simulation_main.py:206-228
    _AUI_TOOLS_TO_REMOVE = {
        "AgentUserInterface__get_last_message_from_user",
        "AgentUserInterface__get_last_message_from_agent",
        "AgentUserInterface__get_last_unread_messages",
        "AgentUserInterface__get_all_messages",
    }

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[EnvironmentCallback]] = None,
        run_oracle: bool = False,
        notification_verbosity: str = "medium",
        filter_aui_tools: bool = False,
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
            filter_aui_tools: If True, remove AgentUserInterface message-retrieval
                tools and set ``wait_for_user_response = False``, matching ARE's
                default agent behavior. Required when using notification-based
                message delivery.
        """
        self._run_oracle = run_oracle
        self._notification_verbosity = notification_verbosity
        self._filter_aui_tools = filter_aui_tools
        self._are_env: Any = None
        self._scenario: Any = None
        self._oracle_traces: Optional[Dict[str, Any]] = None
        self._tool_wrappers: Dict[str, AREToolWrapper] = {}

        super().__init__(task_data, callbacks)
```

Update `create_tools` to support AUI filtering:

```python
    def create_tools(self) -> Dict[str, AREToolWrapper]:
        """Wrap all ARE app tools in AREToolWrapper.

        When ``filter_aui_tools=True``, removes AgentUserInterface
        message-retrieval tools and sets ``wait_for_user_response = False``,
        matching ARE's ``remove_aui_irrelevant_tools()``.

        Returns:
            Dict mapping tool names to AREToolWrapper instances.
        """
        tools: Dict[str, AREToolWrapper] = {}

        if self._are_env is None:
            return tools

        for app in self._are_env.apps.values():
            if self._filter_aui_tools and hasattr(app, "wait_for_user_response"):
                app.wait_for_user_response = False

            for are_tool in app.get_tools():
                if self._filter_aui_tools and are_tool.name in self._AUI_TOOLS_TO_REMOVE:
                    continue
                wrapper = AREToolWrapper(are_tool, self)
                tools[are_tool.name] = wrapper
                self._tool_wrappers[are_tool.name] = wrapper

        return tools
```

Add `get_turn_notifications` and convenience accessors:

```python
    def get_turn_notifications(self) -> Tuple[List[str], bool, bool]:
        """Get notifications for turn transitions, re-queuing env notifications.

        Drains the notification queue, separates by type, re-queues environment
        notifications (so the inner loop's pre-step picks them up), and returns
        user messages and status flags.

        Matches ARE's ``get_notifications()`` in ``are_simulation_main.py:331-359``.

        Returns:
            Tuple of ``(user_messages, has_env_notifications, has_stop)``.

        Raises:
            Any unexpected exception from the notification system propagates.
        """
        if self._are_env is None:
            return [], False, False

        notification_system = getattr(self._are_env, "notification_system", None)
        if notification_system is None:
            return [], False, False

        from datetime import datetime, timezone
        from are.simulation.notification_system import MessageType  # type: ignore[import-not-found]

        sim_time = self.get_simulation_time()
        timestamp = datetime.fromtimestamp(sim_time, tz=timezone.utc)
        unhandled = notification_system.message_queue.get_by_timestamp(timestamp=timestamp)

        if not unhandled:
            return [], False, False

        user_messages: List[str] = []
        has_env = False
        has_stop = False

        for notif in unhandled:
            msg_type = getattr(notif, "message_type", None)
            if msg_type == MessageType.USER_MESSAGE:
                user_messages.append(notif.message)
            elif msg_type == MessageType.ENVIRONMENT_NOTIFICATION:
                notification_system.message_queue.put(notif)
                has_env = True
            elif msg_type == MessageType.ENVIRONMENT_STOP:
                has_stop = True

        return user_messages, has_env, has_stop

    def get_scenario(self) -> Any:
        """Get the ARE scenario object."""
        return self._scenario

    def get_start_time(self) -> Optional[float]:
        """Get the scenario start time.

        Returns:
            Start time as Unix timestamp, or None if not available.
        """
        return self.state.get("start_time")

    def get_notification_system(self) -> Any:
        """Get the ARE notification system.

        Returns:
            ARE NotificationSystem instance, or None if not available.
        """
        if self._are_env is None:
            return None
        return getattr(self._are_env, "notification_system", None)
```

- [ ] **Step 5: Run all AREEnvironment tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/test_are_environment.py -v`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "feat(are): add AUI tool filtering and get_turn_notifications

Add filter_aui_tools parameter to AREEnvironment for notification-
based message delivery. Add get_turn_notifications() for multi-turn
agent loops. Add get_scenario(), get_start_time(), and
get_notification_system() convenience accessors."
```

### Task 6: Run full ARE test suite to verify Phase 1

**Files:**
- Test: `tests/interface/environments/test_are_environment.py`

- [ ] **Step 1: Run all ARE tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/ -v`

Expected: All PASS.

- [ ] **Step 2: Run linter**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run ruff check maseval/interface/environments/`

Expected: No errors.

---

## Phase 2: Gaia2 Simplification

### Task 7: Make Gaia2Environment subclass AREEnvironment

**Files:**
- Modify: `maseval/benchmark/gaia2/environment.py`
- Test: `tests/test_benchmarks/test_gaia2/test_environment.py` (run existing tests, no changes yet)

- [ ] **Step 1: Rewrite Gaia2Environment to subclass AREEnvironment**

Replace `maseval/benchmark/gaia2/environment.py` with:

```python
"""Gaia2 Benchmark - Environment.

MASEval Environment wrapping ARE's simulation.

Original Repository: https://github.com/facebookresearch/meta-agents-research-environments
Code License: MIT

Citation:
    Froger, R., Benhalloum, A., Rusakov, A., et al. (2026). Gaia2: Benchmarking LLM
    Agents on Dynamic and Asynchronous Environments. ICLR 2026.
    https://openreview.net/forum?id=9gw03JpKK4
"""

from typing import Any, Dict, List, Optional

from maseval.interface.environments.are import AREEnvironment
from maseval.interface.environments.are_tool_wrapper import AREToolWrapper


class Gaia2Environment(AREEnvironment):
    """GAIA2 benchmark environment built on AREEnvironment.

    Extends AREEnvironment with GAIA2-specific setup:
    - Delegates to ARE's ``preprocess_scenario()`` for oracle run, judge
      creation, and turn initialization
    - Configures custom judge engine for semantic comparison
    - Filters AUI tools (notification-based message delivery)

    Inherits from AREEnvironment:
    - Tool wrapping with simulation time tracking
    - Notification polling (poll_notifications, get_turn_notifications)
    - Lifecycle control (pause, resume_with_offset, cleanup)
    - Tracing and configuration gathering
    """

    def __init__(
        self,
        task_data: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        judge_engine_config: Optional[Any] = None,
    ):
        """Initialize Gaia2 environment.

        Args:
            task_data: Task data containing:
                - scenario: ARE BenchmarkScenario object
                - capability: Capability type (execution, search, etc.)
                - universe_id: Universe identifier
            callbacks: Optional callbacks
            judge_engine_config: Optional :class:`Gaia2JudgeEngineConfig` controlling
                which LLM model and provider the ARE judge uses for semantic comparison.
        """
        self._judge_engine_config = judge_engine_config
        # Gaia2 always uses notification-based delivery, so filter AUI tools
        super().__init__(
            task_data,
            callbacks=callbacks,
            filter_aui_tools=True,
            notification_verbosity="medium",
        )

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize ARE scenario using preprocess_scenario().

        Delegates to ARE's ``preprocess_scenario()`` for faithful preprocessing:
        SystemApp insertion, duration setting, initialization, oracle run,
        soft reset, judge creation, and turn initialization.

        Args:
            task_data: Task data with scenario, capability, universe_id

        Returns:
            State dictionary with scenario metadata
        """
        try:
            from are.simulation.environment import Environment as AREEnv  # type: ignore[import-not-found]
            from are.simulation.environment import EnvironmentConfig  # type: ignore[import-not-found]
            from are.simulation.notification_system import VerboseNotificationSystem  # type: ignore[import-not-found]
            from are.simulation.scenarios.scenario_imported_from_json.utils import (  # type: ignore[import-not-found]
                get_scenario_duration,
                preprocess_scenario,
            )
            from are.simulation.validation import GraphPerEventJudgeConfig  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ARE (Agent Research Environments) is required for Gaia2 benchmark.\n"
                "Install with: pip install meta-agents-research-environments\n"
                "Or: uv add --optional gaia2 meta-agents-research-environments"
            ) from e

        from are.simulation.scenarios.config import (  # type: ignore[import-not-found]
            MAX_SCENARIO_DURATION,
            MAX_TIME_SCENARIO_DURATION,
        )

        scenario = task_data.get("scenario")
        if scenario is None:
            raise ValueError("Task data must contain 'scenario' with ARE BenchmarkScenario")

        max_duration = get_scenario_duration(scenario, MAX_TIME_SCENARIO_DURATION, MAX_SCENARIO_DURATION)

        # Build judge config
        if self._judge_engine_config is not None:
            from are.simulation.agents.are_simulation_agent_config import (  # type: ignore[import-not-found]
                LLMEngineConfig,
            )
            from are.simulation.validation.configs import create_judge_engine  # type: ignore[import-not-found]

            llm_engine_config = LLMEngineConfig(
                model_name=self._judge_engine_config.model_name,
                provider=self._judge_engine_config.provider,
                endpoint=self._judge_engine_config.endpoint,
            )
            engine = create_judge_engine(llm_engine_config)
            judge_config = GraphPerEventJudgeConfig(engine=engine)
        else:
            judge_config = GraphPerEventJudgeConfig()

        preprocess_scenario(
            scenario=scenario,
            judge_config=judge_config,
            max_scenario_duration=max_duration,
        )

        # Create ARE environment and start simulation
        config = EnvironmentConfig(
            oracle_mode=False,
            duration=scenario.duration,
            time_increment_in_seconds=scenario.time_increment_in_seconds,
        )
        if scenario.start_time and scenario.start_time > 0:
            config.start_time = scenario.start_time

        self._are_env = AREEnv(config, notification_system=VerboseNotificationSystem())
        self._scenario = scenario
        self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

        return {
            "scenario_id": getattr(scenario, "scenario_id", None),
            "duration": scenario.duration,
            "capability": task_data.get("capability"),
            "universe_id": task_data.get("universe_id"),
            "start_time": getattr(scenario, "start_time", None),
        }

    def gather_traces(self) -> Dict[str, Any]:
        """Collect traces with GAIA2-specific fields."""
        traces = super().gather_traces()
        traces["capability"] = self.state.get("capability")
        traces["universe_id"] = self.state.get("universe_id")
        return traces

    def gather_config(self) -> Dict[str, Any]:
        """Gather config with GAIA2-specific fields."""
        config = super().gather_config()
        config["capability"] = self.state.get("capability")
        config["universe_id"] = self.state.get("universe_id")
        return config
```

- [ ] **Step 2: Run the existing Gaia2 environment tests (no changes to tests)**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/test_benchmarks/test_gaia2/test_environment.py -v`

Expected: Most tests PASS. Some may fail due to import path changes (`Gaia2GenericTool` no longer exists as the tool type). Note which tests fail — they will be fixed in Task 8.

- [ ] **Step 3: Commit (WIP — tests may not all pass yet)**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add maseval/benchmark/gaia2/environment.py
git commit -m "refactor(gaia2): make Gaia2Environment subclass AREEnvironment

WIP — Gaia2Environment now inherits tool wrapping, notification
polling, lifecycle control, and cleanup from AREEnvironment.
Only setup_state (preprocess_scenario + judge) and GAIA2-specific
gather_traces/gather_config fields remain as overrides."
```

### Task 8: Update Gaia2 tests and delete Gaia2GenericTool

**Files:**
- Delete: `maseval/benchmark/gaia2/tool_wrapper.py`
- Modify: `tests/test_benchmarks/test_gaia2/test_tool_wrapper.py`
- Modify: `tests/test_benchmarks/test_gaia2/test_environment.py`
- Modify: `tests/test_benchmarks/test_gaia2/conftest.py` (if needed)

- [ ] **Step 1: Check what references Gaia2GenericTool in test files**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && grep -rn "Gaia2GenericTool\|gaia2\.tool_wrapper\|from maseval.benchmark.gaia2.tool_wrapper" tests/`

This tells you every import and reference that needs updating.

- [ ] **Step 2: Update test_tool_wrapper.py imports and assertions**

Replace `from maseval.benchmark.gaia2.tool_wrapper import Gaia2GenericTool` with `from maseval.interface.environments.are_tool_wrapper import AREToolWrapper` throughout. Update class instantiation from `Gaia2GenericTool(mock_are_tool, mock_env)` to `AREToolWrapper(mock_are_tool, mock_env)`. Update assertion class names in type checks.

The test structure stays the same — these tests now validate AREToolWrapper behavior with the same expectations that were validated against Gaia2GenericTool.

- [ ] **Step 3: Update test_environment.py imports**

Replace any `Gaia2GenericTool` references with `AREToolWrapper`. The environment tests that check tool types (e.g., `isinstance(tool, Gaia2GenericTool)`) should check `isinstance(tool, AREToolWrapper)`.

- [ ] **Step 4: Update conftest.py if it references Gaia2GenericTool**

Check and update any fixtures that import or reference `Gaia2GenericTool`.

- [ ] **Step 5: Check for other references across the codebase**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && grep -rn "Gaia2GenericTool\|gaia2\.tool_wrapper" maseval/ --include="*.py"`

Update any remaining references (benchmark.py, __init__.py, etc.).

- [ ] **Step 6: Delete Gaia2GenericTool**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
rm maseval/benchmark/gaia2/tool_wrapper.py
```

- [ ] **Step 7: Run all Gaia2 tests**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/test_benchmarks/test_gaia2/ -v`

Expected: All PASS. If any fail, fix the specific import or assertion and re-run.

- [ ] **Step 8: Commit**

```bash
cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman
git add -A
git commit -m "refactor(gaia2): delete Gaia2GenericTool, use AREToolWrapper

Gaia2GenericTool is functionally identical to AREToolWrapper now
that AREToolWrapper has simulation time tracking and AppToolAdapter.
All Gaia2 tool wrapper tests updated to use AREToolWrapper directly."
```

### Task 9: Final verification — full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all ARE and Gaia2 tests together**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest tests/interface/environments/ tests/test_benchmarks/test_gaia2/ -v`

Expected: All PASS.

- [ ] **Step 2: Run linter on all changed files**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run ruff check maseval/interface/environments/ maseval/benchmark/gaia2/`

Expected: No errors.

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `cd /Users/cornelius/Repositories/maseval/.claude/worktrees/objective-raman && uv run pytest --tb=short -q`

Expected: No regressions outside of ARE/Gaia2 tests.
