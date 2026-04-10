# ARE Integration Quality Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix code quality issues in the ARE integration: add discoverable classmethods, remove invented defaults, add proper types, fix silent failures, separate side effects, and simplify tool wrapper duplication.

**Architecture:** All changes are to `maseval/interface/environments/are.py` and `maseval/interface/environments/are_tool_wrapper.py`. The base `Environment.__init__` requires `environment_data: Dict[str, Any]`, so classmethods build that dict internally. `Gaia2Environment` (the primary subclass) must not break — it overrides `setup_state()` completely and inherits `create_tools()`, lifecycle methods, and accessors.

**Tech Stack:** Python 3.10+, ARE (`meta-agents-research-environments`), pytest, uv

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `maseval/interface/environments/are.py` | Modify | Add classmethods, remove defaults, add types, fix lifecycle, separate AUI mutation, log cleanup errors |
| `maseval/interface/environments/are_tool_wrapper.py` | Modify | Replace `_extract_schema` with `AppTool.to_open_ai()`, add `TYPE_CHECKING` types |
| `tests/interface/environments/test_are_environment.py` | Modify | Add classmethod tests, update shorthand tests (required params), update lifecycle tests (raises), update AUI tests (separate method) |
| `tests/interface/environments/test_are_tool_wrapper.py` | Modify | Update schema tests for new approach |
| `tests/interface/environments/test_are_integration.py` | Modify | Add classmethod integration tests, update shorthand test |

**Files that must NOT change:** `maseval/benchmark/gaia2/environment.py`, `maseval/core/environment.py`

---

### Task 1: Add `TYPE_CHECKING` imports and typed attributes to `AREEnvironment`

Adds proper types to `_are_env`, `_scenario`, and all accessor return types so developers get IDE autocomplete. Uses `TYPE_CHECKING` (already used in `are_tool_wrapper.py`) so there's no runtime dependency on ARE being installed.

**Files:**
- Modify: `maseval/interface/environments/are.py:1-15` (imports), `:69-100` (`__init__`), `:400-427` (accessors)

- [ ] **Step 1: Add TYPE_CHECKING imports and update attribute types**

In `maseval/interface/environments/are.py`, add `TYPE_CHECKING` imports at the top and update the type annotations on `_are_env` and `_scenario`:

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from are.simulation.environment import Environment as _AREEnv  # type: ignore[import-not-found]
    from are.simulation.scenarios.scenario import Scenario as _Scenario  # type: ignore[import-not-found]
    from are.simulation.notification_system import VerboseNotificationSystem as _NotificationSystem  # type: ignore[import-not-found]
```

In `__init__`, change:
```python
# Before
self._are_env: Any = None
self._scenario: Any = None

# After
self._are_env: Optional["_AREEnv"] = None
self._scenario: Optional["_Scenario"] = None
```

Update accessor return types:
```python
# Before
def get_are_environment(self) -> Any:
def get_scenario(self) -> Any:
def get_notification_system(self) -> Any:

# After
def get_are_environment(self) -> Optional["_AREEnv"]:
def get_scenario(self) -> Optional["_Scenario"]:
def get_notification_system(self) -> Optional["_NotificationSystem"]:
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `uv run pytest tests/interface/environments/test_are_environment.py tests/interface/environments/test_are_tool_wrapper.py -v`
Expected: All existing tests PASS (type changes are annotation-only at runtime)

- [ ] **Step 3: Commit**

```bash
git add maseval/interface/environments/are.py
git commit -m "refactor(are): add TYPE_CHECKING types for IDE autocomplete"
```

---

### Task 2: Add `from_scenario` and `from_apps` classmethods

Provides two discoverable construction paths with typed parameters and IDE autocomplete. Both classmethods build `environment_data` dicts internally and call `__init__`. The existing `__init__(environment_data=...)` path remains unchanged for backward compatibility and for subclasses like `Gaia2Environment`.

**Files:**
- Modify: `maseval/interface/environments/are.py:50-100` (class body, after `__init__`)
- Test: `tests/interface/environments/test_are_environment.py`
- Test: `tests/interface/environments/test_are_integration.py`

- [ ] **Step 1: Write failing tests for `from_scenario`**

Add a new test class in `tests/interface/environments/test_are_environment.py`:

```python
class TestAREEnvironmentClassmethods:
    """Tests for from_scenario and from_apps classmethods."""

    @patch("maseval.interface.environments.are._import_are")
    def test_from_scenario_creates_environment(self, mock_import):
        """from_scenario() constructs environment from an ARE Scenario."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment.from_scenario(scenario)

        assert env.state["scenario_id"] == "test-001"
        assert env.state["duration"] == 600
        assert env.state["seed"] == 42
        assert env._are_env is mock_are_env

    @patch("maseval.interface.environments.are._import_are")
    def test_from_scenario_passes_options(self, mock_import):
        """from_scenario() forwards run_oracle, notification_verbosity, filter_aui_tools."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment.from_scenario(
            scenario,
            run_oracle=False,
            notification_verbosity="high",
            filter_aui_tools=True,
        )

        assert env._run_oracle is False
        assert env._notification_verbosity == "high"
        assert env._filter_aui_tools is True
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentClassmethods -v`
Expected: FAIL — `AttributeError: type object 'AREEnvironment' has no attribute 'from_scenario'`

- [ ] **Step 2: Implement `from_scenario`**

Add to `AREEnvironment` class in `maseval/interface/environments/are.py`, after the `__init__` method:

```python
@classmethod
def from_scenario(
    cls,
    scenario: "Any",
    callbacks: Optional[List[EnvironmentCallback]] = None,
    run_oracle: bool = False,
    notification_verbosity: str = "medium",
    filter_aui_tools: bool = False,
) -> "AREEnvironment":
    """Create AREEnvironment from a pre-built ARE Scenario.

    Args:
        scenario: ARE Scenario object (from ``are.simulation.scenarios``).
        callbacks: Optional maseval EnvironmentCallbacks.
        run_oracle: If True, run ARE oracle mode during setup.
        notification_verbosity: ``"low"``, ``"medium"``, or ``"high"``.
        filter_aui_tools: If True, exclude AUI message-retrieval tools.

    Returns:
        Configured AREEnvironment instance.
    """
    return cls(
        environment_data={"scenario": scenario},
        callbacks=callbacks,
        run_oracle=run_oracle,
        notification_verbosity=notification_verbosity,
        filter_aui_tools=filter_aui_tools,
    )
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentClassmethods -v`
Expected: PASS for `from_scenario` tests, FAIL for `from_apps` (not yet written)

- [ ] **Step 3: Write failing tests for `from_apps`**

Add to `TestAREEnvironmentClassmethods` in `tests/interface/environments/test_are_environment.py`:

```python
    @patch("maseval.interface.environments.are._import_are")
    @patch("maseval.interface.environments.are.AREEnvironment._build_scenario_from_shorthand")
    def test_from_apps_creates_environment(self, mock_build, mock_import):
        """from_apps() constructs environment from apps list + required params."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_scenario = _make_mock_scenario()
        mock_build.return_value = mock_scenario
        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        apps = [MagicMock(), MagicMock()]
        env = AREEnvironment.from_apps(
            apps=apps,
            duration=120,
            seed=7,
        )

        assert env._scenario is mock_scenario
        # Verify environment_data was built correctly
        call_args = mock_build.call_args[0][0]
        assert call_args["apps"] is apps
        assert call_args["duration"] == 120
        assert call_args["seed"] == 7

    @patch("maseval.interface.environments.are._import_are")
    @patch("maseval.interface.environments.are.AREEnvironment._build_scenario_from_shorthand")
    def test_from_apps_passes_optional_params(self, mock_build, mock_import):
        """from_apps() forwards optional scenario and environment params."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_scenario = _make_mock_scenario()
        mock_build.return_value = mock_scenario
        mock_are_env = _make_mock_are_env()
        mock_are_mod.Environment.return_value = mock_are_env

        apps = [MagicMock()]
        events = [MagicMock()]
        env = AREEnvironment.from_apps(
            apps=apps,
            duration=60,
            seed=42,
            events=events,
            start_time=100,
            time_increment_in_seconds=5,
            scenario_id="my-test",
            notification_verbosity="high",
        )

        call_args = mock_build.call_args[0][0]
        assert call_args["events"] is events
        assert call_args["start_time"] == 100
        assert call_args["time_increment_in_seconds"] == 5
        assert call_args["scenario_id"] == "my-test"
        assert env._notification_verbosity == "high"
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentClassmethods::test_from_apps_creates_environment -v`
Expected: FAIL — `AttributeError: type object 'AREEnvironment' has no attribute 'from_apps'`

- [ ] **Step 4: Implement `from_apps`**

Add to `AREEnvironment` class, after `from_scenario`:

```python
@classmethod
def from_apps(
    cls,
    apps: "List[Any]",
    duration: int,
    seed: int,
    events: "Optional[List[Any]]" = None,
    start_time: float = 0,
    time_increment_in_seconds: int = 1,
    scenario_id: str = "custom",
    callbacks: Optional[List[EnvironmentCallback]] = None,
    run_oracle: bool = False,
    notification_verbosity: str = "medium",
    filter_aui_tools: bool = False,
) -> "AREEnvironment":
    """Create AREEnvironment from ARE app instances and explicit config.

    Args:
        apps: List of ARE App instances (e.g. ``[CalendarApp(), ContactsApp()]``).
        duration: Scenario duration in seconds. Required — no default.
        seed: Random seed for reproducibility. Required — no default.
        events: Optional list of ARE events to schedule.
        start_time: Simulation start time in seconds.
        time_increment_in_seconds: Fixed tick interval (>= 1 second).
        scenario_id: Identifier for the scenario.
        callbacks: Optional maseval EnvironmentCallbacks.
        run_oracle: If True, run ARE oracle mode during setup.
        notification_verbosity: ``"low"``, ``"medium"``, or ``"high"``.
        filter_aui_tools: If True, exclude AUI message-retrieval tools.

    Returns:
        Configured AREEnvironment instance.
    """
    environment_data: Dict[str, Any] = {
        "apps": apps,
        "duration": duration,
        "seed": seed,
        "start_time": start_time,
        "time_increment_in_seconds": time_increment_in_seconds,
        "scenario_id": scenario_id,
    }
    if events is not None:
        environment_data["events"] = events
    return cls(
        environment_data=environment_data,
        callbacks=callbacks,
        run_oracle=run_oracle,
        notification_verbosity=notification_verbosity,
        filter_aui_tools=filter_aui_tools,
    )
```

- [ ] **Step 5: Run all classmethod tests**

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentClassmethods -v`
Expected: All PASS

- [ ] **Step 6: Add integration test for `from_apps`**

Add to `tests/interface/environments/test_are_integration.py`, inside `TestAREEnvironmentLifecycle`:

```python
    def test_from_apps_classmethod(self):
        """from_apps() creates a working environment with real ARE apps."""
        from maseval.interface.environments.are import AREEnvironment

        apps = [CalendarApp(), ContactsApp(), SystemApp()]
        env = AREEnvironment.from_apps(
            apps=apps,
            duration=30,
            seed=99,
        )
        try:
            assert env.state["duration"] == 30
            assert env.state["seed"] == 99
            assert len(env.tools) > 0
        finally:
            env.cleanup()
```

Run: `uv run pytest tests/interface/environments/test_are_integration.py::TestAREEnvironmentLifecycle::test_from_apps_classmethod -v`
Expected: PASS (or SKIP if ARE not installed)

- [ ] **Step 7: Update class docstring**

Update the `AREEnvironment` class docstring in `maseval/interface/environments/are.py`:

```python
class AREEnvironment(Environment):
    """Generic maseval Environment wrapping ARE's simulation infrastructure.

    Two construction paths:

    1. **From a Scenario:** ``AREEnvironment.from_scenario(scenario)``
    2. **From apps:** ``AREEnvironment.from_apps(apps=[...], duration=60, seed=42)``

    Both paths also accept ``run_oracle``, ``notification_verbosity``,
    ``filter_aui_tools``, and ``callbacks``.

    The raw ``AREEnvironment(environment_data=...)`` constructor is available
    for subclasses that need full control over ``setup_state()``.

    Lifecycle is user-controlled: call ``start()`` before ``run_agents()``,
    ``stop()`` after. ``pause()``/``resume_with_offset()`` control simulation time.
    """
```

- [ ] **Step 8: Run full test suite and commit**

Run: `uv run pytest tests/interface/environments/ -v`
Expected: All PASS

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py tests/interface/environments/test_are_integration.py
git commit -m "feat(are): add from_scenario() and from_apps() classmethods for discoverable construction"
```

---

### Task 3: Remove invented defaults in `_build_scenario_from_shorthand`

Makes `duration` and `seed` required in the shorthand path. When called via `from_apps()`, these are already required parameters. When called via raw `__init__` with `environment_data`, missing keys now raise `KeyError` instead of silently inventing values.

**Files:**
- Modify: `maseval/interface/environments/are.py:155-185` (`_build_scenario_from_shorthand`)
- Modify: `tests/interface/environments/test_are_environment.py` (update shorthand tests)

- [ ] **Step 1: Write failing test for missing required params**

Add to `TestAREEnvironmentShorthandPath` in `tests/interface/environments/test_are_environment.py`:

```python
    @patch("maseval.interface.environments.are._import_are")
    def test_shorthand_requires_duration(self, mock_import):
        """Shorthand path raises KeyError when duration is missing."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        with pytest.raises(KeyError, match="duration"):
            AREEnvironment(environment_data={"apps": [MagicMock()], "seed": 1})

    @patch("maseval.interface.environments.are._import_are")
    def test_shorthand_requires_seed(self, mock_import):
        """Shorthand path raises KeyError when seed is missing."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        with pytest.raises(KeyError, match="seed"):
            AREEnvironment(environment_data={"apps": [MagicMock()], "duration": 60})
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentShorthandPath::test_shorthand_requires_duration -v`
Expected: FAIL — test expects `KeyError` but current code silently defaults

- [ ] **Step 2: Remove defaults from `_build_scenario_from_shorthand`**

In `maseval/interface/environments/are.py`, change `_build_scenario_from_shorthand`:

```python
def _build_scenario_from_shorthand(self, environment_data: Dict[str, Any]) -> Any:
    """Build an ARE Scenario from shorthand environment_data.

    Args:
        environment_data: Dict with ``"apps"`` (required), ``"duration"`` (required),
            ``"seed"`` (required), and optional ``"events"``, ``"start_time"``,
            ``"time_increment_in_seconds"``, ``"scenario_id"``.

    Returns:
        ARE Scenario instance.

    Raises:
        KeyError: If ``"apps"``, ``"duration"``, or ``"seed"`` is missing.
    """
    from are.simulation.scenarios.scenario import Scenario  # type: ignore[import-not-found]

    apps = environment_data["apps"]
    duration = environment_data["duration"]
    seed = environment_data["seed"]
    events = environment_data.get("events", [])
    start_time = environment_data.get("start_time", 0)
    time_increment = environment_data.get("time_increment_in_seconds", 1)

    scenario = Scenario(
        scenario_id=environment_data.get("scenario_id", "custom"),  # ty: ignore[unknown-argument]
        apps=apps,  # ty: ignore[unknown-argument]
        events=events,  # ty: ignore[unknown-argument]
        duration=duration,  # ty: ignore[unknown-argument]
        seed=seed,  # ty: ignore[unknown-argument]
        start_time=start_time,  # ty: ignore[unknown-argument]
        time_increment_in_seconds=time_increment,  # ty: ignore[unknown-argument]
    )
    scenario.initialize()
    return scenario
```

The key changes: `environment_data["duration"]` and `environment_data["seed"]` instead of `.get(..., default)`.

Note: `start_time=0` and `time_increment_in_seconds=1` are infrastructure defaults (not experimental parameters) — these are safe to keep. `events=[]` (empty list) is also safe — it means "no pre-scheduled events." `scenario_id="custom"` is a label, not an experimental parameter.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentShorthandPath -v`
Expected: All PASS

- [ ] **Step 4: Update integration test to always pass required params**

In `tests/interface/environments/test_are_integration.py`, the existing shorthand test at `test_shorthand_path_creates_environment` already passes `duration` and `seed`, so no change needed. Verify:

Run: `uv run pytest tests/interface/environments/test_are_integration.py::TestAREEnvironmentLifecycle::test_shorthand_path_creates_environment -v`
Expected: PASS (or SKIP if ARE not installed)

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): require duration and seed in shorthand path, no invented defaults"
```

---

### Task 4: Remove `getattr` fallbacks in `setup_state`

Access scenario attributes directly so `AttributeError` surfaces immediately if something is wrong, instead of silently using fallback values.

**Files:**
- Modify: `maseval/interface/environments/are.py:130-153` (`setup_state`), `:200-205` (`_run_oracle_mode`)

- [ ] **Step 1: Write failing test for direct attribute access**

The existing test `test_setup_state_with_scenario` already uses a mock scenario that has all attributes. Add a test that verifies `AttributeError` surfaces when an attribute is missing:

Add to `TestAREEnvironmentScenarioPath` in `tests/interface/environments/test_are_environment.py`:

```python
    @patch("maseval.interface.environments.are._import_are")
    def test_setup_state_raises_on_missing_scenario_attributes(self, mock_import):
        """setup_state raises AttributeError if scenario lacks expected attributes."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        # Create scenario missing time_increment_in_seconds
        scenario = MagicMock(spec=[])
        scenario.duration = 600
        scenario.apps = []
        # Deliberately missing: time_increment_in_seconds, scenario_id, seed, start_time

        with pytest.raises(AttributeError):
            AREEnvironment(environment_data={"scenario": scenario})
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentScenarioPath::test_setup_state_raises_on_missing_scenario_attributes -v`
Expected: FAIL — current code uses `getattr` with fallbacks, so no `AttributeError`

- [ ] **Step 2: Replace `getattr` with direct access in `setup_state`**

In `maseval/interface/environments/are.py`, change `setup_state`:

```python
# Before (line 131-137)
config = are_mod.EnvironmentConfig(
    oracle_mode=False,
    duration=scenario.duration,
    time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),
)
if getattr(scenario, "start_time", None) and scenario.start_time > 0:
    config.start_time = scenario.start_time

# After
config = are_mod.EnvironmentConfig(
    oracle_mode=False,
    duration=scenario.duration,
    time_increment_in_seconds=scenario.time_increment_in_seconds,
)
if scenario.start_time and scenario.start_time > 0:
    config.start_time = scenario.start_time
```

And in the state dict return (lines 146-153):

```python
# Before
return {
    "scenario_id": getattr(scenario, "scenario_id", None),
    "duration": scenario.duration,
    "seed": getattr(scenario, "seed", None),
    "start_time": getattr(scenario, "start_time", None),
    "app_names": [getattr(app, "name", str(app)) for app in scenario.apps],
    "oracle_traces": self._oracle_traces,
}

# After
return {
    "scenario_id": scenario.scenario_id,
    "duration": scenario.duration,
    "seed": scenario.seed,
    "start_time": scenario.start_time,
    "app_names": [app.name for app in scenario.apps],
    "oracle_traces": self._oracle_traces,
}
```

- [ ] **Step 3: Replace `getattr` in `_run_oracle_mode`**

In `_run_oracle_mode` (line 204):

```python
# Before
time_increment_in_seconds=getattr(scenario, "time_increment_in_seconds", 1),

# After
time_increment_in_seconds=scenario.time_increment_in_seconds,
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/interface/environments/test_are_environment.py -v`
Expected: All PASS (existing tests use mocks that have all attributes; new test verifies AttributeError)

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): remove getattr fallbacks, let AttributeError surface on bad scenarios"
```

---

### Task 5: Separate AUI mutation from `create_tools`

Extract `app.wait_for_user_response = False` from the tool-iteration loop in `create_tools()` into a dedicated `_configure_aui_apps()` method. The method is still called from `create_tools()` (not `setup_state`), because `Gaia2Environment` overrides `setup_state()` completely without calling `super()` — but inherits `create_tools()`. The goal is clarity: the mutation is now a named, visible step rather than hiding inside a tool-iteration loop.

**Files:**
- Modify: `maseval/interface/environments/are.py:242-263` (`create_tools`)
- Modify: `tests/interface/environments/test_are_environment.py`

- [ ] **Step 1: Write test that _configure_aui_apps is callable independently**

Add to `TestAREEnvironmentAUIFiltering` in `tests/interface/environments/test_are_environment.py`:

```python
    @patch("maseval.interface.environments.are._import_are")
    def test_configure_aui_apps_disables_wait(self, mock_import):
        """_configure_aui_apps() sets wait_for_user_response=False on AUI apps."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod

        aui_app = MagicMock()
        aui_app.name = "AgentUserInterface"
        aui_app.wait_for_user_response = True
        aui_app.get_tools.return_value = []

        email_app = MagicMock()
        email_app.name = "EmailClient"
        del email_app.wait_for_user_response  # non-AUI app has no such attr
        email_app.get_tools.return_value = []

        mock_are_env = MagicMock()
        mock_are_env.apps = {"AgentUserInterface": aui_app, "EmailClient": email_app}
        mock_are_env.current_time = 0.0
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario}, filter_aui_tools=True)

        # AUI app was configured
        assert aui_app.wait_for_user_response is False
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentAUIFiltering::test_configure_aui_apps_disables_wait -v`
Expected: PASS (behavior unchanged, establishing baseline)

- [ ] **Step 2: Extract `_configure_aui_apps` and call from `create_tools`**

In `maseval/interface/environments/are.py`, add a new method:

```python
def _configure_aui_apps(self) -> None:
    """Disable AUI wait_for_user_response when AUI filtering is enabled.

    Separated from ``create_tools()`` for clarity — app mutation is
    a distinct concern from tool wrapping.
    """
    if not self._filter_aui_tools or self._are_env is None:
        return
    for app in self._are_env.apps.values():
        if hasattr(app, "wait_for_user_response"):
            app.wait_for_user_response = False
```

Update `create_tools` to call it at the top, and remove the inline mutation:

```python
def create_tools(self) -> Dict[str, AREToolWrapper]:
    """Wrap all ARE app tools in AREToolWrapper.

    Returns:
        Dict mapping tool names to AREToolWrapper instances.
    """
    tools: Dict[str, AREToolWrapper] = {}

    if self._are_env is None:
        return tools

    self._configure_aui_apps()

    for app in self._are_env.apps.values():
        for are_tool in app.get_tools():
            if self._filter_aui_tools and are_tool.name in self._AUI_TOOLS_TO_REMOVE:
                continue
            wrapper = AREToolWrapper(are_tool, self)
            tools[are_tool.name] = wrapper
            self._tool_wrappers[are_tool.name] = wrapper

    return tools
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentAUIFiltering -v`
Expected: All PASS

Run: `uv run pytest tests/interface/environments/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "refactor(are): extract _configure_aui_apps from create_tools for clarity"
```

---

### Task 6: Simplify `AREToolWrapper` schema extraction

Replace the hand-rolled `_extract_schema()` with `AppTool.to_open_ai()` which already produces the same information. The `to_open_ai()` format nests the schema under `function.parameters`, so we extract that inner part. This removes ~20 lines of duplication.

**Files:**
- Modify: `maseval/interface/environments/are_tool_wrapper.py:53-106`
- Modify: `tests/interface/environments/test_are_tool_wrapper.py`
- Modify: `tests/interface/environments/test_are_environment.py` (remove now-obsolete schema crash test)

- [ ] **Step 1: Write test for new schema extraction behavior**

Add to `TestAREToolWrapper` in `tests/interface/environments/test_are_tool_wrapper.py`:

```python
    def test_input_schema_from_to_open_ai(self):
        """input_schema is extracted from AppTool.to_open_ai() format."""
        are_tool = self._make_mock_are_tool()
        are_tool.to_open_ai.return_value = {
            "type": "function",
            "function": {
                "name": "Calendar__create_event",
                "description": "Create a calendar event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                    },
                    "required": ["title"],
                },
            },
        }
        env = MagicMock()

        wrapper = AREToolWrapper(are_tool, env)

        assert wrapper.input_schema == {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
            },
            "required": ["title"],
        }
```

Run: `uv run pytest tests/interface/environments/test_are_tool_wrapper.py::TestAREToolWrapper::test_input_schema_from_to_open_ai -v`
Expected: FAIL — current `_extract_schema` doesn't use `to_open_ai()`

- [ ] **Step 2: Replace `_extract_schema` with `to_open_ai()` extraction**

In `maseval/interface/environments/are_tool_wrapper.py`, replace:

```python
# Before (lines 75-106)
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
                "type": arg.arg_type,
                "description": getattr(arg, "description", ""),
            }
            if not arg.has_default:
                required.append(param_name)

        return {"properties": properties, "required": required}

# After
        # Extract JSON schema from ARE tool's OpenAI format
        self.input_schema: Dict[str, Any] = self._extract_schema(are_tool)

    @staticmethod
    def _extract_schema(are_tool: Any) -> Dict[str, Any]:
        """Extract JSON schema from ARE tool's OpenAI function calling format.

        Uses ``AppTool.to_open_ai()`` as the canonical schema source.

        Args:
            are_tool: ARE AppTool instance.

        Returns:
            JSON schema dict (the ``parameters`` object from OpenAI format).
        """
        if not hasattr(are_tool, "to_open_ai"):
            return {}
        openai_spec = are_tool.to_open_ai()
        return openai_spec.get("function", {}).get("parameters", {})
```

- [ ] **Step 3: Remove obsolete `test_schema_extraction_crashes_on_missing_arg_type`**

In `tests/interface/environments/test_are_environment.py`, remove the test `test_schema_extraction_crashes_on_missing_arg_type` from `TestAREToolWrapper` — it tested the old hand-rolled extraction which no longer exists.

- [ ] **Step 4: Run all tool wrapper tests**

Run: `uv run pytest tests/interface/environments/test_are_tool_wrapper.py tests/interface/environments/test_are_environment.py::TestAREToolWrapper -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are_tool_wrapper.py tests/interface/environments/test_are_tool_wrapper.py tests/interface/environments/test_are_environment.py
git commit -m "refactor(are): use AppTool.to_open_ai() for schema extraction instead of hand-rolled _extract_schema"
```

---

### Task 7: Fix silent no-ops in lifecycle methods and cleanup

Makes `start()`, `stop()`, `pause()`, `resume_with_offset()` raise `RuntimeError` when `_are_env` is `None`. Makes `cleanup()` log exceptions instead of swallowing them silently.

**Files:**
- Modify: `maseval/interface/environments/are.py:265-295` (lifecycle), `:438-444` (cleanup)
- Modify: `tests/interface/environments/test_are_environment.py`

- [ ] **Step 1: Write failing tests for lifecycle raises**

Add a new test class to `tests/interface/environments/test_are_environment.py`:

```python
class TestAREEnvironmentLifecycleErrors:
    """Tests that lifecycle methods raise on misconfigured environment."""

    @patch("maseval.interface.environments.are._import_are")
    def test_start_raises_when_no_are_env(self, mock_import):
        """start() raises RuntimeError if _are_env is None."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        env._are_env = None

        with pytest.raises(RuntimeError, match="ARE environment not initialized"):
            env.start()

    @patch("maseval.interface.environments.are._import_are")
    def test_stop_raises_when_no_are_env(self, mock_import):
        """stop() raises RuntimeError if _are_env is None."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        env._are_env = None

        with pytest.raises(RuntimeError, match="ARE environment not initialized"):
            env.stop()

    @patch("maseval.interface.environments.are._import_are")
    def test_pause_raises_when_no_are_env(self, mock_import):
        """pause() raises RuntimeError if _are_env is None."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        env._are_env = None

        with pytest.raises(RuntimeError, match="ARE environment not initialized"):
            env.pause()

    @patch("maseval.interface.environments.are._import_are")
    def test_resume_raises_when_no_are_env(self, mock_import):
        """resume_with_offset() raises RuntimeError if _are_env is None."""
        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_mod.Environment.return_value = _make_mock_are_env()

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})
        env._are_env = None

        with pytest.raises(RuntimeError, match="ARE environment not initialized"):
            env.resume_with_offset(5.0)
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentLifecycleErrors -v`
Expected: FAIL — current code silently returns

- [ ] **Step 2: Write test for cleanup logging**

```python
    @patch("maseval.interface.environments.are._import_are")
    def test_cleanup_logs_exception(self, mock_import, caplog):
        """cleanup() logs exceptions instead of swallowing silently."""
        import logging

        mock_are_mod = MagicMock()
        mock_import.return_value = mock_are_mod
        mock_are_env = _make_mock_are_env()
        mock_are_env.stop.side_effect = RuntimeError("stop failed")
        mock_are_mod.Environment.return_value = mock_are_env

        scenario = _make_mock_scenario()
        env = AREEnvironment(environment_data={"scenario": scenario})

        with caplog.at_level(logging.WARNING):
            env.cleanup()  # should not raise

        assert "stop failed" in caplog.text
```

Run: `uv run pytest tests/interface/environments/test_are_environment.py::TestAREEnvironmentLifecycleErrors::test_cleanup_logs_exception -v`
Expected: FAIL — current code has bare `except: pass`

- [ ] **Step 3: Implement lifecycle raises and cleanup logging**

In `maseval/interface/environments/are.py`, add a logger at the top of the file (after imports):

```python
import logging

logger = logging.getLogger(__name__)
```

Replace lifecycle methods:

```python
def start(self) -> None:
    """Start the ARE simulation event loop.

    Call this after environment setup and before running agents.
    Runs the scenario with ``wait_for_end=False`` so control returns
    immediately for agent interaction.

    Raises:
        RuntimeError: If the ARE environment is not initialized.
    """
    if self._are_env is None or self._scenario is None:
        raise RuntimeError("ARE environment not initialized. Cannot start simulation.")
    self._are_env.run(self._scenario, wait_for_end=False, schedule_events=True)

def stop(self) -> None:
    """Stop the ARE simulation event loop.

    Raises:
        RuntimeError: If the ARE environment is not initialized.
    """
    if self._are_env is None:
        raise RuntimeError("ARE environment not initialized. Cannot stop simulation.")
    self._are_env.stop()

def pause(self) -> None:
    """Pause simulation time progression.

    Raises:
        RuntimeError: If the ARE environment is not initialized.
    """
    if self._are_env is None:
        raise RuntimeError("ARE environment not initialized. Cannot pause simulation.")
    self._are_env.pause()

def resume_with_offset(self, offset: float) -> None:
    """Resume simulation with a time offset.

    Args:
        offset: Seconds to advance simulation clock before resuming.

    Raises:
        RuntimeError: If the ARE environment is not initialized.
    """
    if self._are_env is None:
        raise RuntimeError("ARE environment not initialized. Cannot resume simulation.")
    self._are_env.resume_with_offset(offset)
```

Replace cleanup:

```python
def cleanup(self) -> None:
    """Stop ARE simulation. Called by maseval after task completes."""
    if self._are_env is not None:
        try:
            self._are_env.stop()
        except Exception:
            logger.warning("Error during ARE environment cleanup", exc_info=True)
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/interface/environments/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add maseval/interface/environments/are.py tests/interface/environments/test_are_environment.py
git commit -m "fix(are): raise RuntimeError on lifecycle calls without initialized env, log cleanup errors"
```

---

### Task 8: Final verification

Run the full test suite and lint to verify nothing is broken, including Gaia2 tests.

**Files:** None (verification only)

- [ ] **Step 1: Run ARE-specific tests**

Run: `uv run pytest tests/interface/environments/ -v`
Expected: All PASS

- [ ] **Step 2: Run Gaia2 tests (offline only)**

Run: `uv run pytest tests/test_benchmarks/test_gaia2/ -v -m "not credentialed and not live"`
Expected: All PASS (Gaia2Environment inherits updated AREEnvironment methods)

- [ ] **Step 3: Run lint and format**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: Clean

- [ ] **Step 4: Run full default test suite**

Run: `uv run pytest -v`
Expected: All PASS

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: format after ARE quality fixes"
```
