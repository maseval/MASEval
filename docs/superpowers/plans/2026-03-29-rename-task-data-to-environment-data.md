# Rename task_data to environment_data in Tests and Examples

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate confusing `task_data` naming in tests and examples where the value is actually `environment_data`

**Architecture:** Pure rename refactor. No behavioral changes. The production code (base class + all 6 environments + all benchmark instantiations) already uses `environment_data` consistently. This plan fixes tests, fixtures, and examples that still use the old `task_data` naming.

**Tech Stack:** Python, pytest

**Note:** The MACS real_data tests have a **bug** where `{"environment_data": task.environment_data}` wraps environment_data in an extra dict, causing `setup_state` to silently get `tools=[]`. This is fixed as part of the rename.

---

### Task 1: Rename MACS test fixture and local variables

**Files:**
- Modify: `tests/test_benchmarks/test_macs/conftest.py:428-432`
- Modify: `tests/test_benchmarks/test_macs/test_macs_environment.py` (all `sample_task_data` and `task_data` references)
- Modify: `tests/test_benchmarks/test_macs/test_macs_integration.py:136-178` (local `task_data` variables)

- [ ] **Step 1: Rename fixture in conftest.py**

In `tests/test_benchmarks/test_macs/conftest.py`, rename the fixture from `sample_task_data` to `sample_environment_data`:

```python
@pytest.fixture
def sample_environment_data(sample_tool_specs):
    """Sample environment data dict for MACSEnvironment creation."""
    return {
        "tools": sample_tool_specs,
    }
```

- [ ] **Step 2: Rename all references in test_macs_environment.py**

In `tests/test_benchmarks/test_macs/test_macs_environment.py`, replace all occurrences of `sample_task_data` with `sample_environment_data`, and all local variables named `task_data` with `environment_data`. Examples:

```python
# Before
def test_init_extracts_tool_specs(self, macs_model_factory, sample_task_data):
    env = MACSEnvironment(sample_task_data, macs_model_factory)

# After
def test_init_extracts_tool_specs(self, macs_model_factory, sample_environment_data):
    env = MACSEnvironment(sample_environment_data, macs_model_factory)
```

```python
# Before (local variable)
task_data = {"tools": [...]}
env = MACSEnvironment(task_data, macs_model_factory)

# After
environment_data = {"tools": [...]}
env = MACSEnvironment(environment_data, macs_model_factory)
```

- [ ] **Step 3: Rename in test_macs_integration.py**

```python
# Before
task_data = {"tools": [...]}
env = MACSEnvironment(task_data, macs_model_factory)

# After
environment_data = {"tools": [...]}
env = MACSEnvironment(environment_data, macs_model_factory)
```

- [ ] **Step 4: Run MACS tests to verify**

Run: `uv run pytest tests/test_benchmarks/test_macs/test_macs_environment.py tests/test_benchmarks/test_macs/test_macs_integration.py -v`
Expected: All tests PASS (no behavioral change, only renames)

- [ ] **Step 5: Commit**

```bash
git add tests/test_benchmarks/test_macs/conftest.py tests/test_benchmarks/test_macs/test_macs_environment.py tests/test_benchmarks/test_macs/test_macs_integration.py
git commit -m "test(macs): rename task_data to environment_data in test fixtures and variables"
```

---

### Task 2: Fix MACS real_data test bug and rename

**Files:**
- Modify: `tests/test_benchmarks/test_macs/test_macs_integration_real_data.py:64,86`

- [ ] **Step 1: Fix the wrapping bug and rename**

Lines 64 and 86 currently pass `{"environment_data": task.environment_data}` which wraps environment_data in an extra dict. `MACSEnvironment.setup_state` does `environment_data.get("tools", [])` on this, finding no `"tools"` key, silently producing an empty tools list. Fix by passing `task.environment_data` directly:

```python
# Before (line 64)
env = MACSEnvironment({"environment_data": task.environment_data}, macs_model_factory)

# After
env = MACSEnvironment(task.environment_data, macs_model_factory)
```

```python
# Before (line 86)
env = MACSEnvironment({"environment_data": task.environment_data}, macs_model_factory)

# After
env = MACSEnvironment(task.environment_data, macs_model_factory)
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_benchmarks/test_macs/test_macs_integration_real_data.py
git commit -m "fix(macs): pass environment_data directly instead of wrapping in extra dict

The old code wrapped task.environment_data in {\"environment_data\": ...},
causing setup_state to silently get tools=[] via .get(\"tools\", [])."
```

---

### Task 3: Rename task_data in TAU2 test

**Files:**
- Modify: `tests/test_benchmarks/test_tau2/test_environment.py:1142-1143`

- [ ] **Step 1: Rename local variable**

```python
# Before
task_data = {"domain": "retail"}
constructor = get_environment_constructor(task_data)

# After
environment_data = {"domain": "retail"}
constructor = get_environment_constructor(environment_data)
```

- [ ] **Step 2: Run TAU2 tests to verify**

Run: `uv run pytest tests/test_benchmarks/test_tau2/test_environment.py -v -k "test_replay"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_benchmarks/test_tau2/test_environment.py
git commit -m "test(tau2): rename task_data to environment_data for clarity"
```

---

### Task 4: Update examples

**Files:**
- Modify: `examples/five_a_day_benchmark/five_a_day_benchmark.py:135-153`
- Modify: `examples/five_a_day_benchmark/five_a_day_benchmark.ipynb` (corresponding cells)
- Modify: `examples/introduction/tutorial.ipynb` (cells using `task_data`)
- Modify: `docs/guides/usage-tracking.md:325-326`

- [ ] **Step 1: Update five_a_day_benchmark.py**

The FiveADayEnvironment constructor and setup_state use `task_data` as both parameter name and local dict key. This is a user-facing example that should model correct naming.

```python
# Before (line 135)
def __init__(self, task_data: Dict[str, Any], framework: str, callbacks: Optional[List] = None):
    ...
    super().__init__(task_data, callbacks)

def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    ...
    env_data = task_data["environment_data"].copy()

# After
def __init__(self, environment_data: Dict[str, Any], framework: str, callbacks: Optional[List] = None):
    ...
    super().__init__(environment_data, callbacks)

def setup_state(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
    ...
    env_data = environment_data.copy()
```

Also update the instantiation site (around line 743) where `task_data` dict is constructed. Note: this example builds a custom dict with `{"environment_data": {...}}` and passes the whole thing — it needs to pass just the inner environment_data dict directly, matching how the base class now works.

- [ ] **Step 2: Update five_a_day_benchmark.ipynb**

Mirror the same changes from step 1 in the notebook version.

- [ ] **Step 3: Update tutorial.ipynb**

The tutorial constructs a custom Environment subclass with `setup_state(self, task_data)`. Update parameter name to `environment_data`. Also rename the `task_data` exploration variable (where it indexes into the tasks list) — this one is actually a Task dict from the data loader, so it can stay as `task` or `task_dict` to distinguish from environment_data.

- [ ] **Step 4: Update docs/guides/usage-tracking.md**

```python
# Before
def __init__(self, task_data):
    super().__init__(task_data)

# After
def __init__(self, environment_data):
    super().__init__(environment_data)
```

- [ ] **Step 5: Commit**

```bash
git add examples/ docs/guides/usage-tracking.md
git commit -m "docs: rename task_data to environment_data in examples and guides"
```

---

### Task 5: Verify all tests pass

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run linting**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No issues

- [ ] **Step 3: Search for remaining task_data references**

Run: `grep -rn "task_data" maseval/ tests/ examples/ docs/ --include="*.py" --include="*.md"`

Verify that remaining `task_data` references are either:
- `self._task_data` in multiagentbench.py (different concept — evaluation task data, not environment constructor param)
- `task_data` in data loader test files (local variable for raw JSON task data before it becomes a Task object)
- Notebook cell outputs (not editable source)
