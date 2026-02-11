# Testing Plan: GAIA2 & MultiAgentBench Alignment

## Context

Commit `e6d8a03` ("Improved Testing Infrastructure") established a two-tier testing
pattern for benchmarks (Tau2 and MACS). GAIA2 and MultiAgentBench do not yet follow
this pattern. This plan brings them into alignment.

### What Tau2 & MACS Have That GAIA2 & MultiAgentBench Don't

| Capability                                          | Tau2 | MACS | GAIA2  | MABench |
| --------------------------------------------------- | ---- | ---- | ------ | ------- |
| `@benchmark` marker on all tests                    | Yes  | Yes  | Yes    | **No**  |
| `@live` marker on network/data tests                | Yes  | Yes  | **No** | **No**  |
| `@slow` marker on heavy download tests              | Yes  | Yes  | **No** | **No**  |
| `test_data_integrity.py` (tmp-dir, self-contained)  | Yes  | Yes  | **No** | **No**  |
| Session-scoped conftest fixture for domain tests    | Yes  | Yes  | **No** | **No**  |
| Real-data integration tests                         | Yes  | Yes  | **No** | **No**  |
| Parametrized across domains/capabilities            | Yes  | Yes  | **No** | **No**  |
| Descriptive data-availability assertions            | Yes  | Yes  | **No** | **No**  |

### Two Patterns for Real-Data Tests

The improved infrastructure uses two distinct patterns for tests that touch real data.
Both are described in `tests/README.md` under "Benchmark tests":

**Pattern A — Self-contained tmp-dir download (data integrity & integration tests)**

`test_data_integrity.py` and `test_*_integration_real_data.py` define their own
module/class-scoped fixture that downloads into `tmp_path_factory`. They validate
upstream data freshness without relying on cached state. Marked
`live + slow + benchmark`.

Examples:
- `test_tau2/test_data_integrity.py` → class-scoped `_download_data` into tmp dir
- `test_macs/test_data_integrity.py` → module-scoped `macs_data_dir` into tmp dir
- `test_macs/test_macs_integration_real_data.py` → module-scoped `real_macs_data`

**Pattern B — Session-scoped conftest fixture (domain tool & environment tests)**

A session-scoped fixture in `conftest.py` downloads to the **package's default data
directory** (cached across runs). Downstream fixtures cascade from it. Tests that
depend on this fixture are marked `live + benchmark`.

Examples:
- `test_tau2/conftest.py::ensure_tau2_data` → `retail_db` → `retail_toolkit`
- `test_macs/conftest.py::ensure_macs_templates`

---

## Plan: GAIA2

### Existing State

GAIA2 has 7 test files with good mock coverage:
- `test_evaluator.py` — single/multi-turn judge, GSR metrics
- `test_environment.py` — scenario extraction, tool wrapping, cleanup
- `test_benchmark.py` — lifecycle, seeding, agent-agnostic design
- `test_default_agent.py` — ReAct loop, action parsing, termination
- `test_tool_wrapper.py` — invocation tracking, tracing
- `test_data_loader.py` — constants, validation, model ID config
- `conftest.py` — MockARETool, MockAREEnvironment, MockGraphPerEventJudge

All test classes are marked `@pytest.mark.benchmark`. No tests are marked `@live` or
`@slow`. No real data is downloaded or tested.

### Changes

#### 1. New file: `test_data_integrity.py` (Pattern A)

Self-contained data integrity tests that download from HuggingFace into a tmp dir.

```
pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark, pytest.mark.gaia2]
```

Module-scoped fixture:
```python
@pytest.fixture(scope="module")
def gaia2_data(tmp_path_factory):
    """Download GAIA2 validation split into a temporary directory."""
    from maseval.benchmark.gaia2.data_loader import load_tasks
    tasks = load_tasks(split="validation")
    return tasks
```

Test classes:

- **TestGaia2DatasetIntegrity**
  - `test_validation_split_loads` — `load_tasks("validation")` returns data
  - `test_minimum_task_count` — dataset has >= expected number of tasks
  - `test_required_fields_present` — every task has `scenario`, `oracle_events`, `capability`
  - `test_oracle_events_non_empty` — every task has at least one oracle event
  - `test_scenario_is_deserializable` — ARE scenario field is valid (not empty/null) for a sample

- **TestGaia2CapabilityCoverage**
  - `@pytest.mark.parametrize("capability", VALID_CAPABILITIES)`
  - `test_capability_has_tasks` — every declared capability has >= 1 task in the dataset

#### 2. New file: `test_integration.py` (Pattern A)

Real-data integration tests exercising the GAIA2 pipeline with real tasks but a
DummyModelAdapter (no API keys needed).

```
pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark, pytest.mark.gaia2]
```

Module-scoped fixture reuses the `gaia2_data` approach (download into tmp dir, or
reuse the data integrity fixture via a shared conftest fixture).

Test classes:

- **TestGaia2EnvironmentWithRealData**
  - `test_environment_setup_from_real_task` — `Gaia2Environment` from a real task
    creates without error, `setup_state()` succeeds
  - `test_real_tools_are_wrapped` — tools created from a real scenario are
    `Gaia2GenericTool` instances with name, description, and callable inputs
  - `test_real_tools_have_valid_schema` — every wrapped tool's `inputs` is a dict
    with expected structure (not empty, has descriptions)

- **TestDefaultAgentWithRealTools**
  - `test_agent_builds_system_prompt_with_real_tools` — `DefaultGaia2Agent` constructed
    from real task tools has a system prompt that mentions real tool names
  - `test_single_step_execution` — run agent for 1 iteration with a canned ReAct
    response targeting a real tool name, verify tool invocation is recorded

- **TestGaia2EvaluatorWithRealOracleEvents**
  - `test_evaluator_processes_real_oracle_events` — `Gaia2Evaluator` with a real task's
    `oracle_events` and a mock judge runs without error
  - `test_evaluator_returns_scoreable_result` — result has expected fields (gsr, status)

- **TestGaia2PipelineSmoke**
  - `test_full_pipeline_single_task` — `Gaia2Benchmark.run()` on one real task with
    `DummyModelAdapter` produces a `TaskResult` with expected structure
    (status in known statuses, traces dict present, eval dict present)

#### 3. Update `conftest.py` — Session-scoped fixture (Pattern B)

Add a session-scoped fixture for tests that want real ARE tools/environments without
re-downloading each time:

```python
@pytest.fixture(scope="session")
def ensure_gaia2_data():
    """Download GAIA2 validation data to the package's default cache.

    Tests that need real data should depend on this and be marked @pytest.mark.live.
    """
    from maseval.benchmark.gaia2.data_loader import load_tasks
    tasks = load_tasks(split="validation")
    return tasks
```

This enables future domain-specific test files (e.g., testing specific ARE app tools)
to depend on `ensure_gaia2_data` without each file re-downloading.

#### 4. No changes to existing test files

The existing mock-based tests are solid Tier 1 tests. They should remain as-is with
their `@benchmark` marker and no `@live`/`@slow` markers.

---

## Plan: MultiAgentBench

### Existing State

MultiAgentBench has 6 test files with extensive coverage (~3,200 lines):
- `test_evaluator.py` — domain-specific evaluation, parsing, metrics
- `test_benchmark.py` — lifecycle, seeding, MARBLE integration, coordination modes
- `test_data_loader.py` — JSONL loading, domain info, MARBLE download, werewolf config
- `test_environment.py` — infrastructure checks, MARBLE env delegation, tool wrapping
- `test_marble_adapter.py` — agent wrapping, action/communication logging
- `conftest.py` — task data fixtures, agent adapter, concrete benchmark

**Critical gap:** No test classes are marked `@pytest.mark.benchmark`. This means:
- `pytest -m benchmark` misses all MultiAgentBench tests
- `pytest -m core` incorrectly includes them
- The CI `test-benchmark` job doesn't run them

No tests are marked `@live` or `@slow`. No real MARBLE data is downloaded or tested.

### Changes

#### 1. Add `@pytest.mark.benchmark` to ALL existing test classes

Every test class/function in `test_evaluator.py`, `test_benchmark.py`,
`test_data_loader.py`, `test_environment.py`, and `test_marble_adapter.py` needs
`@pytest.mark.benchmark`. This is the highest-priority change — it fixes the CI
pipeline visibility.

Preferred approach: use `pytestmark` at file level where all tests in a file are
benchmark tests:

```python
pytestmark = pytest.mark.benchmark
```

#### 2. New file: `test_data_integrity.py` (Pattern A)

Self-contained data integrity tests that clone/verify MARBLE into a tmp dir.

```
pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]
```

Module-scoped fixture:
```python
@pytest.fixture(scope="module")
def marble_data(tmp_path_factory):
    """Clone and verify MARBLE data into a temporary directory."""
    from maseval.benchmark.multiagentbench.data_loader import (
        ensure_marble_exists,
    )
    data_dir = tmp_path_factory.mktemp("marble_data")
    marble_dir = ensure_marble_exists(data_dir=data_dir, auto_download=True)
    return marble_dir
```

Test classes:

- **TestMarbleDataPresence**
  - `@pytest.mark.parametrize("domain", VALID_DOMAINS)`
  - `test_domain_directory_exists` — domain directory exists in MARBLE
  - `test_domain_has_task_data` — JSONL/config files present per domain

- **TestMarbleTaskStructure**
  - `@pytest.mark.parametrize("domain", VALID_DOMAINS - {"werewolf"})`
  - `test_minimum_task_count` — each domain has >= expected minimum tasks
  - `test_required_fields` — each task has `scenario`, `task_id`, `task`, `agents`,
    `relationships`
  - `test_agent_structure` — each agent has `agent_id`

- **TestMarbleWerewolfConfigs**
  - `test_werewolf_config_files_exist` — config YAML files present
  - `test_werewolf_configs_parse` — YAML files parse correctly
  - `test_werewolf_configs_have_roles` — configs contain expected roles

#### 3. New file: `test_integration_real_data.py` (Pattern A)

Real-data integration tests, following the MACS `test_macs_integration_real_data.py`
pattern.

```
pytestmark = [pytest.mark.live, pytest.mark.slow, pytest.mark.benchmark]
```

Module-scoped fixture (same as data integrity or shared via conftest).

Test classes:

- **TestMultiAgentBenchRealDataLoading**
  - `@pytest.mark.parametrize("domain", VALID_DOMAINS)`
  - `test_load_tasks_returns_tasks` — `load_tasks(domain)` returns non-empty TaskQueue
  - `test_tasks_have_agents` — every loaded task has >= 1 agent in its config
  - `test_configure_model_ids` — `configure_model_ids()` modifies tasks in place

- **TestMultiAgentBenchRealEnvironment**
  - `@pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)`
    where `NON_INFRA_DOMAINS = VALID_DOMAINS - {"database", "minecraft"}`
  - `test_environment_setup` — `MultiAgentBenchEnvironment` initializes from real task
  - `test_environment_state` — `setup_state()` extracts domain and max_iterations
  - `test_environment_traces` — `gather_traces()` returns dict with expected keys

- **TestMultiAgentBenchRealEvaluation**
  - `@pytest.mark.parametrize("domain", VALID_DOMAINS - {"minecraft"})`
  - `test_evaluator_creation` — evaluator created from real task's domain and
    `DummyModelAdapter` without error
  - `test_evaluator_processes_structure` — evaluator's `filter_traces()` processes
    a synthetic trace structure without error

- **TestMultiAgentBenchPipelineSmoke**
  - `@pytest.mark.parametrize("domain", NON_INFRA_DOMAINS)`
  - `test_full_pipeline_single_task` — benchmark `.run()` on one real task with
    `DummyModelAdapter` produces results with expected structure. Uses descriptive
    assertion messages:
    ```python
    assert len(results) > 0, (
        f"No results for domain '{domain}'. "
        "Check test_data_integrity tests first."
    )
    ```

#### 4. Update `conftest.py` — Session-scoped fixture (Pattern B)

```python
@pytest.fixture(scope="session")
def ensure_marble_data():
    """Clone MARBLE data once per session.

    Tests that need real data should depend on this and be marked @pytest.mark.live.
    """
    from maseval.benchmark.multiagentbench.data_loader import ensure_marble_exists
    marble_dir = ensure_marble_exists(auto_download=True)
    return marble_dir
```

#### 5. Descriptive assertion messages in existing tests

Update assertion messages in existing environment/evaluator tests where data absence
could cause confusing failures, linking to data integrity tests. This is low priority
and can be done opportunistically.

---

## CI Updates

### `.github/workflows/test.yml`

Add caching for GAIA2 and MARBLE data in the `test-slow` job:

```yaml
- name: Cache GAIA2 data
  uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface/  # or wherever HF datasets cache
    key: gaia2-data-${{ hashFiles('maseval/benchmark/gaia2/data_loader.py') }}

- name: Cache MARBLE data
  uses: actions/cache@v4
  with:
    path: maseval/benchmark/multiagentbench/marble/
    key: marble-data-${{ hashFiles('maseval/benchmark/multiagentbench/data_loader.py') }}
```

### `tests/README.md`

Update the tree to include `test_multiagentbench/`:

```
└── test_benchmarks/
    ├── test_tau2/              # Tau2 benchmark + data integrity
    ├── test_macs/              # MACS benchmark + data integrity
    ├── test_gaia2/             # GAIA2 benchmark + data integrity
    └── test_multiagentbench/   # MultiAgentBench + data integrity
```

---

## Summary of New Files

| Benchmark      | New File                       | Pattern | Markers                      |
| -------------- | ------------------------------ | ------- | ---------------------------- |
| GAIA2          | `test_data_integrity.py`       | A       | `live + slow + benchmark + gaia2` |
| GAIA2          | `test_integration.py`          | A       | `live + slow + benchmark + gaia2` |
| MultiAgentBench| `test_data_integrity.py`       | A       | `live + slow + benchmark`    |
| MultiAgentBench| `test_integration_real_data.py` | A      | `live + slow + benchmark`    |

## Summary of Modified Files

| File | Change |
| ---- | ------ |
| `test_multiagentbench/test_evaluator.py` | Add `pytestmark = pytest.mark.benchmark` |
| `test_multiagentbench/test_benchmark.py` | Add `pytestmark = pytest.mark.benchmark` |
| `test_multiagentbench/test_data_loader.py` | Add `pytestmark = pytest.mark.benchmark` |
| `test_multiagentbench/test_environment.py` | Add `pytestmark = pytest.mark.benchmark` |
| `test_multiagentbench/test_marble_adapter.py` | Add `pytestmark = pytest.mark.benchmark` |
| `test_gaia2/conftest.py` | Add `ensure_gaia2_data` session fixture |
| `test_multiagentbench/conftest.py` | Add `ensure_marble_data` session fixture |
| `.github/workflows/test.yml` | Add data caching for GAIA2 + MARBLE |
| `tests/README.md` | Add `test_multiagentbench/` to tree |

## Priority Order

1. **Add `@benchmark` to all MultiAgentBench tests** — fixes CI pipeline visibility (broken now)
2. **`test_data_integrity.py` for both** — foundation for catching upstream data regressions
3. **`test_integration*.py` for both** — catches real-data issues mocks can't
4. **Session-scoped conftest fixtures** — enables future domain-specific test files
5. **CI caching** — makes the slow tier practical in CI
6. **README update** — keeps documentation aligned
