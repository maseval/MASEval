# Testing Strategy Plan for MASEval

## Problem Statement

MASEval benchmarks are insufficiently tested because:
1. **LLM API calls** - Tests requiring real API calls are either skipped or mocked entirely
2. **Data downloads** - Benchmark data comes from external sources (GitHub, HuggingFace) and downloads are not validated
3. **Slow operations** - No mechanism to separate fast unit tests from slow/live/credentialed tests
4. **Conditional skips** - Many Tau2 tests skip when database fixtures lack data (45+ skipped tests)

The result: bugs in real integrations go undetected until runtime.

---

## Current Test Infrastructure

### Existing Pytest Markers

| Marker | Purpose | Used In |
|--------|---------|---------|
| `core` | Tests without optional dependencies | Core library, runs on all Python versions |
| `interface` | Tests requiring optional deps (smolagents, langgraph, etc.) | Framework integrations |
| `contract` | Cross-implementation contract tests | Adapter contracts |
| `benchmark` | Benchmark-specific tests | All benchmark test files |
| `smolagents`, `langgraph`, `llamaindex`, `gaia2`, `camel` | Framework-specific markers | Individual framework tests |

### Current CI Pipeline (`.github/workflows/test.yml`)

1. **test-core**: Runs `@pytest.mark.core` only (fast, no optional deps)
2. **test-benchmark**: Runs `@pytest.mark.benchmark` only
3. **test-all**: Runs everything after core and benchmark pass

### Existing Mocking Infrastructure

The codebase has comprehensive mocks in `tests/conftest.py`:
- `DummyModelAdapter` - Returns predefined responses, cycles through them
- `DummyAgent`, `DummyAgentAdapter` - Mock agent implementations
- `DummyEnvironment`, `DummyUser`, `DummyBenchmark` - Mock benchmark components
- Framework-specific mocks for CAMEL, Smolagents

**Problem**: These mocks are excellent for unit tests but don't validate that real APIs still work.

### Current Skip Patterns

**Tau2 domain tests** contain 45+ conditional skips like:
```python
if not users:
    pytest.skip("No users in test database")
```

**Optional dependencies** use `pytest.importorskip()` which is appropriate.

---

## Proposed Testing Strategy

### Composable Marker Design

Instead of hierarchical tiers, new markers represent **orthogonal properties** that can be combined on any test. A single test can carry multiple markers describing its requirements.

#### New Markers

| Marker | Meaning | Why excluded from default |
|--------|---------|--------------------------|
| `live` | Needs network access (downloads, external APIs) | May fail without network / adds latency |
| `credentialed` | Needs API keys, costs money per call | No keys in CI by default |
| `slow` | Takes >30 seconds | Too slow for every commit |
| `smoke` | Full end-to-end pipeline validation | Pre-release only |

These compose with existing markers (`core`, `interface`, `contract`, `benchmark`, framework-specific):

```python
@pytest.mark.benchmark
@pytest.mark.live
@pytest.mark.slow
def test_download_gaia2_dataset():
    """Needs network, takes a while."""
    ...

@pytest.mark.interface
@pytest.mark.live
@pytest.mark.credentialed
def test_openai_tool_calling():
    """Needs network + API key, but fast."""
    ...

@pytest.mark.smoke
@pytest.mark.credentialed
def test_full_benchmark_run():
    """End-to-end pre-release validation."""
    ...
```

#### Marker Implication: `credentialed` implies `live`

Since you can't call a paid API without network, `credentialed` logically implies `live`. To enforce this automatically, add a hook to `conftest.py`:

```python
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("credentialed") and not item.get_closest_marker("live"):
            item.add_marker(pytest.mark.live)
```

This ensures `-m "not live"` always gives a fully offline run.

#### New Markers in `pyproject.toml`

```toml
# In pyproject.toml [tool.pytest.ini_options]
markers = [
    # ... existing markers (core, interface, contract, benchmark, etc.) ...
    "live: Tests requiring network access (downloads, external APIs)",
    "credentialed: Tests requiring API keys (implies live, costs money)",
    "slow: Tests taking >30 seconds (data downloads, large datasets)",
    "smoke: Full end-to-end pipeline validation (pre-release only)",
]
```

#### Default Exclusion

Update `addopts` to exclude expensive tests by default:
```toml
addopts = "-ra -q -m 'not (slow or credentialed or smoke)'"
```

Note: CLI `-m` **replaces** `addopts` rather than composing with it, so `pytest -m credentialed` runs all credentialed tests including slow/smoke ones. This is generally the desired behavior when explicitly selecting a marker.

---

## Implementation Plan

### Phase 1: Marker Infrastructure

1. Add new markers (`live`, `credentialed`, `slow`, `smoke`) to `pyproject.toml`
2. Update `addopts` to exclude `slow`, `credentialed`, and `smoke` by default
3. Add `pytest_collection_modifyitems` hook to `conftest.py` to enforce `credentialed` → `live` implication
4. Create `tests/markers.py` with skip decorators for missing API keys:
   - `requires_openai` - skips if `OPENAI_API_KEY` not set
   - `requires_anthropic` - skips if `ANTHROPIC_API_KEY` not set
   - `requires_google` - skips if `GOOGLE_API_KEY` not set

### Phase 2: Data Loading Validation

1. **Mark data download tests as `@pytest.mark.live` and `@pytest.mark.slow`** — they need network and take time
2. **Add dataset caching in CI** using `actions/cache` on the data directory to avoid re-downloading every run
3. **Add integrity checks** after downloads:
   - Verify expected files exist
   - Validate JSON schemas
   - Check database tables have minimum required data
4. **Fix Tau2 conditional skips**:
   - Option A: Seed test databases with guaranteed fixture data
   - Option B: Convert skips to `xfail` with clear reasons
   - Option C: Create separate "requires data" marker

### Phase 3: HTTP Mocking for CI

For CI environments without API keys, use HTTP-level mocking:

1. **Use `responses` or `respx` library** to intercept HTTP calls
2. **Create response fixtures** that match real API response schemas
3. **Run mocked tests on every PR**, real API tests on schedule

This defines the expected API contract via mocks first (TDD-style), then Phase 4 validates those contracts against real APIs.

### Phase 4: Real LLM API Tests

Create minimal tests marked `@pytest.mark.credentialed` that validate API contracts:

1. **One test per provider** (OpenAI, Anthropic, Google, HuggingFace, LiteLLM)
2. **Minimal token usage** - Use cheapest models, simple prompts like "Say 'test'"
3. **Validate response structure** - Check that `ChatResponse` fields are populated correctly
4. **Test tool calling format** - Ensure tool call JSON structure matches expectations
5. **Mark appropriately** - Each test gets `credentialed` plus relevant existing markers (e.g. `interface` for framework-specific tests)

**Cost estimate**: <$1/month for daily credentialed tests using cheapest models.

This provides two levels of confidence:
- Mocked tests (Phase 3) catch code-level regressions (fast, every PR)
- Real API tests (Phase 4) catch API contract changes (daily, with keys)

### Phase 5: CI/CD Updates

Update GitHub Actions workflow with composable marker filter expressions:

| Job | Trigger | Filter Expression | Description |
|-----|---------|-------------------|-------------|
| test-fast | Every push/PR | `-m "not (slow or credentialed or smoke)"` | Fast offline tests |
| test-credentialed | Daily schedule | `-m "credentialed and not smoke"` | Real API tests (with secrets) |
| test-slow | Weekly schedule | `-m "slow and not credentialed"` | Data download + integrity (with cache) |
| test-smoke | Manual dispatch | `-m smoke` | Full end-to-end pre-release |

Store API keys as GitHub Secrets, only available to scheduled/manual runs.
Add `actions/cache` for dataset directories in the `test-slow` job.

---

## Addressing Specific Issues

### Issue: Tau2 45+ Skipped Tests

**Root cause**: Tests depend on database fixtures that may be empty.

**Solution options**:
1. **Seed fixtures** - Ensure `ensure_tau2_data()` creates minimum required entities
2. **Separate marker** - `@pytest.mark.requires_tau2_data` for tests needing full data
3. **Use xfail** - Mark as expected failures with `reason="Requires seeded database"`

**Recommendation**: Option 1 (seed fixtures) is cleanest long-term.

### Issue: LLM Calls Only Mocked

**Root cause**: No mechanism to test real APIs without breaking fast CI.

**Solution**:
- Keep mocked tests for fast CI (`core`, `interface`, `contract`, `benchmark` markers)
- Add `credentialed` marker for real API tests
- Run credentialed tests on schedule with API keys from secrets

### Issue: Data Downloads Not Validated

**Root cause**: Session fixtures download but don't verify completeness.

**Solution**:
- Mark download tests with `live` + `slow`
- Add post-download validation: file existence, JSON validity, minimum record counts
- Cache datasets in CI with `actions/cache` so repeat runs skip the download
- Fail loudly if data is corrupt or incomplete

---

## Execution Order

1. **Week 1**: Add composable markers (`live`, `credentialed`, `slow`, `smoke`) to pyproject.toml, update addopts, add implication hook
2. **Week 2**: Add data integrity tests (`live` + `slow`), fix Tau2 skips
3. **Week 3**: Add HTTP mocking with `responses` library (define API contracts via mocks)
4. **Week 4**: Create credentialed tests for each LLM provider (validate mocks against real APIs)
5. **Week 5**: Update CI workflow with filter expressions, add dataset caching

---

## Success Criteria

- [ ] `pytest` (default) completes in <5 minutes with `-m "not (slow or credentialed or smoke)"`
- [ ] `pytest -m credentialed` validates all LLM provider APIs
- [ ] `pytest -m "live and slow"` validates all data downloads
- [ ] `pytest -m "not live"` gives a fully offline run
- [ ] Tau2 skipped tests reduced from 45+ to <5
- [ ] CI runs credentialed tests daily, slow tests weekly
- [ ] Pre-release smoke tests validate full pipeline
