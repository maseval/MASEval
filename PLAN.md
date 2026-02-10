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

### Phase 1: Marker Infrastructure — DONE

1. ~~Add new markers (`live`, `credentialed`, `slow`, `smoke`) to `pyproject.toml`~~ — Done
2. ~~Update `addopts` to exclude `slow`, `credentialed`, and `smoke` by default~~ — Done
3. ~~Add `pytest_collection_modifyitems` hook to `conftest.py` to enforce `credentialed` → `live` implication~~ — Done
4. ~~Create `tests/markers.py` with skip decorators for missing API keys~~ — Done
   - `requires_openai` - skips if `OPENAI_API_KEY` not set
   - `requires_anthropic` - skips if `ANTHROPIC_API_KEY` not set
   - `requires_google` - skips if `GOOGLE_API_KEY` not set

**Note**: `tests/markers.py` exists but is not directly importable because `tests/` lacks `__init__.py`. Test files that need these decorators define them inline. Consider adding `tests/__init__.py` or moving decorators to a conftest.py in a future cleanup.

### Phase 2: Data Loading Validation — DONE

1. ~~**Mark data download tests as `@pytest.mark.live` and `@pytest.mark.slow`**~~ — Done
2. **Add dataset caching in CI** using `actions/cache` on the data directory to avoid re-downloading every run — Deferred to Phase 5
3. ~~**Add integrity checks** after downloads~~ — Done
   - ~~Verify expected files exist~~ — Done
   - ~~Validate JSON schemas~~ — Done
   - ~~Check database tables have minimum required data~~ — Done
   - Tests: `tests/test_benchmarks/test_tau2/test_data_integrity.py` (23 tests), `tests/test_benchmarks/test_macs/test_data_integrity.py` (24 tests)
4. **Fix Tau2 conditional skips** — Not started
   - Option A: Seed test databases with guaranteed fixture data
   - Option B: Convert skips to `xfail` with clear reasons
   - Option C: Create separate "requires data" marker
   - One test (`test_airline_db_has_nonfree_baggages`) marked `xfail` due to upstream v0.2.0 data gap

### Phase 3: HTTP Mocking for CI — DONE

For CI environments without API keys, use HTTP-level mocking:

1. ~~**Use `respx` library** to intercept HTTP calls~~ — Done (`respx>=0.22.0` added to dev dependencies)
2. ~~**Create response fixtures** that match real API response schemas~~ — Done
3. ~~**Run mocked tests on every PR**~~ — Tests run in default suite (no `live`/`credentialed` marker)

Tests: `tests/test_interface/test_model_integration/test_api_contracts.py` (10 tests)
- OpenAI: text response, tool calls, seed propagation (3 tests)
- Anthropic: text response, tool use, system message extraction (3 tests)
- Google GenAI: text response, function calls (2 tests)
- LiteLLM: text response, tool calls (2 tests, mocked at `litellm.completion` level since LiteLLM is a routing layer)

This defines the expected API contract via mocks first (TDD-style), then Phase 4 validates those contracts against real APIs.

### Phase 4: Real LLM API Tests — DONE

~~Create minimal tests marked `@pytest.mark.credentialed` that validate API contracts:~~

1. ~~**One test per provider**~~ — Done for 4 API-based providers (see deviation note below)
2. ~~**Minimal token usage** - Use cheapest models, simple prompts like "Say 'test'"~~ — Done
3. ~~**Validate response structure** - Check that `ChatResponse` fields are populated correctly~~ — Done
4. ~~**Test tool calling format** - Ensure tool call JSON structure matches expectations~~ — Done
5. ~~**Mark appropriately** - Each test gets `credentialed` plus relevant existing markers~~ — Done

Tests: `tests/test_interface/test_model_integration/test_live_api.py` (8 tests)
- OpenAI (`gpt-4o-mini`): text + tool call (2 tests, requires `OPENAI_API_KEY`)
- Anthropic (`claude-3-5-haiku-20241022`): text + tool use (2 tests, requires `ANTHROPIC_API_KEY`)
- Google GenAI (`gemini-2.0-flash`): text + function call (2 tests, requires `GOOGLE_API_KEY`)
- LiteLLM → OpenAI (`gpt-4o-mini`): text + tool call (2 tests, requires `OPENAI_API_KEY`)

**Plan deviation**: HuggingFace was excluded. The HuggingFace adapter uses local `transformers` models (no API, no network, no API keys) — it doesn't fit the `credentialed` test pattern.

**Cost estimate**: <$1/month for daily credentialed tests using cheapest models.

This provides two levels of confidence:
- Mocked tests (Phase 3) catch code-level regressions (fast, every PR)
- Real API tests (Phase 4) catch API contract changes (daily, with keys)

### Phase 5: CI/CD Updates — DONE

~~Update `.github/workflows/test.yml` to leverage the composable marker system.~~

#### Current workflow (kept as-is)

These existing jobs remain unchanged — they already exclude `slow`, `credentialed`, and `smoke` via `addopts`:

| Job | Trigger | Python versions | What it runs |
|-----|---------|-----------------|--------------|
| test-core | Every push/PR | 3.10–3.14 | `-m core` (fast, no optional deps) |
| test-benchmark | Every push/PR | 3.10–3.14 | `-m benchmark` |
| test-all | After core+benchmark | 3.10–3.14 | `pytest -v` (uses `addopts`, excludes slow/credentialed/smoke) |
| coverage | Every push/PR | 3.12 | `pytest` with coverage reporting |

#### New jobs added

| Job | Trigger | Python | Filter | Gate | Description |
|-----|---------|--------|--------|------|-------------|
| ~~**test-slow**~~ | Every push/PR | **3.12 only** | `-m "slow and not credentialed"` | None | Data download + integrity (47 tests). No secrets needed. Single version because download behavior doesn't vary across Python versions. `actions/cache` on `maseval/benchmark/tau2/data/`, `maseval/benchmark/macs/data/`, and `maseval/benchmark/macs/prompt_templates/` keyed by data loader file hashes. |
| ~~**test-credentialed**~~ | Every push/PR | **3.12 only** | `-m "credentialed and not smoke"` | **GitHub Environment approval** | Real API tests (8 tests). Uses `credentialed-tests` GitHub Environment with required reviewers. Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) only exposed after maintainer approval. |

**No `test-smoke` job for now** — no smoke tests exist yet. Add when smoke tests are written.

#### GitHub Environment setup (manual step — NOT YET DONE)

Create a `credentialed-tests` Environment in repo Settings → Environments:
1. Add required reviewer(s) (maintainer)
2. Add secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
3. The job references `environment: credentialed-tests` — GitHub shows "Waiting for review" on PRs
4. Maintainer approves → job runs with access to secrets
5. If not approved, the check stays pending without blocking the PR

#### Why not scheduled runs?

- **Slow tests on every PR** catch data pipeline breakage before merge, not days later
- **Credentialed tests behind approval** give the same control as a schedule but with better feedback — results appear on the PR that introduced the change, not in a disconnected nightly run

---

## Addressing Specific Issues

### Issue: Tau2 45+ Skipped Tests — NOT STARTED

**Root cause**: Tests depend on database fixtures that may be empty.

**Solution options**:
1. **Seed fixtures** - Ensure `ensure_tau2_data()` creates minimum required entities
2. **Separate marker** - `@pytest.mark.requires_tau2_data` for tests needing full data
3. **Use xfail** - Mark as expected failures with `reason="Requires seeded database"`

**Recommendation**: Option 1 (seed fixtures) is cleanest long-term.

### Issue: LLM Calls Only Mocked — RESOLVED

**Root cause**: No mechanism to test real APIs without breaking fast CI.

**Solution** (implemented in Phases 3 & 4):
- Keep mocked tests for fast CI (`core`, `interface`, `contract`, `benchmark` markers)
- Mocked HTTP tests (Phase 3) run in default suite — no keys needed
- `credentialed` marker for real API tests (Phase 4) — excluded by default
- Credentialed tests on PRs behind GitHub Environment approval (Phase 5)

### Issue: Data Downloads Not Validated — PARTIALLY RESOLVED

**Root cause**: Session fixtures download but don't verify completeness.

**Solution** (implemented in Phase 2):
- ~~Mark download tests with `live` + `slow`~~ — Done
- ~~Add post-download validation: file existence, JSON validity, minimum record counts~~ — Done
- Cache datasets in CI with `actions/cache` so repeat runs skip the download — Deferred to Phase 5
- ~~Fail loudly if data is corrupt or incomplete~~ — Done

---

## Execution Order

1. ~~**Week 1**: Add composable markers (`live`, `credentialed`, `slow`, `smoke`) to pyproject.toml, update addopts, add implication hook~~ — DONE (Phase 1)
2. ~~**Week 2**: Add data integrity tests (`live` + `slow`), fix Tau2 skips~~ — DONE (Phase 2, Tau2 skips not yet addressed)
3. ~~**Week 3**: Add HTTP mocking with `respx` library (define API contracts via mocks)~~ — DONE (Phase 3)
4. ~~**Week 4**: Create credentialed tests for each LLM provider (validate mocks against real APIs)~~ — DONE (Phase 4)
5. ~~**Week 5**: Add `test-slow` and `test-credentialed` CI jobs~~ — DONE (Phase 5). GitHub Environment setup is a manual step (see Phase 5).

---

## Success Criteria

- [x] `pytest` (default) completes in <5 minutes with `-m "not (slow or credentialed or smoke)"` — ~31s for 1830 tests
- [x] `pytest -m credentialed` validates all LLM provider APIs — 8 tests across 4 providers
- [x] `pytest -m "live and slow"` validates all data downloads — 47 tests for Tau2 + MACS
- [x] `pytest -m "not live"` gives a fully offline run — enforced via `credentialed` → `live` implication hook
- [ ] Tau2 skipped tests reduced from 45+ to <5 — Not yet addressed
- [x] CI runs slow tests on every push/PR (single Python version) — Phase 5
- [x] CI runs credentialed tests on PR with maintainer approval via GitHub Environment — Phase 5 (environment setup is manual)
- [ ] Pre-release smoke tests validate full pipeline — Future (no smoke tests written yet)
