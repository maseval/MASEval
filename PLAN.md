# Testing Strategy Plan for MASEval

## Problem Statement

MASEval benchmarks are insufficiently tested because:
1. **LLM API calls** - Tests requiring real API calls are either skipped or mocked entirely
2. **Data downloads** - Benchmark data comes from external sources (GitHub, HuggingFace) and downloads are not validated
3. **Slow operations** - No mechanism to separate fast unit tests from slow integration tests
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

### New Test Tiers

| Tier | Marker | Speed | Dependencies | When to Run |
|------|--------|-------|--------------|-------------|
| 1. Unit | `core` (existing) | <100ms/test | None (mocked) | Every commit |
| 2. Integration | `integration` (new) | 1-30s/test | API keys | Daily scheduled |
| 3. Slow | `slow` (new) | 30s-5min/test | Data downloads | Weekly scheduled |
| 4. Smoke | `smoke` (new) | 5-30min total | Full pipeline | Before releases |

### New Markers to Add

```toml
# In pyproject.toml [tool.pytest.ini_options]
markers = [
    # ... existing markers ...
    "slow: Tests taking >30 seconds (data downloads, large datasets)",
    "integration: Tests requiring real API keys",
    "smoke: Full end-to-end tests before releases",
]
```

### Default Exclusion

Update `addopts` to exclude slow tests by default:
```toml
addopts = "-ra -q -m 'not slow and not integration and not smoke'"
```

This ensures `pytest` runs fast by default while `pytest -m integration` explicitly runs integration tests.

---

## Implementation Plan

### Phase 1: Marker Infrastructure

1. Add new markers (`slow`, `integration`, `smoke`) to `pyproject.toml`
2. Update `addopts` to exclude slow/integration/smoke by default
3. Create `tests/markers.py` with skip decorators for missing API keys:
   - `requires_openai` - skips if `OPENAI_API_KEY` not set
   - `requires_anthropic` - skips if `ANTHROPIC_API_KEY` not set
   - `requires_google` - skips if `GOOGLE_API_KEY` not set

### Phase 2: Real LLM API Tests

Create minimal integration tests that validate API contracts:

1. **One test per provider** (OpenAI, Anthropic, Google, HuggingFace, LiteLLM)
2. **Minimal token usage** - Use cheapest models, simple prompts like "Say 'test'"
3. **Validate response structure** - Check that `ChatResponse` fields are populated correctly
4. **Test tool calling format** - Ensure tool call JSON structure matches expectations

**Cost estimate**: <$1/month for daily integration tests using cheapest models.

### Phase 3: Data Loading Validation

1. **Mark data download tests as `@pytest.mark.slow`**
2. **Add integrity checks** after downloads:
   - Verify expected files exist
   - Validate JSON schemas
   - Check database tables have minimum required data
3. **Fix Tau2 conditional skips**:
   - Option A: Seed test databases with guaranteed fixture data
   - Option B: Convert skips to `xfail` with clear reasons
   - Option C: Create separate "requires data" marker

### Phase 4: HTTP Mocking for CI

For CI environments without API keys, use HTTP-level mocking:

1. **Use `responses` or `respx` library** to intercept HTTP calls
2. **Create response fixtures** that match real API response schemas
3. **Run mocked tests on every PR**, real API tests on schedule

This provides two levels of confidence:
- Mocked tests catch code-level regressions (fast, every PR)
- Real API tests catch API contract changes (daily, with keys)

### Phase 5: CI/CD Updates

Update GitHub Actions workflow:

| Job | Trigger | Tests Run |
|-----|---------|-----------|
| test-unit | Every push/PR | `core` and `benchmark` (excluding slow/integration) |
| test-integration | Daily schedule | `integration` marker (with API secrets) |
| test-slow | Weekly schedule | `slow` marker |
| test-smoke | Manual dispatch | `smoke` marker (pre-release validation) |

Store API keys as GitHub Secrets, only available to scheduled/manual runs.

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
- Keep mocked tests for fast CI (`core`, `benchmark` markers)
- Add `integration` marker for real API tests
- Run integration tests on schedule with API keys from secrets

### Issue: Data Downloads Not Validated

**Root cause**: Session fixtures download but don't verify completeness.

**Solution**:
- Add post-download validation in slow tests
- Check file existence, JSON validity, minimum record counts
- Fail loudly if data is corrupt or incomplete

---

## Execution Order

1. **Week 1**: Add markers to pyproject.toml, update addopts
2. **Week 2**: Create integration tests for each LLM provider
3. **Week 3**: Add data integrity tests, fix Tau2 skips
4. **Week 4**: Add HTTP mocking with `responses` library
5. **Week 5**: Update CI workflow with scheduled jobs

---

## Success Criteria

- [ ] `pytest` (default) completes in <5 minutes
- [ ] `pytest -m integration` validates all LLM provider APIs
- [ ] `pytest -m slow` validates all data downloads
- [ ] Tau2 skipped tests reduced from 45+ to <5
- [ ] CI runs integration tests daily, slow tests weekly
- [ ] Pre-release smoke tests validate full pipeline
