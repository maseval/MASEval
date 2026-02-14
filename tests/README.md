# Testing Strategy

## Overview

Tests are organized by **what they test** and **what they need to run**:

- **What they test**: `core`, `interface`, `contract`, `benchmark`, plus framework-specific markers (`smolagents`, `langgraph`, `llamaindex`, `gaia2`, `camel`)
- **What they need**: `live` (network), `credentialed` (API keys), `slow` (>30s), `smoke` (full pipeline)

These markers compose freely. A test can be both `benchmark` and `slow`, or `interface` and `credentialed`.

## Running Tests Locally

```bash
# Default — fast tests only (excludes slow, credentialed, smoke)
uv run pytest -v

# Core tests only (no optional dependencies needed)
uv run pytest -m core -v

# Specific markers
uv run pytest -m benchmark -v
uv run pytest -m smolagents -v
uv run pytest -m contract -v

# Data download + integrity validation (needs network, takes time)
uv run pytest -m "live and slow" -v

# Live API tests (needs API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)
uv run pytest -m credentialed -v

# Fully offline run (no network at all)
uv run pytest -m "not live" -v
```

**Note:** The default `addopts` in `pyproject.toml` excludes `slow`, `credentialed`, and `smoke`. When you pass `-m` explicitly, it replaces the default filter.

## Test Markers

Defined in `pyproject.toml`:

| Marker                                                    | Purpose                                   |
| --------------------------------------------------------- | ----------------------------------------- |
| `core`                                                    | No optional dependencies needed           |
| `interface`                                               | Requires optional dependencies            |
| `contract`                                                | Cross-implementation behavioral contracts |
| `benchmark`                                               | Benchmark-specific tests                  |
| `smolagents`, `langgraph`, `llamaindex`, `gaia2`, `camel` | Framework-specific                        |
| `live`                                                    | Requires network access                   |
| `credentialed`                                            | Requires API keys (implies `live`)        |
| `slow`                                                    | Takes >30 seconds                         |
| `smoke`                                                   | Full end-to-end pipeline validation       |

**Marker implication:** `credentialed` automatically implies `live` via a hook in `conftest.py`. This ensures `-m "not live"` always gives a fully offline run.

## CI Pipeline

Jobs in `.github/workflows/test.yml`. Each test job collects coverage data (from Python 3.12 only); the final coverage job merges them into one combined report.

| Job                | Python    | What it runs                                 | Gate                |
| ------------------ | --------- | -------------------------------------------- | ------------------- |
| test-core          | 3.10–3.14 | `-m core` (no optional deps)                 | —                   |
| test-benchmark     | 3.10–3.14 | `-m "benchmark and not (slow or live)"`      | —                   |
| test-core-optional | 3.10–3.14 | `-m core` (with optional deps)               | —                   |
| test-interface     | 3.10–3.14 | `-m interface`                               | —                   |
| test-slow          | 3.12      | `-m "(slow or live) and not credentialed"`   | —                   |
| test-credentialed  | 3.12      | `-m "credentialed and not smoke"` (disabled) | Maintainer approval |
| coverage           | 3.12      | Combines coverage from all jobs above        | After all test jobs |

Contributors don't need API keys — the default suite and slow tests run without them.

### Detecting orphaned tests

Every test must carry at least one marker that maps to a CI job. To find tests that would be missed:

```bash
uv run pytest --collect-only -m "not (core or benchmark or interface or slow or live or credentialed or smoke)"
```

If this reports any collected tests, add the appropriate marker (usually `pytestmark = pytest.mark.core` or `pytest.mark.benchmark`) to the file.

## Test Organization

```
tests/
├── conftest.py                 # Shared fixtures, marker implication hook
├── markers.py                  # Skip decorators for missing API keys
├── test_core/                  # Unit tests (no optional deps, marked core)
├── test_interface/             # Integration tests (marked interface + framework)
│   ├── test_agent_integration/ # Framework-specific agent adapters
│   └── test_model_integration/ # Provider-specific model adapters + API contracts
├── test_contract/              # Cross-implementation contract tests (marked contract)
└── test_benchmarks/            # Benchmark tests (marked benchmark)
    ├── test_tau2/              # Tau2 benchmark + data integrity
    ├── test_macs/              # MACS benchmark + data integrity
    ├── test_gaia2/             # GAIA2 benchmark + data integrity
    └── test_multiagentbench/   # MultiAgentBench + data integrity
```

### `test_core/` — Unit tests

Bottom-up tests for core classes in isolation. No optional dependencies.

### `test_interface/` — Integration tests

Tests for framework-specific adapters. Includes HTTP-mocked API contract tests (run by default) and live API tests (marked `credentialed`).

### `test_contract/` — Contract tests

Top-down tests validating that all implementations of the same abstraction behave identically. These are the most critical tests for MASEval's framework-agnostic promise.

### `test_benchmarks/` — Benchmark tests

Benchmark tests follow a **two-tier pattern**:

**Tier 1: Structural tests (offline, `benchmark` marker only)**

Tests that work without downloaded data or network access:

- Import protection: `maseval` runs without benchmark optional dependencies
- Graceful errors: descriptive error when benchmark code is accessed without deps
- Interface checks: class methods exist, types correct, invalid inputs rejected
- Mock-based tests: benchmark pipeline tested with `DummyModelAdapter` and synthetic fixtures

**Tier 2: Real data tests (`benchmark` + `live` markers)**

Tests that download and use actual benchmark data:

- Environment/tool tests: create real environments, execute tools on real databases
- Data loading pipeline: `load_tasks`, `load_domain_config`, etc.
- Data integrity validation (also marked `slow`): schema checks, minimum record counts, field structure

#### Data download pattern

Benchmarks use `ensure_data_exists()` to download data to the **package's default data directory** (not temp dirs). This function caches — it skips download if files already exist. A session-scoped pytest fixture (e.g., `ensure_tau2_data`, `ensure_macs_templates`) triggers the download once per test session.

Tests that need real data should:

1. Depend on the download fixture (`ensure_tau2_data`, `ensure_macs_templates`, etc.)
2. Be marked `@pytest.mark.live`
3. Use simple constructors — e.g., `Tau2Environment({"domain": "retail"})` — since data is already in the default location

Tests that don't need data (structural, mock-based) should NOT depend on the download fixture and should NOT be marked `live`.

#### How to decide: mock or real data?

This is a judgment call. As a guideline:

- If the test validates **structure, types, or error handling** → Tier 1 (offline)
- If the test operates on **real database records, files, or network resources** → Tier 2 (`live`)
- Don't force synthetic fixtures where they add complexity without value. If something needs real data, test it with real data.

Data integrity tests (verifying downloaded data is complete and well-formed) are also marked `slow` since they trigger downloads.

## Patterns

```python
# Mark entire file
pytestmark = pytest.mark.core

# Mark individual test
@pytest.mark.interface
def test_something():
    pass

# Compose markers
@pytest.mark.benchmark
@pytest.mark.live
@pytest.mark.slow
def test_download_data():
    pass

# Skip if optional dependency missing
pytest.importorskip("smolagents")

# Skip if API key missing
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
```

## Notes

- Credentialed tests require maintainer approval via GitHub Environment.
