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

Six jobs in `.github/workflows/test.yml`:

| Job               | Python    | What it runs                      | Gate                   |
| ----------------- | --------- | --------------------------------- | ---------------------- |
| test-core         | 3.10–3.14 | `-m core`                         | —                      |
| test-benchmark    | 3.10–3.14 | `-m benchmark`                    | —                      |
| test-all          | 3.10–3.14 | `pytest -v` (default filter)      | After core + benchmark |
| test-slow         | 3.12      | `-m "slow and not credentialed"`  | —                      |
| test-credentialed | 3.12      | `-m "credentialed and not smoke"` | Maintainer approval    |
| coverage          | 3.12      | Full suite with coverage report   | —                      |

Contributors don't need API keys — the default suite and slow tests run without them.

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
    └── test_gaia2/             # GAIA2 benchmark
```

### `test_core/` — Unit tests

Bottom-up tests for core classes in isolation. No optional dependencies.

### `test_interface/` — Integration tests

Tests for framework-specific adapters. Includes HTTP-mocked API contract tests (run by default) and live API tests (marked `credentialed`).

### `test_contract/` — Contract tests

Top-down tests validating that all implementations of the same abstraction behave identically. These are the most critical tests for MASEval's framework-agnostic promise.

### `test_benchmarks/` — Benchmark tests

Benchmark implementations and data integrity validation. Data integrity tests are marked `live` + `slow`.

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
