# Interface package

Adapters and lightweight integration shims for external libraries live under
`maseval.interface`. Examples: `maseval.interface.smolagents`,
`maseval.interface.langraph`, `maseval.interface.inference.openai`.

## Key rules

- Keep adapters small: they translate between MASEval abstractions (Task,
  Environment, AgentAdapter, Benchmark) and an external API.
- Avoid heavy imports at module import time; import optional dependencies lazily
  inside functions or methods and raise helpful ImportError messages when missing.
- Prefer explicit, focused modules over monolithic files that mix many adapters.

### Import patterns

**Typing imports** are fine to use with `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import SomeType  # Only used for type hints
```

**Classes that inherit from framework types** must go in a separate `<framework>_optional.py` file. This allows the main module to work without the framework installed (for type checking, documentation, and import error messages).

Example: `smolagents.py` contains `SmolAgentAdapter` (inherits from MASEval types), while `smolagents_optional.py` contains `SmolAgentUserSimulationInputTool` (inherits from `smolagents.UserInputTool`).

The main module imports from the optional module lazily:

```python
def get_tool(self):
    from maseval.interface.agents.smolagents_optional import SmolAgentUserSimulationInputTool
    return SmolAgentUserSimulationInputTool(self)
```

## Short layout

maseval/interface/

- **init**.py # package-level doc, optional imports
- <integration>.py or <integration>/ # adapters may be a single module (e.g. `langgraph.py`) or a subpackage/directory (e.g. `agents/`, `inference/`) containing multiple helper modules

## Docs and tests

- Document optional dependencies in `pyproject.toml` extras (e.g. `maseval[smolagents]`).
- Add adapter documentation under `docs/interface/` using mkdocstrings to
  render adapter classes when importable (dont forget to edit `mkdocs.yml`, if necessary). Also add the module to `docs/reference/....md` under `Interface`.
- Tests for adapters should live under `tests/interface/` and should skip or mark
  failures gracefully when optional dependencies are not installed.

## Rationale

Centralizing adapters under `maseval.interface` keeps integration code discoverable
and separate from core logic. This makes it easier to maintain optional deps and
avoid surprising import-time side effects.

Important: never import from `maseval.interface` into `maseval.core` or `maseval.utils`.
