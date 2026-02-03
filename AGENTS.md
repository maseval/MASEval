# AGENTS.md

## Project Overview

MASEval is an orchestration library for benchmarking LLM-based multi-agent systems. Think of it as PyTorch Lightning for agent evaluation—it provides the execution engine while users implement agent logic.

**Key Architecture Rule:** Strict separation between `maseval/core` (minimal dependencies) and `maseval/interface` (optional integrations). Core must NEVER import from interface.

The library is in early development, so breaking changes that are parsimonous are strongly preferred.

## Setup Commands

```bash
# Sync environment with all dependencies (creates .venv automatically)
uv sync --all-extras --all-groups

# Activate environment
source .venv/bin/activate

# Or use uv run for individual commands (no activation needed)
uv run python examples/amazon_collab.py
uv run pytest tests/
```

## Code Style and Quality

- Line length: 144 characters
- Tool: `ruff`
- All checks must pass in CI before merge

```bash
# Format code
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check . --fix
```

## Testing Instructions

- Tests use pytest markers: `core`, `interface`, `smolagents`, `langgraph`, `contract`
- All tests must pass before PR merge
- Add/update tests for code changes
- Fix type errors and lint issues until suite is green

```bash
# Run all tests
uv run pytest -v

# Core tests only (minimal dependencies)
uv run pytest -m core -v

# Specific integration tests
uv run pytest -m smolagents -v
uv run pytest -m interface -v
```

## Coverage

View coverage by feature area (auto-discovers benchmarks/interfaces):

```bash
uv run python scripts/coverage_by_feature.py
```

Manual coverage for specific modules:

```bash
pytest --cov=maseval.core.agent --cov-report=term-missing
```

## Typing

### Type Checker

The project uses the `ty` type checker.

```bash
# Check types across the project
uv run ty check

# View documentation
uv run ty --help
```

Documentation: [https://docs.astral.sh/ty/](https://docs.astral.sh/ty/)

### Philosophy & Priorities

**Types exist to help users and catch bugs—not to satisfy theoretical purity.**

This library uses type hinting to:

- Provide better IDE autocomplete and error detection
- Document expected types clearly
- Catch real errors before runtime

However, **pragmatism over pedantry**: if a typing pattern improves usability and robustness, use it—even if it technically violates some typing rule.

**Example:** `MACSBenchmark` narrows its environment type to `MACSEnvironment` (instead of generic `Environment`). This violates strict subtyping rules but provides users with:

- Precise autocomplete for MACS-specific methods
- Clear documentation that MACS requires its own environment
- Better error messages during development
- Prevents mixing incompatible components (e.g., `Tau2Environment` cannot be passed to `MACSBenchmark`)

Unless there's a graceful alternative that preserves usability, choose the pattern that helps users most.

**Guiding principle:** Orient yourself on existing patterns in the codebase. Consistency matters more than theoretical correctness.

### Syntax Rules

- **Unions:** Use `A | B` notation (not `Union[A, B]`)
- **Optional:** Prefer `Optional[X]` over `X | None` for explicitness
- **Collections:** Use `List[...]`, `Dict[..., ...]`, `Sequence[...]` instead of `list`, `dict`, `sequence`

**Example:**

```python
def process_data(
    items: List[str],
    config: Optional[Dict[str, Any]] = None
) -> str | int:
    ...
```

## Dependency Management

Three types of dependencies:

- **Core** (`[project.dependencies]`): Required, installed with `pip install maseval`. Keep minimal!
- **Optional** (`[project.optional-dependencies]`): Published for end users. Framework integrations like `smolagents`, `langgraph`, plus convenience bundles like `all` and `examples`. Uses self-references (e.g., `maseval[all]`) to avoid duplication - this is a standard Python packaging feature.
- **Dev Groups** (`[dependency-groups]`): NOT published. Only for contributors. Tools like `pytest`, `ruff`, `mkdocs`.

```bash
# Add core dependency (use sparingly!)
uv add <package-name>

# Add optional dependency for end users (e.g., framework integrations)
uv add --optional <extra-name> <package-name>

# Add development dependency (not published - dev tools only)
uv add --group dev <package-name>

# Remove any dependency
uv remove <package-name>
```

**Important:** `uv add` automatically updates both `pyproject.toml` and `uv.lock`. Never edit lockfile manually.

**Understanding `uv sync` options:**

- `uv sync`: Installs only core dependencies
- `uv sync --extra <name>`: Adds specific optional dependency (e.g., `--extra smolagents`)
- `uv sync --all-extras`: Installs ALL optional dependencies (includes `smolagents`, `langgraph`, etc.)
- `uv sync --group <name>`: Adds specific dev group (e.g., `--group dev`)
- `uv sync --all-groups`: Installs ALL dev groups (`dev`, `docs`)
- `uv sync --all-extras --all-groups`: Full contributor setup with everything

**Note:** Type checking requires `--all-extras` to install all framework packages needed for checking types across all interfaces.

## Architecture Rules

**Core vs Interface Separation (CRITICAL)**

- `maseval/core/`: Essential library logic. NO optional dependencies. Must work with minimal installation.
- `maseval/interface/`: Adapters for external frameworks. ALL dependencies are optional.

**NEVER:**

- Import `maseval/interface` from `maseval/core`
- Add optional dependencies to core
- This is enforced in CI and will fail the build

**Adding new framework integrations:**

1. Create adapter in `maseval/interface/<library>/`
2. Add dependency to `[project.optional-dependencies]`
3. Add tests in `tests/test_interface/`
4. Mark tests with appropriate pytest marker
5. Keep adapters small and well-documented

**Framework Adapter Pattern:**

When implementing adapters for external frameworks, **always use the framework's native message storage as the source of truth**:

**Pattern 1: Persistent State (smolagents)**

```python
class MyFrameworkAdapter(AgentAdapter):
    def get_messages(self) -> MessageHistory:
        """Dynamically fetch from framework's internal storage."""
        # Get from framework (e.g., agent.memory, agent.messages)
        framework_messages = self.agent.get_messages()

        # Convert and return immediately (no caching)
        return self._convert_messages(framework_messages)

    def _run_agent(self, query: str) -> MessageHistory:
        # Run agent (updates framework's internal state)
        self.agent.run(query)

        # Return fresh history
        return self.get_messages()
```

**Pattern 2: Stateless/Result-based (LangGraph)**

```python
def __init__(self, agent_instance, name, callbacks=None, config=None):
    super().__init__(agent_instance, name, callbacks)
    self._last_result = None  # Cache for stateless mode
    self._config = config  # For stateful mode if supported

def get_messages(self) -> MessageHistory:
    # Try persistent state first (if framework supports it)
    if self._config and hasattr(self.agent, 'get_state'):
        state = self.agent.get_state(self._config)
        return self._convert_messages(state.values['messages'])

    # Fall back to cached result
    if self._last_result:
        return self._convert_messages(self._last_result['messages'])

    return MessageHistory()

def _run_agent(self, query: str) -> MessageHistory:
    result = self.agent.invoke(query, config=self._config)
    self._last_result = result  # Cache result
    return self.get_messages()
```

**Why no caching?** Conversion is cheap, and caching creates consistency issues if framework state changes. The framework's internal storage is the single source of truth. For stateless frameworks, cache only the last result and prefer fetching from persistent state if available.

## Documentation

- Built with MkDocs Material theme + mkdocstrings
- Source files in `docs/`, config in `mkdocs.yml`

```bash
# Build with strict checking (catches broken links)
mkdocs build --strict

# Serve locally at http://127.0.0.1:8000
mkdocs serve
```

## Contribution Workflow

1. Create a feature branch (never commit to `main`)
2. Make changes following code style guidelines
3. Run formatters and linters: `ruff format . && ruff check . --fix`
4. Run tests: `pytest -v`
5. Update documentation if needed
6. Open PR against `main` branch
7. Request review from `cemde`
8. Ensure all CI checks pass

**CI Pipeline:** GitHub Actions runs formatting checks, linting, and test suite across Python versions and OS. All checks must pass before merge.

## For VS Code Agents Specifically

**Important:** VS Code agents (GitHub Copilot in VS Code) do NOT have direct access to run terminal commands like `uv`, `python`, `pytest`, or `ruff`.

When you need to run commands:

- Ask the user to execute commands
- Provide clear explanations of what each command does
- Don't attempt to run `uv`, `python`, `pytest`, or similar commands directly, it does not work.
- You CAN read/write files, and execute simple terminal commands such as `ls`, `cd`, `grep` etc.

Example workflow:

1. Make code changes using file edit tools
2. Ask user to run: `ruff format . && ruff check . --fix`
3. Ask user to run: `pytest -v` to verify changes
4. Read test output if user provides it

## Project-Specific Conventions

- Minimum Python: 3.10 (Primary development: 3.12)
- Package manager: `uv` exclusively (don't use `pip install` in development)
- Commit guidelines: Reference issues, keep commits focused, run full test suite before pushing

## Common Tasks Quick Reference

```bash
# Fresh environment setup
uv sync --all-extras --all-groups

# Before committing
uv run ruff format . && uv run ruff check . --fix && uv run pytest -v && uv run ty check

# Run example
uv run python examples/amazon_collab.py

# Update after pulling changes
uv sync --all-extras --all-groups

# Add optional dependency
uv add --optional <extra-name> <package-name>

# Check specific test file
uv run pytest tests/test_core/test_agent.py -v
```

## Security and Confidentiality

**IMPORTANT:** This project contains confidential research material.

- DO NOT publicly distribute code or data
- DO NOT publish without explicit permission
- DO NOT share copyrighted third-party benchmark data

## Changelog

When you complete a task, document your changes in the Changelog. Multiple tasks contribute to a single PR, and PRs are compiled into release changelogs.

### User-Facing Documentation

Write changelog entries from the **user's perspective** - describe what the change means for someone using the library, not what you did internally. Focus on features, fixes, and improvements they'll notice or benefit from.

### Task-Level Documentation

Add an entry for your completed task under the `## Unreleased` section.

### Important Rules

- If you modified something already listed under "Added" in `Unreleased`, **update that existing entry** instead of adding a new one under "Changed"
- Keep entries focused on user impact, not implementation details
- Multiple task entries will be grouped together under the same PR
- PR changelogs are then compiled into release notes between versions

### Format

Brief description of the user-facing change (PR: #PR_NUMBER_PLACEHOLDER)

### Example (User-Facing)

**Good:**

- Added support for custom retry strategies in API client with argument `retry` for `Client.__init__`. (PR: #13)
- Fixed timeout errors when processing large datasets in `func` (PR: #4)

**Bad (not user-focused):**

- Refactored retry logic into separate module
- Updated error handling in data_processor.py

## Docstrings

Write docstrings for **users**, not about your implementation process.

### Rules

- Describe what the code does and how to use it
- Explain parameters, return values, and behavior
- Never write narratives: "I did...", "First we...", "Then I..."
- Never include quality claims: "rigorously tested", "well-optimized"
- Omit implementation details users don't need

### Bad (narrative, claims, implementation details)

```
def calculate_average(numbers: list) -> float:
    """
    I implemented this to calculate averages. First I sum the numbers,
    then divide by count. Rigorously tested and optimized.
    """
```

### Good (clear, user-focused)

```
def calculate_average(numbers: list) -> float:
    """
    Calculate the arithmetic mean of numbers.

    Args:
        numbers: List of numeric values

    Returns:
        Average as float

    Raises:
        ValueError: If list is empty
    """
```

### mkdocs Rendering

This project uses mkdocstrings to render docstrings as HTML. Follow these rules to ensure proper rendering:

**Lists require a blank line before them:**

```python
# Bad - renders as one paragraph
"""Subclasses must provide:
- method_one(): Description
- method_two(): Description
"""

# Good - renders as proper bullet list
"""Subclasses must provide:

- `method_one()` - Description
- `method_two()` - Description
"""
```

**Return descriptions must be single-line** (multi-line creates multiple table rows):

```python
# Bad
"""
Returns:
    TerminationReason indicating why is_done() returns True,
    or NOT_TERMINATED if the interaction is still ongoing.
"""

# Good
"""
Returns:
    Why `is_done()` returns True, or `NOT_TERMINATED` if still ongoing.
"""
```

**For dictionary returns, document fields in the docstring body** using "Output fields:":

```python
# Bad - creates multiple table rows in Returns
"""
Returns:
    Dictionary containing:
    - `name` - User identifier
    - `profile` - User profile data
"""

# Good - fields in body, single-line Returns
"""
Gather execution traces from this user.

Output fields:

- `name` - User identifier
- `profile` - User profile data
- `message_count` - Number of messages in history

Returns:
    Dictionary containing user state and interaction data.
"""
```

**HTML-like strings must be in backticks** (otherwise stripped as HTML):

```python
# Bad - </stop> disappears
"""Uses "</stop>" to signal satisfaction."""

# Good
"""Uses `"</stop>"` to signal satisfaction."""
```

**Use backticks for code references** - method names, parameters, and values: `` `is_done()` ``, `` `stop_tokens` ``, `` `None` ``

## Early-Release Status

**This project is early-release. Clean, maintainable code is the priority - not backwards compatibility.**

- Break APIs if it improves design
- Refactor poor implementations
- Remove technical debt as soon as you identify it
- Don't preserve bad patterns for compatibility reasons
- Focus on getting it right, not keeping it the same

We have zero obligation to maintain backwards compatibility. If you find code messy, propose a fix.
