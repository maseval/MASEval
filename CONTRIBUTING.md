# Contributing to MASEval

First off, thank you for considering contributing to MASEval! It's people like you that make our community great. This document provides a guide for making contributions.

## Guiding Principles

Before diving into the technical details, here are the core principles that guide our development process.

### Contribution Workflow

We follow a standard GitHub workflow. Following these steps makes it easier to review and merge your changes.

1.  **Create a new branch** for your feature or bugfix. Do not commit directly to `main`.
2.  **Make your changes** following our code style (see below).
3.  **Update CHANGELOG.md**: Add a brief entry under `[Unreleased]` in the appropriate section (Added/Changed/Fixed/Removed).
4.  **Open a Pull Request** against the `main` branch - our PR template will guide you through the checklist.
5.  **Request a review** from `cemde`.
6.  Ensure all **automated tests pass** before your PR can be merged. Our automated checks are detailed in the technical section below.

### Architectural Principles

The `maseval` package is designed with a strict separation between its core logic and optional integrations. Understanding this is key to contributing effectively.

1.  **`maseval/core`**: This is the heart of the library. It contains the essential logic and **must not** have any optional dependencies. It should be fully functional with a minimal installation.

2.  **`maseval/interface`**: This contains adapters for other multi-agent frameworks (like `crewai`, `langgraph`, etc.). All dependencies for these integrations are optional.

> [!WARNING]
> Code in `maseval/core` **must never** import from `maseval/interface`. This separation is critical to keep the core package lightweight and dependency-free. Breaking this rule will cause the library to fail.

## Technical Guide for Contributors

This section provides the technical details you'll need to get started with coding.

### 1. Setting up Your Development Environment

We use `uv` for fast and reliable package management. The best way to ensure a consistent environment is to sync it with the project's lockfile.

```bash
# Sync your environment with all dependencies (including dev tools and optional dependencies)
# This command will automatically create a virtual environment if one doesn't exist
uv sync --all-extras --all-groups
```

> **Note**: `uv sync` will automatically create a `.venv` directory if it doesn't exist, so there's no need to run `uv venv` separately. The `--all-extras` flag includes all optional dependencies (like framework integrations), and `--all-groups` includes development tools like `ruff` and `pytest`.

#### Running Commands

You have two options for running commands in your development environment:

**Option 1: Activate the virtual environment** (traditional approach)

```bash
# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Now you can run commands directly
python examples/amazon_collab.py
pytest tests/
ruff format .
```

**Option 2: Use `uv run`** (no activation needed)

```bash
# Run commands directly with uv run
uv run python examples/amazon_collab.py
uv run pytest tests/
uv run ruff format .
```

Both approaches work equally well! Use whichever you prefer. The `uv run` approach is convenient if you don't want to activate the environment, as it automatically uses the correct virtual environment for each command.

### 2. Code Style and Linting

We use `ruff` to enforce a consistent code style. Before committing, please run the formatter and linter.

```bash
# Format the codebase
ruff format .

# Lint the codebase and fix what can be fixed automatically
ruff check . --fix
```

If you haven't activated your virtual environment, you can use `uv run ruff format .` and `uv run ruff check . --fix` instead.

For convenience, you can enable **pre-commit hooks** to automatically format and lint code on every commit:

```bash
uv run pre-commit install
```

This is optional—CI will catch any issues regardless. But if enabled, the hooks will:
- **Format** code with `ruff format` (using project settings from `pyproject.toml`)
- **Lint and auto-fix** issues with `ruff check --fix`

> **Note**: The pre-commit hooks intentionally skip removing unused imports (`F401`) and unused variables (`F841`) to avoid disrupting work-in-progress code. Run `uv run ruff check . --fix` manually before opening a PR to clean these up.

### 3. Dependency Management

Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. Understanding the different dependency types is important:

#### Dependency Types

**Core Dependencies (`[project.dependencies]`)**

- Required for the package to function at all
- Installed by default when users run `pip install maseval`
- Example: `rich` (used for console output in the core library)

**Optional Dependencies (`[project.optional-dependencies]`)**

- Published with the package and installable by end users
- Used for optional features that users can choose to install
- Installed via `pip install maseval[smolagents]` or `pip install maseval[langgraph]`
- Examples: `smolagents`, `langgraph`, `openai` (framework integrations and inference engines)

**Dependency Groups (`[dependency-groups]`)**

- **Not** published with the package (development-only)
- Used by contributors for development, testing, and documentation
- Installed via `uv sync --group dev` or `uv sync --all-groups`
- Examples: `pytest`, `ruff`, `mkdocs` (development and documentation tools)

> **Key Difference**: Optional dependencies are for **end users** who want additional features. Dependency groups are for **contributors** who need development tools. Only optional dependencies are published to PyPI.

#### Adding or Changing Dependencies

If you need to add or change dependencies, use `uv add`. This command automatically updates both `pyproject.toml` and `uv.lock`.

```bash
# Add a core dependency (required for the package to work)
uv add <package-name>

# Add an optional dependency to a specific extra group
uv add --optional <extra-name> <package-name>

# Add a development dependency to a group
uv add --group dev <package-name>

# Remove a dependency
uv remove <package-name>
```

After updating dependencies in a Pull Request, other developers can get the changes simply by running `uv sync --all-extras --all-groups`.

### 4. Building Documentation

Our documentation is built with `MkDocs`. To preview your changes locally:

```bash
# Build the documentation with strict checking
mkdocs build --strict

# Serve the documentation locally at http://127.0.0.1:8000
mkdocs serve
```

> **Tip**: You can also use `uv run mkdocs build --strict` and `uv run mkdocs serve` if you prefer not to activate the environment.

#### View Source Code Links

API reference pages should include links to source files on GitHub. Use the following pattern:

```markdown
[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/path/to/YOUR_NEW_CLASS.py){ .md-source-file align=right }

::: maseval.path.to.YOUR_NEW_CLASS
```

This renders a right-aligned GitHub link above the auto-generated API documentation. See `docs/reference/agent.md` for a complete example.

#### Including Jupyter Notebooks in Documentation

You may include jupyter notebook examples directly into the documentation. We use the `mkdocs-jupyter` plugin to render Jupyter notebooks in the documentation. To avoid duplicating notebook files, we use **symbolic links**.

**Example**: The notebook `examples/aws_collab/amazon_collab.ipynb` is included in the docs via a symlink at `docs/examples/amazon_collab.ipynb`.

To add a new notebook to the documentation:

1. **Create a symlink** from the notebook's location to the `docs/` directory:

   ```bash
   ln -sf ../../examples/your_notebook_dir/notebook.ipynb docs/examples/notebook.ipynb
   ```

2. **Add the notebook to the nav** in `mkdocs.yml`:

   ```yaml
   nav:
     - Examples:
         - Your Notebook: examples/notebook.ipynb
   ```

3. **Commit the symlink** to version control:
   ```bash
   git add docs/examples/notebook.ipynb
   ```

This approach ensures that:

- The notebook remains in its original location (e.g., with related data files)
- Documentation always reflects the latest notebook version
- No manual copying/syncing is required

### 5. Continuous Integration with GitHub Actions

When you open a Pull Request, a series of automated checks will run using **GitHub Actions (GHA)**. This is our continuous integration (CI) pipeline, and it ensures that all contributions meet our quality standards.

The pipeline automatically performs the following tasks:

- **Linting and Formatting**: Verifies that your code adheres to our style guide using `ruff`.
- **Testing**: Runs the entire test suite across different Python versions and operating systems. This includes tests for both the core package and the optional integrations.
- **Type Checking**: Validates type annotations using `ty`.
- **Documentation**: Ensures documentation builds without errors using `mkdocs`.

**All checks must pass** before your Pull Request can be merged. You can view the progress and logs of these checks directly on your Pull Request page in GitHub.

> **Note:** You don't need to run all these checks locally - CI will catch issues. However, running `uv run ruff format && uv run ruff check` before pushing can save you time.

### 6. Implementing Framework Adapters

When creating adapters for external agent frameworks (in `maseval/interface/agents/`), follow these best practices to ensure consistency and reliability:

#### Message History Pattern

**Always use the framework's native message storage as the source of truth.** Do not cache converted messages in the adapter, as this can lead to inconsistencies if the framework's internal state changes.

**Correct Pattern** (SmolAgents example):

```python
class SmolAgentAdapter(AgentAdapter):
    def get_message_history(self) -> MessageHistory:
        """Dynamically fetch and convert messages from framework's memory."""
        # Get messages from framework's internal storage
        smol_messages = self.agent.write_memory_to_messages()

        # Convert and return (no caching)
        return self._convert_smolagents_messages(smol_messages)

    def _run_agent(self, query: str) -> MessageHistory:
        # Run the agent (updates framework's internal memory)
        self.agent.run(query)

        # Return by calling get_message_history() to fetch latest
        return self.get_message_history()
```

**Why this matters:**

1. **Single Source of Truth**: The framework maintains the canonical message history
2. **Always Current**: Each call to `get_message_history()` fetches the latest state
3. **No Sync Issues**: No risk of cached copy becoming stale
4. **Cheap Conversion**: Message format conversion is typically very fast, so caching provides minimal benefit

**Anti-pattern to avoid**:

```python
# ❌ DON'T DO THIS - Cached copy can become stale
def _run_agent(self, query: str) -> MessageHistory:
    self.agent.run(query)
    history = self._convert_messages(self.agent.messages)
    self._cached_history = history  # Bad: creates stale cache
    return history

def get_message_history(self) -> MessageHistory:
    return self._cached_history  # Bad: returns potentially stale data
```

#### Framework Integration Checklist

When adding support for a new framework:

- [ ] Override `get_messages()` to fetch from framework's native storage
- [ ] Implement `_run_agent()` to execute agent and return fresh history
- [ ] Create conversion method (e.g., `_convert_X_messages()`) for message format
- [ ] Handle tool calls and tool responses if the framework supports them
- [ ] Add optional dependency to `pyproject.toml` under `[project.optional-dependencies]`
- [ ] Add conditional import in `maseval/interface/agents/__init__.py`
- [ ] Write integration tests in `tests/test_interface/`
- [ ] Update documentation with usage examples
- [ ] Provide a `logs` property inside the `AgentAdapter`.

#### Framework-Specific Patterns

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

        # Return by calling get_messages() to fetch latest
        return self.get_messages()
```

**Why This Works:**

1. **Single Source of Truth**: Framework's internal storage is authoritative
2. **Always Current**: Each call to `get_messages()` fetches the latest state

**Stateless/Result-based (LangGraph pattern)**:

Frameworks that return results without persistent state can cache the last result:

```python
def __init__(self, agent_instance, name, callbacks=None, config=None):
    super().__init__(agent_instance, name, callbacks)
    self._last_result = None
    self._config = config  # For stateful mode if supported

def get_messages(self) -> MessageHistory:
    # Try fetching from persistent state if configured
    if self._config and hasattr(self.agent, 'get_state'):
        state = self.agent.get_state(self._config)
        return self._convert_messages(state.values['messages'])

    # Fall back to cached result
    if self._last_result:
        return self._convert_messages(self._last_result['messages'])

    return MessageHistory()

def _run_agent(self, query: str) -> MessageHistory:
    result = self.agent.invoke(query, config=self._config)
    self._last_result = result  # Cache for stateless mode
    return self.get_messages()
```

The key principle: **Always try to fetch from the framework's source of truth first**, fall back to caching only when the framework doesn't provide persistent state access.
