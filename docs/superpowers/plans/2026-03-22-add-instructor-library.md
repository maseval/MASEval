# Add Instructor Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the [instructor](https://github.com/567-labs/instructor) library as core infrastructure for structured LLM output handling — both internally and as a user-facing API. This is not a patch on top of existing code; it is a clean replacement. The custom JSON extraction, schema flattening, and retry logic are removed, not wrapped with fallbacks.

**Why:** Reliable structured output from unreliable models is critical for researchers who use cheap/small models due to cost constraints. Hand-rolled JSON parsing and retry logic is finicky, under-tested, and reimplemented in multiple places. Instructor provides a battle-tested foundation (3M+ monthly downloads) for validation, retries with error feedback to the model, and multi-provider support. By making it core infrastructure, every future structured output need builds on this foundation rather than reinventing it.

**Design principles:**
1. **Clean replacement, not a compatibility layer.** Per AGENTS.md: "Clean, maintainable code is the priority — not backwards compatibility." Old code (`_extract_json_object`, `_flatten_schema`, manual retry loops) is deleted, not preserved as fallbacks. If instructor handles it, the old path is gone.
2. **Infrastructure for future work.** This is a seed ecosystem — the integration points are designed so that upcoming features (partial streaming, custom validators, fallback models) plug in naturally.
3. **Follow existing patterns.** The `ModelAdapter` / provider adapter pattern stays. Instructor slots into this architecture cleanly via `_structured_chat()` overrides in each provider adapter.

**Architecture:** Instructor wraps provider clients via `instructor.from_provider()` to add `response_model` support with automatic validation and retries. All providers use a unified API: `client.chat.completions.create(response_model=..., messages=...)`. We integrate at the `ModelAdapter` level: each adapter creates an instructor-patched client alongside the raw client. The public `chat()` method gains an optional `response_model` parameter. Simulators switch fully to `response_model` — no legacy JSON parsing fallback. Tau2 schema flattening uses instructor's `openai_schema()` as a base.

**Tech Stack:** Python, instructor (>=1.14.0), pydantic (>=2.10.6, already a core dep)

**Key instructor API facts (verified against v1.14.4):**
- `instructor.from_provider("provider/model")` — unified client creation (no `from_gemini` or `from_anthropic`)
- `instructor.from_openai(client)` — OpenAI-specific wrapping
- `instructor.from_litellm(completion_fn)` — LiteLLM wrapping
- All wrapped clients use `client.chat.completions.create(response_model=..., messages=...)` uniformly
- `instructor.openai_schema(MyModel)` — generate clean OpenAI-compatible schemas (returns object with `.openai_schema` dict containing `name`, `description`, `parameters`)
- Note: `openai_schema()` still produces `anyOf` for `Optional` fields — we keep `_flatten_schema()` as a thin utility for providers (like Google GenAI) that reject `anyOf`

**Project conventions (from AGENTS.md):**
- Use `uv add` for dependencies, `uv run` for commands, never `pip install`
- Union syntax: `A | B`, optionals: `Optional[X]`, collections: `List`, `Dict`
- Core (`maseval/core/`) must NOT import from interface (`maseval/interface/`)
- `just all` before committing (format + lint + typecheck + test)

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `pyproject.toml` | Dependencies | Modify: add `instructor>=1.14.0` to core deps via `uv add` |
| `maseval/core/model.py` | ModelAdapter base + ChatResponse | Modify: add `response_model` param to `chat()`, add `_structured_chat()`, add `structured_response` field to ChatResponse |
| `maseval/core/instructor.py` | Instructor integration helpers | Create: `create_instructor_client()` helper, `flatten_model_schema()` |
| `maseval/interface/inference/openai.py` | OpenAI adapter | Modify: create instructor client, override `_structured_chat()` |
| `maseval/interface/inference/anthropic.py` | Anthropic adapter | Modify: create instructor client, override `_structured_chat()` |
| `maseval/interface/inference/google_genai.py` | Google adapter | Modify: create instructor client, override `_structured_chat()` |
| `maseval/interface/inference/litellm.py` | LiteLLM adapter | Modify: create instructor client, override `_structured_chat()` |
| `maseval/core/simulator.py` | LLM simulators | Modify: add Pydantic response models, use `response_model` in simulators with legacy fallback |
| `maseval/benchmark/tau2/tau2.py` | Tau2 benchmark | Modify: replace `_flatten_schema()` usage in both `_build_tool_definitions()` (line 897) and `_get_tool_definitions()` (line 1231) with `flatten_model_schema()` |
| `tests/test_core/test_instructor_integration.py` | Instructor integration tests | Create: test `response_model` on ModelAdapter |
| `tests/test_core/test_llm_simulator.py` | Existing simulator tests | Modify: add response model tests, verify existing tests pass |
| `CHANGELOG.md` | Changelog | Modify: add entry under Unreleased |

---

## Task 1: Add instructor dependency and create integration module

**Files:**
- Modify: `pyproject.toml:24-29`
- Create: `maseval/core/instructor.py`
- Test: `tests/test_core/test_instructor_integration.py`

- [ ] **Step 1: Write failing test for instructor import**

```python
# tests/test_core/test_instructor_integration.py
"""Test instructor library integration."""
import pytest


@pytest.mark.core
class TestInstructorAvailable:
    """Verify instructor is importable as a core dependency."""

    def test_instructor_importable(self):
        """instructor should be importable since it's a core dep."""
        import instructor
        assert hasattr(instructor, "from_openai")

    def test_instructor_helpers_importable(self):
        """maseval.core.instructor helpers should be importable."""
        from maseval.core.instructor import create_instructor_client
        assert callable(create_instructor_client)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_instructor_integration.py -v -x`
Expected: FAIL — instructor not installed, module not found

- [ ] **Step 3: Add instructor to core dependencies**

Run: `uv add instructor`

This updates both `pyproject.toml` and `uv.lock` automatically.

- [ ] **Step 4: Create maseval/core/instructor.py**

```python
"""Instructor library integration for structured LLM outputs.

Provides helpers to create instructor-patched clients from provider SDK clients
and to generate flattened JSON schemas from Pydantic models.

Instructor adds ``response_model`` support with automatic validation and retries
to any supported LLM provider.

Example:
    ```python
    from maseval.core.instructor import create_instructor_client

    # Wrap an OpenAI client
    import openai
    client = openai.OpenAI()
    instructor_client = create_instructor_client(client, provider="openai")
    ```
"""

from __future__ import annotations

from typing import Any, Optional, Dict


def create_instructor_client(
    client: Any,
    provider: str,
    mode: Optional[str] = None,
) -> Any:
    """Create an instructor-patched client from a provider SDK client.

    All patched clients expose a unified API:
    ``client.chat.completions.create(response_model=..., messages=...)``.

    Args:
        client: The provider SDK client instance (e.g., ``openai.OpenAI()``,
            ``anthropic.Anthropic()``). For LiteLLM, pass ``litellm.completion``.
        provider: Provider name. One of: ``"openai"``, ``"litellm"``.
            For other providers, use ``instructor.from_provider()`` directly.
        mode: Optional instructor mode override. If None, uses the default
            for the provider.

    Returns:
        An instructor-patched client supporting ``response_model``.

    Raises:
        ValueError: If provider is not recognized.
    """
    import instructor

    kwargs: Dict[str, Any] = {}
    if mode is not None:
        kwargs["mode"] = getattr(instructor.Mode, mode.upper(), mode)

    if provider == "openai":
        return instructor.from_openai(client, **kwargs)
    elif provider == "litellm":
        return instructor.from_litellm(client, **kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Use instructor.from_provider() directly for other providers."
        )


def flatten_model_schema(model: type) -> Dict[str, Any]:
    """Generate a flattened JSON schema from a Pydantic model.

    Uses instructor's ``openai_schema`` to produce a clean schema, then
    applies additional flattening to remove ``anyOf`` (for ``Optional``
    fields) and other constructs that some providers reject.

    This replaces the manual ``_flatten_schema()`` function that was
    previously needed to post-process Pydantic v2 schemas.

    Args:
        model: A Pydantic BaseModel subclass.

    Returns:
        A flat JSON schema dict suitable for LLM tool parameters.
    """
    import instructor

    schema_obj = instructor.openai_schema(model)
    schema = schema_obj.openai_schema["parameters"]

    # instructor's openai_schema still produces anyOf for Optional fields.
    # Flatten those for provider compatibility (especially Google GenAI).
    return _resolve_schema(schema)


def _resolve_schema(node: Any) -> Any:
    """Recursively resolve anyOf and strip unsupported keys from a schema."""
    if not isinstance(node, dict):
        return node

    _STRIP_KEYS = {"$defs", "additionalProperties", "title", "default"}

    # Simplify anyOf (Optional[X] -> X with nullable)
    if "anyOf" in node:
        variants = node["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        if len(non_null) == 1:
            resolved = _resolve_schema(non_null[0])
            if isinstance(resolved, dict):
                resolved["nullable"] = True
                if "description" in node and "description" not in resolved:
                    resolved["description"] = node["description"]
            return resolved
        if non_null:
            return _resolve_schema(non_null[0])

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIP_KEYS or key == "anyOf":
            continue
        if isinstance(value, dict):
            out[key] = _resolve_schema(value)
        elif isinstance(value, list):
            out[key] = [_resolve_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_instructor_integration.py -v -x`
Expected: PASS

- [ ] **Step 6: Run full existing test suite**

Run: `uv run pytest tests/test_core/ -v --tb=short`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock maseval/core/instructor.py tests/test_core/test_instructor_integration.py
git commit -m "feat: add instructor as core dependency with integration helpers"
```

---

## Task 2: Add response_model support to ChatResponse and ModelAdapter base

**Files:**
- Modify: `maseval/core/model.py:62-135` (ChatResponse)
- Modify: `maseval/core/model.py:207-342` (ModelAdapter.chat)
- Test: `tests/test_core/test_instructor_integration.py`

- [ ] **Step 1: Write failing tests for response_model support**

Append to `tests/test_core/test_instructor_integration.py`:

```python
from pydantic import BaseModel
from conftest import DummyModelAdapter
from maseval.core.model import ChatResponse


class WeatherResponse(BaseModel):
    city: str
    temperature: float
    unit: str


@pytest.mark.core
class TestChatResponseStructured:
    """Test ChatResponse with structured_response field."""

    def test_chat_response_has_structured_response_field(self):
        """ChatResponse should have an optional structured_response field."""
        resp = ChatResponse(content='{"city": "Paris"}')
        assert resp.structured_response is None

    def test_chat_response_with_structured_response(self):
        """ChatResponse can hold a parsed Pydantic model."""
        weather = WeatherResponse(city="Paris", temperature=20.0, unit="celsius")
        resp = ChatResponse(content='{"city": "Paris"}', structured_response=weather)
        assert resp.structured_response is not None
        assert resp.structured_response.city == "Paris"


@pytest.mark.core
class TestModelAdapterResponseModel:
    """Test ModelAdapter.chat() with response_model parameter."""

    def test_chat_accepts_response_model_param(self):
        """chat() should accept a response_model keyword argument."""
        import inspect
        from maseval.core.model import ModelAdapter
        sig = inspect.signature(ModelAdapter.chat)
        assert "response_model" in sig.parameters

    def test_chat_without_response_model_unchanged(self):
        """chat() without response_model behaves exactly as before."""
        model = DummyModelAdapter(responses=["Hello"])
        result = model.chat([{"role": "user", "content": "Hi"}])
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello"
        assert result.structured_response is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_instructor_integration.py::TestChatResponseStructured -v -x`
Expected: FAIL — `structured_response` field doesn't exist

- [ ] **Step 3: Add structured_response field to ChatResponse**

In `maseval/core/model.py`, add to the `ChatResponse` dataclass (after `stop_reason` on line 105):

```python
    structured_response: Optional[Any] = None
```

Update the docstring to include:

```
        structured_response: The validated Pydantic model instance when
            ``response_model`` was used with ``chat()``. None otherwise.
```

- [ ] **Step 4: Add response_model and max_retries parameters to ModelAdapter.chat()**

In `maseval/core/model.py`, modify the `chat()` method signature (lines 207-214) to:

```python
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], MessageHistory],
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_model: Optional[type] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> ChatResponse:
```

Update the docstring Args section to include:

```
            response_model: Optional Pydantic BaseModel class. When provided,
                the model's response is validated against this schema and
                returned in ``ChatResponse.structured_response``. Uses
                instructor for automatic validation and retries.
            max_retries: Number of retries on validation failure when using
                ``response_model``. Default is 3. Ignored without ``response_model``.
```

In the `try` block (around line 284), add branching for response_model:

```python
        try:
            if response_model is not None:
                result = self._structured_chat(
                    messages_list,
                    response_model=response_model,
                    max_retries=max_retries,
                    generation_params=generation_params,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            else:
                result = self._chat_impl(
                    messages_list,
                    generation_params=generation_params,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
```

- [ ] **Step 5: Add _structured_chat() method to ModelAdapter**

Add after `_chat_impl` (around line 367):

```python
    def _structured_chat(
        self,
        messages: List[Dict[str, Any]],
        response_model: type,
        max_retries: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Internal structured chat using instructor.

        Subclasses that support instructor should override this method.
        The default implementation falls back to ``_chat_impl`` and attempts
        manual JSON parsing of the response content.

        Args:
            messages: List of message dicts.
            response_model: Pydantic model class for response validation.
            max_retries: Number of retries on validation failure.
            generation_params: Generation parameters.
            tools: Tool definitions, if any.
            tool_choice: Tool choice setting, if any.
            **kwargs: Additional arguments.

        Returns:
            ChatResponse with ``structured_response`` populated.
        """
        # Base class raises — subclasses must override with their
        # instructor-patched client. No silent fallback to unstructured output.
        raise NotImplementedError(
            f"{type(self).__name__} does not support response_model. "
            f"Override _structured_chat() with an instructor-patched client."
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_instructor_integration.py -v -x`
Expected: PASS

- [ ] **Step 7: Run full existing test suite**

Run: `uv run pytest tests/test_core/ -v --tb=short`
Expected: All existing tests still PASS (the new `structured_response` field defaults to None)

- [ ] **Step 8: Commit**

```bash
git add maseval/core/model.py tests/test_core/test_instructor_integration.py
git commit -m "feat: add response_model support to ModelAdapter.chat() and ChatResponse"
```

---

## Task 3: Implement instructor support in provider adapters

**Files:**
- Modify: `maseval/interface/inference/openai.py`
- Modify: `maseval/interface/inference/anthropic.py`
- Modify: `maseval/interface/inference/google_genai.py`
- Modify: `maseval/interface/inference/litellm.py`
- Test: `tests/test_core/test_instructor_integration.py`

**Important:** All provider adapters use instructor's unified API after wrapping. `instructor.from_provider("provider/model")` returns an `Instructor` instance where all calls go through `client.chat.completions.create(response_model=..., messages=...)` regardless of the underlying provider. For OpenAI, we use `instructor.from_openai(client)`. For LiteLLM, `instructor.from_litellm(litellm.completion)`. For Anthropic and Google, we use `instructor.from_provider()` since there are no dedicated `from_anthropic`/`from_gemini` functions in current instructor.

- [ ] **Step 1: Write failing tests for provider adapter instructor support**

Append to `tests/test_core/test_instructor_integration.py`:

```python
from unittest.mock import MagicMock


@pytest.mark.core
class TestOpenAIInstructorSupport:
    """Test OpenAI adapter creates instructor client."""

    def test_openai_adapter_has_instructor_client(self):
        """OpenAIModelAdapter should create an instructor-patched client."""
        from maseval.interface.inference import OpenAIModelAdapter
        mock_client = MagicMock()
        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4")
        assert hasattr(adapter, "_instructor_client")

    def test_openai_adapter_structured_chat_uses_instructor(self):
        """OpenAIModelAdapter._structured_chat should use instructor client."""
        from maseval.interface.inference import OpenAIModelAdapter
        mock_client = MagicMock()
        adapter = OpenAIModelAdapter(client=mock_client, model_id="gpt-4")

        # Mock the instructor client
        mock_response = WeatherResponse(city="Paris", temperature=20.0, unit="celsius")
        adapter._instructor_client = MagicMock()
        adapter._instructor_client.chat.completions.create.return_value = mock_response

        result = adapter.chat(
            [{"role": "user", "content": "Weather in Paris?"}],
            response_model=WeatherResponse,
        )

        assert result.structured_response is not None
        assert result.structured_response.city == "Paris"
        adapter._instructor_client.chat.completions.create.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_instructor_integration.py::TestOpenAIInstructorSupport -v -x`
Expected: FAIL — no `_instructor_client` attribute

- [ ] **Step 3: Add instructor support to OpenAIModelAdapter**

In `maseval/interface/inference/openai.py`:

Add to `__init__` (after existing setup, around line 92):
```python
        # Create instructor-patched client for structured outputs
        from maseval.core.instructor import create_instructor_client
        self._instructor_client = create_instructor_client(client, provider="openai")
```

Add `_structured_chat` override (after `_chat_impl`):
```python
    def _structured_chat(
        self,
        messages: List[Dict[str, Any]],
        response_model: type,
        max_retries: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Use instructor for structured output with validation and retries."""
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed

        result = self._instructor_client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **params,
        )

        # result is a validated Pydantic model instance
        return ChatResponse(
            content=result.model_dump_json(),
            structured_response=result,
            role="assistant",
            model=self._model_id,
        )
```

- [ ] **Step 4: Add instructor support to AnthropicModelAdapter**

In `maseval/interface/inference/anthropic.py`:

Add to `__init__` (after existing setup, around line 109):
```python
        # Create instructor-patched client for structured outputs
        import instructor
        self._instructor_client = instructor.from_provider("anthropic/" + model_id)
```

Note: We use `from_provider` since there's no `from_anthropic` in current instructor.

Add `_structured_chat` override:
```python
    def _structured_chat(
        self,
        messages: List[Dict[str, Any]],
        response_model: type,
        max_retries: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Use instructor for structured output with validation and retries."""
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        max_tokens = params.pop("max_tokens", self._max_tokens)
        params["max_tokens"] = max_tokens

        result = self._instructor_client.chat.completions.create(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            **params,
        )

        return ChatResponse(
            content=result.model_dump_json(),
            structured_response=result,
            role="assistant",
            model=self._model_id,
        )
```

- [ ] **Step 5: Add instructor support to GoogleGenAIModelAdapter**

In `maseval/interface/inference/google_genai.py`:

Add to `__init__` (after existing setup, around line 85):
```python
        # Create instructor-patched client for structured outputs
        import instructor
        self._instructor_client = instructor.from_provider("gemini/" + model_id)
```

Add `_structured_chat` override:
```python
    def _structured_chat(
        self,
        messages: List[Dict[str, Any]],
        response_model: type,
        max_retries: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Use instructor for structured output with validation and retries."""
        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed

        result = self._instructor_client.chat.completions.create(
            response_model=response_model,
            messages=messages,
            max_retries=max_retries,
            **params,
        )

        return ChatResponse(
            content=result.model_dump_json(),
            structured_response=result,
            role="assistant",
            model=self._model_id,
        )
```

- [ ] **Step 6: Add instructor support to LiteLLMModelAdapter**

In `maseval/interface/inference/litellm.py`:

Add to `__init__` (after existing setup, around line 101):
```python
        # Create instructor-patched completion function for structured outputs.
        # Deferred to first use since litellm is an optional import.
        self._instructor_client = None
```

Add helper + `_structured_chat` override:
```python
    def _get_instructor_client(self) -> Any:
        """Lazily create instructor-patched LiteLLM client."""
        if self._instructor_client is None:
            try:
                import litellm
            except ImportError as e:
                raise ImportError("LiteLLM is not installed. Install with: pip install maseval[litellm]") from e
            from maseval.core.instructor import create_instructor_client
            self._instructor_client = create_instructor_client(litellm.completion, provider="litellm")
        return self._instructor_client

    def _structured_chat(
        self,
        messages: List[Dict[str, Any]],
        response_model: type,
        max_retries: int = 3,
        generation_params: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Use instructor for structured output with validation and retries."""
        client = self._get_instructor_client()

        params = dict(self._default_generation_params)
        if generation_params:
            params.update(generation_params)
        params.update(kwargs)

        if self._seed is not None and "seed" not in params:
            params["seed"] = self._seed
        if self._api_key:
            params["api_key"] = self._api_key
        if self._api_base:
            params["api_base"] = self._api_base

        result = client(
            model=self._model_id,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **params,
        )

        return ChatResponse(
            content=result.model_dump_json(),
            structured_response=result,
            role="assistant",
            model=self._model_id,
        )
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_core/test_instructor_integration.py -v -x`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `uv run pytest tests/test_core/ -v --tb=short`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add maseval/interface/inference/openai.py maseval/interface/inference/anthropic.py maseval/interface/inference/google_genai.py maseval/interface/inference/litellm.py tests/test_core/test_instructor_integration.py
git commit -m "feat: add instructor support to all provider adapters"
```

---

## Task 4: Rework simulators to use instructor

**Files:**
- Modify: `maseval/core/simulator.py`
- Test: `tests/test_core/test_llm_simulator.py`

**Context:** Simulators currently use `model.generate()` (text-in/text-out) and manually parse JSON using `_extract_json_object()` + `json.loads()`. With instructor, we switch fully to `model.chat(messages=[...], response_model=OutputModel)` to get validated Pydantic models directly. The old `_extract_json_object()`, `_parse_output()`, and manual retry logic are deleted — instructor handles validation and retries.

- [ ] **Step 1: Write failing test for Pydantic response models**

Append to `tests/test_core/test_llm_simulator.py`:

```python
@pytest.mark.core
class TestSimulatorResponseModels:
    """Test that simulator response Pydantic models work correctly."""

    def test_tool_simulator_response_model_exists(self):
        from maseval.core.simulator import ToolSimulatorResponse
        resp = ToolSimulatorResponse(text="success", details={"key": "value"})
        assert resp.text == "success"
        assert resp.details == {"key": "value"}

    def test_user_simulator_response_model_exists(self):
        from maseval.core.simulator import UserSimulatorResponse
        resp = UserSimulatorResponse(text="I need help")
        assert resp.text == "I need help"

    def test_agentic_user_simulator_response_model_exists(self):
        from maseval.core.simulator import AgenticUserSimulatorResponse
        resp = AgenticUserSimulatorResponse(
            text="Let me check",
            tool_calls=[{"name": "check_status", "arguments": {}}],
        )
        assert resp.text == "Let me check"
        assert len(resp.tool_calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_llm_simulator.py::TestSimulatorResponseModels -v -x`
Expected: FAIL — models don't exist yet

- [ ] **Step 3: Add Pydantic response models to simulator.py**

In `maseval/core/simulator.py`, after imports (before `_extract_json_object`), add:

```python
from pydantic import BaseModel, Field


class ToolSimulatorResponse(BaseModel):
    """Expected output format for ToolLLMSimulator."""
    text: str = Field(default="", description="Human-readable description of the tool's output")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured tool output data")


class UserSimulatorResponse(BaseModel):
    """Expected output format for UserLLMSimulator."""
    text: str = Field(default="", description="The user's response text")


class AgenticUserSimulatorResponse(BaseModel):
    """Expected output format for AgenticUserLLMSimulator."""
    text: str = Field(default="", description="The user's response text")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="List of tool calls")
```

- [ ] **Step 4: Add _response_model and _parse_structured_response to LLMSimulator base**

In `LLMSimulator` class, add class attribute:
```python
    _response_model: Optional[type] = None
```

Add method:
```python
    def _parse_structured_response(self, response: Any) -> Any:
        """Convert instructor-validated response to expected return format.

        Override in subclasses to convert the Pydantic model instance
        to the format expected by callers.
        """
        return response
```

- [ ] **Step 5: Rewrite LLMSimulator.__call__ to use instructor directly**

Replace the inner loop body. No legacy fallback — instructor handles validation and retries via `response_model`. Delete `_extract_json_object()` and `_parse_output()` methods entirely.

```python
            try:
                chat_result = self.model.chat(
                    messages=[{"role": "user", "content": prompt}],
                    response_model=self._response_model,
                    max_retries=self.max_try,
                    generation_params=generation_params,
                )
                parsed_result = self._parse_structured_response(chat_result.structured_response)
                entry["raw_output"] = chat_result.content
                entry["parsed_output"] = parsed_result
                entry["status"] = SimulatorCallStatus.Successful.value
            except Exception as e:
                entry["raw_output"] = None
                entry["status"] = SimulatorCallStatus.ModelCallError.value
                entry["error"] = str(e)
            self.logs.append(entry)
```

- [ ] **Step 6: Delete legacy parsing code**

Remove from `simulator.py`:
- `_extract_json_object()` function (lines 13-27)
- `_parse_output()` methods from all simulator subclasses
- Manual JSON retry logic in `__call__` (the old `json.loads` / `json.JSONDecodeError` paths)

- [ ] **Step 6: Wire up response models in simulator subclasses**

In `ToolLLMSimulator`:
```python
    _response_model = ToolSimulatorResponse

    def _parse_structured_response(self, response: ToolSimulatorResponse) -> tuple[str, Dict[str, Any]]:
        return response.text, response.details
```

In `UserLLMSimulator`:
```python
    _response_model = UserSimulatorResponse

    def _parse_structured_response(self, response: UserSimulatorResponse) -> str:
        return response.text
```

In `AgenticUserLLMSimulator`:
```python
    _response_model = AgenticUserSimulatorResponse

    def _parse_structured_response(self, response: AgenticUserSimulatorResponse) -> Tuple[str, List[Dict[str, Any]]]:
        return response.text, response.tool_calls
```

- [ ] **Step 7: Run all simulator tests**

Run: `uv run pytest tests/test_core/test_llm_simulator.py -v --tb=short`
Expected: All existing tests still PASS (DummyModelAdapter returns text via `generate()`, so the fallback path is exercised)

- [ ] **Step 8: Run full core test suite**

Run: `uv run pytest tests/test_core/ -v --tb=short`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add maseval/core/simulator.py tests/test_core/test_llm_simulator.py
git commit -m "feat: add instructor-based structured output to simulators with legacy fallback"
```

---

## Task 5: Replace _flatten_schema in Tau2

**Files:**
- Modify: `maseval/benchmark/tau2/tau2.py:781-902` and line 1231
- Test: `tests/test_core/test_instructor_integration.py`

**Context:** `_flatten_schema()` is called in two places:
1. `_build_tool_definitions()` at line 897
2. `_get_tool_definitions()` at line 1231

Both must be updated to use `flatten_model_schema()` from `maseval.core.instructor`.

- [ ] **Step 1: Write failing test for instructor-based schema generation**

Append to `tests/test_core/test_instructor_integration.py`:

```python
@pytest.mark.core
class TestInstructorSchemaGeneration:
    """Test that flatten_model_schema produces clean schemas."""

    def test_generates_clean_schema(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        from maseval.core.instructor import flatten_model_schema

        class OrderParams(BaseModel):
            order_id: str = Field(description="The order ID")
            status: Optional[str] = Field(default=None, description="Filter by status")

        flat = flatten_model_schema(OrderParams)
        assert "$ref" not in str(flat)
        assert "$defs" not in str(flat)
        assert "anyOf" not in str(flat)
        assert flat["type"] == "object"
        assert "order_id" in flat["properties"]

    def test_handles_nested_models(self):
        from pydantic import BaseModel, Field
        from maseval.core.instructor import flatten_model_schema

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        flat = flatten_model_schema(Person)
        assert "$ref" not in str(flat)
        assert "address" in flat["properties"]
```

- [ ] **Step 2: Run test to verify it passes (flatten_model_schema was created in Task 1)**

Run: `uv run pytest tests/test_core/test_instructor_integration.py::TestInstructorSchemaGeneration -v -x`
Expected: PASS (function already exists from Task 1)

- [ ] **Step 3: Replace _flatten_schema calls in tau2.py**

In `maseval/benchmark/tau2/tau2.py`:

1. Add import at the top of the `_build_tool_definitions` function (replace the existing `from typing import Any as TypingAny` line area):
```python
    from maseval.core.instructor import flatten_model_schema
```

2. Replace line 897:
```python
# Before:
"parameters": _flatten_schema(params_model.model_json_schema()),
# After:
"parameters": flatten_model_schema(params_model),
```

3. Replace line 1231 (in `_get_tool_definitions()`):
```python
# Before:
"parameters": _flatten_schema(params_model.model_json_schema()),
# After:
from maseval.core.instructor import flatten_model_schema
...
"parameters": flatten_model_schema(params_model),
```

(Add the import once at the top of `_get_tool_definitions`, not inline at line 1231.)

- [ ] **Step 4: Delete _flatten_schema() function**

Remove the `_flatten_schema()` function (lines 781-834) from `tau2.py`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_core/ -v --tb=short`
Expected: PASS

Run: `uv run pytest tests/ -v --tb=short -m "not (slow or credentialed or smoke)"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add maseval/benchmark/tau2/tau2.py tests/test_core/test_instructor_integration.py
git commit -m "feat: replace _flatten_schema with instructor-based schema generation in Tau2"
```

---

## Task 6: Update exports and changelog

**Files:**
- Modify: `maseval/core/__init__.py` (if it has explicit exports)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Check and update exports**

Check what's currently exported from `maseval/core/__init__.py` and `maseval/__init__.py`. If they have explicit `__all__` or import statements, add:

```python
from .instructor import create_instructor_client, flatten_model_schema
```

Also export the simulator response models if they're useful to users:
```python
from .simulator import ToolSimulatorResponse, UserSimulatorResponse, AgenticUserSimulatorResponse
```

- [ ] **Step 2: Update CHANGELOG.md**

Add under `## Unreleased`:

```markdown
### Added

- Added `instructor` as a core dependency for structured LLM output handling with automatic validation and retries.
- Added `response_model` parameter to `ModelAdapter.chat()` — pass a Pydantic `BaseModel` class to get validated structured outputs via `ChatResponse.structured_response`.
- Added `structured_response` field to `ChatResponse` for accessing parsed Pydantic model instances.
- Added `maseval.core.instructor` module with `create_instructor_client()` and `flatten_model_schema()` helpers.
- Added Pydantic response models for simulators: `ToolSimulatorResponse`, `UserSimulatorResponse`, `AgenticUserSimulatorResponse`.
- Simulators now use instructor for structured output parsing with automatic fallback to legacy JSON extraction.

### Changed

- Replaced manual `_flatten_schema()` in Tau2 benchmark with instructor-based `flatten_model_schema()`.
```

- [ ] **Step 3: Commit**

```bash
git add maseval/core/__init__.py maseval/__init__.py CHANGELOG.md
git commit -m "chore: update exports and changelog for instructor integration"
```

---

## Task 7: Final validation

- [ ] **Step 1: Run linter and formatter**

Run: `uv run ruff format . && uv run ruff check . --fix`

- [ ] **Step 2: Run type checker**

Run: `uv run ty check`

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short -m "not (slow or credentialed or smoke)"`
Expected: All PASS

- [ ] **Step 4: Verify end-to-end import**

Run:
```bash
uv run python3 -c "
from maseval.core.instructor import create_instructor_client, flatten_model_schema
from maseval.core.model import ChatResponse, ModelAdapter
from maseval.core.simulator import ToolSimulatorResponse, UserSimulatorResponse, AgenticUserSimulatorResponse
print('All imports successful')

from pydantic import BaseModel, Field
class TestModel(BaseModel):
    name: str = Field(description='A name')
    age: int = Field(description='An age')

schema = flatten_model_schema(TestModel)
print(f'Schema: {schema}')
assert 'anyOf' not in str(schema)
print('Schema generation works correctly')
"
```

- [ ] **Step 5: Run just all (format + lint + typecheck + test)**

Run: `just all`

- [ ] **Step 6: Review git log**

Run: `git log --oneline main..HEAD`
Expected: Clean series of feature commits

- [ ] **Step 7: Final cleanup commit if needed**

```bash
git status
# Only commit if there are changes
git diff --cached --quiet || git commit -m "chore: final cleanup for instructor integration"
```
