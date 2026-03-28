"""Tests for maseval.core.callback_handler.CallbackHandler."""

import pytest

from maseval.core.callback_handler import CallbackHandler

pytestmark = pytest.mark.core


class TestCallbackHandler:
    def test_register_and_invoke(self):
        handler = CallbackHandler()
        results = []
        handler.register(lambda x: results.append(x))
        handler.invoke("hello")
        assert results == ["hello"]

    def test_invoke_multiple_callbacks(self):
        handler = CallbackHandler()
        log = []
        handler.register(lambda: log.append("a"))
        handler.register(lambda: log.append("b"))
        handler.invoke()
        assert log == ["a", "b"]

    def test_deregister(self):
        handler = CallbackHandler()
        log = []
        cb = lambda: log.append("x")  # noqa: E731
        handler.register(cb)
        handler.deregister(cb)
        handler.invoke()
        assert log == []

    def test_deregister_nonexistent_raises(self):
        handler = CallbackHandler()
        with pytest.raises(ValueError):
            handler.deregister(lambda: None)

    def test_invoke_passes_kwargs(self):
        handler = CallbackHandler()
        captured = {}
        handler.register(lambda **kw: captured.update(kw))
        handler.invoke(key="val")
        assert captured == {"key": "val"}
