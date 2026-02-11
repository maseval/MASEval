"""Reusable skip decorators for tests requiring external credentials.

Usage::

    from tests.markers import requires_openai

    @pytest.mark.credentialed
    @requires_openai
    def test_openai_chat():
        ...

Each decorator checks for the corresponding environment variable and skips
the test with a clear message when it is absent.  They do **not** add the
``credentialed`` marker automatically — apply it yourself so that the marker
filter in ``addopts`` can exclude these tests without having to inspect
environment variables.
"""

import os

import pytest

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

requires_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
