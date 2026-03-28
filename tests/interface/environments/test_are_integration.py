"""Integration tests for AREEnvironment (requires ARE installed)."""

import pytest

try:
    import are  # noqa: F401
    HAS_ARE = True
except ImportError:
    HAS_ARE = False


@pytest.mark.are
@pytest.mark.skipif(not HAS_ARE, reason="ARE not installed")
class TestAREEnvironmentIntegration:
    """Integration tests that exercise real ARE infrastructure."""

    def test_import_works(self):
        """AREEnvironment can be imported when ARE is installed."""
        from maseval.interface.environments.are import AREEnvironment
        assert AREEnvironment is not None

    def test_tool_wrapper_import(self):
        """AREToolWrapper can be imported when ARE is installed."""
        from maseval.interface.environments.are_tool_wrapper import AREToolWrapper
        assert AREToolWrapper is not None

    def test_package_init_exports(self):
        """Package __init__ exports AREEnvironment when ARE is installed."""
        from maseval.interface.environments import AREEnvironment, AREToolWrapper
        assert AREEnvironment is not None
        assert AREToolWrapper is not None
