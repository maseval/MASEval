"""Gaia2 Benchmark - Tool Wrapper.

Backward compatibility module. The canonical tool wrapper is now
AREToolWrapper in maseval.interface.environments.are_tool_wrapper.

Original Repository: https://github.com/facebookresearch/meta-agents-research-environments
Code License: MIT
"""

from typing import TYPE_CHECKING, Any, Dict, List

from maseval.interface.environments.are_tool_wrapper import AREToolWrapper

if TYPE_CHECKING:
    from maseval.benchmark.gaia2.environment import Gaia2Environment

# Backward compatibility alias
Gaia2GenericTool = AREToolWrapper


def wrap_are_tools(
    are_tools: List[Any],
    environment: "Gaia2Environment",
) -> Dict[str, AREToolWrapper]:
    """Wrap multiple ARE tools for MASEval.

    Args:
        are_tools: List of ARE AppTool instances
        environment: The Gaia2Environment these tools belong to

    Returns:
        Dictionary mapping tool names to wrapped tools
    """
    wrapped: Dict[str, AREToolWrapper] = {}
    for tool in are_tools:
        wrapper = AREToolWrapper(tool, environment)
        wrapped[wrapper.name] = wrapper
    return wrapped
