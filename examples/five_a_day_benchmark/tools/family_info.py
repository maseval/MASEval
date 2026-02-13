"""Family information tool - single purpose."""

from typing import Any

from .base import BaseTool, ToolResult


class FamilyInfoGetChildrenTool(BaseTool):
    """Get list of children from family information."""

    def __init__(self, family_data: dict[str, Any]):
        super().__init__(
            "family_info_get_children",
            "Get list of children with their names and ages",
            tool_args=[],
        )
        self.children = family_data.get("children", [])

    def execute(self, **kwargs) -> ToolResult:
        """Get list of children."""
        return ToolResult(
            success=True,
            data={"children": self.children, "count": len(self.children)},
        )


class FamilyInfoToolCollection:
    """Family information tool collection factory.

    Currently only contains one tool, but structured as a collection
    for consistency and future extensibility.
    """

    def __init__(self, family_data: dict[str, Any]):
        self.family_data = family_data

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all family info sub-tools."""
        return [
            FamilyInfoGetChildrenTool(self.family_data),
        ]
