"""MCP Calendar Tool Collection - Model Context Protocol Integration

Demonstrates MCP integration by wrapping calendar operations as framework-agnostic tools.
Shows how maseval enables agents to work with external systems via standardized protocols.

Tools:
- {name}_get_events: Get events in date range
- {name}_check_availability: Check if time slot is available
- {name}_add_event: Add new event to calendar
"""

from typing import Any, Dict

from .base import BaseTool, ToolResult


class MCPCalendarState:
    """Shared state for MCP calendar tools."""

    def __init__(self, name: str, calendar_data: Dict[str, Any]):
        self.name = name
        self.calendar_data = calendar_data


class MCPCalendarGetEventsTool(BaseTool):
    """Get events in date range via MCP."""

    def __init__(self, mcp_state: MCPCalendarState):
        super().__init__(
            f"{mcp_state.name}_get_events",
            f"Get events from {mcp_state.name} calendar with optional start_date and end_date filters (YYYY-MM-DD)",
            tool_args=["start_date", "end_date"],
        )
        self.state = mcp_state

    def execute(self, **kwargs) -> ToolResult:
        """Get events in date range."""
        start_date = kwargs.get("start_date", "")
        end_date = kwargs.get("end_date", "")

        try:
            events = self.state.calendar_data.get("events", [])
            if start_date and end_date:
                filtered = [e for e in events if start_date <= e["date"] <= end_date]
            elif start_date:
                filtered = [e for e in events if e["date"] >= start_date]
            elif end_date:
                filtered = [e for e in events if e["date"] <= end_date]
            else:
                filtered = events

            return ToolResult(success=True, data={"events": filtered, "count": len(filtered)}, metadata={"mcp": True, "action": "get_events"})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class MCPCalendarCheckAvailabilityTool(BaseTool):
    """Check if time slot is available via MCP."""

    def __init__(self, mcp_state: MCPCalendarState):
        super().__init__(
            f"{mcp_state.name}_check_availability",
            f"Check if time slot is available in {mcp_state.name} calendar (requires date, start_time, end_time in HH:MM format)",
            tool_args=["date", "start_time", "end_time"],
        )
        self.state = mcp_state

    def execute(self, **kwargs) -> ToolResult:
        """Check if time slot is available."""
        date = kwargs.get("date", "")
        start_time = kwargs.get("start_time", "")
        end_time = kwargs.get("end_time", "")

        try:
            events = self.state.calendar_data.get("events", [])
            conflicts = [e for e in events if e["date"] == date and not (end_time <= e["start_time"] or start_time >= e["end_time"])]

            return ToolResult(
                success=True,
                data={"available": len(conflicts) == 0, "conflicts": conflicts},
                metadata={"mcp": True, "action": "check_availability"},
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class MCPCalendarAddEventTool(BaseTool):
    """Add new event to calendar via MCP."""

    def __init__(self, mcp_state: MCPCalendarState):
        super().__init__(
            f"{mcp_state.name}_add_event",
            f"Add new event to {mcp_state.name} calendar (requires date, start_time, end_time, title; optional description)",
            tool_args=["date", "start_time", "end_time", "title", "description"],
        )
        self.state = mcp_state

    def execute(self, **kwargs) -> ToolResult:
        """Add a new event to calendar."""
        date = kwargs.get("date", "")
        start_time = kwargs.get("start_time", "")
        end_time = kwargs.get("end_time", "")
        title = kwargs.get("title", "")
        description = kwargs.get("description", "")

        try:
            if not all([date, start_time, end_time, title]):
                return ToolResult(success=False, data=None, error="Missing required fields: date, start_time, end_time, title")

            event = {
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "title": title,
                "description": description,
            }
            self.state.calendar_data.setdefault("events", []).append(event)

            return ToolResult(success=True, data={"success": True, "event": event}, metadata={"mcp": True, "action": "add_event"})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))


class MCPCalendarToolCollection:
    """MCP Calendar tool collection factory.

    Usage:
        mcp_state = MCPCalendarState(name, calendar_data)
        collection = MCPCalendarToolCollection(mcp_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, mcp_state: MCPCalendarState):
        self.state = mcp_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all MCP calendar sub-tools."""
        return [
            MCPCalendarGetEventsTool(self.state),
            MCPCalendarCheckAvailabilityTool(self.state),
            MCPCalendarAddEventTool(self.state),
        ]
