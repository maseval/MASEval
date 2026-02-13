"""MCP Calendar Server - Model Context Protocol Implementation

Demonstrates MCP server implementation using the official MCP Python SDK.
Exposes calendar operations as MCP tools for agent access.
"""

from typing import Any

from mcp.server.fastmcp import FastMCP


def create_calendar_server(calendar_name: str, calendar_data: dict[str, Any]) -> FastMCP:
    """Create an MCP calendar server with the given calendar data.

    Args:
        calendar_name: Name of the calendar (e.g., "my_calendar", "other_calendar")
        calendar_data: Calendar events dict with "events" list

    Returns:
        FastMCP server instance
    """
    mcp = FastMCP(f"calendar-{calendar_name}")

    @mcp.tool()
    def get_events(start_date: str = "", end_date: str = "") -> dict[str, Any]:
        """Get calendar events in a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive). Empty means no lower bound.
            end_date: End date in YYYY-MM-DD format (inclusive). Empty means no upper bound.

        Returns:
            Dictionary with events list and count
        """
        events = calendar_data.get("events", [])

        # Filter by date range
        if start_date and end_date:
            filtered = [e for e in events if start_date <= e["date"] <= end_date]
        elif start_date:
            filtered = [e for e in events if e["date"] >= start_date]
        elif end_date:
            filtered = [e for e in events if e["date"] <= end_date]
        else:
            filtered = events

        return {"events": filtered, "count": len(filtered)}

    @mcp.tool()
    def check_availability(date: str, start_time: str, end_time: str) -> dict[str, Any]:
        """Check if a time slot is available on the calendar.

        Args:
            date: Date in YYYY-MM-DD format
            start_time: Start time in HH:MM format (24-hour)
            end_time: End time in HH:MM format (24-hour)

        Returns:
            Dictionary with availability status and any conflicting events
        """
        events = calendar_data.get("events", [])

        # Find conflicts: overlapping events on the same date
        conflicts = []
        for event in events:
            if event["date"] == date:
                # Check for time overlap: NOT (end_time <= event_start OR start_time >= event_end)
                event_start = event["start_time"]
                event_end = event["end_time"]
                if not (end_time <= event_start or start_time >= event_end):
                    conflicts.append(event)

        return {"available": len(conflicts) == 0, "conflicts": conflicts}

    @mcp.tool()
    def add_event(date: str, start_time: str, end_time: str, title: str, description: str = "") -> dict[str, Any]:
        """Add a new event to the calendar.

        Args:
            date: Date in YYYY-MM-DD format
            start_time: Start time in HH:MM format (24-hour)
            end_time: End time in HH:MM format (24-hour)
            title: Event title
            description: Optional event description

        Returns:
            Dictionary with success status and the created event
        """
        event = {
            "date": date,
            "start_time": start_time,
            "end_time": end_time,
            "title": title,
            "description": description,
        }

        # Add to calendar data
        calendar_data.setdefault("events", []).append(event)

        return {"success": True, "event": event}

    return mcp
