"""Calendar tool collection with shared state.

Tools:
- calendar_get_availability: Get calendar availability for date range
- calendar_get_date_availability: Get availability for specific date
"""

from dataclasses import dataclass

from .base import BaseTool, ToolResult


@dataclass
class TimeSlot:
    """Time slot representation."""

    start: str
    end: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"start": self.start, "end": self.end}


class CalendarState:
    """Shared state for calendar tools."""

    def __init__(self, calendar_id: str, availability_data: dict[str, list[dict[str, str]]]):
        self.calendar_id = calendar_id
        self.availability: dict[str, list[TimeSlot]] = {}

        # Load availability
        for date, slots in availability_data.items():
            self.availability[date] = [TimeSlot(start=s["start"], end=s["end"]) for s in slots]


class CalendarGetAvailabilityTool(BaseTool):
    """Get calendar availability for date range."""

    def __init__(self, calendar_state: CalendarState):
        super().__init__(
            "calendar_get_availability",
            f"Get {calendar_state.calendar_id} calendar availability with optional start_date and end_date filters (YYYY-MM-DD)",
            tool_args=["start_date", "end_date"],
        )
        self.state = calendar_state

    def execute(self, **kwargs) -> ToolResult:
        """Get availability for date range."""
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        if not start_date and not end_date:
            # Return all availability
            result = {date: [slot.to_dict() for slot in slots] for date, slots in self.state.availability.items()}
        else:
            # Filter by date range
            result = {}
            for date, slots in self.state.availability.items():
                if start_date and date < start_date:
                    continue
                if end_date and date > end_date:
                    continue
                result[date] = [slot.to_dict() for slot in slots]

        return ToolResult(
            success=True,
            data={"availability": result, "calendar_id": self.state.calendar_id},
        )


class CalendarGetDateAvailabilityTool(BaseTool):
    """Get availability for a specific date."""

    def __init__(self, calendar_state: CalendarState):
        super().__init__(
            "calendar_get_date_availability",
            f"Get {calendar_state.calendar_id} calendar availability for a specific date (YYYY-MM-DD)",
            tool_args=["date"],
        )
        self.state = calendar_state

    def execute(self, **kwargs) -> ToolResult:
        """Get availability for a specific date."""
        date = kwargs.get("date")

        if not date:
            return ToolResult(success=False, data=None, error="date is required")

        if date in self.state.availability:
            slots = [slot.to_dict() for slot in self.state.availability[date]]
            return ToolResult(
                success=True,
                data={
                    "date": date,
                    "slots": slots,
                },
            )

        return ToolResult(
            success=False,
            data=None,
            error=f"No availability found for {date}",
        )


class CalendarToolCollection:
    """Calendar tool collection factory.

    Usage:
        calendar_state = CalendarState(calendar_id, availability_data)
        collection = CalendarToolCollection(calendar_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, calendar_state: CalendarState):
        self.state = calendar_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all calendar sub-tools."""
        return [
            CalendarGetAvailabilityTool(self.state),
            CalendarGetDateAvailabilityTool(self.state),
        ]
