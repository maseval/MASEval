"""Hotel search tool collection.

Tools:
- hotel_search_search: Search hotels by criteria (price, distance, wifi)
- hotel_search_get: Get specific hotel by ID
- hotel_search_get_all: Get all available hotels
"""

from typing import Any

from .base import BaseTool, ToolResult


class HotelSearchState:
    """Shared state for hotel search tools."""

    def __init__(self, hotels_data: list[dict[str, Any]]):
        self.hotels = hotels_data


class HotelSearchSearchTool(BaseTool):
    """Search hotels by criteria."""

    def __init__(self, hotel_state: HotelSearchState):
        super().__init__(
            "hotel_search_search",
            "Search hotels with optional filters: max_price (per night), max_distance (km to venue), min_wifi (Mbps speed)",
            tool_args=["max_price", "max_distance", "min_wifi"],
        )
        self.state = hotel_state

    def execute(self, **kwargs) -> ToolResult:
        """Search hotels with filters."""
        max_price = kwargs.get("max_price")
        max_distance = kwargs.get("max_distance")
        min_wifi = kwargs.get("min_wifi")

        filtered = self.state.hotels

        if max_price is not None:
            filtered = [h for h in filtered if h["price_per_night"] <= max_price]
        if max_distance is not None:
            filtered = [h for h in filtered if h["distance_to_venue_km"] <= max_distance]
        if min_wifi is not None:
            filtered = [h for h in filtered if h["wifi_speed_mbps"] >= min_wifi]

        return ToolResult(
            success=True,
            data={"hotels": filtered, "count": len(filtered)},
        )


class HotelSearchGetTool(BaseTool):
    """Get specific hotel by ID."""

    def __init__(self, hotel_state: HotelSearchState):
        super().__init__(
            "hotel_search_get",
            "Get specific hotel details by hotel ID",
            tool_args=["hotel_id"],
        )
        self.state = hotel_state

    def execute(self, **kwargs) -> ToolResult:
        """Get specific hotel by ID."""
        hotel_id = kwargs.get("hotel_id")

        if not hotel_id:
            return ToolResult(success=False, data=None, error="hotel_id is required")

        for hotel in self.state.hotels:
            if hotel["id"] == hotel_id:
                return ToolResult(success=True, data=hotel)

        return ToolResult(success=False, data=None, error=f"Hotel {hotel_id} not found")


class HotelSearchGetAllTool(BaseTool):
    """Get all available hotels."""

    def __init__(self, hotel_state: HotelSearchState):
        super().__init__(
            "hotel_search_get_all",
            "Get all available hotels without any filters",
            tool_args=[],
        )
        self.state = hotel_state

    def execute(self, **kwargs) -> ToolResult:
        """Get all available hotels."""
        return ToolResult(
            success=True,
            data={"hotels": self.state.hotels, "count": len(self.state.hotels)},
        )


class HotelSearchToolCollection:
    """Hotel search tool collection factory.

    Usage:
        hotel_state = HotelSearchState(hotels_data)
        collection = HotelSearchToolCollection(hotel_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, hotel_search_state: HotelSearchState):
        self.state = hotel_search_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all hotel search sub-tools."""
        return [
            HotelSearchSearchTool(self.state),
            HotelSearchGetTool(self.state),
            HotelSearchGetAllTool(self.state),
        ]
