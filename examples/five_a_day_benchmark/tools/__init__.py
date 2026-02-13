"""Tools for the 5-a-day benchmark."""

from typing import Any, Callable, Dict, List

from .base import BaseTool, ToolResult, ToolInvocation
from .email import EmailToolCollection, EmailState
from .banking import BankingToolCollection, BankingState
from .calculator import CalculatorToolCollection
from .code_execution import CodeExecutionToolCollection, CodeExecutionState, get_safe_python_exec_environment
from .family_info import FamilyInfoToolCollection
from .stock_price import StockPriceToolCollection
from .calendar import CalendarToolCollection, CalendarState
from .fitness import RunningAppToolCollection, GymTrackerToolCollection, RunningAppState, GymTrackerState
from .hotel_search import HotelSearchToolCollection, HotelSearchState
from .mcp_calendar import MCPCalendarToolCollection, MCPCalendarState

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolInvocation",
    "EmailToolCollection",
    "BankingToolCollection",
    "CalculatorToolCollection",
    "CodeExecutionToolCollection",
    "FamilyInfoToolCollection",
    "StockPriceToolCollection",
    "CalendarToolCollection",
    "RunningAppToolCollection",
    "GymTrackerToolCollection",
    "HotelSearchToolCollection",
    "MCPCalendarToolCollection",
    "get_states",
    "filter_tool_adapters_by_prefix",
    "get_safe_python_exec_environment",
]


def get_states(tool_names: List[str], env_data: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize state objects for tools that require them.

    Args:
        tool_names: List of tool names needed for this task (from env_data["tools"])
        env_data: Environment data dictionary from task configuration

    Returns:
        Dictionary mapping state keys to initialized state objects.
        Only includes states for tools that are actually needed.

    Example:
        env_data = {"tools": ["email", "banking"], "email_inbox": [...], "banking": {...}}
        states = get_states(env_data["tools"], env_data)
        # Returns: {"email_state": EmailState(...), "banking_state": BankingState(...)}
    """
    # Map tool names to the state keys they require
    states_required_by_tool: Dict[str, List[str]] = {
        "email": ["email_state"],
        "banking": ["banking_state"],
        "calendar": ["calendar_state"],
        "running_app": ["running_app_state"],
        "gym_tracker": ["gym_tracker_state"],
        "my_calendar_mcp": ["my_calendar_mcp_state"],
        "other_calendar_mcp": ["other_calendar_mcp_state"],
        "python_executor": ["python_executor_state"],
        "hotel_search": ["hotel_search_state"],
    }

    # Map state keys to their initializer functions
    state_initializers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "email_state": lambda data: EmailState(data["user_email"], data["email_inbox"]),
        "banking_state": lambda data: BankingState(
            data["banking"]["bank_transactions"],
            data["banking"]["current_balance"],
            data["banking"].get("assets", {}),
        ),
        "calendar_state": lambda data: CalendarState(data.get("my_calendar_name", "my_calendar"), data["my_calendar_availability"]),
        "running_app_state": lambda data: RunningAppState(data["running_activities"]),
        "gym_tracker_state": lambda data: GymTrackerState(data["gym_activities"]),
        "my_calendar_mcp_state": lambda data: _create_mcp_calendar_state("my_calendar_mcp", data, "my_calendar_mcp_availability"),
        "other_calendar_mcp_state": lambda data: _create_mcp_calendar_state("other_calendar_mcp", data, "other_calendar_mcp_availability"),
        "python_executor_state": lambda data: CodeExecutionState(data.get("test_cases")),
        "hotel_search_state": lambda data: HotelSearchState(data["hotels"]),
    }

    # Collect all required state keys for the requested tools
    required_state_keys = set()
    for tool_name in tool_names:
        if tool_name in states_required_by_tool:
            required_state_keys.update(states_required_by_tool[tool_name])

    # Initialize only the required states
    initialized_states = {}
    for state_key in required_state_keys:
        if state_key in state_initializers:
            initialized_states[state_key] = state_initializers[state_key](env_data)

    return initialized_states


def _create_mcp_calendar_state(calendar_name: str, env_data: Dict[str, Any], availability_key: str) -> MCPCalendarState:
    """Helper to create MCP calendar state with converted event format.

    Converts from availability format: {"2025-12-02": [{"start": "09:00", "end": "11:00"}]}
    To MCP events format: {"events": [{"date": "2025-12-02", "start_time": "09:00", ...}]}
    """
    availability = env_data[availability_key]
    events = [
        {
            "date": date,
            "start_time": slot["start"],
            "end_time": slot["end"],
            "title": "Busy",
        }
        for date, slots in availability.items()
        for slot in slots
    ]
    return MCPCalendarState(calendar_name, {"events": events})


def filter_tool_adapters_by_prefix(adapters: Dict[str, Any], tool_names: List[str]) -> Dict[str, Any]:
    """Filter tool adapters by exact name prefix.

    Args:
        adapters: Dict of tool adapters to filter, keyed by tool name
        tool_names: Tool/collection names to match (e.g., ["banking", "calculator"])

    Returns:
        Filtered dict of adapters matching the specified names
    """
    if not tool_names:
        return {}

    filtered: Dict[str, Any] = {}
    for adapter_name, adapter in adapters.items():
        for tool_name in tool_names:
            if adapter_name.startswith(f"{tool_name}_"):
                filtered[adapter_name] = adapter
                break

    return filtered
