"""Fitness tracking tool collections with shared state.

Tools:
- running_app_get_activities: Get running/cardio activities with optional date filters
- running_app_get_activity: Get specific activity by ID
- gym_tracker_get_workouts: Get gym/strength workouts with optional date filters
- gym_tracker_get_workout: Get specific workout by ID
"""

from dataclasses import dataclass
from typing import Any

from .base import BaseTool, ToolResult


@dataclass
class Activity:
    """Fitness activity representation."""

    id: str
    date: str
    activity: str
    duration_minutes: int
    type: str
    distance_km: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "date": self.date,
            "activity": self.activity,
            "duration_minutes": self.duration_minutes,
            "type": self.type,
        }
        if self.distance_km is not None:
            result["distance_km"] = self.distance_km
        return result


class RunningAppState:
    """Shared state for running app tools."""

    def __init__(self, activities_data: list[dict[str, Any]]):
        self.activities: list[Activity] = []

        # Load activities
        for i, activity_data in enumerate(activities_data):
            self.activities.append(
                Activity(
                    id=str(i + 1),
                    date=activity_data["date"],
                    activity=activity_data["activity"],
                    duration_minutes=activity_data["duration_minutes"],
                    distance_km=activity_data.get("distance_km"),
                    type=activity_data["type"],
                )
            )


class RunningAppGetActivitiesTool(BaseTool):
    """Get running/cardio activities with optional date filters."""

    def __init__(self, running_state: RunningAppState):
        super().__init__(
            "running_app_get_activities",
            "Get running and cardio activities with optional start_date and end_date filters (YYYY-MM-DD)",
            tool_args=["start_date", "end_date"],
        )
        self.state = running_state

    def execute(self, **kwargs) -> ToolResult:
        """Get activities in date range."""
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        filtered = self.state.activities

        if start_date:
            filtered = [a for a in filtered if a.date >= start_date]
        if end_date:
            filtered = [a for a in filtered if a.date <= end_date]

        return ToolResult(
            success=True,
            data={
                "activities": [a.to_dict() for a in filtered],
                "count": len(filtered),
            },
        )


class RunningAppGetActivityTool(BaseTool):
    """Get specific activity by ID."""

    def __init__(self, running_state: RunningAppState):
        super().__init__(
            "running_app_get_activity",
            "Get specific running/cardio activity by ID",
            tool_args=["activity_id"],
        )
        self.state = running_state

    def execute(self, **kwargs) -> ToolResult:
        """Get specific activity by ID."""
        activity_id = kwargs.get("activity_id")

        if not activity_id:
            return ToolResult(success=False, data=None, error="activity_id is required")

        for activity in self.state.activities:
            if activity.id == activity_id:
                return ToolResult(success=True, data=activity.to_dict())

        return ToolResult(success=False, data=None, error=f"Activity {activity_id} not found")


class RunningAppToolCollection:
    """Running app tool collection factory.

    Usage:
        running_state = RunningAppState(activities_data)
        collection = RunningAppToolCollection(running_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, running_state: RunningAppState):
        self.state = running_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all running app sub-tools."""
        return [
            RunningAppGetActivitiesTool(self.state),
            RunningAppGetActivityTool(self.state),
        ]


class GymTrackerState:
    """Shared state for gym tracker tools."""

    def __init__(self, workouts_data: list[dict[str, Any]]):
        self.workouts: list[Activity] = []

        # Load workouts
        for i, workout_data in enumerate(workouts_data):
            self.workouts.append(
                Activity(
                    id=str(i + 1),
                    date=workout_data["date"],
                    activity=workout_data["activity"],
                    duration_minutes=workout_data["duration_minutes"],
                    type=workout_data["type"],
                    distance_km=None,
                )
            )


class GymTrackerGetWorkoutsTool(BaseTool):
    """Get gym/strength workouts with optional date filters."""

    def __init__(self, gym_state: GymTrackerState):
        super().__init__(
            "gym_tracker_get_workouts",
            "Get gym and strength workouts with optional start_date and end_date filters (YYYY-MM-DD)",
            tool_args=["start_date", "end_date"],
        )
        self.state = gym_state

    def execute(self, **kwargs) -> ToolResult:
        """Get workouts in date range."""
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        filtered = self.state.workouts

        if start_date:
            filtered = [w for w in filtered if w.date >= start_date]
        if end_date:
            filtered = [w for w in filtered if w.date <= end_date]

        return ToolResult(
            success=True,
            data={
                "workouts": [w.to_dict() for w in filtered],
                "count": len(filtered),
            },
        )


class GymTrackerGetWorkoutTool(BaseTool):
    """Get specific workout by ID."""

    def __init__(self, gym_state: GymTrackerState):
        super().__init__(
            "gym_tracker_get_workout",
            "Get specific gym/strength workout by ID",
            tool_args=["workout_id"],
        )
        self.state = gym_state

    def execute(self, **kwargs) -> ToolResult:
        """Get specific workout by ID."""
        workout_id = kwargs.get("workout_id")

        if not workout_id:
            return ToolResult(success=False, data=None, error="workout_id is required")

        for workout in self.state.workouts:
            if workout.id == workout_id:
                return ToolResult(success=True, data=workout.to_dict())

        return ToolResult(success=False, data=None, error=f"Workout {workout_id} not found")


class GymTrackerToolCollection:
    """Gym tracker tool collection factory.

    Usage:
        gym_state = GymTrackerState(workouts_data)
        collection = GymTrackerToolCollection(gym_state)
        tools = collection.get_sub_tools()
    """

    def __init__(self, gym_state: GymTrackerState):
        self.state = gym_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all gym tracker sub-tools."""
        return [
            GymTrackerGetWorkoutsTool(self.state),
            GymTrackerGetWorkoutTool(self.state),
        ]
