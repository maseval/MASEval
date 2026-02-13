"""Evaluators for Task 4: Hotel Optimization.

Task: User asks to find the best hotel based on distance, WiFi, and price priorities.
Success criteria: Agent selects hotel close to the mathematical optimum.
"""

import re
from typing import Any, Dict, List, Optional

from maseval import Evaluator, Environment, Task, User
from .utils import normalize_final_answer


class OptimizationQualityEvaluator(Evaluator):
    """Evaluates how close agent's hotel choice is to the optimal solution.

    Evaluation type: Optimization (ranking)
    Measures: Did the agent find the best (or near-best) hotel?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.eval_data = task.evaluation_data
        self.hotels = task.environment_data.get("hotels", [])
        self.weights = task.environment_data.get("user_priorities", {"distance_weight": 0.4, "wifi_weight": 0.35, "price_weight": 0.25})
        self.hotel_scores = self._calculate_all_hotel_scores()

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """No tool traces needed for this evaluator."""
        return {}

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate optimization quality."""

        if not final_answer:
            return {"found_optimal_hotel": False, "optimization_score": 0.0, "error": "No final answer found"}

        full_response = normalize_final_answer(final_answer)

        if not full_response:
            return {"found_optimal_hotel": False, "optimization_score": 0.0, "error": "No assistant response found"}

        chosen_hotel_id = self._extract_hotel_choice(full_response)

        if not chosen_hotel_id:
            return {"found_optimal_hotel": False, "chosen_hotel": None, "optimization_score": 0.0, "error": "No hotel choice found in response"}

        # Get scores sorted by quality
        sorted_hotels = sorted(self.hotel_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_hotel_id = sorted_hotels[0][0]
        chosen_rank = self._get_hotel_rank(chosen_hotel_id, sorted_hotels)

        # Score based on proximity to optimal
        if chosen_rank == 1:
            score = 1.0
        elif chosen_rank == 2:
            score = 0.7
        elif chosen_rank == 3:
            score = 0.5
        else:
            score = max(0.1, 0.5 - (chosen_rank - 3) * 0.1)

        return {
            "found_optimal_hotel": chosen_rank == 1,
            "chosen_hotel": chosen_hotel_id,
            "chosen_hotel_rank": chosen_rank,
            "optimal_hotel": optimal_hotel_id,
            "optimization_score": score,
            "top_3_hotels": sorted_hotels[:3],
        }

    def _calculate_all_hotel_scores(self) -> Dict[str, float]:
        """Calculate optimization scores for all hotels."""
        max_price = max(hotel["price_per_night"] for hotel in self.hotels)

        scores = {}
        for hotel in self.hotels:
            # Normalize each criterion (0-1 scale)
            distance_score = max(0, 1.0 - hotel["distance_to_venue_km"] / 5.0)
            wifi_score = min(1.0, hotel["wifi_speed_mbps"] / 500.0)
            price_score = max(0, 1.0 - hotel["price_per_night"] / max_price)

            # Weighted sum
            score = (
                distance_score * self.weights["distance_weight"]
                + wifi_score * self.weights["wifi_weight"]
                + price_score * self.weights["price_weight"]
            )
            scores[hotel["id"]] = score

        return scores

    def _extract_hotel_choice(self, response: str) -> Optional[str]:
        """Extract hotel ID from response.

        Uses first hotel mention as the primary recommendation,
        since agents typically state their choice first before alternatives.
        """
        # Look for hotel IDs (H001, H002, etc.)
        hotel_id_pattern = r"\bH0*(\d{1,2})\b"
        matches = re.findall(hotel_id_pattern, response.upper())

        if matches:
            hotel_num = int(matches[0])  # Use first match, not last
            return f"H{hotel_num:03d}"

        # Try to find hotel name
        for hotel in self.hotels:
            if hotel["name"] in response:
                return hotel["id"]

        return None

    def _get_hotel_rank(self, hotel_id: str, sorted_hotels: List[tuple]) -> int:
        """Get rank of hotel (1-indexed)."""
        for i, (hid, _) in enumerate(sorted_hotels, 1):
            if hid == hotel_id:
                return i
        return len(sorted_hotels) + 1


class SearchStrategyEvaluator(Evaluator):
    """Evaluates the search strategy employed by the agent.

    Evaluation type: Completeness check
    Measures: Did the agent examine hotels systematically?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.total_hotels = len(task.environment_data.get("hotels", []))

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to relevant tool traces only."""
        tools = traces.get("tools", {})
        return {
            "hotel_search_search": tools.get("hotel_search_search", {}),
            "hotel_search_get_all": tools.get("hotel_search_get_all", {}),
            "calculator": tools.get("calculator_calculate", {}),
        }

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate search strategy."""

        # Check if any hotel_search tool was used (search, get, or get_all)
        used_hotel_search = bool(traces.get("hotel_search_search", {}).get("invocations", [])) or bool(
            traces.get("hotel_search_get_all", {}).get("invocations", [])
        )

        # Check if calculator was used (for scoring)
        used_calculator = bool(traces.get("calculator", {}).get("invocations", []))

        # Calculate strategy score
        score_components = [int(used_hotel_search), int(used_calculator)]
        strategy_score = sum(score_components) / len(score_components)

        return {"used_hotel_search": used_hotel_search, "used_calculator": used_calculator, "search_strategy_score": strategy_score}
