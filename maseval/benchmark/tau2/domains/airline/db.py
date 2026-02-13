"""Tau 2 Benchmark - Airline Domain Database.

Database class for the airline reservation domain.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/airline/data_model.py:FlightDB
"""

from typing import Any, Dict

from pydantic import Field

from maseval.benchmark.tau2.domains.base import DB
from maseval.benchmark.tau2.domains.airline.models import Flight, Reservation, User


class AirlineDB(DB):
    """Database containing all airline-related data including flights, users, and reservations.

    Adapted from: tau2-bench src/tau2/domains/airline/data_model.py:FlightDB
    """

    flights: Dict[str, Flight] = Field(description="Dictionary of all flights indexed by flight number")
    users: Dict[str, User] = Field(description="Dictionary of all users indexed by user ID")
    reservations: Dict[str, Reservation] = Field(description="Dictionary of all reservations indexed by reservation ID")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the airline database.

        Returns:
            Dictionary with counts of flights, flight instances, users, and reservations
        """
        num_flights = len(self.flights)
        num_flights_instances = sum(len(flight.dates) for flight in self.flights.values())
        num_users = len(self.users)
        num_reservations = len(self.reservations)
        return {
            "num_flights": num_flights,
            "num_flights_instances": num_flights_instances,
            "num_users": num_users,
            "num_reservations": num_reservations,
        }
