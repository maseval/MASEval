"""Tau 2 Benchmark - Airline Domain.

Flight reservation management domain with booking and modification operations:
- Search direct and one-stop flights
- Book reservations with passengers and payment
- Modify flights, passengers, baggage
- Cancel reservations
- Send compensation certificates

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted components:
- models.py: Adapted from src/tau2/domains/airline/data_model.py
- db.py: Adapted from src/tau2/domains/airline/data_model.py:FlightDB
- tools.py: Adapted from src/tau2/domains/airline/tools.py
"""

from .models import (
    Address,
    AirportCode,
    CabinClass,
    Certificate,
    CreditCard,
    DirectFlight,
    Flight,
    FlightBase,
    FlightDateStatus,
    FlightDateStatusAvailable,
    FlightDateStatusCancelled,
    FlightDateStatusDelayed,
    FlightDateStatusFlying,
    FlightDateStatusLanded,
    FlightDateStatusOnTime,
    FlightInfo,
    FlightType,
    GiftCard,
    Insurance,
    MembershipLevel,
    Name,
    Passenger,
    Payment,
    PaymentMethod,
    Reservation,
    ReservationFlight,
    User,
)
from .db import AirlineDB
from .tools import AirlineTools

__all__ = [
    # Models
    "Address",
    "AirportCode",
    "CabinClass",
    "Certificate",
    "CreditCard",
    "DirectFlight",
    "Flight",
    "FlightBase",
    "FlightDateStatus",
    "FlightDateStatusAvailable",
    "FlightDateStatusCancelled",
    "FlightDateStatusDelayed",
    "FlightDateStatusFlying",
    "FlightDateStatusLanded",
    "FlightDateStatusOnTime",
    "FlightInfo",
    "FlightType",
    "GiftCard",
    "Insurance",
    "MembershipLevel",
    "Name",
    "Passenger",
    "Payment",
    "PaymentMethod",
    "Reservation",
    "ReservationFlight",
    "User",
    # Database
    "AirlineDB",
    # Tools
    "AirlineTools",
]
