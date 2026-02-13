"""Tau 2 Benchmark - Airline Domain Models.

Pydantic models for the airline reservation domain entities.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/airline/data_model.py
"""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Type Aliases
# =============================================================================

FlightType = Literal["round_trip", "one_way"]
CabinClass = Literal["business", "economy", "basic_economy"]
Insurance = Literal["yes", "no"]
MembershipLevel = Literal["gold", "silver", "regular"]


# =============================================================================
# Basic Models
# =============================================================================


class AirportCode(BaseModel):
    """Airport information with IATA code and city name."""

    iata: str = Field(description="IATA code")
    city: str = Field(description="City name")


class Name(BaseModel):
    """Person's full name."""

    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class Address(BaseModel):
    """Physical address."""

    address1: str = Field(description="Primary address line")
    address2: Optional[str] = Field(None, description="Secondary address line (optional)")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


class Passenger(BaseModel):
    """Passenger information for a reservation."""

    first_name: str = Field(description="Passenger's first name")
    last_name: str = Field(description="Passenger's last name")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")


# =============================================================================
# Payment Models
# =============================================================================


class Payment(BaseModel):
    """A payment transaction."""

    payment_id: str = Field(description="Unique identifier for the payment")
    amount: int = Field(description="Payment amount in dollars")


class PaymentMethodBase(BaseModel):
    """Base class for payment methods."""

    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    """Credit card payment method."""

    source: Literal["credit_card"] = Field(description="Indicates this is a credit card payment method")
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class GiftCard(PaymentMethodBase):
    """Gift card payment method."""

    source: Literal["gift_card"] = Field(description="Indicates this is a gift card payment method")
    amount: float = Field(description="Gift card value amount")
    id: str = Field(description="Unique identifier for the gift card")


class Certificate(PaymentMethodBase):
    """Certificate payment method (compensation voucher)."""

    source: Literal["certificate"] = Field(description="Indicates this is a certificate payment method")
    amount: float = Field(description="Certificate value amount")


PaymentMethod = Union[CreditCard, GiftCard, Certificate]


# =============================================================================
# Flight Status Models
# =============================================================================


class FlightDateStatusAvailable(BaseModel):
    """Flight is available for booking."""

    status: Literal["available"] = Field(description="Indicates flight is available for booking")
    available_seats: Dict[CabinClass, int] = Field(description="Available seats by class")
    prices: Dict[CabinClass, int] = Field(description="Current prices by class")


class FlightDateStatusOnTime(BaseModel):
    """Flight is on time."""

    status: Literal["on time"] = Field(description="Indicates flight is on time")
    estimated_departure_time_est: str = Field(description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS")
    estimated_arrival_time_est: str = Field(description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS")


class FlightDateStatusFlying(BaseModel):
    """Flight is currently in the air."""

    status: Literal["flying"] = Field(description="Indicates flight is in flight")
    actual_departure_time_est: str = Field(description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS")
    estimated_arrival_time_est: str = Field(description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS")


class FlightDateStatusLanded(BaseModel):
    """Flight has landed."""

    status: Literal["landed"] = Field(description="Indicates flight has landed")
    actual_departure_time_est: str = Field(description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS")
    actual_arrival_time_est: str = Field(description="Actual arrival time in EST in the format YYYY-MM-DDTHH:MM:SS")


class FlightDateStatusCancelled(BaseModel):
    """Flight was cancelled."""

    status: Literal["cancelled"] = Field(description="Indicates flight was cancelled")


class FlightDateStatusDelayed(BaseModel):
    """Flight is delayed."""

    status: Literal["delayed"] = Field(description="Indicates flight was delayed")
    estimated_departure_time_est: str = Field(description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS")
    estimated_arrival_time_est: str = Field(description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS")


FlightDateStatus = Union[
    FlightDateStatusAvailable,
    FlightDateStatusLanded,
    FlightDateStatusCancelled,
    FlightDateStatusDelayed,
    FlightDateStatusFlying,
    FlightDateStatusOnTime,
]


# =============================================================================
# Flight Models
# =============================================================================


class FlightBase(BaseModel):
    """Base flight information."""

    flight_number: str = Field(description="Unique flight identifier")
    origin: str = Field(description="IATA code for origin airport")
    destination: str = Field(description="IATA code for destination airport")


class Flight(FlightBase):
    """Complete flight information with schedule and date-specific status."""

    scheduled_departure_time_est: str = Field(description="Scheduled departure time in EST in the format HH:MM:SS")
    scheduled_arrival_time_est: str = Field(description="Scheduled arrival time in EST in the format HH:MM:SS")
    dates: Dict[str, FlightDateStatus] = Field(description="Flight status by date (YYYY-MM-DD)")


class DirectFlight(FlightBase):
    """Direct flight search result."""

    status: Literal["available"] = Field(description="Indicates flight is available for booking")
    scheduled_departure_time_est: str = Field(description="Scheduled departure time in EST in the format HH:MM:SS")
    scheduled_arrival_time_est: str = Field(description="Scheduled arrival time in EST in the format HH:MM:SS")
    date: Optional[str] = Field(description="Flight date in YYYY-MM-DD format", default=None)
    available_seats: Dict[CabinClass, int] = Field(description="Available seats by class")
    prices: Dict[CabinClass, int] = Field(description="Current prices by class")


class ReservationFlight(FlightBase):
    """Flight segment in a reservation."""

    date: str = Field(description="Flight date in YYYY-MM-DD format")
    price: int = Field(description="Flight price in dollars.")


class FlightInfo(BaseModel):
    """Flight reference for booking/updating."""

    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD'.")


# =============================================================================
# User Model
# =============================================================================


class User(BaseModel):
    """User account with payment methods and saved passengers."""

    user_id: str = Field(description="Unique identifier for the user")
    name: Name = Field(description="User's full name")
    address: Address = Field(description="User's address information")
    email: str = Field(description="User's email address")
    dob: str = Field(description="User's date of birth in the format YYYY-MM-DD")
    payment_methods: Dict[str, PaymentMethod] = Field(description="User's saved payment methods")
    saved_passengers: List[Passenger] = Field(description="User's saved passenger information")
    membership: MembershipLevel = Field(description="User's membership level")
    reservations: List[str] = Field(description="List of user's reservation IDs")


# =============================================================================
# Reservation Model
# =============================================================================


class Reservation(BaseModel):
    """Flight reservation with passengers and payment history."""

    reservation_id: str = Field(description="Unique identifier for the reservation")
    user_id: str = Field(description="ID of the user who made the reservation")
    origin: str = Field(description="IATA code for trip origin")
    destination: str = Field(description="IATA code for trip destination")
    flight_type: FlightType = Field(description="Type of trip")
    cabin: CabinClass = Field(description="Selected cabin class")
    flights: List[ReservationFlight] = Field(description="List of flights in the reservation")
    passengers: List[Passenger] = Field(description="List of passengers on the reservation")
    payment_history: List[Payment] = Field(description="History of payments for this reservation")
    created_at: str = Field(description="Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS")
    total_baggages: int = Field(description="Total number of bags in reservation")
    nonfree_baggages: int = Field(description="Number of paid bags in reservation")
    insurance: Insurance = Field(description="Whether travel insurance was purchased")
    status: Optional[Literal["cancelled"]] = Field(description="Status of the reservation", default=None)
