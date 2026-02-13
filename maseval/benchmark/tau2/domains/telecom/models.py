"""Tau 2 Benchmark - Telecom Domain Models.

Pydantic models for the telecom customer service domain entities.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/telecom/data_model.py
"""

import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Default start date used in tau2-bench
DEFAULT_START_DATE = datetime.date(2025, 1, 1)


# =============================================================================
# Basic Models
# =============================================================================


class Address(BaseModel):
    """Physical address."""

    model_config = ConfigDict(extra="forbid")

    street: str = Field(description="Street address including house/apartment number")
    city: str = Field(description="City name")
    state: str = Field(description="State or province code (e.g., CA, NY)")
    zip_code: str = Field(description="Postal/ZIP code")


# =============================================================================
# Plan Models
# =============================================================================


class Plan(BaseModel):
    """Service plan with data limits and pricing."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="Unique identifier for the plan")
    name: str = Field(description="Display name of the plan")
    data_limit_gb: float = Field(description="Monthly data allowance in gigabytes (GB)")
    price_per_month: float = Field(description="Monthly price of the plan in USD")
    data_refueling_price_per_gb: float = Field(description="Price per gigabyte for data refueling")


# =============================================================================
# Device Models
# =============================================================================


class DeviceType(str, Enum):
    """Type of device."""

    PHONE = "phone"
    ROUTER = "router"
    TABLET = "tablet"
    WATCH = "watch"
    OTHER = "other"


class Device(BaseModel):
    """Device that can be associated with a line."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(description="Unique identifier for the device")
    device_type: DeviceType = Field(description="Type/category of the device")
    model: str = Field(description="Model name/number of the device")
    imei: Optional[str] = Field(None, description="International Mobile Equipment Identity number")
    is_esim_capable: bool = Field(description="Whether the device supports eSIM technology")
    activated: bool = Field(False, description="Whether the device has been activated on the network")
    activation_date: Optional[datetime.datetime] = Field(None, description="Date and time when the device was activated")
    last_esim_transfer_date: Optional[datetime.datetime] = Field(None, description="Last date an eSIM profile was transferred to this device")


# =============================================================================
# Line Models
# =============================================================================


class LineStatus(str, Enum):
    """Status of a phone/data line."""

    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_ACTIVATION = "Pending Activation"
    CLOSED = "Closed"


class Line(BaseModel):
    """Phone/data line associated with a customer."""

    model_config = ConfigDict(extra="forbid")

    line_id: str = Field(description="Unique identifier for the line")
    phone_number: str = Field(description="Phone number associated with the line")
    status: LineStatus = Field(LineStatus.PENDING_ACTIVATION, description="Current status of the line")
    plan_id: str = Field(description="Plan associated with this line")
    device_id: Optional[str] = Field(None, description="Device associated with this line")
    data_used_gb: float = Field(0.0, description="Data used in the current billing cycle in gigabytes")
    data_refueling_gb: float = Field(0.0, description="Data refueled in the current billing cycle in gigabytes")
    roaming_enabled: bool = Field(False, description="Whether international roaming is enabled for this line")
    contract_end_date: Optional[datetime.date] = Field(None, description="End date of the current contract")
    last_plan_change_date: Optional[datetime.date] = Field(None, description="Date of the most recent plan change")
    last_sim_replacement_date: Optional[datetime.date] = Field(None, description="Date of the most recent SIM card replacement")
    suspension_start_date: Optional[datetime.date] = Field(None, description="Start date of the current suspension period")


# =============================================================================
# Billing Models
# =============================================================================


class LineItem(BaseModel):
    """Individual charge or credit on a bill."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="Descriptive text for the line item")
    amount: float = Field(description="Monetary amount in USD (positive for charges, negative for credits)")
    date: datetime.date = Field(description="Date the line item was applied")
    item_type: str = Field(description="Category of the line item (e.g., Plan Charge, Overage, Fee, Credit, Payment)")


class BillStatus(str, Enum):
    """Status of a bill."""

    DRAFT = "Draft"
    ISSUED = "Issued"
    AWAITING_PAYMENT = "Awaiting Payment"
    PAID = "Paid"
    OVERDUE = "Overdue"
    DISPUTED = "Disputed"


class Bill(BaseModel):
    """Customer bill for a billing period."""

    model_config = ConfigDict(extra="forbid")

    bill_id: str = Field(description="Unique identifier for the bill")
    customer_id: str = Field(description="ID of the customer this bill belongs to")
    period_start: datetime.date = Field(description="Start date of the billing period")
    period_end: datetime.date = Field(description="End date of the billing period")
    issue_date: datetime.date = Field(description="Date the bill was issued/generated")
    total_due: float = Field(description="Total amount due in USD")
    due_date: datetime.date = Field(description="Date by which payment is due")
    line_items: List[LineItem] = Field(default_factory=list, description="Individual charges, credits, and payments")
    status: BillStatus = Field(BillStatus.DRAFT, description="Current status of the bill")


# =============================================================================
# Customer Models
# =============================================================================


class AccountStatus(str, Enum):
    """Status of a customer account."""

    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_VERIFICATION = "Pending Verification"
    CLOSED = "Closed"


class PaymentMethodType(str, Enum):
    """Type of payment method."""

    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    PAYPAL = "PayPal"


class PaymentMethod(BaseModel):
    """Stored payment method for a customer."""

    model_config = ConfigDict(extra="forbid")

    method_type: PaymentMethodType = Field(description="Type of payment method")
    account_number_last_4: str = Field(description="Last 4 digits of the account number")
    expiration_date: str = Field(description="The expiration date of the payment method in the format MM/YYYY")


class Customer(BaseModel):
    """Customer account with lines, bills, and payment methods."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(description="Unique identifier for the customer")
    full_name: str = Field(description="Customer's full name")
    date_of_birth: str = Field(description="Customer's date of birth for identity verification (format: YYYY-MM-DD)")
    email: str = Field(description="Customer's email address")
    phone_number: str = Field(description="Customer's primary contact phone number")
    address: Address = Field(description="Customer's billing address")
    account_status: AccountStatus = Field(AccountStatus.PENDING_VERIFICATION, description="Current status of the customer account")
    payment_methods: List[PaymentMethod] = Field(default_factory=list, description="Stored payment methods for this customer")
    line_ids: List[str] = Field(default_factory=list, description="Phone/data lines owned by this customer")
    bill_ids: List[str] = Field(default_factory=list, description="Bills associated with this customer")
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.combine(DEFAULT_START_DATE, datetime.time()),
        description="Date and time when the customer account was created",
    )
    last_extension_date: Optional[datetime.date] = Field(
        None, description="Date of the most recent payment extension (used for quarterly limit check)"
    )
    goodwill_credit_used_this_year: float = Field(0.0, description="Amount of goodwill credit used in the current calendar year")
