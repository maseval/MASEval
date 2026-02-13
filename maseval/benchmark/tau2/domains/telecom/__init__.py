"""Tau 2 Benchmark - Telecom Domain.

Telecom customer service domain with account and billing operations:
- Customer account management
- Line/plan management
- Billing and payments
- Device management
- Tech support (with user tools for device simulation)

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted components:
- models.py: Adapted from src/tau2/domains/telecom/data_model.py
- db.py: Adapted from src/tau2/domains/telecom/data_model.py:TelecomDB
- tools.py: Adapted from src/tau2/domains/telecom/tools.py
"""

from .models import (
    AccountStatus,
    Address,
    Bill,
    BillStatus,
    Customer,
    Device,
    DeviceType,
    Line,
    LineItem,
    LineStatus,
    PaymentMethod,
    PaymentMethodType,
    Plan,
)
from .db import TelecomDB
from .tools import TelecomTools
from .user_tools import TelecomUserTools
from .user_models import TelecomUserDB

__all__ = [
    # Models
    "AccountStatus",
    "Address",
    "Bill",
    "BillStatus",
    "Customer",
    "Device",
    "DeviceType",
    "Line",
    "LineItem",
    "LineStatus",
    "PaymentMethod",
    "PaymentMethodType",
    "Plan",
    # Database
    "TelecomDB",
    "TelecomUserDB",
    # Tools
    "TelecomTools",
    "TelecomUserTools",
]
