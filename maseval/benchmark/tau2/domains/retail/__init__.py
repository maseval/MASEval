"""Tau 2 Benchmark - Retail Domain.

E-commerce customer service domain with order management operations:
- Cancel pending orders
- Modify pending orders (items, address, payment)
- Exchange delivered items
- Return delivered items

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted components:
- models.py: Adapted from src/tau2/domains/retail/data_model.py
- db.py: Adapted from src/tau2/domains/retail/data_model.py:RetailDB
- tools.py: Adapted from src/tau2/domains/retail/tools.py
"""

from .models import (
    CancelReason,
    CreditCard,
    GiftCard,
    Order,
    OrderFullfilment,
    OrderItem,
    OrderPayment,
    OrderPaymentType,
    OrderStatus,
    Paypal,
    PaymentMethod,
    Product,
    User,
    UserAddress,
    UserName,
    Variant,
)
from .db import RetailDB
from .tools import RetailTools

__all__ = [
    # Models
    "Variant",
    "Product",
    "UserName",
    "UserAddress",
    "CreditCard",
    "Paypal",
    "GiftCard",
    "PaymentMethod",
    "User",
    "OrderFullfilment",
    "OrderItem",
    "OrderPayment",
    "OrderPaymentType",
    "OrderStatus",
    "CancelReason",
    "Order",
    # Database
    "RetailDB",
    # Tools
    "RetailTools",
]
