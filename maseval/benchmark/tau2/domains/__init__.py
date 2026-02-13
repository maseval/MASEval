"""Tau 2 Benchmark - Domain Implementations.

This package contains domain-specific implementations for the tau2-bench benchmark.

Domains:
    - retail: E-commerce order management (cancel, modify, exchange, return)
    - airline: Flight reservation management (book, modify, cancel)
    - telecom: Telecom customer service (billing, plans, tech support)

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)
"""

from .base import DB, ToolKitBase, ToolType, is_tool

VALID_DOMAINS = ("retail", "airline", "telecom")

__all__ = [
    "VALID_DOMAINS",
    "DB",
    "ToolKitBase",
    "ToolType",
    "is_tool",
]
