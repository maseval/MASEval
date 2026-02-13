"""Tau 2 Benchmark - Telecom Domain Database.

Database class for the telecom customer service domain.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/telecom/data_model.py:TelecomDB

Note: Unlike retail/airline, telecom uses Lists instead of Dicts for entities.
This matches the upstream tau2-bench structure.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from maseval.benchmark.tau2.domains.base import DB
from maseval.benchmark.tau2.domains.telecom.models import Bill, Customer, Device, Line, Plan
from maseval.benchmark.tau2.domains.telecom.user_models import TelecomUserDB


class TelecomDB(DB):
    """Database containing all telecom-related data.

    Note: Uses Lists instead of Dicts for entities, matching upstream tau2-bench.

    Adapted from: tau2-bench src/tau2/domains/telecom/data_model.py:TelecomDB
    """

    plans: List[Plan] = Field(default_factory=list, description="Available service plans")
    customers: List[Customer] = Field(default_factory=list, description="All customers in the system")
    lines: List[Line] = Field(default_factory=list, description="All lines in the system")
    bills: List[Bill] = Field(default_factory=list, description="All bills in the system")
    devices: List[Device] = Field(default_factory=list, description="All devices in the system")

    # User-side state (device and surroundings)
    # Optional because it's only used when user simulation is active
    user_db: Optional[TelecomUserDB] = Field(None, description="User device and environment state")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the telecom database.

        Returns:
            Dictionary with counts of plans, customers, lines, bills, devices, and payment methods
        """
        num_plans = len(self.plans)
        num_customers = len(self.customers)
        num_lines = len(self.lines)
        num_bills = len(self.bills)
        num_devices = len(self.devices)
        num_payment_methods = sum(len(customer.payment_methods) for customer in self.customers)

        stats = {
            "num_plans": num_plans,
            "num_customers": num_customers,
            "num_lines": num_lines,
            "num_bills": num_bills,
            "num_devices": num_devices,
            "num_payment_methods": num_payment_methods,
        }

        if self.user_db:
            stats["user_db"] = {
                "device_on": self.user_db.device.is_on,
                "network_status": self.user_db.device.network_status.value,
                "sim_status": self.user_db.device.sim_status.value,
            }

        return stats
