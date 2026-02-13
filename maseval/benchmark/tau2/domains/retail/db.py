"""Tau 2 Benchmark - Retail Domain Database.

Database class for the retail e-commerce domain.

Original benchmark: https://github.com/sierra-research/tau2-bench
Version: v0.2.0 (commit f8de30c, 2025-10-06)
Copyright (c) 2025 Sierra Research (MIT License)

Adapted from: src/tau2/domains/retail/data_model.py:RetailDB
"""

from typing import Any, Dict

from pydantic import Field

from maseval.benchmark.tau2.domains.base import DB
from maseval.benchmark.tau2.domains.retail.models import Order, Product, User


class RetailDB(DB):
    """Database containing all retail-related data including products, users and orders.

    Adapted from: tau2-bench src/tau2/domains/retail/data_model.py:RetailDB
    """

    products: Dict[str, Product] = Field(description="Dictionary of all products indexed by product ID")
    users: Dict[str, User] = Field(description="Dictionary of all users indexed by user ID")
    orders: Dict[str, Order] = Field(description="Dictionary of all orders indexed by order ID")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retail database.

        Returns:
            Dictionary with counts of products, users, orders, and total items
        """
        num_products = len(self.products)
        num_users = len(self.users)
        num_orders = len(self.orders)
        total_num_items = sum(len(product.variants) for product in self.products.values())
        return {
            "num_products": num_products,
            "num_users": num_users,
            "num_orders": num_orders,
            "total_num_items": total_num_items,
        }
