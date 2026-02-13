"""Banking tool collection with shared state across sub-tools.

Tools:
- banking_get_balance: Get current account balance
- banking_get_transactions: Get transaction history with optional date filtering
- banking_get_transaction: Get specific transaction by ID
- banking_get_asset: Get asset information by name
"""

from dataclasses import dataclass
from typing import Any

from .base import BaseTool, ToolResult


@dataclass
class Transaction:
    """Bank transaction structure."""

    id: str
    date: str
    description: str
    amount: float
    type: str  # "deposit" or "expense"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "date": self.date,
            "description": self.description,
            "amount": self.amount,
            "type": self.type,
        }


class BankingState:
    """Shared state for all banking tools.

    Maintains balance, transactions, and assets across all sub-tool invocations.
    """

    def __init__(
        self,
        transactions_data: list[dict[str, Any]],
        current_balance: float,
        assets_data: dict[str, Any] | None = None,
    ):
        self.current_balance = current_balance
        self.transactions: list[Transaction] = []
        self.assets = assets_data or {}

        # Load transactions
        for i, txn_data in enumerate(transactions_data):
            self.transactions.append(
                Transaction(
                    id=str(i + 1),
                    date=txn_data["date"],
                    description=txn_data["description"],
                    amount=txn_data["amount"],
                    type=txn_data["type"],
                )
            )


class BankingGetBalanceTool(BaseTool):
    """Get current account balance."""

    def __init__(self, banking_state: BankingState):
        super().__init__(
            "banking_get_balance",
            "Get current bank account balance",
            tool_args=[],
        )
        self.state = banking_state

    def execute(self, **kwargs) -> ToolResult:
        """Get current account balance."""
        return ToolResult(
            success=True,
            data={"balance": self.state.current_balance, "currency": "USD"},
        )


class BankingGetTransactionsTool(BaseTool):
    """Get transaction history with optional date filtering."""

    def __init__(self, banking_state: BankingState):
        super().__init__(
            "banking_get_transactions",
            "Get transaction history with optional start_date and end_date filters (YYYY-MM-DD format)",
            tool_args=["start_date", "end_date"],
        )
        self.state = banking_state

    def execute(self, **kwargs) -> ToolResult:
        """Get transaction history with optional date filtering."""
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        filtered = self.state.transactions

        if start_date:
            filtered = [t for t in filtered if t.date >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.date <= end_date]

        return ToolResult(
            success=True,
            data=[t.to_dict() for t in filtered],
            metadata={"count": len(filtered)},
        )


class BankingGetTransactionTool(BaseTool):
    """Get specific transaction by ID."""

    def __init__(self, banking_state: BankingState):
        super().__init__(
            "banking_get_transaction",
            "Get specific transaction details by transaction ID",
            tool_args=["transaction_id"],
        )
        self.state = banking_state

    def execute(self, **kwargs) -> ToolResult:
        """Get specific transaction by ID."""
        transaction_id = kwargs.get("transaction_id")

        if not transaction_id:
            return ToolResult(success=False, data=None, error="transaction_id is required")

        for txn in self.state.transactions:
            if txn.id == transaction_id:
                return ToolResult(success=True, data=txn.to_dict())

        return ToolResult(success=False, data=None, error=f"Transaction {transaction_id} not found")


class BankingGetAssetTool(BaseTool):
    """Get number of shares owned for a stock ticker."""

    def __init__(self, banking_state: BankingState):
        super().__init__(
            "banking_get_asset",
            "Get the number of shares you own for a stock ticker symbol (e.g., 'AAPL' returns how many Apple shares you own)",
            tool_args=["asset_name"],
        )
        self.state = banking_state

    def execute(self, **kwargs) -> ToolResult:
        """Get asset information."""
        asset_name = kwargs.get("asset_name")

        if not asset_name:
            return ToolResult(success=False, data=None, error="asset_name is required")

        if asset_name in self.state.assets:
            return ToolResult(
                success=True,
                data={"asset": asset_name, "value": self.state.assets[asset_name]},
            )

        return ToolResult(success=False, data=None, error=f"Asset {asset_name} not found")


class BankingToolCollection:
    """Banking tool collection factory.

    Creates a shared state and returns all banking sub-tools that share that state.
    This ensures transaction history and balance are consistent across all operations.

    Usage:
        banking_state = BankingState(transactions_data, balance, assets_data)
        collection = BankingToolCollection(banking_state)
        tools = collection.get_sub_tools()
        # All tools share the same banking state
    """

    def __init__(self, banking_state: BankingState):
        self.state = banking_state

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all banking sub-tools with shared state."""
        return [
            BankingGetBalanceTool(self.state),
            BankingGetTransactionsTool(self.state),
            BankingGetTransactionTool(self.state),
            BankingGetAssetTool(self.state),
        ]
