"""Stock price lookup tool - single purpose."""

from .base import BaseTool, ToolResult


class StockPriceGetTool(BaseTool):
    """Get stock price by symbol."""

    def __init__(self, price_data: dict[str, float]):
        super().__init__(
            "stock_price_get",
            "Look up current stock price by symbol (e.g. 'AAPL', 'GOOGL')",
            tool_args=["symbol"],
        )
        self.prices = price_data

    def execute(self, **kwargs) -> ToolResult:
        """Get stock price for symbol."""
        symbol = kwargs.get("symbol")

        if not symbol:
            return ToolResult(success=False, data=None, error="symbol is required")

        symbol = symbol.upper()
        if symbol in self.prices:
            return ToolResult(
                success=True,
                data={
                    "symbol": symbol,
                    "price": self.prices[symbol],
                    "currency": "USD",
                },
            )

        return ToolResult(success=False, data=None, error=f"Symbol {symbol} not found")


class StockPriceToolCollection:
    """Stock price tool collection factory.

    Currently only contains one tool, but structured as a collection
    for consistency with other tool patterns.
    """

    def __init__(self, price_data: dict[str, float]):
        self.price_data = price_data

    def get_sub_tools(self) -> list[BaseTool]:
        """Return all stock price sub-tools."""
        return [
            StockPriceGetTool(self.price_data),
        ]
