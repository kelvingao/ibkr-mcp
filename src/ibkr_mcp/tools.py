"""
ibkr-mcp MCP tools entrypoints.

This module exposes thin MCP tool wrappers that delegate to the shared
IBKRContext implementation in ``ibkr_mcp.server``.
"""

from typing import Any, Dict, Optional, cast
from mcp.server.fastmcp import Context
from ibkr_mcp.server import IBKRContext, mcp


def _ctx(ctx: Context) -> IBKRContext:
    """Helper to extract the strongly-typed IBKRContext from FastMCP Context."""
    return cast(IBKRContext, ctx.request_context.lifespan_context)


@mcp.tool(description="Retrieve account summary for a specific account or all accounts")
async def get_account_summary(ctx: Context, account: str = "") -> list[dict[str, Any]]:
    """
    Get account summary for a specific account or all accounts.

    Delegates to IBKRContext.get_account_summary.
    """
    return await _ctx(ctx).get_account_summary(account)


@mcp.tool(description="Retrieve portfolio for a specific account or the connected account")
async def get_portfolio(ctx: Context, account: str = "") -> list[dict[str, Any]]:
    """
    Get portfolio for a specific account or the connected account.

    Delegates to IBKRContext.get_portfolio.
    """
    return await _ctx(ctx).get_portfolio(account)


@mcp.tool(description="Retrieve normalised positions for a specific account or all accounts")
async def get_positions(ctx: Context, account: str = "") -> Dict[str, Any]:
    """
    Get normalised positions for a specific account or all accounts.

    Delegates to IBKRContext.get_positions, which owns the actual
    implementation that talks to IBKR and normalises the results.
    """
    return await _ctx(ctx).get_positions(account)


@mcp.tool(description="Retrieve portfolio greeks summary")
async def get_greeks_summary(ctx: Context, account: str = "") -> Dict[str, Any]:
    """
    Load current positions and return a portfolio greeks summary.

    Delegates to IBKRContext.get_greeks_summary.
    """
    return await _ctx(ctx).get_greeks_summary(account)


@mcp.tool(description="Retrieve historical IB news for a symbol or conid")
async def get_historical_news(
    ctx: Context,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_count: int = 10,
    exchange: str = "SMART",
    currency: str = "USD",
    conid: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get historical news for a stock by symbol or conid between two dates.

    Delegates to IBKRContext.get_historical_news.
    """
    return await _ctx(ctx).get_historical_news(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        max_count=max_count,
        exchange=exchange,
        currency=currency,
        conid=conid,
    )


__all__ = [
    "get_account_summary",
    "get_portfolio",
    "get_positions",
    "get_greeks_summary",
    "get_historical_news",
]

