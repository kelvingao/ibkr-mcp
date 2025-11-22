"""
ibkr-mcp related MCP tools.
"""

from mcp.server.fastmcp import Context
from ibkr_mcp.server import mcp

from .common.greeks import GreekCalculator, compute_concentration
from .common.positions import PositionLoader, PositionSource
from typing import Dict, Any


@mcp.tool()
async def get_account_summary(ctx: Context, account: str = "") -> list[dict[str, Any]]:
    """
    Get account summary for specific account or all accounts.

    Args:
        ctx: The MCP request context, providing access to the IB client via
            the lifespan context.
        account: The account name to retrieve summary for. If empty, summary
            for all accounts is returned.
    Return:
        A list of account summary items for all accounts, converted to plain
        dictionaries so they can be serialized over MCP.
    """
    ib = ctx.request_context.lifespan_context.ib
    if not ib.isConnected():
        raise RuntimeError("Not connected to Interactive Brokers")

    account_summary = await ib.accountSummaryAsync(account)
    if account_summary is None:
        return []

    items: list[dict[str, Any]] = []
    for v in account_summary:
        items.append({
             "tag": v.tag,
             "value": v.value,
             "currency": v.currency,
             "account": v.account
        })

    return items

@mcp.tool()
async def get_portfolio(ctx: Context, account: str = "") -> list[dict[str, Any]]:
    """
    Get portfolio for specific account or the connected account.

    Args:
        ctx: The MCP request context, providing access to the IB client via
            the lifespan context.
        account: The account name to retrieve portfolio for. If empty, portfolio
            for the connected account is returned.
    Returns:
        A list of portfolio items for the account, converted to plain
        dictionaries so they can be serialized over MCP.
    """
    # The IB client is stored on the lifespan context created in ibkr_lifespan.
    ib = ctx.request_context.lifespan_context.ib

    if not ib.isConnected():
        raise RuntimeError("Not connected to Interactive Brokers")

    portfolio = ib.portfolio(account)
    if portfolio is None:
        return []

    items: list[dict[str, Any]] = []
    for p in portfolio:
        items.append(
            {
                "account": p.account,
                "contract": p.contract,
                "position": p.position,
                "marketPrice": p.marketPrice,
                "marketValue": p.marketValue,
                "averageCost": p.averageCost,
                "unrealizedPNL": p.unrealizedPNL,
                "realizedPNL": p.realizedPNL,
            }
        )

    return items

# @mcp.tool()
# async def get_positions(ctx: Context, account: str = "") -> list[dict[str, Any]]:
#     """
#     Get positions for specific account or all accounts.

#     Args:
#         ctx: The MCP request context, providing access to the IB client via
#             the lifespan context.

#     Returns:
#         A list of position items for all accounts, converted to plain
#         dictionaries so they can be serialized over MCP.
#     """
#     ib = ctx.request_context.lifespan_context.ib

#     if not ib.isConnected():
#         raise RuntimeError("Not connected to Interactive Brokers")

#     positions = ib.positions(account)
#     if positions is None:
#         return []

#     items: list[dict[str, Any]] = []
#     for pos in positions:
#         items.append(
#             {
#                 "account": pos.account,
#                 "contract": pos.contract,
#                 "position": pos.position,
#                 "averageCost": pos.avgCost,
#             }
#         )

#     return items


@mcp.tool()
async def get_positions(
    ctx: Context,
    account: str = ""
) -> Dict[str, Any]:
    """
    Load current positions and return a portfolio greeks summary.

    This tool:
    - fetches positions for the given account from IBKR
    - computes option greeks per underlying
    - aggregates portfolio-wide greeks totals
    - computes a simple concentration metric by gross exposure
    """
    ib = ctx.request_context.lifespan_context.ib

    if not ib.isConnected():
        raise RuntimeError("Not connected to Interactive Brokers")
    
    loader = PositionLoader(
        PositionSource(ib=ib)
    )
    positions = loader.load(account)

    return {
        "positions": positions.to_dict(orient="records")
    }


@mcp.tool()
async def get_greeks_summary(
    ctx: Context,
    account: str = ""
) -> Dict[str, Any]:
    """
    Load current positions and return a portfolio greeks summary.

    This tool:
    - fetches positions for the given account from IBKR
    - computes option greeks per underlying
    - aggregates portfolio-wide greeks totals
    - computes a simple concentration metric by gross exposure
    """
    ib = ctx.request_context.lifespan_context.ib

    if not ib.isConnected():
        raise RuntimeError("Not connected to Interactive Brokers")
    
    loader = PositionLoader(
        PositionSource(ib=ib)
    )
    positions = loader.load(account)

    return {
        "positions": positions.to_dict(orient="records")
    }

    calculator = GreekCalculator(ib)
    greek_summary = calculator.compute(positions)
    concentration, concentration_series = compute_concentration(positions)

    # Convert DataFrames to dictionaries for JSON serialization
    return {
        "greek_summary": {
            "per_symbol": greek_summary.per_symbol.to_dict(orient="records"),
            "totals": greek_summary.totals
        },
        "concentration": {
            "per_symbol": concentration.to_dict(orient="records"),
            "series": concentration_series.to_dict()
        },
        "position_count": len(positions)
    }
