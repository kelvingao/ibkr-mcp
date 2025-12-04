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


@mcp.tool(description="Scan option market and generate strategy trade signals")
async def scan_option_signals(
    ctx: Context,
    symbols: list[str] | None = None,
    strategies: list[str] | None = None,
    run_mode: str = "live",
    market_data_type: str = "LIVE",
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """
    Scan a batch of symbols for options strategy trade signals.

    This delegates to IBKRContext.scan_option_signals, which uses the external
    optionscanner project to:
    - load configuration and discover strategies
    - fetch option chains from IBKR or local parquet snapshots
    - execute each strategy and normalise the resulting signals
    """
    return await _ctx(ctx).scan_option_signals(
        symbols=symbols,
        strategies=strategies,
        run_mode=run_mode,
        market_data_type=market_data_type,
        config_path=config_path,
    )


@mcp.tool(description="Fetch option chain snapshots for the provided symbols")
async def get_option_chains(
    ctx: Context,
    symbols: list[str],
    mode: str = "live",
    market_data_type: str = "LIVE",
    persist: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve option chain snapshots either live from IBKR or from the local cache.
    """
    return await _ctx(ctx).get_option_chains(
        symbols=symbols,
        mode=mode,
        market_data_type=market_data_type,
        persist=persist,
    )


@mcp.tool(description="Evaluate portfolio risk using configured limits")
async def evaluate_portfolio_risk(
    ctx: Context,
    account: str = "",
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "risk.yaml",
) -> Dict[str, Any]:
    """
    Evaluate portfolio risk based on current positions and a risk configuration.

    Configuration resolution order:
    - if ``config`` is provided, it is used directly (no file loading)
    - otherwise, a YAML file is loaded using ``config_path`` or IBKR_MCP_RISK_CONFIG

    This delegates to IBKRContext.evaluate_portfolio_risk and returns:
    - current greeks and concentration metrics
    - loaded risk limits and roll rules
    - a list of breached risk constraints
    """
    return await _ctx(ctx).evaluate_portfolio_risk(
        account=account,
        config=config,
        config_path=config_path,
    )


@mcp.tool(description="Generate playbook-style adjustment actions for current portfolio")
async def generate_playbook_actions(
    ctx: Context,
    account: str = "",
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "risk.yaml",
) -> Dict[str, Any]:
    """
    Generate strategy playbook actions based on current positions and risk rules.

    This delegates to IBKRContext.generate_playbook_actions and returns a list of
    human-readable adjustment suggestions along with the underlying risk context.
    """
    return await _ctx(ctx).generate_playbook_actions(
        account=account,
        config=config,
        config_path=config_path,
    )


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
    "get_option_chains",
    "scan_option_signals",
    "get_historical_news",
    "evaluate_portfolio_risk",
    "generate_playbook_actions",
]
