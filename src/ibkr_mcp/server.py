"""
FastMCP MCP server entry point, packaged under ibkr_mcp.
"""

import os
from datetime import datetime, timedelta

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

from dotenv import load_dotenv
from ib_async import IB, Contract, Stock, Forex, Future, Option
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware

from ibkr_mcp.common.positions import PositionLoader, PositionSource
from ibkr_mcp.common.greeks import GreekCalculator, compute_concentration

load_dotenv()

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4001"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "0"))
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "")

@dataclass
class IBKRContext:
    """
    Lifespan context storing the shared IB client.
    """

    ib: IB

    def _ensure_connected(self) -> None:
        if not self.ib.isConnected():
            raise RuntimeError("Not connected to Interactive Brokers")

    @staticmethod
    def _create_contract(
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        if symbol.isdigit():
            return Contract(conId=int(symbol))
        if sec_type == "STK":
            return Stock(symbol=symbol, exchange=exchange, currency=currency)
        if sec_type in ("FOREX", "CASH"):
            return Forex(pair=symbol)
        if sec_type == "FUT":
            return Future(symbol=symbol, exchange=exchange)
        if sec_type == "OPT":
            # Option expects strike as float, not currency as 3rd arg
            return Option(symbol=symbol, exchange=exchange, currency=currency)
        return Contract(
            symbol=symbol, secType=sec_type, exchange=exchange, currency=currency
        )

    def _flatten_contracts(self, contracts: list[Any]) -> list[Contract]:
        # Recursively flatten nested contract lists and filter out None
        result: list[Contract] = []
        for c in contracts:
            if isinstance(c, Contract):
                result.append(c)
            elif isinstance(c, list):
                result.extend(self._flatten_contracts(c))
        return result
    
    @staticmethod
    def _format_markdown_list(items: list[str], ordered: bool = False) -> str:
        """Format items as a markdown list."""
        if not items:
            return ""

        if ordered:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        else:
            return "\n".join(f"- {item}" for item in items)





    async def get_account_summary(self, account: str = "") -> list[dict]:
        self._ensure_connected()
        raw = await self.ib.accountSummaryAsync(account)
        return [
            {
                "tag": v.tag,
                "currency": v.currency,
                "account": v.account,
                "value": v.value,
            }
            for v in raw or []
        ]

    async def get_positions(self, account: str = "") -> dict:
        """
        Load current positions for the given account using the shared IB client.

        Returns a dictionary containing a list of normalised position records.
        """
        self._ensure_connected()
        loader = PositionLoader(PositionSource(ib=self.ib))
        positions = loader.load(account)
        return {
            "positions": positions.to_dict(orient="records")
        }

    async def get_portfolio(self, account: str = "") -> list[dict]:
        """
        Get portfolio for a specific account or the connected account.

        Returns a list of portfolio position dictionaries.
        """
        self._ensure_connected()
        portfolio = self.ib.portfolio(account)
        if portfolio is None:
            return []

        items: list[dict] = []
        for p in portfolio:
            items.append(
                {
                    "account": getattr(p, "account", ""),
                    "contract": getattr(p, "contract", None),
                    "position": getattr(p, "position", 0),
                    "marketPrice": getattr(p, "marketPrice", 0.0),
                    "marketValue": getattr(p, "marketValue", 0.0),
                    "averageCost": getattr(p, "averageCost", 0.0),
                    "unrealizedPNL": getattr(p, "unrealizedPNL", 0.0),
                    "realizedPNL": getattr(p, "realizedPNL", 0.0),
                }
            )
        return items

    async def get_greeks_summary(self, account: str = "") -> Dict[str, Any]:
        """
        Load current positions and return a portfolio greeks summary.

        This method:
        - fetches positions for the given account from IBKR
        - computes option greeks per underlying
        - aggregates portfolio-wide greeks totals
        - computes a simple concentration metric by gross exposure
        """
        self._ensure_connected()
        loader = PositionLoader(PositionSource(ib=self.ib))
        positions = loader.load(account)

        calculator = GreekCalculator(self.ib)
        greek_summary = calculator.compute(positions)
        concentration, concentration_series = compute_concentration(positions)

        return {
            "greek_summary": {
                "per_symbol": greek_summary.per_symbol.to_dict(orient="records"),
                "totals": greek_summary.totals,
            },
            "concentration": {
                "per_symbol": concentration.to_dict(orient="records"),
                "series": concentration_series.to_dict(),
            },
            "position_count": len(positions),
        }

    async def get_historical_news(
        self,
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
        Mirrors the behaviour of the original news tool, but centralised here.
        """
        self._ensure_connected()
        ib = self.ib

        # Validate and set default dates
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Validate date format and order
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                return {
                    "ok": False,
                    "error_type": "INVALID_DATE_RANGE",
                    "message": f"Start date {start_date} must be before end date {end_date}",
                    "symbol": symbol,
                }
        except ValueError as exc:
            return {
                "ok": False,
                "error_type": "INVALID_DATE_FORMAT",
                "message": f"Date format must be YYYY-MM-DD: {exc}",
                "symbol": symbol,
            }

        # Validate max_count
        max_count = max(1, min(100, max_count))

        try:
            # Use conid if provided, otherwise qualify symbol
            if conid:
                contract = Contract(conId=conid)
                contracts = [contract]
            else:
                contract = Contract(
                    symbol=symbol,
                    secType="STK",
                    exchange=exchange,
                    currency=currency,
                )
                contracts_raw = await ib.qualifyContractsAsync(contract)
                contracts = []
                for contract_list in contracts_raw:
                    if isinstance(contract_list, list):
                        contracts.extend(contract_list)
                    else:
                        contracts.append(contract_list)

            if not contracts:
                return {
                    "ok": False,
                    "error_type": "NO_CONTRACT",
                    "message": f"No contract found for {symbol}",
                    "symbol": symbol,
                }

            c = contracts[0]
            actual_conid = getattr(c, "conId", 0)

            # Common news providers; adjust as needed
            news_provider_codes = "BRFG+BRFUPDN+DJ-N+DJ-RT+DJ-RTA+DJ-RTE+DJ-RTG+DJNL+FLY"

            news = await ib.reqHistoricalNewsAsync(
                actual_conid,
                news_provider_codes,
                start_date,
                end_date,
                max_count,
            )

            if not news:
                return {
                    "ok": True,
                    "symbol": symbol,
                    "conid": actual_conid,
                    "period": {"start": start_date, "end": end_date},
                    "articles": [],
                    "total_count": 0,
                }

            # Process news articles
            articles: list[Dict[str, Any]] = []
            if isinstance(news, list):
                for article in news[:max_count]:
                    articles.append(
                        {
                            "headline": getattr(article, "headline", ""),
                            "timestamp": getattr(article, "time", ""),
                            "provider": getattr(article, "providerCode", ""),
                            "article_id": getattr(article, "articleId", ""),
                        }
                    )

            return {
                "ok": True,
                "symbol": symbol,
                "conid": actual_conid,
                "period": {"start": start_date, "end": end_date},
                "articles": articles,
                "total_count": len(articles),
            }

        except Exception as exc:
            return {
                "ok": False,
                "error_type": "IB_ERROR",
                "message": f"Error getting historical news: {exc}",
                "symbol": symbol,
            }

@asynccontextmanager
async def ibkr_lifespan(server: FastMCP) -> AsyncIterator[IBKRContext]:
    """
    Manages the IB client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        IBKRContext: The context containing the IB client.
    """
    ib = IB()
    await ib.connectAsync(
        IBKR_HOST,
        IBKR_PORT,
        clientId=IBKR_CLIENT_ID,
        account=IBKR_ACCOUNT
    )

    ib.reqMarketDataType(2)

    try:
        yield IBKRContext(ib=ib)
    finally:
        ib.disconnect()


# Create an MCP server
mcp = FastMCP(
    name="ibkr-mcp",
    lifespan=ibkr_lifespan,
)

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting resource."""
    return f"Hello, {name}!"


@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt."""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."

async def run_http_with_cors() -> None:
    """
    Run the FastMCP HTTP server with CORS enabled so that browsers can
    successfully perform OPTIONS /mcp/ preflight requests.
    """
    import uvicorn

    # Base Streamable HTTP app from FastMCP
    app = mcp.streamable_http_app()

    # Allow configuration via environment; default to wildcard for local use
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    if raw_origins == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [
            origin.strip()
            for origin in raw_origins.split(",")
            if origin.strip()
        ]

    # Attach CORS middleware. This will handle OPTIONS /mcp/ preflight
    # and add the appropriate CORS headers to responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # HTTP host/port are configured via environment to avoid relying on
    # FastMCP settings that may not exist in older mcp versions.
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8050"))

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)
    await server.serve()

async def serve() -> None:
    """Async entry point for running the MCP server over stdio."""
    try:
        transport = os.getenv("TRANSPORT", "stdio")
        if transport == 'http':
            # Run the MCP server with streamable HTTP transport
            await run_http_with_cors()
        elif transport == 'stdio':
            # Run the MCP server with stdio transport
            await mcp.run_stdio_async()
        else:
            raise ValueError(f"Unsupported transport: {transport}. Use 'http' or 'stdio'.")
    finally:
        print("🔄 Server shutting down...")
        mcp.get_context().lifespan_context.ib.disconnect()
