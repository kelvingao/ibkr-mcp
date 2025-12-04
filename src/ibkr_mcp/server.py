"""
FastMCP MCP server entry point, packaged under ibkr_mcp.
"""

import os

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from dotenv import load_dotenv
from ib_async import IB, Contract, Stock, Forex, Future, Option
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware

from ibkr_mcp.services import AccountService, NewsService, OptionDataService, RiskService

load_dotenv()

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4001"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "0"))
IBKR_ACCOUNT = os.getenv("IBKR_ACCOUNT", "")
OPTION_DATA_DIR = Path(os.getenv("IBKR_MCP_OPTION_DATA_DIR", "optiondata"))
OPTION_HISTORY_DIR = Path(os.getenv("IBKR_MCP_OPTION_HISTORY_DIR", "historydata"))
DEFAULT_MARKET_DATA_TYPE = os.getenv("IBKR_MCP_MARKET_DATA_TYPE", "LIVE")

@dataclass
class IBKRContext:
    """
    Lifespan context storing the shared IB client.
    """

    ib: IB
    account_service: AccountService = field(init=False)
    risk_service: RiskService = field(init=False)
    news_service: NewsService = field(init=False)
    option_data_service: OptionDataService = field(init=False)

    def __post_init__(self) -> None:
        self.account_service = AccountService(self.ib, self._ensure_connected)
        self.risk_service = RiskService(self.ib, self._ensure_connected)
        self.news_service = NewsService(self.ib, self._ensure_connected)
        self.option_data_service = OptionDataService(
            self.ib,
            self._ensure_connected,
            data_dir=OPTION_DATA_DIR,
            history_dir=OPTION_HISTORY_DIR,
            default_market_data_type=DEFAULT_MARKET_DATA_TYPE,
        )

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
        return await self.account_service.get_account_summary(account)

    async def get_positions(self, account: str = "") -> dict:
        """
        Load current positions for the given account using the shared IB client.

        Returns a dictionary containing a list of normalised position records.
        """
        return self.account_service.get_positions(account)

    async def get_portfolio(self, account: str = "") -> list[dict]:
        """
        Get portfolio for a specific account or the connected account.

        Returns a list of portfolio position dictionaries.
        """
        return self.account_service.get_portfolio(account)

    async def get_greeks_summary(self, account: str = "") -> Dict[str, Any]:
        """
        Load current positions and return a portfolio greeks summary.

        This method:
        - fetches positions for the given account from IBKR
        - computes option greeks per underlying
        - aggregates portfolio-wide greeks totals
        - computes a simple concentration metric by gross exposure
        """
        return await self.risk_service.get_greeks_summary(account)

    async def evaluate_portfolio_risk(
        self,
        account: str = "",
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "risk.yaml",
    ) -> Dict[str, Any]:
        """
        Load positions, compute greeks and concentration, and evaluate risk limits.

        The risk configuration can be provided directly as a dict (via MCP tool
        arguments) or loaded from a YAML file. Lookup order is:
        - the explicitly provided ``config`` argument, if not None
        - the explicitly provided ``config_path`` argument, if not empty
        - the IBKR_MCP_RISK_CONFIG environment variable
        - a local "risk.yaml" file in the working directory
        """
        return await self.risk_service.evaluate_portfolio_risk(
            account=account,
            config=config,
            config_path=config_path,
        )

    async def generate_playbook_actions(
        self,
        account: str = "",
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "risk.yaml",
    ) -> Dict[str, Any]:
        """
        Generate strategy playbook actions based on current positions and risk rules.

        This reuses the same risk configuration resolution as evaluate_portfolio_risk
        and applies PlaybookEngine to derive human-readable adjustment suggestions.
        """
        return await self.risk_service.generate_playbook_actions(
            account=account,
            config=config,
            config_path=config_path,
        )

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
        return await self.news_service.get_historical_news(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_count=max_count,
            exchange=exchange,
            currency=currency,
            conid=conid,
        )

    async def get_option_chains(
        self,
        symbols: Optional[list[str]] = None,
        mode: str = "live",
        market_data_type: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch option chain snapshots either via live IBKR data or local cache.
        """
        return await self.option_data_service.fetch_option_chains(
            symbols or [],
            mode=mode,
            market_data_type=market_data_type,
            persist=persist,
        )

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
