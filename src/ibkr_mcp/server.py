"""
FastMCP MCP server entry point, packaged under ibkr_mcp.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from dotenv import load_dotenv
from ib_async import IB
from mcp.server.fastmcp import FastMCP

load_dotenv()


@dataclass
class IBKRContext:
    """
    Lifespan context storing the shared IB client.
    """

    ib: IB


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
    await ib.connectAsync("127.0.0.1", 4001, clientId=9898, account="U9860850")

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


async def serve() -> None:
    """Async entry point for running the MCP server over stdio."""
    await mcp.run_stdio_async()
