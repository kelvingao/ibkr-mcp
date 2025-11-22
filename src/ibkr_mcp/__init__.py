"""
ibkr_mcp package.

The main MCP server entry point is exposed as `ibkr_mcp.main`, so you can use
it as a console script target or run with `python -m ibkr_mcp`.
"""
from .server import mcp, serve

# Import tools package so its @mcp.tool() registrations are applied.
from . import tools as _tools  # noqa: F4011

def main():
    import asyncio
    asyncio.run(serve())


if __name__ == "__main__":
    main()
