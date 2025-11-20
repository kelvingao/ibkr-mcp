"""
ibkr_mcp package.

The main MCP server entry point is exposed as `ibkr_mcp.main`, so you can use
it as a console script target or run with `python -m ibkr_mcp`.
"""
from .server import serve

def main():
    import asyncio
    asyncio.run(serve())


if __name__ == "__main__":
    main()

