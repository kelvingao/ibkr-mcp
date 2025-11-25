"""
ibkr_mcp package.

The main MCP server entry point is exposed as `ibkr_mcp.main`, so you can use
it as a console script target or run with `python -m ibkr_mcp`.
"""

from .server import mcp, serve  # noqa: F401

# Import tools module explicitly so its @mcp.tool() decorators are registered.
# This happens after the server (and its `mcp` instance) is fully initialised,
# which avoids circular imports between server and tools.
from . import tools as _tools  # noqa: F401


def main() -> None:
    import asyncio

    asyncio.run(serve())


if __name__ == "__main__":
    main()
