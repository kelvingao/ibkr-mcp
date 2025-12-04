"""Common utilities for IBKR MCP (positions, greeks, risk rules, etc.).

This package hosts reusable domain logic that can be used by the
MCP server, tools, and external clients.
"""

from .greeks import GreekCalculator, compute_concentration  # noqa: F401
from .positions import PositionLoader, PositionSource, NORMALISED_COLUMNS  # noqa: F401
from .rules import RiskBreach, RiskEvaluator, RiskLimitConfig  # noqa: F401
from .playbooks import PlaybookContext, PlaybookEngine  # noqa: F401

__all__ = [
    "GreekCalculator",
    "compute_concentration",
    "PositionLoader",
    "PositionSource",
    "NORMALISED_COLUMNS",
    "RiskEvaluator",
    "RiskBreach",
    "RiskLimitConfig",
    "PlaybookContext",
    "PlaybookEngine",
]
