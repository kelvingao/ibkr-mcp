"""Strategy package for option trading strategies."""

from .base import BaseOptionStrategy
from .strategy_put_credit_spread import PutCreditSpreadStrategy

__all__ = ["BaseOptionStrategy", "PutCreditSpreadStrategy"]
