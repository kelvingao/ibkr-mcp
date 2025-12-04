"""Base classes and utilities for NautilusTrader option strategies."""
from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Optional

from loguru import logger

try:
    from nautilus_trader.trading.strategy import Strategy
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "NautilusTrader must be installed to use the strategy framework."
    ) from exc


@dataclass(slots=True)
class TradeSignal:
    """Represents a trade signal emitted by a strategy."""

    symbol: str
    expiry: datetime
    strike: float
    option_type: str
    direction: str
    rationale: str


class BaseOptionStrategy(Strategy, abc.ABC):
    """Abstract base class for option strategies using NautilusTrader."""

    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if name is not None:
            self.name = name
        elif not hasattr(self, "name") or getattr(self, "name", None) in (None, ""):
            self.name = self.__class__.__name__
        logger.debug("Initialized strategy: {name}", name=self.name)

    @abc.abstractmethod
    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """Process incoming data and return trade signals."""

    def emit_signal(self, signal: TradeSignal) -> TradeSignal:
        """Utility for logging and returning signals."""

        logger.info(
            "Signal emitted | strategy={strategy} symbol={symbol} strike={strike} expiry={expiry} type={opt_type} direction={direction} rationale={rationale}",
            strategy=self.name,
            symbol=signal.symbol,
            strike=signal.strike,
            expiry=signal.expiry.isoformat(),
            opt_type=signal.option_type,
            direction=signal.direction,
            rationale=signal.rationale,
        )
        return signal

    @staticmethod
    def _snapshot_value(snapshot: Any, key: str, default: Any = None) -> Any:
        if snapshot is None:
            return default
        if isinstance(snapshot, Mapping):
            return snapshot.get(key, default)
        return getattr(snapshot, key, default)

    @staticmethod
    def _snapshot_options(snapshot: Any) -> List[Any]:
        options = BaseOptionStrategy._snapshot_value(snapshot, "options")
        if not options:
            return []
        if isinstance(options, list):
            return options
        try:
            return list(options)
        except TypeError:
            return []

    def _resolve_underlying_price(self, snapshot: Any, chain: Any) -> Optional[float]:
        price = self._snapshot_value(snapshot, "underlying_price")
        if price is None and chain is not None:
            try:
                if "underlying_price" in chain and len(chain):
                    series = chain["underlying_price"]
                    price = series.iloc[0] if hasattr(series, "iloc") else series[0]
            except Exception:
                price = None
        if price is None:
            return None
        try:
            return float(price)
        except (TypeError, ValueError):
            return None


__all__ = ["BaseOptionStrategy", "TradeSignal"]
