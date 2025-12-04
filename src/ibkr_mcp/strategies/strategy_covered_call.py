"""Covered call strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List

import pandas as pd
from loguru import logger

from .base import BaseOptionStrategy, TradeSignal


class CoveredCallStrategy(BaseOptionStrategy):
    """Suggests covered calls based on yield thresholds."""

    def __init__(
        self,
        min_days_to_expiry: int = 21,
        min_annualized_yield: float = 0.12,
        enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_days_to_expiry = min_days_to_expiry
        self.min_annualized_yield = min_annualized_yield
        self.enabled = enabled

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        if not self.enabled:
            logger.debug("CoveredCallStrategy disabled; skipping evaluation")
            return []
        signals: List[TradeSignal] = []
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue
            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None or underlying_price <= 0:
                continue
            for expiry, subset in chain.groupby("expiry"):
                days_to_expiry = max((expiry - datetime.now(timezone.utc)).days, 1)
                if days_to_expiry < self.min_days_to_expiry:
                    continue
                otm_calls = subset[(subset["option_type"] == "CALL") & (subset["strike"] >= underlying_price * 1.05)]
                if otm_calls.empty:
                    continue
                otm_calls = otm_calls.sort_values("strike")
                best = None
                best_yield = 0.0
                for _, row in otm_calls.iterrows():
                    premium = float(row.get("bid", row.get("mark", 0.0)))
                    annualized_yield = (premium / underlying_price) * (365 / days_to_expiry)
                    if annualized_yield > best_yield:
                        best_yield = annualized_yield
                        best = row
                if best is None or best_yield < self.min_annualized_yield:
                    continue
                rationale = (
                    f"Covered call premium {best.get('bid', best.get('mark', 0.0)):.2f} with annualized yield {best_yield:.2%}"
                )
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=subset["symbol"].iloc[0],
                            expiry=expiry,
                            strike=float(best["strike"]),
                            option_type="CALL",
                            direction="SHORT_CALL",
                            rationale=rationale,
                        )
                    )
                )
        return signals

    def _to_dataframe(self, snapshot: Any) -> pd.DataFrame:
        if isinstance(snapshot, pd.DataFrame):
            df = snapshot.copy()
        elif hasattr(snapshot, "to_pandas"):
            df = snapshot.to_pandas()
        else:
            df = pd.DataFrame(self._snapshot_options(snapshot))
        if df.empty:
            return df
        if "expiry" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["expiry"]):
            df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        symbol = self._snapshot_value(snapshot, "symbol")
        if "symbol" not in df.columns and symbol is not None:
            df["symbol"] = symbol
        underlying = self._snapshot_value(snapshot, "underlying_price")
        if "underlying_price" not in df.columns and underlying is not None:
            df["underlying_price"] = underlying
        return df
