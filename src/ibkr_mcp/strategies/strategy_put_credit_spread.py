"""Put credit spread strategy tuned for modest uptrends."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from market_state import MarketState, MarketStateProvider

from .base import BaseOptionStrategy, TradeSignal


class PutCreditSpreadStrategy(BaseOptionStrategy):
    """Constructs bull put spreads when the market trends higher but remains choppy."""

    def __init__(
        self,
        spread_width: float = 5.0,
        min_days_to_expiry: int = 18,
        max_days_to_expiry: int = 45,
        min_credit: float = 0.75,
        short_target_delta: float = 0.25,
        short_otm_pct: float = 0.05,
        market_state_provider: Optional[MarketStateProvider] = None,
        required_states: Optional[Sequence[MarketState]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.spread_width = float(spread_width)
        self.min_days_to_expiry = int(min_days_to_expiry)
        self.max_days_to_expiry = int(max_days_to_expiry)
        self.min_credit = float(min_credit)
        self.short_target_delta = abs(float(short_target_delta))
        self.short_otm_pct = float(short_otm_pct)
        self.market_state_provider = market_state_provider
        default_states: Tuple[MarketState, ...] = (MarketState.BULL, MarketState.UPTREND)
        self.required_states = tuple(required_states) if required_states else default_states

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        now = datetime.now(timezone.utc)
        signals: List[TradeSignal] = []
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty or "expiry" not in chain.columns:
                continue
            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None or underlying_price <= 0:
                continue
            symbol = str(chain["symbol"].iloc[0]) if "symbol" in chain.columns else str(self._snapshot_value(snapshot, "symbol"))
            state = self._get_market_state(symbol)
            if not self._state_allows(symbol, state):
                continue

            for expiry, subset in chain.groupby("expiry"):
                expiry_ts = self._normalize_expiry(expiry)
                if expiry_ts is None:
                    continue
                days_to_expiry = max((expiry_ts - now).days, 0)
                if days_to_expiry < self.min_days_to_expiry or days_to_expiry > self.max_days_to_expiry:
                    continue

                signal = self._build_credit_spread(
                    subset=subset,
                    symbol=symbol,
                    expiry=expiry_ts,
                    days_to_expiry=days_to_expiry,
                    underlying_price=underlying_price,
                    state=state,
                )
                if signal is not None:
                    signals.append(signal)

        return signals

    def _build_credit_spread(
        self,
        subset: pd.DataFrame,
        symbol: Optional[str],
        expiry: pd.Timestamp,
        days_to_expiry: int,
        underlying_price: float,
        state: Optional[MarketState],
    ) -> Optional[TradeSignal]:
        puts = subset.copy()
        if "option_type" not in puts.columns:
            return None
        puts["option_type"] = puts["option_type"].str.upper()
        puts = puts[puts["option_type"] == "PUT"]
        if puts.empty:
            return None

        short_put = self._select_short_put(puts, underlying_price)
        if short_put is None:
            return None
        long_put = self._select_long_put(puts, float(short_put["strike"]))
        if long_put is None:
            return None

        short_price = self._price_from_row(short_put, primary="bid")
        long_price = self._price_from_row(long_put, primary="ask")
        if short_price is None or long_price is None:
            return None
        net_credit = short_price - long_price
        if net_credit < self.min_credit:
            logger.info(
                "Skipping put credit spread due to low credit | symbol={symbol} credit={credit:.2f} min={minimum:.2f}",
                symbol=symbol,
                credit=net_credit,
                minimum=self.min_credit,
            )
            return None

        width = float(short_put["strike"]) - float(long_put["strike"])
        if width <= 0:
            return None
        max_loss = width - net_credit
        if max_loss <= 0:
            return None
        risk_reward = net_credit / max_loss if max_loss > 0 else 0.0

        rationale = (
            f"Bull put spread: short {short_put['strike']:.2f}P long {long_put['strike']:.2f}P "
            f"credit {net_credit:.2f} width {width:.2f} DTE {days_to_expiry} "
            f"state {(state.value if state else 'unknown')} R/R {risk_reward:.2f}"
        )

        return self.emit_signal(
            TradeSignal(
                symbol=str(symbol or short_put.get("symbol", "")),
                expiry=expiry,
                strike=float(short_put["strike"]),
                option_type="PUT",
                direction="BULL_PUT_CREDIT_SPREAD",
                rationale=rationale,
            )
        )

    def _select_short_put(self, puts: pd.DataFrame, underlying_price: float) -> Optional[pd.Series]:
        candidates = puts.copy()
        if "strike" not in candidates.columns:
            return None
        candidates["strike"] = pd.to_numeric(candidates["strike"], errors="coerce")
        candidates = candidates.dropna(subset=["strike"])
        if candidates.empty:
            return None
        otm_candidates = candidates[candidates["strike"] <= underlying_price]
        if otm_candidates.empty:
            return None
        preferred_floor = underlying_price * (1 - self.short_otm_pct)
        preferred = otm_candidates[otm_candidates["strike"] >= preferred_floor]
        target_frame = preferred if not preferred.empty else otm_candidates
        if "delta" in target_frame.columns and not target_frame["delta"].isna().all():
            deltas = target_frame["delta"].abs()
            idx = (deltas - self.short_target_delta).abs().argsort()
            return target_frame.iloc[idx[:1]].iloc[0]
        target_frame = target_frame.sort_values("strike", ascending=False)
        return target_frame.iloc[0]

    def _select_long_put(self, puts: pd.DataFrame, short_strike: float) -> Optional[pd.Series]:
        candidates = puts.copy()
        candidates["strike"] = pd.to_numeric(candidates["strike"], errors="coerce")
        candidates = candidates.dropna(subset=["strike"])
        if candidates.empty:
            return None
        otm = candidates[candidates["strike"] <= short_strike - self.spread_width]
        if otm.empty:
            return None
        otm = otm.sort_values("strike", ascending=False)
        return otm.iloc[0]

    def _get_market_state(self, symbol: Optional[str]) -> Optional[MarketState]:
        if not symbol or not self.market_state_provider:
            return None
        try:
            state = self.market_state_provider.get_state(symbol)
            logger.info(
                "Market state fetched for put credit spread | symbol={symbol} state={state}",
                symbol=symbol,
                state=state.value if state else "unknown",
            )
            return state
        except Exception:
            logger.exception("Failed to obtain market state | symbol={symbol}", symbol=symbol)
            return None

    def _state_allows(self, symbol: Optional[str], state: Optional[MarketState]) -> bool:
        if state is None or not self.required_states:
            return True
        if state in self.required_states:
            logger.info(
                "Market state filter passed for put credit spread | symbol={symbol} state={state}",
                symbol=symbol,
                state=state.value,
            )
            return True
        required = ",".join(item.value for item in self.required_states)
        logger.info(
            "Skipping put credit spread due to market state | symbol={symbol} state={state} required={required}",
            symbol=symbol,
            state=state.value if state else "unknown",
            required=required,
        )
        return False

    def _normalize_expiry(self, value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        expiry = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
        if getattr(expiry, "tzinfo", None) is None:
            expiry = expiry.tz_localize(timezone.utc)
        else:
            expiry = expiry.tz_convert(timezone.utc)
        return expiry

    @staticmethod
    def _price_from_row(
        row: pd.Series,
        primary: str,
        fallbacks: Tuple[str, ...] = ("mark", "mid", "last"),
    ) -> Optional[float]:
        for field in (primary, *fallbacks):
            if field not in row:
                continue
            value = row.get(field)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return None

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


__all__ = ["PutCreditSpreadStrategy"]
