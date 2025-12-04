"""Vertical spread option strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from loguru import logger

from market_state import MarketState, MarketStateProvider

from .base import BaseOptionStrategy, TradeSignal


class VerticalSpreadStrategy(BaseOptionStrategy):
    """Constructs bullish or bearish vertical spreads with directional and risk filters."""

    def __init__(
        self,
        spread_width: float = 5.0,
        min_days_to_expiry: int = 14,
        max_days_to_expiry: int = 45,
        target_days_to_expiry: Optional[int] = None,
        max_iv_rank: float = 0.30,
        min_risk_reward_ratio: float = 1.0,
        bullish_ma_field: Optional[str] = "moving_average_30",
        bearish_iv_rank_trigger: float = 0.50,
        market_state_provider: Optional[MarketStateProvider] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spread_width = spread_width
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.preferred_days_to_expiry = target_days_to_expiry or max_days_to_expiry
        self.max_iv_rank = max_iv_rank
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.bullish_ma_fields = self._ma_field_candidates(bullish_ma_field)
        self.bearish_iv_rank_trigger = bearish_iv_rank_trigger
        self.market_state_provider = market_state_provider

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        now = datetime.now(timezone.utc)
        signals: List[TradeSignal] = []
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None:
                continue

            iv_rank = self._extract_iv_rank(snapshot, chain)
            if iv_rank is not None and iv_rank > self.max_iv_rank:
                symbol = chain["symbol"].iloc[0] if "symbol" in chain.columns else "UNKNOWN"
                logger.info(
                    "Skipping symbol due to high IV rank | symbol={symbol} iv_rank={iv_rank:.2%} max={max_iv_rank:.2%}",
                    symbol=symbol,
                    iv_rank=iv_rank,
                    max_iv_rank=self.max_iv_rank,
                )
                continue

            expiry = self._select_expiry(chain["expiry"].dropna().unique(), now)
            if expiry is None:
                continue
            subset = chain[chain["expiry"] == expiry]
            if subset.empty:
                continue

            symbol = str(subset["symbol"].iloc[0]) if "symbol" in subset.columns else None
            days_to_expiry = max((expiry - now).days, 1)
            state = self._get_market_state(symbol)
            ma_value, ma_label = self._resolve_moving_average(snapshot, chain)
            directional = self._evaluate_directional_bias(
                underlying_price, iv_rank, ma_value
            )

            logger.info(
                "Evaluating spreads | symbol={symbol} expiry={expiry} DTE={dte} state={state} iv_rank={iv_rank} MA={ma_label}:{ma_value}",
                symbol=symbol,
                expiry=expiry.date(),
                dte=days_to_expiry,
                state=state.value if state else "unknown",
                iv_rank=f"{iv_rank:.2%}" if iv_rank is not None else "n/a",
                ma_label=ma_label or "n/a",
                ma_value=f"{ma_value:.2f}" if ma_value is not None else "n/a",
            )

            call_subset = subset[subset["option_type"] == "CALL"].sort_values("strike")
            put_subset = subset[subset["option_type"] == "PUT"].sort_values("strike")

            if (
                directional.is_bullish
                and not call_subset.empty
                and self._state_allows(symbol, state, (MarketState.BULL, MarketState.UPTREND))
            ):
                atm_call = self._find_atm_option(call_subset, underlying_price)
                if atm_call is not None:
                    signal = self._build_debit_spread_signal(
                        subset=call_subset,
                        atm_leg=atm_call,
                        select_higher=True,
                        symbol=symbol,
                        expiry=expiry,
                        direction="BULL_CALL_DEBIT_SPREAD",
                        option_type="CALL",
                        underlying_price=underlying_price,
                        days_to_expiry=days_to_expiry,
                        ma_value=ma_value,
                        ma_label=ma_label,
                    )
                    if signal is not None:
                        signals.append(signal)

            if (
                directional.is_bearish
                and not put_subset.empty
                and self._state_allows(symbol, state, MarketState.BEAR)
            ):
                atm_put = self._find_atm_option(put_subset, underlying_price)
                if atm_put is not None:
                    signal = self._build_debit_spread_signal(
                        subset=put_subset,
                        atm_leg=atm_put,
                        select_higher=False,
                        symbol=symbol,
                        expiry=expiry,
                        direction="BEAR_PUT_DEBIT_SPREAD",
                        option_type="PUT",
                        underlying_price=underlying_price,
                        days_to_expiry=days_to_expiry,
                        ma_value=ma_value,
                        ma_label=ma_label,
                    )
                    if signal is not None:
                        signals.append(signal)

        return signals

    def _build_debit_spread_signal(
        self,
        subset: pd.DataFrame,
        atm_leg: pd.Series,
        select_higher: bool,
        symbol: Optional[str],
        expiry: pd.Timestamp,
        direction: str,
        option_type: str,
        underlying_price: float,
        days_to_expiry: int,
        ma_value: Optional[float],
        ma_label: Optional[str],
    ) -> Optional[TradeSignal]:
        if select_higher:
            otm_subset = subset[subset["strike"] >= atm_leg["strike"] + self.spread_width].sort_values("strike")
        else:
            otm_subset = subset[subset["strike"] <= atm_leg["strike"] - self.spread_width].sort_values(
                "strike", ascending=False
            )
        if otm_subset.empty:
            return None
        short_leg = otm_subset.iloc[0]
        metrics = self._calculate_spread_metrics(atm_leg, short_leg)
        if metrics is None:
            return None
        net_debit, max_profit, max_loss = metrics
        if max_loss <= 0:
            return None
        risk_reward = max_profit / max_loss
        if risk_reward < self.min_risk_reward_ratio:
            logger.info(
                "Skipping spread due to low R/R | symbol={symbol} direction={direction} R/R={ratio:.2f} min={min_ratio:.2f}",
                symbol=symbol,
                direction=direction,
                ratio=risk_reward,
                min_ratio=self.min_risk_reward_ratio,
            )
            return None

        context_parts = [
            f"DTE {days_to_expiry}",
            f"net debit {net_debit:.2f}",
            f"R/R {risk_reward:.2f}",
        ]
        if ma_value is not None and ma_label:
            relation = ">" if underlying_price > ma_value else "<"
            context_parts.append(f"price {relation} {ma_label} {ma_value:.2f}")
        rationale = (
            f"{direction.replace('_', ' ').title()}: long {atm_leg['strike']:.2f}{option_type[0]} "
            f"short {short_leg['strike']:.2f}{option_type[0]}, "
            + ", ".join(context_parts)
        )

        return self.emit_signal(
            TradeSignal(
                symbol=str(symbol or subset['symbol'].iloc[0]),
                expiry=expiry,
                strike=float(short_leg["strike"]),
                option_type=option_type,
                direction=direction,
                rationale=rationale,
            )
        )

    def _select_expiry(self, expiries: Sequence[pd.Timestamp], now: datetime) -> Optional[pd.Timestamp]:
        candidate = None
        best_distance = None
        for expiry in expiries:
            expiry_ts = pd.Timestamp(expiry, tz="UTC") if not isinstance(expiry, pd.Timestamp) else expiry
            if getattr(expiry_ts, "tzinfo", None) is None:
                expiry_ts = expiry_ts.tz_localize(timezone.utc)
            days = (expiry_ts - now).days
            if days < self.min_days_to_expiry or days > self.max_days_to_expiry:
                continue
            distance = abs(days - self.preferred_days_to_expiry)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                candidate = expiry_ts
        return candidate

    def _get_market_state(self, symbol: Optional[str]) -> Optional[MarketState]:
        if not symbol or not self.market_state_provider:
            return None
        try:
            state = self.market_state_provider.get_state(symbol)
            logger.info(
                "Market state fetched | symbol={symbol} state={state}",
                symbol=symbol,
                state=state.value if state else "unknown",
            )
            return state
        except Exception:
            logger.exception("Failed to obtain market state | symbol={symbol}", symbol=symbol)
            return None

    @staticmethod
    def _state_allows(
        symbol: Optional[str],
        state: Optional[MarketState],
        required: Union[MarketState, Iterable[MarketState]],
    ) -> bool:
        if isinstance(required, MarketState):
            allowed: Tuple[MarketState, ...] = (required,)
        else:
            allowed = tuple(required)
        if not allowed:
            return True
        if state is None:
            return True
        if state in allowed:
            allowed_values = ",".join(item.value for item in allowed)
            logger.info(
                "Market state filter passed | symbol={symbol} state={state} allowed={allowed}",
                symbol=symbol,
                state=state.value,
                allowed=allowed_values,
            )
            return True
        required_str = ",".join(item.value for item in allowed)
        logger.info(
            "Skipping spread due to market state | symbol={symbol} state={state} required={required}",
            symbol=symbol,
            state=state.value,
            required=required_str,
        )
        return False

    def _evaluate_directional_bias(
        self,
        underlying_price: float,
        iv_rank: Optional[float],
        ma_value: Optional[float],
    ) -> "DirectionalBias":
        is_bullish = ma_value is None or underlying_price > ma_value
        is_bearish = ma_value is not None and underlying_price < ma_value
        if iv_rank is not None and iv_rank >= self.bearish_iv_rank_trigger:
            is_bearish = True
        return DirectionalBias(is_bullish=is_bullish, is_bearish=is_bearish)

    def _resolve_moving_average(self, snapshot: Any, chain: pd.DataFrame) -> Tuple[Optional[float], Optional[str]]:
        for field in self.bullish_ma_fields:
            value = self._safe_float(self._snapshot_value(snapshot, field))
            if value is not None:
                return value, field
            if field in chain.columns:
                series = chain[field].dropna()
                if not series.empty:
                    return self._safe_float(series.iloc[0]), field
        return None, None

    def _extract_iv_rank(self, snapshot: Any, chain: pd.DataFrame) -> Optional[float]:
        value = self._safe_float(self._snapshot_value(snapshot, "iv_rank"))
        if value is not None:
            return value
        if "iv_rank" in chain.columns:
            series = chain["iv_rank"].dropna()
            if not series.empty:
                return self._safe_float(series.iloc[0])
        return None

    @staticmethod
    def _ma_field_candidates(primary: Optional[str]) -> Tuple[str, ...]:
        defaults = (
            "moving_average_30",
            "ma30",
            "ma_30",
            "sma30",
            "sma_30",
            "moving_average_50",
            "ma50",
            "ma_50",
            "sma50",
            "sma_50",
        )
        if not primary:
            return defaults
        ordered = [primary]
        ordered.extend([field for field in defaults if field != primary])
        return tuple(ordered)

    @staticmethod
    def _find_atm_option(options: pd.DataFrame, underlying_price: float) -> Optional[pd.Series]:
        if options.empty:
            return None
        idx = (options["strike"] - underlying_price).abs().argsort()
        return options.iloc[idx[:1]].iloc[0]

    def _calculate_spread_metrics(
        self,
        long_leg: pd.Series,
        short_leg: pd.Series,
    ) -> Optional[Tuple[float, float, float]]:
        long_price = self._price_from_row(long_leg, primary="ask")
        short_price = self._price_from_row(short_leg, primary="bid")
        if long_price is None or short_price is None:
            return None
        net_debit = long_price - short_price
        if net_debit <= 0:
            return None
        long_strike = self._safe_float(long_leg.get("strike"))
        short_strike = self._safe_float(short_leg.get("strike"))
        if long_strike is None or short_strike is None:
            return None
        strike_diff = abs(short_strike - long_strike)
        if strike_diff <= 0:
            return None
        max_profit = strike_diff - net_debit
        if max_profit <= 0:
            return None
        max_loss = net_debit
        return net_debit, max_profit, max_loss

    def _price_from_row(
        self,
        row: pd.Series,
        primary: str,
        fallbacks: Tuple[str, ...] = ("mark", "last"),
    ) -> Optional[float]:
        for field in (primary, *fallbacks):
            value = self._safe_float(row.get(field))
            if value is not None and value > 0:
                return value
        return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
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


class DirectionalBias:
    __slots__ = ("is_bullish", "is_bearish")

    def __init__(self, is_bullish: bool, is_bearish: bool) -> None:
        self.is_bullish = is_bullish
        self.is_bearish = is_bearish
