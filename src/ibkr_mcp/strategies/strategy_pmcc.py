"""Poor Man's Covered Call (PMCC) strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from loguru import logger

from .base import BaseOptionStrategy, TradeSignal


class PoorMansCoveredCallStrategy(BaseOptionStrategy):
    """Identify Poor Man's Covered Call (PMCC) opportunities."""

    def __init__(
        self,
        leaps_min_days: int = 240,
        leaps_delta_threshold: float = 0.7,
        leaps_max_strike_pct: float = 0.9,
        max_leaps_extrinsic_pct: float = 0.35,
        leaps_max_theta_abs: float = 0.04,
        short_min_days: int = 21,
        short_max_days: int = 60,
        short_otm_pct: float = 0.05,
        short_delta_range: Tuple[float, float] = (0.2, 0.45),
        short_min_theta_abs: float = 0.04,
        min_return_on_capital: float = 0.08, # minimum return on capital for PMCC trade ideas, it could be lower to 0.08-0.12
        max_trade_ideas: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.leaps_min_days = leaps_min_days
        self.leaps_delta_threshold = leaps_delta_threshold
        self.leaps_max_strike_pct = leaps_max_strike_pct
        self.max_leaps_extrinsic_pct = max_leaps_extrinsic_pct
        self.leaps_max_theta_abs = abs(leaps_max_theta_abs)
        self.short_min_days = short_min_days
        self.short_max_days = short_max_days
        self.short_otm_pct = short_otm_pct
        self.short_delta_range = tuple(sorted(abs(x) for x in short_delta_range))
        self.short_min_theta_abs = abs(short_min_theta_abs)
        self.min_return_on_capital = min_return_on_capital
        self.max_trade_ideas = max_trade_ideas

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        now = pd.Timestamp(datetime.now(timezone.utc))
        signals: List[TradeSignal] = []

        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                continue

            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None or underlying_price <= 0:
                continue

            prepared = chain.copy()
            required_cols = {"option_type", "expiry", "strike"}
            if not required_cols.issubset(prepared.columns):
                continue
            prepared["option_type"] = prepared["option_type"].str.upper()
            prepared["expiry"] = pd.to_datetime(prepared["expiry"], utc=True)
            prepared["days_to_expiry"] = (prepared["expiry"] - now).dt.days
            for greek in ("delta", "theta", "implied_volatility"):
                if greek in prepared.columns:
                    prepared[greek] = pd.to_numeric(prepared[greek], errors="coerce")
            for price_col in ("bid", "ask", "mark", "last"):
                if price_col in prepared.columns:
                    prepared[price_col] = pd.to_numeric(prepared[price_col], errors="coerce")
            prepared = prepared.fillna({"bid": 0.0, "ask": 0.0, "mark": 0.0, "last": 0.0})

            calls = prepared[prepared["option_type"] == "CALL"].copy()
            if calls.empty:
                continue

            leaps_candidates = self._filter_leaps(calls, underlying_price)
            if leaps_candidates.empty:
                continue

            short_candidates = self._filter_short_calls(calls, underlying_price)
            if short_candidates.empty:
                continue

            trade_candidates: List[Dict[str, Any]] = []
            for _, leaps in leaps_candidates.iterrows():
                combos = self._evaluate_short_candidates(leaps, short_candidates, underlying_price)
                trade_candidates.extend(combos)

            if not trade_candidates:
                continue

            ranked_candidates = self._rank_candidates(trade_candidates)
            self._log_ranked_ideas(ranked_candidates)

            for candidate in ranked_candidates[: self.max_trade_ideas]:
                signals.extend(self._emit_pmcc_signals(candidate))
        return signals

    def _filter_leaps(self, calls: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
        leaps = calls[calls["days_to_expiry"] >= self.leaps_min_days].copy()
        if leaps.empty:
            return leaps
        leaps = leaps[leaps["strike"] <= underlying_price * self.leaps_max_strike_pct]
        if leaps.empty:
            return leaps
        if "delta" in leaps.columns:
            delta_series = leaps["delta"].fillna(0.0).abs()
            leaps = leaps[delta_series >= self.leaps_delta_threshold]
        if "theta" in leaps.columns:
            theta_abs = leaps["theta"].abs().fillna(self.leaps_max_theta_abs + 1.0)
            leaps = leaps[theta_abs <= self.leaps_max_theta_abs]
        leaps["price"] = leaps.apply(self._option_price_long, axis=1)
        leaps = leaps[leaps["price"] > 0]
        if leaps.empty:
            return leaps
        intrinsic = (underlying_price - leaps["strike"]).clip(lower=0.0)
        extrinsic = leaps["price"] - intrinsic
        extrinsic_pct = extrinsic / underlying_price
        leaps = leaps[extrinsic_pct <= self.max_leaps_extrinsic_pct]
        leaps = leaps.assign(intrinsic=intrinsic, extrinsic=extrinsic)
        return leaps

    def _filter_short_calls(self, calls: pd.DataFrame, underlying_price: float) -> pd.DataFrame:
        shorts = calls[
            (calls["days_to_expiry"] >= self.short_min_days)
            & (calls["days_to_expiry"] <= self.short_max_days)
            & (calls["strike"] >= underlying_price * (1 + self.short_otm_pct))
        ].copy()
        if shorts.empty:
            return shorts
        delta_low, delta_high = self.short_delta_range
        if "delta" in shorts.columns:
            delta_abs = shorts["delta"].abs().fillna(0.0)
            shorts = shorts[(delta_abs >= delta_low) & (delta_abs <= delta_high)]
        if "theta" in shorts.columns:
            theta_abs = shorts["theta"].abs().fillna(0.0)
            shorts = shorts[theta_abs >= self.short_min_theta_abs]
        shorts["price"] = shorts.apply(self._option_price_short, axis=1)
        shorts = shorts[shorts["price"] > 0]
        return shorts

    def _evaluate_short_candidates(
        self,
        leaps: pd.Series,
        short_candidates: pd.DataFrame,
        underlying_price: float,
    ) -> List[Dict[str, Any]]:
        eligible = short_candidates[
            (short_candidates["expiry"] < leaps["expiry"])
            & (short_candidates["strike"] >= leaps["strike"])
        ]
        if eligible.empty:
            return []

        leaps_price = float(leaps.get("price", 0.0))
        leaps_theta = float(leaps.get("theta", 0.0) or 0.0)
        leaps_delta = float(leaps.get("delta", 0.0) or 0.0)
        leaps_days = int(leaps.get("days_to_expiry", 0) or 0)
        symbol = str(leaps.get("symbol", ""))
        candidates: List[Dict[str, Any]] = []
        for _, short in eligible.iterrows():
            credit = float(short.get("price", 0.0))
            if credit <= 0:
                continue
            net_debit = leaps_price - credit
            if net_debit <= 0:
                continue
            roc = credit / net_debit
            if roc < self.min_return_on_capital:
                continue
            short_theta = float(short.get("theta", 0.0) or 0.0)
            short_delta = float(short.get("delta", 0.0) or 0.0)
            short_days = int(short.get("days_to_expiry", 0) or 0)
            annualized_roi = roc
            if short_days > 0:
                annualized_roi = roc * (365 / short_days)
            score = self._compute_score(
                roc=roc,
                leaps_theta=leaps_theta,
                short_theta=short_theta,
                leaps_delta=leaps_delta,
                short_delta=short_delta,
            )
            candidates.append(
                {
                    "symbol": symbol or str(short.get("symbol", "")),
                    "leaps": leaps,
                    "short": short,
                    "net_debit": net_debit,
                    "credit": credit,
                    "roc": roc,
                    "annualized_roi": annualized_roi,
                    "score": score,
                    "underlying_price": underlying_price,
                    "leaps_theta": leaps_theta,
                    "short_theta": short_theta,
                    "leaps_delta": leaps_delta,
                    "short_delta": short_delta,
                    "leaps_days": leaps_days,
                    "short_days": short_days,
                }
            )
        return candidates

    @staticmethod
    def _compute_score(
        roc: float,
        leaps_theta: float,
        short_theta: float,
        leaps_delta: float,
        short_delta: float,
    ) -> float:
        theta_score = (1 + abs(short_theta)) / (1 + abs(leaps_theta))
        delta_alignment = max(0.0, abs(leaps_delta) - abs(short_delta))
        return roc * theta_score * (1 + delta_alignment)

    @staticmethod
    def _rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda item: (item.get("score", 0.0), item.get("roc", 0.0)),
            reverse=True,
        )

    def _log_ranked_ideas(self, candidates: List[Dict[str, Any]]) -> None:
        for rank, candidate in enumerate(candidates[: self.max_trade_ideas], start=1):
            leaps = candidate["leaps"]
            short = candidate["short"]
            logger.info(
                "PMCC idea rank={rank} symbol={symbol} net_debit={net:.2f} credit={credit:.2f} "
                "ROI={roi:.2%} annualized={annual:.2%} underlying={underlying:.2f} "
                "LEAPS expiry={leaps_exp:%Y-%m-%d} strike={leaps_strike:.2f} Δ={leaps_delta:.2f} Θ={leaps_theta:.2f} "
                "SHORT expiry={short_exp:%Y-%m-%d} strike={short_strike:.2f} Δ={short_delta:.2f} Θ={short_theta:.2f}",
                rank=rank,
                symbol=candidate["symbol"],
                net=candidate["net_debit"],
                credit=candidate["credit"],
                roi=candidate["roc"],
                annual=candidate["annualized_roi"],
                underlying=candidate["underlying_price"],
                leaps_exp=pd.Timestamp(leaps["expiry"]).to_pydatetime(),
                leaps_strike=float(leaps["strike"]),
                leaps_delta=float(candidate["leaps_delta"]),
                leaps_theta=float(candidate["leaps_theta"]),
                short_exp=pd.Timestamp(short["expiry"]).to_pydatetime(),
                short_strike=float(short["strike"]),
                short_delta=float(candidate["short_delta"]),
                short_theta=float(candidate["short_theta"]),
            )

    def _emit_pmcc_signals(self, candidate: Dict[str, Any]) -> List[TradeSignal]:
        leaps = candidate["leaps"]
        short = candidate["short"]
        leaps_expiry = pd.Timestamp(leaps["expiry"]).to_pydatetime()
        short_expiry = pd.Timestamp(short["expiry"]).to_pydatetime()
        symbol = candidate["symbol"]
        rationale = (
            "PMCC idea | net_debit={net:.2f} credit={credit:.2f} ROI={roi:.2%} "
            "annualized={annual:.2%} underlying={underlying:.2f} "
            "LEAPS Δ={leaps_delta:.2f} Θ={leaps_theta:.2f} "
            "SHORT Δ={short_delta:.2f} Θ={short_theta:.2f}"
        ).format(
            net=candidate["net_debit"],
            credit=candidate["credit"],
            roi=candidate["roc"],
            annual=candidate["annualized_roi"],
            underlying=candidate["underlying_price"],
            leaps_delta=candidate["leaps_delta"],
            leaps_theta=candidate["leaps_theta"],
            short_delta=candidate["short_delta"],
            short_theta=candidate["short_theta"],
        )

        signals = [
            self.emit_signal(
                TradeSignal(
                    symbol=symbol,
                    expiry=leaps_expiry,
                    strike=float(leaps["strike"]),
                    option_type="CALL",
                    direction="LONG_PMCC_LEAPS",
                    rationale=rationale,
                )
            ),
            self.emit_signal(
                TradeSignal(
                    symbol=symbol,
                    expiry=short_expiry,
                    strike=float(short["strike"]),
                    option_type="CALL",
                    direction="SHORT_PMCC_CALL",
                    rationale=rationale,
                )
            ),
        ]
        return signals

    @staticmethod
    def _option_price_long(row: pd.Series) -> float:
        return PoorMansCoveredCallStrategy._option_price(row, prefer="ask")

    @staticmethod
    def _option_price_short(row: pd.Series) -> float:
        return PoorMansCoveredCallStrategy._option_price(row, prefer="bid")

    @staticmethod
    def _option_price(row: pd.Series, prefer: str = "bid") -> float:
        if prefer == "ask":
            priority = ("ask", "mark", "mid", "last", "bid")
        else:
            priority = ("bid", "mark", "mid", "last", "ask")
        for key in priority:
            if key in row:
                value = row.get(key, 0.0)
                if pd.notna(value) and float(value) > 0:
                    return float(value)
        return 0.0

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


__all__ = ["PoorMansCoveredCallStrategy"]
