"""Iron condor strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from .base import BaseOptionStrategy, TradeSignal


class IronCondorStrategy(BaseOptionStrategy):
    """Identifies neutral volatility setups for iron condors."""

    def __init__(
        self,
        target_delta: float = 0.16,  # absolute delta target per short strike
        premium_threshold: float = 1.0,  # minimum absolute credit (per condor) worth trading, in dollars
        min_credit_pct: float = 0.01,  # minimum credit as % of underlying (guards against low premium vs. risk)
        max_expiries_per_symbol: int = 3,  # cap how many expirations per symbol we evaluate each run
        delta_tolerance: float = 0.05,  # acceptable +/- range when searching for target delta strikes
        spread_width: float = 10.0,  # distance between short and long legs on each side (in dollars)
        allowed_symbols: Optional[Iterable[str]] = None,  # whitelist underlyings (defaults to SPY/QQQ/IWM)
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_delta = float(abs(target_delta))
        self.premium_threshold = premium_threshold
        self.min_credit_pct = min_credit_pct
        self.max_expiries_per_symbol = max_expiries_per_symbol
        self.delta_tolerance = max(delta_tolerance, 0.0)
        self.spread_width = max(spread_width, 0.01)
        default_symbols: Set[str] = {"SPY", "QQQ", "IWM", "NVDA", "AMD", "AAPL", "MSFT", "TSLA"}
        self.allowed_symbols = {symbol.upper() for symbol in (allowed_symbols or default_symbols)}

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for snapshot in data:
            chain = self._to_dataframe(snapshot)
            if chain.empty:
                logger.debug("IronCondor skip snapshot: empty chain")
                continue
            symbol = str(chain["symbol"].iloc[0]).upper()
            if symbol not in self.allowed_symbols:
                logger.debug("IronCondor skip snapshot: symbol not allowed | symbol={symbol}", symbol=symbol)
                continue
            underlying_price = self._resolve_underlying_price(snapshot, chain)
            if underlying_price is None or underlying_price <= 0:
                logger.debug(
                    "IronCondor skip snapshot: invalid underlying | symbol={symbol} price={price}",
                    symbol=symbol,
                    price=underlying_price,
                )
                continue
            chain = chain.sort_values("strike")
            expiry_groups = chain.groupby("expiry")
            processed_expiries = 0
            for expiry, subset in expiry_groups:
                if (expiry - datetime.now(timezone.utc)).days < 21:
                    logger.debug(
                        "IronCondor skip expiry: too few days to expiry | symbol={symbol} expiry={expiry}",
                        symbol=symbol,
                        expiry=expiry.isoformat(),
                    )
                    continue
                if self.max_expiries_per_symbol and processed_expiries >= self.max_expiries_per_symbol:
                    break
                calls = subset[subset["option_type"] == "CALL"]
                puts = subset[subset["option_type"] == "PUT"]
                if calls.empty or puts.empty:
                    logger.debug(
                        "IronCondor skip expiry: missing option side | symbol={symbol} expiry={expiry} calls_empty={calls} puts_empty={puts}",
                        symbol=symbol,
                        expiry=expiry.isoformat(),
                        calls=calls.empty,
                        puts=puts.empty,
                    )
                    continue
                condor, total_credit, max_loss = self._build_condor(
                    subset["symbol"].iloc[0],
                    expiry,
                    calls,
                    puts,
                    underlying_price,
                )
                if condor is None:
                    continue
                if total_credit <= 0.0 or max_loss <= 0.0:
                    logger.debug(
                        "IronCondor skip condor: invalid payoff | symbol={symbol} expiry={expiry} total_credit={credit} max_loss={loss}",
                        symbol=symbol,
                        expiry=expiry.isoformat(),
                        credit=total_credit,
                        loss=max_loss,
                    )
                    continue
                symbol = condor["symbol"]
                denominator = self.spread_width - total_credit
                if denominator <= 0:
                    logger.debug(
                        "IronCondor skip condor: invalid denominator | symbol={symbol} expiry={expiry} total_credit={credit} spread_width={width}",
                        symbol=symbol,
                        expiry=expiry.isoformat(),
                        credit=total_credit,
                        width=self.spread_width,
                    )
                    continue
                ror = total_credit / denominator
                current_best = best_by_symbol.get(symbol)
                if current_best is None or ror > current_best.get("ror", 0.0):
                    best_by_symbol[symbol] = {"condor": condor, "ror": ror}
                processed_expiries += 1
        signals: List[TradeSignal] = []
        for entry in best_by_symbol.values():
            condor = entry["condor"]
            rationale = condor["rationale"]
            expiry = condor["expiry"]
            symbol = condor["symbol"]

            def add_signal(option_row: pd.Series, direction: str) -> None:
                signals.append(
                    self.emit_signal(
                        TradeSignal(
                            symbol=symbol,
                            expiry=expiry,
                            strike=float(option_row["strike"]),
                            option_type=str(option_row["option_type"]),
                            direction=direction,
                            rationale=rationale,
                        )
                    )
                )

            add_signal(condor["short_call"], "SHORT_CONDOR_CALL")
            add_signal(condor["long_call"], "LONG_CONDOR_CALL")
            add_signal(condor["short_put"], "SHORT_CONDOR_PUT")
            add_signal(condor["long_put"], "LONG_CONDOR_PUT")
        return signals

    def _build_condor(
        self,
        symbol: str,
        expiry: pd.Timestamp,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        underlying_price: float,
    ) -> Tuple[Optional[Dict[str, Any]], float, float]:
        calls = calls.sort_values("strike").reset_index(drop=True).copy()
        puts = puts.sort_values("strike").reset_index(drop=True).copy()

        if len(calls) < 2 or len(puts) < 2:
            self._log_rejection("insufficient_legs", symbol, expiry, calls=len(calls), puts=len(puts))
            return None, 0.0, 0.0

        for frame in (calls, puts):
            frame["delta"] = pd.to_numeric(frame.get("delta", pd.Series(index=frame.index)), errors="coerce")

        put_delta_target = -self.target_delta
        put_min = put_delta_target - self.delta_tolerance
        put_max = put_delta_target + self.delta_tolerance
        put_candidates = puts[puts["delta"].between(put_min, put_max)]
        if put_candidates.empty:
            self._log_rejection(
                "no_put_candidate",
                symbol,
                expiry,
                delta_target=put_delta_target,
                tolerance=self.delta_tolerance,
            )
            return None, 0.0, 0.0
        short_put = puts.loc[(put_candidates["delta"] - put_delta_target).abs().idxmin()]

        call_delta_target = self.target_delta
        call_min = call_delta_target - self.delta_tolerance
        call_max = call_delta_target + self.delta_tolerance
        call_candidates = calls[calls["delta"].between(call_min, call_max)]
        if call_candidates.empty:
            self._log_rejection(
                "no_call_candidate",
                symbol,
                expiry,
                delta_target=call_delta_target,
                tolerance=self.delta_tolerance,
            )
            return None, 0.0, 0.0
        short_call = calls.loc[(call_candidates["delta"] - call_delta_target).abs().idxmin()]

        long_put_strike = float(short_put["strike"]) - self.spread_width
        long_call_strike = float(short_call["strike"]) + self.spread_width

        long_put_matches = puts[abs(puts["strike"] - long_put_strike) < 1e-6]
        if long_put_matches.empty:
            self._log_rejection(
                "missing_long_put",
                symbol,
                expiry,
                target_strike=long_put_strike,
            )
            return None, 0.0, 0.0
        long_put = long_put_matches.iloc[0]

        long_call_matches = calls[abs(calls["strike"] - long_call_strike) < 1e-6]
        if long_call_matches.empty:
            self._log_rejection(
                "missing_long_call",
                symbol,
                expiry,
                target_strike=long_call_strike,
            )
            return None, 0.0, 0.0
        long_call = long_call_matches.iloc[0]

        if float(short_put["strike"]) >= float(short_call["strike"]):
            self._log_rejection(
                "invalid_strike_order",
                symbol,
                expiry,
                short_put=float(short_put["strike"]),
                short_call=float(short_call["strike"]),
            )
            return None, 0.0, 0.0

        credit_call = self._price(short_call, "bid") - self._price(long_call, "ask")
        credit_put = self._price(short_put, "bid") - self._price(long_put, "ask")
        total_credit = credit_call + credit_put

        if total_credit < self.premium_threshold:
            self._log_rejection(
                "credit_below_threshold",
                symbol,
                expiry,
                credit=total_credit,
                threshold=self.premium_threshold,
            )
            return None, 0.0, 0.0
        credit_pct = total_credit / underlying_price
        if credit_pct < self.min_credit_pct:
            self._log_rejection(
                "credit_pct_below_min",
                symbol,
                expiry,
                credit_pct=credit_pct,
                min_pct=self.min_credit_pct,
            )
            return None, 0.0, 0.0

        rationale = (
            f"Iron condor credit {total_credit:.2f} ({credit_pct:.2%}) | "
            f"target delta ≈{self.target_delta:.2f} | "
            f"call delta {short_call.get('delta', 0):.2f} / put delta {short_put.get('delta', 0):.2f} | "
            f"{expiry.date()} strikes {short_put['strike']}-{long_put['strike']} / "
            f"{short_call['strike']}-{long_call['strike']}"
        )

        condor_payload = {
            "symbol": symbol,
            "expiry": expiry,
            "short_call": short_call,
            "long_call": long_call,
            "short_put": short_put,
            "long_put": long_put,
            "total_credit": float(total_credit),
            "rationale": rationale,
        }
        max_loss = (self.spread_width * 100) - (total_credit * 100)
        return condor_payload, float(total_credit), float(max_loss)

    def _price(self, row: pd.Series, preference: str) -> float:
        fields = ("bid", "ask") if preference == "bid" else ("ask", "bid")
        candidates = list(fields) + ["mark", "price"]
        for field in candidates:
            if field in row and pd.notna(row[field]):
                value = float(row[field])
                if value > 0:
                    return value
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
        return df

    def _log_rejection(
        self,
        reason: str,
        symbol: Optional[str],
        expiry: Optional[pd.Timestamp],
        **metrics: Any,
    ) -> None:
        logger.debug(
            "IronCondor filter | reason={reason} symbol={symbol} expiry={expiry} context={context}",
            reason=reason,
            symbol=symbol,
            expiry=expiry.isoformat() if hasattr(expiry, "isoformat") else expiry,
            context=metrics or None,
        )
