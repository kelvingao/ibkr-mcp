"""Greek calculation utilities for the portfolio manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from ib_async import IB, Option
from loguru import logger


GREEK_COLUMNS = ["delta", "gamma", "vega", "theta", "rho"]


@dataclass(slots=True)
class GreekSummary:
    """Container for per-symbol and portfolio-wide Greeks."""

    per_symbol: pd.DataFrame
    totals: Dict[str, float]


class GreekCalculator:
    """Computes or fetches greeks for the loaded positions."""

    def __init__(self, ib: Optional[IB]) -> None:
        self._ib = ib

    def compute(self, positions: pd.DataFrame) -> GreekSummary:
        if positions.empty:
            per_symbol = pd.DataFrame(columns=["underlying", *GREEK_COLUMNS])
            totals = {column: 0.0 for column in GREEK_COLUMNS}
            return GreekSummary(per_symbol=per_symbol, totals=totals)

        enriched = positions.copy()
        missing_columns = [column for column in GREEK_COLUMNS if column not in enriched.columns]
        if missing_columns:
            for column in missing_columns:
                enriched[column] = 0.0
            self._populate_greeks_from_ib(enriched)

        for column in GREEK_COLUMNS:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
            enriched[column] *= enriched.get("quantity", 0.0).astype(float)
            multiplier = (
                enriched.get("multiplier", 1.0)
                .astype(float)
                .replace(0.0, 1.0)
                .fillna(1.0)
            )
            enriched[column] *= multiplier

        per_symbol = (
            enriched.groupby("underlying", as_index=False)[GREEK_COLUMNS]
            .sum()
            .sort_values("underlying")
        )
        totals = {column: float(enriched[column].sum()) for column in GREEK_COLUMNS}
        logger.info("Computed portfolio Greeks totals: {totals}", totals=totals)
        return GreekSummary(per_symbol=per_symbol, totals=totals)

    def _populate_greeks_from_ib(self, df: pd.DataFrame) -> None:
        if self._ib is None:
            logger.debug("No IBKR connection available for greek enrichment")
            return
        if not getattr(self._ib, "isConnected", lambda: False)():
            logger.debug("IBKR client is not connected; skipping greek enrichment")
            return
        logger.info("Fetching missing greeks from IBKR")
        for idx, row in df.iterrows():
            if row.get("sec_type", "OPT").upper() != "OPT":
                continue
            try:
                contract = self._build_option_contract(row)
            except ValueError as exc:
                logger.warning(
                    "Skipping position due to invalid contract data | symbol={symbol} expiry={expiry} strike={strike} reason={error}",
                    symbol=row.get("symbol"),
                    expiry=row.get("expiry"),
                    strike=row.get("strike"),
                    error=exc,
                )
                continue
            try:
                ticker = self._ib.reqMktData(contract, "", True, False)
                try:
                    self._ib.sleep(0.1)
                except Exception:
                    pass
                greeks = getattr(ticker, "modelGreeks", None)
                if greeks is None:
                    continue
                for column in GREEK_COLUMNS:
                    current = df.at[idx, column]
                    if current:
                        continue
                    value = getattr(greeks, column, None)
                    if value is None:
                        continue
                    df.at[idx, column] = float(value)
            except Exception as exc:
                logger.warning(
                    "Unable to retrieve greeks for contract | symbol={symbol} expiry={expiry} strike={strike} reason={error}",
                    symbol=row.get("symbol"),
                    expiry=row.get("expiry"),
                    strike=row.get("strike"),
                    error=exc,
                )

    def _build_option_contract(self, row: pd.Series) -> Option:
        expiry = str(row.get("expiry", ""))
        symbol = str(row.get("underlying") or row.get("symbol") or "")
        if not symbol:
            raise ValueError("missing symbol")
        try:
            strike = float(row.get("strike", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid strike") from exc
        right = str(row.get("right", "")) or "C"
        raw_multiplier = row.get("multiplier", 100)
        multiplier_value: float
        try:
            multiplier_value = float(raw_multiplier)
        except (TypeError, ValueError):
            multiplier_value = 100.0
        if pd.isna(multiplier_value) or multiplier_value <= 0.0:
            multiplier_value = 100.0
        multiplier = str(int(multiplier_value))
        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange="SMART",
            multiplier=multiplier,
            currency="USD",
        )
        return contract


def compute_concentration(positions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if positions.empty:
        return (
            pd.DataFrame(columns=["underlying", "gross_exposure", "gross_pct"]),
            pd.Series(dtype=float),
        )
    exposures = positions.copy()
    value_col = "market_value" if "market_value" in exposures.columns else "cost_basis"
    exposures[value_col] = pd.to_numeric(exposures.get(value_col, 0.0), errors="coerce").fillna(0.0)
    exposures["abs_value"] = exposures[value_col].abs()
    per_symbol = (
        exposures.groupby("underlying", as_index=False)["abs_value"].sum().rename(columns={"abs_value": "gross_exposure"})
    )
    gross_total = float(per_symbol["gross_exposure"].sum())
    if gross_total == 0.0:
        per_symbol["gross_pct"] = 0.0
    else:
        per_symbol["gross_pct"] = per_symbol["gross_exposure"] / gross_total
    per_symbol.sort_values("gross_exposure", ascending=False, inplace=True)
    concentration_series = per_symbol.set_index("underlying")["gross_pct"]
    return per_symbol, concentration_series


__all__ = ["GreekCalculator", "GreekSummary", "compute_concentration", "GREEK_COLUMNS"]
