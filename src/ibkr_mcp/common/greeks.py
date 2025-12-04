"""Greek calculation utilities for the portfolio manager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from ib_async import Contract, IB, Option
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
            per_symbol = pd.DataFrame(
                columns=[
                    "underlying",
                    *GREEK_COLUMNS,
                    "delta_exposure_net",
                    "delta_exposure_long",
                    "delta_exposure_short",
                ]
            )
            totals = {
                **{column: 0.0 for column in GREEK_COLUMNS},
                "delta_exposure_net": 0.0,
                "delta_exposure_long": 0.0,
                "delta_exposure_short": 0.0,
            }
            return GreekSummary(per_symbol=per_symbol, totals=totals)

        enriched = positions.copy()
        missing_columns = [column for column in GREEK_COLUMNS if column not in enriched.columns]
        if missing_columns:
            for column in missing_columns:
                enriched[column] = 0.0
            self._ib.reqMarketDataType(2)
            self._populate_greeks_from_ib(enriched)

        sec_type_series = enriched.get("sec_type")
        stock_mask = pd.Series(False, index=enriched.index)
        if sec_type_series is not None:
            sec_type_normalised = sec_type_series.astype(str).str.upper().fillna("")
            stock_mask = sec_type_normalised.eq("STK")
            if stock_mask.any() and "delta" in enriched.columns:
                stock_deltas = pd.to_numeric(
                    enriched.loc[stock_mask, "delta"], errors="coerce"
                )
                needs_default = stock_deltas.isna() | (stock_deltas == 0.0)
                if needs_default.any():
                    enriched.loc[stock_deltas.index[needs_default], "delta"] = 1.0

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

        price = pd.to_numeric(
            enriched.get("underlying_price", pd.Series(pd.NA, index=enriched.index)),
            errors="coerce",
        )
        fallback_needed = price.isna() | (price <= 0.0)
        if fallback_needed.any():
            market_price = pd.to_numeric(
                enriched.get("market_price", pd.Series(0.0, index=enriched.index)),
                errors="coerce",
            ).fillna(0.0)
            stock_fallback = fallback_needed & stock_mask
            if stock_fallback.any():
                price = price.where(~stock_fallback, market_price)
        price = price.fillna(0.0).clip(lower=0.0)
        delta_exposure_net = enriched["delta"] * price
        enriched["delta_exposure_net"] = delta_exposure_net
        enriched["delta_exposure_long"] = delta_exposure_net.clip(lower=0.0)
        enriched["delta_exposure_short"] = delta_exposure_net.clip(upper=0.0)

        aggregate_columns = [
            *GREEK_COLUMNS,
            "delta_exposure_net",
            "delta_exposure_long",
            "delta_exposure_short",
        ]
        per_symbol = (
            enriched.groupby("underlying", as_index=False)[aggregate_columns]
            .sum()
            .sort_values("underlying")
        )
        totals = {column: float(enriched[column].sum()) for column in GREEK_COLUMNS}
        totals["delta_exposure_net"] = float(enriched["delta_exposure_net"].sum())
        totals["delta_exposure_long"] = float(enriched["delta_exposure_long"].sum())
        totals["delta_exposure_short"] = float(enriched["delta_exposure_short"].sum())
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
        if "underlying_price" not in df.columns:
            df["underlying_price"] = pd.NA
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
                underlying_price = getattr(greeks, "undPrice", None)
                if underlying_price is not None:
                    current_price = df.at[idx, "underlying_price"]
                    if pd.isna(current_price) or (
                        isinstance(current_price, str) and not current_price.strip()
                    ):
                        try:
                            df.at[idx, "underlying_price"] = float(underlying_price)
                        except (TypeError, ValueError):
                            pass
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

    def _build_option_contract(self, row: pd.Series) -> Contract:
        # Prefer using conId when available to avoid re-qualifying contracts and
        # to satisfy ib_async's requirement that hashable contracts have a conId.
        conid = row.get("conid")
        if pd.notna(conid) and conid:
            try:
                conid_int = int(conid)
                if conid_int > 0:
                    return Contract(conId=conid_int, exchange="SMART")
            except (TypeError, ValueError):
                # Fall back to constructing by fields if conid is malformed
                pass

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
        contract: Contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange="SMART",
            multiplier=multiplier,
            currency="USD",
        )

        # If possible, qualify the contract to populate conId so that ib_async
        # can safely hash it when used in reqMktData and similar calls.
        if self._ib is None:
            logger.debug("No IBKR connection available for contract qualification")
            return contract

        try:
            self._ib.qualifyContracts(contract)
        except Exception:
            # Qualification is a best-effort improvement; failures are
            # handled by the caller when reqMktData raises.
            pass

        return contract


def compute_concentration(positions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    exposure_columns = ["underlying", "gross_exposure", "gross_pct"]
    if positions.empty:
        return (
            pd.DataFrame(columns=exposure_columns),
            pd.Series(dtype=float),
            {
                "gross_total": 0.0,
            },
        )

    exposures = positions.copy()
    value_col = "market_value" if "market_value" in exposures.columns else "cost_basis"
    exposures[value_col] = pd.to_numeric(exposures.get(value_col, 0.0), errors="coerce").fillna(0.0)
    exposures["abs_value"] = exposures[value_col].abs()
    per_symbol = (
        exposures.groupby("underlying", as_index=False)["abs_value"]
        .sum()
        .rename(columns={"abs_value": "gross_exposure"})
        .sort_values("gross_exposure", ascending=False)
    )

    gross_total = float(per_symbol["gross_exposure"].sum())
    per_symbol["gross_pct"] = 0.0 if gross_total == 0.0 else per_symbol["gross_exposure"] / gross_total
    per_symbol = per_symbol[["underlying", "gross_exposure", "gross_pct"]]

    concentration_series = per_symbol.set_index("underlying")["gross_pct"]
    totals = {
        "gross_total": gross_total,
    }
    return per_symbol, concentration_series, totals

def compute_delta_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=["underlying", "delta_exposure"])
    
    


__all__ = ["GreekCalculator", "GreekSummary", "compute_concentration", "GREEK_COLUMNS"]
