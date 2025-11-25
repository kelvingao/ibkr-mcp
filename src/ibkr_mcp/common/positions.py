"""Position loading and normalisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from ib_async import Contract, IB
from loguru import logger


NORMALISED_COLUMNS = [
    "account",
    "underlying",
    "symbol",
    "expiry",
    "right",
    "strike",
    "sec_type",
    "conid",
    "multiplier",
    "quantity",
    "avg_price",
    "market_price",
    "market_value",
    "cost_basis",
    "open_interest",
    "bid_ask_spread_pct",
    "strategy",
    "source",
]


@dataclass(slots=True)
class PositionSource:
    """Describes the origin of portfolio positions."""

    ib: Optional[IB] = None
    log_dir: Path = Path("./logs")
    results_dir: Path = Path("./results")


class PositionLoader:
    """Loads open positions from multiple sources and normalises to a DataFrame."""

    def __init__(self, source: PositionSource) -> None:
        self._source = source
        self._log_dir = source.log_dir
        self._results_dir = source.results_dir

    def load(self, account: str = "") -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        ib = self._source.ib
        if ib is not None:
            try:
                frames.append(self._load_ib_positions(ib, account))
            except Exception:
                logger.exception("Failed to load positions from IBKR")
        frames.extend(self._load_logged_positions())
        if not frames:
            logger.warning("No portfolio positions were loaded")
            return pd.DataFrame(columns=NORMALISED_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        df = self._normalise(df)
        logger.info("Loaded {count} portfolio positions", count=len(df))
        return df

    def _load_ib_positions(self, ib: IB, account: str = "") -> pd.DataFrame:
        logger.info("Loading positions from IBKR account")
        positions = ib.positions(account)
        rows: List[dict] = []
        try:
            ib_portfolio_items = ib.portfolio()
        except Exception:
            ib_portfolio_items = []
        for position in positions:
            contract: Contract = position.contract
            multiplier = float(getattr(contract, "multiplier", 1) or 1)
            sec_type = getattr(contract, "secType", "")
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
            right = getattr(contract, "right", "")
            strike = float(getattr(contract, "strike", 0.0) or 0.0)
            symbol = getattr(contract, "symbol", "")
            market_price = None
            market_value = None
            key = getattr(contract, "conId", None)
            if key is not None:
                ib_portfolio = next(
                    (item for item in ib_portfolio_items if item.contract.conId == key), None
                )
                if ib_portfolio is not None:
                    market_price = float(getattr(ib_portfolio, "marketPrice", 0.0) or 0.0)
                    market_value = float(getattr(ib_portfolio, "marketValue", 0.0) or 0.0)
            row = {
                "account": getattr(position, "account", ""),
                "conid": key,
                "underlying": symbol,
                "symbol": getattr(contract, "localSymbol", symbol) or symbol,
                "expiry": expiry,
                "right": right,
                "strike": strike,
                "sec_type": sec_type,
                "multiplier": multiplier,
                "quantity": float(getattr(position, "position", 0.0) or 0.0),
                "avg_price": float(getattr(position, "avgCost", 0.0) or 0.0) / multiplier,
                "market_price": market_price,
                "market_value": market_value,
                "cost_basis": float(getattr(position, "avgCost", 0.0) or 0.0)
                * float(getattr(position, "position", 0.0) or 0.0)
                / multiplier,
                "strategy": None,
                "source": "ibkr",
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def _load_logged_positions(self) -> List[pd.DataFrame]:
        paths = list(self._iter_position_logs())
        frames: List[pd.DataFrame] = []
        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                logger.warning(
                    "Unable to read logged positions | path={path} reason={error}",
                    path=str(path),
                    error=exc,
                )
                continue
            df["source"] = df.get("source", "log")
            frames.append(df)
        return frames

    def _iter_position_logs(self) -> Iterable[Path]:
        patterns = ("positions_*.csv", "portfolio_*.csv")
        for directory in {self._log_dir, self._results_dir}:
            if not directory.exists():
                continue
            for pattern in patterns:
                for path in sorted(directory.glob(pattern)):
                    stem = path.stem.lower()
                    if "summary" in stem:
                        logger.debug(
                            "Skipping summary file while loading positions | path={path}",
                            path=str(path),
                        )
                        continue
                    yield path

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in NORMALISED_COLUMNS:
            if column not in df.columns:
                df[column] = None
        df["underlying"] = df["underlying"].fillna(df["symbol"]).astype(str).str.upper()
        df["symbol"] = df["symbol"].astype(str)
        df["sec_type"] = df["sec_type"].fillna("OPT").astype(str)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
        df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").replace(0, 1.0)
        numeric_cols = [
            "conid",
            "avg_price",
            "market_price",
            "market_value",
            "cost_basis",
            "open_interest",
            "bid_ask_spread_pct",
        ]
        for column in numeric_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["strategy"] = df["strategy"].astype(str).str.lower().replace({"nan": None, "": None})
        df["source"] = df["source"].fillna("unknown").astype(str)
        columns = [column for column in NORMALISED_COLUMNS if column in df.columns]
        return df[columns]


__all__ = ["PositionLoader", "PositionSource", "NORMALISED_COLUMNS"]
