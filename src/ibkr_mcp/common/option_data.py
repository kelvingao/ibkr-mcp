"""Option chain data access layer used by the scanner."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from ib_async import Contract, IB, Option, Stock
from loguru import logger
from zoneinfo import ZoneInfo


MARKET_DATA_TYPE_CODES = {
    "LIVE": 1,
    "FROZEN": 2,
}
MARKET_DATA_CODE_TO_NAME = {code: name for name, code in MARKET_DATA_TYPE_CODES.items()}

# Hard-coded venue preferences for symbols that require a specific routing.
OPTION_EXCHANGE_OVERRIDES: Dict[str, str] = {
    "NVDA": "SMART",
    "AAPL": "CBOE",
    "TSLA": "CBOE",
    "META": "CBOE",
    "AMZN": "NASDAQOM",
    "MSFT": "CBOE",
    "GOOG": "CBOE",
    "NFLX": "CBOE",
    "AMD": "CBOE",
    "JPM": "BOX",
}

FALLBACK_OPTION_EXCHANGES = [
    "SMART",
    "CBOE",
    "BOX",
    "NASDAQOM",
    "ARCA",
    "BATSOP",
]


@dataclass(slots=True)
class OptionChainSnapshot:
    """Container for a single option chain snapshot."""

    symbol: str
    underlying_price: float
    timestamp: datetime
    options: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None

    def to_pandas(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.options)
        if frame.empty:
            return frame
        frame["symbol"] = self.symbol
        frame["underlying_price"] = self.underlying_price
        frame["timestamp"] = self.timestamp
        frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
        if self.context:
            for key, value in self.context.items():
                frame[key] = value
        return frame


class BaseDataFetcher:
    """Interface used by executors that need to load option chains."""

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:  # pragma: no cover - abstract
        raise NotImplementedError


class IBKRDataFetcher(BaseDataFetcher):
    """Fetches live option data from IBKR using ib_async."""

    def __init__(
        self,
        ib: IB,
        data_dir: Path,
        market_data_type: str,
        history_dir: Optional[Path] = None,
        persist: bool = True,
    ) -> None:
        self._ib = ib
        self.data_dir = data_dir
        self.market_data_type = market_data_type.upper()
        if self.market_data_type not in MARKET_DATA_TYPE_CODES:
            raise ValueError(
                f"Unsupported market data type '{market_data_type}'. "
                f"Choose one of {sorted(MARKET_DATA_TYPE_CODES)}."
            )
        self._market_data_type_code = MARKET_DATA_TYPE_CODES[self.market_data_type]
        self._current_market_data_type_code: Optional[int] = None
        self._history_timezone = ZoneInfo("America/Los_Angeles")
        self._max_expiries = 32  # cover ≈3 months of weekly expirations
        self._expiry_horizon_days = 92
        self._leaps_min_days = 240
        self._leaps_max_days = 300
        self._max_strikes_per_side = 8
        self._contracts_per_chunk = 40
        self._persist_enabled = persist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = history_dir or Path("historydata")
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ib(self) -> IB:
        """Expose the underlying ib_async client."""
        return self._ib

    def _set_market_data_type(self, code: int, *, reason: Optional[str] = None) -> None:
        if self._current_market_data_type_code == code:
            return
        self._ib.reqMarketDataType(code)
        self._current_market_data_type_code = code
        message = MARKET_DATA_CODE_TO_NAME.get(code, str(code))
        if reason:
            logger.warning(
                "Set IBKR market data type to {market_data_type} ({reason})",
                market_data_type=message,
                reason=reason,
            )
        else:
            logger.info("Set IBKR market data type to {market_data_type}", market_data_type=message)

    async def _request_tickers(self, contracts: Sequence[Contract]) -> List[Any]:
        if not contracts:
            return []
        try:
            self._set_market_data_type(self._market_data_type_code)
            if hasattr(self._ib, "reqTickersAsync"):
                tickers = await self._ib.reqTickersAsync(*contracts)
            else:
                tickers = self._ib.reqTickers(*contracts)
        except Exception as exc:
            error_code = getattr(exc, "errorCode", None)
            if error_code == 354:
                raise RuntimeError(
                    "IBKR rejected the market data request (error 354). "
                    "Ensure your account has the correct LIVE or FROZEN data entitlements."
                ) from exc
            raise
        return list(tickers or [])

    async def _request_underlying_ticker(self, contract: Contract) -> Any:
        tickers = await self._request_tickers([contract])
        if not tickers:
            raise RuntimeError(f"No market data returned for underlying {contract.symbol}")
        return tickers[0]

    async def _qualify_contracts(self, contracts: Sequence[Option]) -> List[Contract]:
        qualified: List[Contract] = []
        for idx in range(0, len(contracts), self._contracts_per_chunk):
            chunk = contracts[idx : idx + self._contracts_per_chunk]
            if not chunk:
                continue
            try:
                chunk_result = await self._ib.qualifyContractsAsync(*chunk)
            except Exception as exc:
                logger.opt(exception=exc).warning(
                    "Unable to qualify option contracts for {symbol}",
                    symbol=chunk[0].symbol,
                )
                continue
            qualified.extend(contract for contract in chunk_result if isinstance(contract, Contract))
        return qualified

    @staticmethod
    def _meaningful_price(value: float) -> bool:
        return math.isfinite(value) and value > 0.0

    def _extract_underlying_price(self, ticker: Any, symbol: str) -> float:
        candidate = float(ticker.last or ticker.close or ticker.marketPrice() or 0.0)
        if self._meaningful_price(candidate):
            return candidate
        raise RuntimeError(f"Unable to determine underlying price for {symbol}")

    async def _load_chain_metadata(self, stock_contract: Contract) -> Any:
        params = await self._ib.reqSecDefOptParamsAsync(
            stock_contract.symbol,
            "",
            stock_contract.secType,
            stock_contract.conId,
        )
        if not params:
            raise RuntimeError(f"No option chain metadata returned for {stock_contract.symbol}")
        override_exchange = OPTION_EXCHANGE_OVERRIDES.get(stock_contract.symbol, "")
        if override_exchange:
            for chain in params:
                if (chain.exchange or "").upper() == override_exchange:
                    return chain
        return max(params, key=lambda p: len(p.strikes or []))

    @staticmethod
    def _sanitize_floats(values: Sequence[Any]) -> List[float]:
        cleaned: List[float] = []
        for value in values:
            if value in ("", None):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                cleaned.append(numeric)
        return cleaned

    def _select_expiries(self, expirations: Sequence[str]) -> List[str]:
        today = datetime.now(timezone.utc).date()
        horizon = today + timedelta(days=self._expiry_horizon_days)
        leaps_start = today + timedelta(days=self._leaps_min_days)
        leaps_end = today + timedelta(days=self._leaps_max_days)

        near_term: List[Tuple[date, str]] = []
        leaps: List[Tuple[date, str]] = []
        others: List[Tuple[date, str]] = []

        for expiry in expirations:
            if not expiry:
                continue
            try:
                expiry_date = datetime.strptime(expiry, "%Y%m%d").date()
            except ValueError:
                continue

            bucket = others
            if today <= expiry_date <= horizon:
                bucket = near_term
            elif leaps_start <= expiry_date <= leaps_end:
                bucket = leaps
            bucket.append((expiry_date, expiry))

        if not (near_term or leaps or others):
            raise RuntimeError("Option chain metadata did not include expirations")

        key = lambda item: item[0]
        near_term.sort(key=key)
        leaps.sort(key=key)
        others.sort(key=key)

        selections: List[Tuple[date, str]] = []
        leaps_budget = min(len(leaps), self._max_expiries)
        near_budget = max(self._max_expiries - leaps_budget, 0)

        selections.extend(near_term[:near_budget])
        selections.extend(leaps[:leaps_budget])

        remaining_slots = self._max_expiries - len(selections)
        if remaining_slots > 0:
            remaining_candidates = near_term[near_budget:] + leaps[leaps_budget:] + others
            selections.extend(remaining_candidates[:remaining_slots])

        return [expiry for _, expiry in selections]

    def _select_strikes(self, strikes: Sequence[Any], reference_price: float) -> List[float]:
        cleaned = sorted(set(self._sanitize_floats(strikes)))
        if not cleaned:
            raise RuntimeError("Option chain metadata did not include strikes")
        sorted_by_distance = sorted(cleaned, key=lambda strike: (abs(strike - reference_price), strike))
        max_contracts = self._max_strikes_per_side * 2
        selection: List[float] = []
        for strike in sorted_by_distance:
            selection.append(strike)
            if len(selection) >= max_contracts:
                break
        selection.sort()
        return selection

    def _build_exchange_order(self, symbol: str, chain_exchange: Optional[str]) -> List[str]:
        exchanges: List[str] = []
        for exchange in [OPTION_EXCHANGE_OVERRIDES.get(symbol), chain_exchange, *FALLBACK_OPTION_EXCHANGES]:
            if exchange and exchange not in exchanges:
                exchanges.append(exchange)
        return exchanges

    def _build_option_contracts(
        self,
        symbol: str,
        expiry: str,
        strikes: Sequence[float],
        exchange: str,
        trading_class: Optional[str],
    ) -> List[Option]:
        contracts: List[Option] = []
        for strike in strikes:
            for right in ("C", "P"):
                contract = Option(symbol, expiry, strike, right, exchange, currency="USD")
                if trading_class and exchange not in ("SMART", ""):
                    contract.tradingClass = trading_class
                contracts.append(contract)
        return contracts

    def _quotes_to_rows(
        self,
        tickers: Sequence[Any],
        symbol: str,
    ) -> Dict[int, Dict[str, Any]]:
        rows: Dict[int, Dict[str, Any]] = {}
        for ticker in tickers:
            contract = ticker.contract
            if not isinstance(contract, Option):
                continue
            expiry = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")
            mark = float(ticker.midpoint() or 0.0)
            bid = float(ticker.bid or 0.0)
            ask = float(ticker.ask or 0.0)
            delta = getattr(ticker.modelGreeks, "delta", 0.0) if ticker.modelGreeks else 0.0
            gamma = getattr(ticker.modelGreeks, "gamma", 0.0) if ticker.modelGreeks else 0.0
            vega = getattr(ticker.modelGreeks, "vega", 0.0) if ticker.modelGreeks else 0.0
            theta = getattr(ticker.modelGreeks, "theta", 0.0) if ticker.modelGreeks else 0.0
            rho = getattr(ticker.modelGreeks, "rho", 0.0) if ticker.modelGreeks else 0.0
            imp_vol = getattr(ticker.modelGreeks, "impliedVol", 0.0) if ticker.modelGreeks else 0.0
            rows[contract.conId] = {
                "symbol": symbol,
                "expiry": expiry,
                "strike": float(contract.strike),
                "option_type": "CALL" if contract.right == "C" else "PUT",
                "bid": bid,
                "ask": ask,
                "mark": mark,
                "delta": float(delta or 0.0),
                "gamma": float(gamma or 0.0),
                "vega": float(vega or 0.0),
                "theta": float(theta or 0.0),
                "rho": float(rho or 0.0),
                "implied_volatility": float(imp_vol or 0.0),
            }
        return rows

    async def fetch_option_chain(self, symbol: str) -> OptionChainSnapshot:
        stock = Stock(symbol, "SMART", "USD")
        qualified_stock = await self._ib.qualifyContractsAsync(stock)
        if qualified_stock:
            stock = qualified_stock[0]
        ticker = await self._request_underlying_ticker(stock)
        underlying_price = self._extract_underlying_price(ticker, symbol)

        chain = await self._load_chain_metadata(stock)
        expiries = self._select_expiries(chain.expirations)
        strikes = self._select_strikes(chain.strikes, underlying_price)
        trading_class = getattr(chain, "tradingClass", "") or ""

        rows_by_conid: Dict[int, Dict[str, Any]] = {}
        for expiry in expiries:
            for exchange in self._build_exchange_order(symbol, chain.exchange):
                contracts = self._build_option_contracts(symbol, expiry, strikes, exchange, trading_class)
                qualified = await self._qualify_contracts(contracts)
                if not qualified:
                    logger.debug(
                        "No qualified option contracts for {symbol} expiry={expiry} exchange={exchange}",
                        symbol=symbol,
                        expiry=expiry,
                        exchange=exchange,
                    )
                    continue
                tickers = await self._request_tickers(qualified)
                rows = self._quotes_to_rows(tickers, symbol)
                if not rows:
                    logger.debug(
                        "No option quotes returned for {symbol} expiry={expiry} exchange={exchange}",
                        symbol=symbol,
                        expiry=expiry,
                        exchange=exchange,
                    )
                    continue
                rows_by_conid.update(rows)
                # Once we get data for the first viable exchange we keep moving to next expiry.
                break

        if not rows_by_conid:
            raise RuntimeError(f"No option contracts selected for {symbol}")

        snapshot = OptionChainSnapshot(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=datetime.now(timezone.utc),
            options=list(rows_by_conid.values()),
        )
        self._persist_snapshot(snapshot)
        return snapshot

    def _persist_snapshot(self, snapshot: OptionChainSnapshot) -> None:
        if not self._persist_enabled:
            return
        frame = snapshot.to_pandas()
        if frame.empty:
            logger.warning("Skipping persistence for {symbol}; snapshot contained no rows", symbol=snapshot.symbol)
            return

        frame = frame.copy()
        frame["price"] = frame.get("mark", 0.0)

        timestamp_utc = snapshot.timestamp.replace(tzinfo=timezone.utc)
        timestamp_str = timestamp_utc.strftime("%Y%m%d_%H%M%S")
        file_path = self.data_dir / f"{snapshot.symbol}_{timestamp_str}.parquet"
        frame.to_parquet(file_path, index=False)
        logger.info("Saved option snapshot to {path}", path=str(file_path))

        timestamp_local = timestamp_utc.astimezone(self._history_timezone)
        frame["timestamp"] = timestamp_local.isoformat()
        history_path = self.history_dir / f"{timestamp_local.strftime('%Y%m%d')}.csv"
        header = not history_path.exists()
        frame.to_csv(history_path, mode="a", header=header, index=False)
        logger.info("Appended option snapshot to history file {path}", path=str(history_path))

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:
        tasks = [self.fetch_option_chain(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[OptionChainSnapshot] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.opt(exception=result).error(
                    "Failed to fetch data for {symbol}: {error}",
                    symbol=symbol,
                    error=result,
                )
                continue
            snapshots.append(result)
        return snapshots


class LocalDataFetcher(BaseDataFetcher):
    """Loads previously persisted option snapshots from disk."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    async def fetch_all(self, symbols: Iterable[str]) -> List[OptionChainSnapshot]:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, self._load_snapshot, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        snapshots: List[OptionChainSnapshot] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.opt(exception=result).error(
                    "Failed to load local data for {symbol}: {error}",
                    symbol=symbol,
                    error=result,
                )
                continue
            snapshots.append(result)
        return snapshots

    def _load_snapshot(self, symbol: str) -> OptionChainSnapshot:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Local data directory '{self.data_dir}' does not exist")
        pattern = f"{symbol}_*.parquet"
        matches = sorted(self.data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No local snapshot found for {symbol}. Expected files matching {pattern} in {self.data_dir}"
            )
        latest = matches[-1]
        frame = pd.read_parquet(latest)
        if frame.empty:
            raise ValueError(f"Local snapshot {latest} is empty")
        timestamp = pd.to_datetime(frame["timestamp"].iloc[0])
        underlying_price = float(frame["underlying_price"].iloc[0])
        options = frame.drop(columns=[col for col in ("symbol", "underlying_price", "timestamp") if col in frame.columns])
        return OptionChainSnapshot(
            symbol=symbol,
            underlying_price=underlying_price,
            timestamp=timestamp.to_pydatetime(),
            options=options.to_dict(orient="records"),
        )


__all__ = [
    "BaseDataFetcher",
    "IBKRDataFetcher",
    "LocalDataFetcher",
    "OptionChainSnapshot",
    "MARKET_DATA_TYPE_CODES",
    "MARKET_DATA_CODE_TO_NAME",
]
