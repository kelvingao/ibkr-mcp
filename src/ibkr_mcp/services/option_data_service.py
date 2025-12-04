"""Option chain retrieval helpers backed by the shared IB client."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from loguru import logger

from ibkr_mcp.common.option_data import (
    IBKRDataFetcher,
    LocalDataFetcher,
    OptionChainSnapshot,
)

from .base import IBService


@dataclass(slots=True)
class OptionDataService(IBService):
    """Service wrapper that exposes option chain retrieval through MCP."""

    data_dir: Path
    history_dir: Path
    default_market_data_type: str = "LIVE"

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_option_chains(
        self,
        symbols: Sequence[str],
        mode: str = "live",
        market_data_type: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Retrieve option chain snapshots either live or from disk."""
        self._ensure_connected()
        normalized_symbols = self._normalise_symbols(symbols)
        if not normalized_symbols:
            return {
                "ok": False,
                "error_type": "NO_SYMBOLS",
                "message": "At least one symbol is required",
            }

        run_mode = (mode or "live").lower()
        if run_mode not in ("live", "local"):
            return {
                "ok": False,
                "error_type": "INVALID_MODE",
                "message": f"Unsupported mode '{mode}'. Use 'live' or 'local'.",
            }

        effective_market_type = (market_data_type or self.default_market_data_type).upper()

        logger.info(
            "Fetching option chains mode={mode} symbols={count} market_data_type={market_type}",
            mode=run_mode,
            count=len(normalized_symbols),
            market_type=effective_market_type,
        )

        try:
            if run_mode == "local":
                fetcher = LocalDataFetcher(self.data_dir)
            else:
                fetcher = IBKRDataFetcher(
                    ib=self.ib,
                    data_dir=self.data_dir,
                    market_data_type=effective_market_type,
                    history_dir=self.history_dir,
                    persist=persist,
                )
            snapshots = await fetcher.fetch_all(normalized_symbols)
        except Exception as exc:  # pragma: no cover - IB/network failures
            logger.opt(exception=exc).error("Option chain retrieval failed")
            return {
                "ok": False,
                "error_type": "FETCH_FAILED",
                "message": str(exc),
            }

        missing = sorted(self._missing_symbols(normalized_symbols, snapshots))
        return {
            "ok": True,
            "mode": run_mode,
            "market_data_type": effective_market_type,
            "symbol_count": len(normalized_symbols),
            "snapshot_count": len(snapshots),
            "snapshots": [self._snapshot_to_dict(s) for s in snapshots],
            "missing_symbols": missing,
            "persisted": persist if run_mode == "live" else False,
            "data_dir": str(self.data_dir),
        }

    @staticmethod
    def _normalise_symbols(symbols: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        for symbol in symbols:
            if not symbol:
                continue
            cleaned.append(symbol.strip().upper())
        # Preserve order but drop duplicates.
        seen: Set[str] = set()
        ordered: List[str] = []
        for symbol in cleaned:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
        return ordered

    @staticmethod
    def _missing_symbols(symbols: Sequence[str], snapshots: Sequence[OptionChainSnapshot]) -> Set[str]:
        have = {snap.symbol for snap in snapshots}
        return {symbol for symbol in symbols if symbol not in have}

    @staticmethod
    def _snapshot_to_dict(snapshot: OptionChainSnapshot) -> Dict[str, Any]:
        timestamp = snapshot.timestamp.astimezone(timezone.utc).isoformat()
        return {
            "symbol": snapshot.symbol,
            "underlying_price": snapshot.underlying_price,
            "timestamp": timestamp,
            "option_count": len(snapshot.options),
            "options": snapshot.options,
            "context": snapshot.context or {},
        }
