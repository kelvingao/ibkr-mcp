"""News retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ib_async import Contract

from .base import IBService


@dataclass(slots=True)
class NewsService(IBService):
    """Wraps historical news lookups."""

    async def get_historical_news(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_count: int = 10,
        exchange: str = "SMART",
        currency: str = "USD",
        conid: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_connected()
        ib = self.ib

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                return {
                    "ok": False,
                    "error_type": "INVALID_DATE_RANGE",
                    "message": f"Start date {start_date} must be before end date {end_date}",
                    "symbol": symbol,
                }
        except ValueError as exc:
            return {
                "ok": False,
                "error_type": "INVALID_DATE_FORMAT",
                "message": f"Date format must be YYYY-MM-DD: {exc}",
                "symbol": symbol,
            }

        max_count = max(1, min(100, max_count))

        try:
            if conid:
                contracts = [Contract(conId=conid)]
            else:
                contract = Contract(
                    symbol=symbol,
                    secType="STK",
                    exchange=exchange,
                    currency=currency,
                )
                contracts_raw = await ib.qualifyContractsAsync(contract)
                contracts = self._flatten_contracts(contracts_raw)

            if not contracts:
                return {
                    "ok": False,
                    "error_type": "NO_CONTRACT",
                    "message": f"No contract found for {symbol}",
                    "symbol": symbol,
                }

            actual_conid = getattr(contracts[0], "conId", 0)

            news = await ib.reqHistoricalNewsAsync(
                actual_conid,
                "BRFG+BRFUPDN+DJ-N+DJ-RT+DJ-RTA+DJ-RTE+DJ-RTG+DJNL+FLY",
                start_date,
                end_date,
                max_count,
            )

            articles: List[Dict[str, Any]] = []
            if isinstance(news, list):
                for article in news[:max_count]:
                    articles.append(
                        {
                            "headline": getattr(article, "headline", ""),
                            "timestamp": getattr(article, "time", ""),
                            "provider": getattr(article, "providerCode", ""),
                            "article_id": getattr(article, "articleId", ""),
                        }
                    )

            return {
                "ok": True,
                "symbol": symbol,
                "conid": actual_conid,
                "period": {"start": start_date, "end": end_date},
                "articles": articles,
                "total_count": len(articles),
            }

        except Exception as exc:  # pragma: no cover - IBKR network call
            return {
                "ok": False,
                "error_type": "IB_ERROR",
                "message": f"Error getting historical news: {exc}",
                "symbol": symbol,
            }

    def _flatten_contracts(self, contracts: Any) -> List[Contract]:
        result: List[Contract] = []
        if contracts is None:
            return result
        if isinstance(contracts, Contract):
            result.append(contracts)
            return result
        if isinstance(contracts, list):
            for item in contracts:
                result.extend(self._flatten_contracts(item))
        return result
