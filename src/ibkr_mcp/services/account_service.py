"""Account and portfolio related helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ibkr_mcp.common.positions import PositionLoader, PositionSource

from .base import IBService


@dataclass(slots=True)
class AccountService(IBService):
    """Encapsulates account summary, positions, and portfolio helpers."""

    async def get_account_summary(self, account: str = "") -> list[dict[str, Any]]:
        self._ensure_connected()
        raw = await self.ib.accountSummaryAsync(account)
        return [
            {
                "tag": v.tag,
                "currency": v.currency,
                "account": v.account,
                "value": v.value,
            }
            for v in raw or []
        ]

    def get_positions(self, account: str = "") -> dict[str, Any]:
        self._ensure_connected()
        loader = PositionLoader(PositionSource(ib=self.ib))
        positions = loader.load(account)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in positions.to_dict(orient="records"):
            normalized = self._normalise_position_row(row)
            account_id = normalized.get("account_id") or "default"
            grouped[account_id].append(normalized)
        return {"positions": dict(grouped)}

    @staticmethod
    def _normalise_position_row(row: dict[str, Any]) -> dict[str, Any]:
        """Format loader records with the field names the UI expects."""

        def clean(value: Any) -> Any:
            if isinstance(value, float) and (value != value):
                return None
            return value

        account_id_raw = (
            row.get("account")
            or row.get("account_id")
            or row.get("accountId")
            or ""
        )
        account_id = str(account_id_raw).strip() or "default"
        quantity = clean(row.get("quantity"))
        avg_price = clean(row.get("avg_price"))
        market_price = clean(row.get("market_price"))
        market_value = clean(row.get("market_value"))

        mapped: dict[str, Any] = {
            "account_id": account_id,
            "symbol": row.get("symbol") or row.get("underlying") or "",
            "underlying": row.get("underlying"),
            "sec_type": row.get("sec_type"),
            "conid": clean(row.get("conid")),
            "expiry": row.get("expiry"),
            "right": row.get("right"),
            "strike": clean(row.get("strike")),
            "multiplier": clean(row.get("multiplier")),
            "quantity": quantity,
            "avg_price": avg_price,
            "avg_cost": avg_price,
            "market_price": market_price,
            "market_value": market_value,
            "cost_basis": clean(row.get("cost_basis")),
            "strategy": row.get("strategy"),
            "source": row.get("source"),
        }
        return mapped

    def get_portfolio(self, account: str = "") -> list[dict[str, Any]]:
        self._ensure_connected()
        portfolio = self.ib.portfolio(account)
        if portfolio is None:
            return []

        items: list[dict[str, Any]] = []
        for p in portfolio:
            items.append(
                {
                    "account": getattr(p, "account", ""),
                    "contract": getattr(p, "contract", None),
                    "position": getattr(p, "position", 0),
                    "marketPrice": getattr(p, "marketPrice", 0.0),
                    "marketValue": getattr(p, "marketValue", 0.0),
                    "averageCost": getattr(p, "averageCost", 0.0),
                    "unrealizedPNL": getattr(p, "unrealizedPNL", 0.0),
                    "realizedPNL": getattr(p, "realizedPNL", 0.0),
                }
            )
        return items
