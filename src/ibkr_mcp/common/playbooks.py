"""Strategy playbook helpers for generating adjustment actions.

This module is adapted from the optionscanner project's portfolio
manager so that the same playbook logic can be used via MCP tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from ibkr_mcp.common.rules import RiskBreach


@dataclass(slots=True)
class PlaybookContext:
    """Provides context for generating playbook actions."""

    roll_rules: Dict[str, Dict[str, float]]


class PlaybookEngine:
    """Dispatches playbook logic per strategy type."""

    def __init__(self, context: PlaybookContext) -> None:
        self._context = context

    def generate(self, positions: pd.DataFrame, breaches: Iterable[RiskBreach]) -> List[str]:
        """Generate adjustment actions for the given positions and breaches."""
        if positions.empty:
            return []
        actions: List[str] = []
        breach_by_symbol: Dict[str, List[RiskBreach]] = {}

        # Index breaches by underlying symbol (or portfolio-wide)
        for breach in breaches:
            symbol = (breach.symbol or "portfolio").upper()
            breach_by_symbol.setdefault(symbol, []).append(breach)

        # Dispatch per strategy type
        for strategy, group in positions.groupby("strategy"):
            if not strategy:
                continue
            strategy_key = str(strategy).lower()
            handler = getattr(self, f"_handle_{strategy_key}", None)
            if handler is None:
                continue
            symbol = str(group["underlying"].iloc[0])
            symbol_breaches = breach_by_symbol.get(symbol.upper(), [])
            actions.extend(handler(symbol, group, symbol_breaches))
        return actions

    def _handle_pmcc(
        self,
        symbol: str,
        positions: pd.DataFrame,
        breaches: Iterable[RiskBreach],
    ) -> List[str]:
        """Generate actions for poor man's covered call (PMCC) positions."""
        rules = self._context.roll_rules.get("pmcc", {})
        take_profit_pct = float(rules.get("take_profit_pct", 0.6))
        roll_delta = float(rules.get("roll_up_if_short_delta_gt", 0.45))
        roll_days = int(rules.get("roll_out_days", 21))

        # Focus on short legs when deciding whether to roll
        short_legs = positions[pd.to_numeric(positions["quantity"], errors="coerce") < 0]
        if short_legs.empty:
            return []
        short = short_legs.iloc[0]

        realized_pct = float(short.get("pnl_pct", take_profit_pct))
        delta = abs(float(short.get("delta", 0.0) or 0.0))
        reason: List[str] = []

        if realized_pct >= take_profit_pct:
            reason.append(f"{realized_pct:.0%} credit captured")

        for breach in breaches:
            if "vega" in breach.metric.lower():
                reason.append("vega limit breach")
                break

        if delta >= roll_delta:
            reason.append(f"short delta {delta:.2f}")

        if not reason:
            reason.append("maintain core collar")

        description = (
            f"PMCC {symbol}: close short {short.get('symbol')} (+{take_profit_pct:.0%} credit), "
            f"re-sell near-{int(roll_delta * 100)}Δ call {roll_days}DTE"
        )
        return [f"{description} ({'; '.join(reason)})"]

    def _handle_condor(
        self,
        symbol: str,
        positions: pd.DataFrame,
        breaches: Iterable[RiskBreach],
    ) -> List[str]:
        """Generate actions for iron condor / condor positions."""
        rules = self._context.roll_rules.get("condor", {})
        take_profit_pct = float(rules.get("take_profit_pct", 0.6))

        strikes = pd.to_numeric(positions.get("strike"), errors="coerce")
        width = float(strikes.max() - strikes.min()) if not strikes.empty else 0.0
        reason: List[str] = [f"width {width:.0f}"] if width == width else []  # NaN guard

        for breach in breaches:
            if "gamma" in breach.metric.lower():
                reason.append("gamma elevated")
                break

        description = f"{symbol} Condor: harvest profits near {take_profit_pct:.0%} credit"
        if reason:
            description += f" ({'; '.join(reason)})"
        return [description]


__all__ = ["PlaybookEngine", "PlaybookContext"]

