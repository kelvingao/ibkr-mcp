"""Risk rule evaluation utilities for the IBKR MCP server.

This module is adapted from the optionscanner project's portfolio
manager so that the same risk configuration and evaluation logic can
be used via MCP tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ibkr_mcp.common.greeks import GREEK_COLUMNS


@dataclass(slots=True)
class RiskLimitConfig:
    """Container for parsed risk configuration."""

    limits: Dict[str, object]
    roll_rules: Dict[str, Dict[str, float]]


@dataclass(slots=True)
class RiskBreach:
    """Represents a breached risk constraint."""

    metric: str
    symbol: Optional[str]
    value: float
    limit: float
    detail: str

    def format_for_report(self) -> str:
        target = self.symbol or "portfolio"
        return f"{self.metric} ({target}) {self.value:.2f} > {self.limit:.2f}"


class RiskEvaluator:
    """Applies configured risk rules to greek totals and exposures."""

    def __init__(self, config: RiskLimitConfig) -> None:
        self._config = config

    def evaluate(
        self,
        greek_summary: pd.DataFrame,
        greek_totals: Dict[str, float],
        concentration: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> List[RiskBreach]:
        """Evaluate all configured risk rules and return any breaches."""
        breaches: List[RiskBreach] = []
        limit_config = self._config.limits
        breaches.extend(self._check_symbol_greek_limits(greek_summary, limit_config))
        breaches.extend(self._check_total_greek_limits(greek_totals, limit_config))
        breaches.extend(self._check_theta_floor(greek_totals, limit_config))
        breaches.extend(self._check_concentration(concentration, limit_config))
        breaches.extend(self._check_liquidity(positions, limit_config))
        logger.info("Evaluated risk rules | breaches={count}", count=len(breaches))
        return breaches

    def _check_symbol_greek_limits(
        self,
        greek_summary: pd.DataFrame,
        limit_config: Dict[str, object],
    ) -> List[RiskBreach]:
        breaches: List[RiskBreach] = []
        for greek in GREEK_COLUMNS:
            greek_limits = limit_config.get(greek, {})
            if not isinstance(greek_limits, dict):
                continue
            per_symbol_limit = greek_limits.get("per_symbol")
            if per_symbol_limit is None:
                continue
            for _, row in greek_summary.iterrows():
                value = float(row.get(greek, 0.0))
                if abs(value) > float(per_symbol_limit):
                    breaches.append(
                        RiskBreach(
                            metric=f"{greek.capitalize()} per-symbol",
                            symbol=str(row.get("underlying")),
                            value=value,
                            limit=float(per_symbol_limit),
                            detail=f"{row.get('underlying')} {greek} {value:.2f} exceeds {per_symbol_limit}",
                        )
                    )
        return breaches

    def _check_total_greek_limits(
        self,
        greek_totals: Dict[str, float],
        limit_config: Dict[str, object],
    ) -> List[RiskBreach]:
        breaches: List[RiskBreach] = []
        for greek in GREEK_COLUMNS:
            greek_limits = limit_config.get(greek, {})
            if not isinstance(greek_limits, dict):
                continue
            total_limit = greek_limits.get("total")
            if total_limit is None:
                continue
            value = float(greek_totals.get(greek, 0.0))
            if abs(value) > float(total_limit):
                breaches.append(
                    RiskBreach(
                        metric=f"{greek.capitalize()} total",
                        symbol=None,
                        value=value,
                        limit=float(total_limit),
                        detail=f"Portfolio {greek} {value:.2f} exceeds {total_limit}",
                    )
                )
        return breaches

    def _check_theta_floor(
        self,
        greek_totals: Dict[str, float],
        limit_config: Dict[str, object],
    ) -> List[RiskBreach]:
        theta_min = limit_config.get("theta_min")
        if theta_min is None:
            return []
        if isinstance(theta_min, dict):
            theta_min = theta_min.get("value")
        if theta_min is None:
            return []
        value = float(greek_totals.get("theta", 0.0))
        if value < float(theta_min):
            return [
                RiskBreach(
                    metric="Theta floor",
                    symbol=None,
                    value=value,
                    limit=float(theta_min),
                    detail=f"Portfolio theta {value:.2f} is below minimum {theta_min}",
                )
            ]
        return []

    def _check_concentration(
        self,
        concentration: pd.DataFrame,
        limit_config: Dict[str, object],
    ) -> List[RiskBreach]:
        concentration_config = limit_config.get("concentration", {})
        if not isinstance(concentration_config, dict):
            return []
        limit = concentration_config.get("max_symbol_pct_gross")
        if limit is None:
            return []
        breaches: List[RiskBreach] = []
        for _, row in concentration.iterrows():
            pct = float(row.get("gross_pct", 0.0))
            if pct > float(limit):
                breaches.append(
                    RiskBreach(
                        metric="Concentration",
                        symbol=str(row.get("underlying")),
                        value=pct,
                        limit=float(limit),
                        detail=f"{row.get('underlying')} gross pct {pct:.2%} exceeds {float(limit):.2%}",
                    )
                )
        return breaches

    def _check_liquidity(
        self,
        positions: pd.DataFrame,
        limit_config: Dict[str, object],
    ) -> List[RiskBreach]:
        liquidity_config = limit_config.get("liquidity", {})
        if not isinstance(liquidity_config, dict):
            return []
        min_open_interest = liquidity_config.get("min_open_interest")
        max_spread = liquidity_config.get("max_bid_ask_spread_pct")
        breaches: List[RiskBreach] = []
        if min_open_interest is not None and "open_interest" in positions.columns:
            illiquid = positions[
                pd.to_numeric(positions["open_interest"], errors="coerce")
                < float(min_open_interest)
            ]
            for _, row in illiquid.iterrows():
                breaches.append(
                    RiskBreach(
                        metric="Open interest",
                        symbol=str(row.get("symbol")),
                        value=float(row.get("open_interest", 0.0)),
                        limit=float(min_open_interest),
                        detail=f"{row.get('symbol')} open interest {row.get('open_interest')} < {min_open_interest}",
                    )
                )
        if max_spread is not None and "bid_ask_spread_pct" in positions.columns:
            wide = positions[
                pd.to_numeric(positions["bid_ask_spread_pct"], errors="coerce")
                > float(max_spread)
            ]
            for _, row in wide.iterrows():
                breaches.append(
                    RiskBreach(
                        metric="Bid/ask spread",
                        symbol=str(row.get("symbol")),
                        value=float(row.get("bid_ask_spread_pct", 0.0)),
                        limit=float(max_spread),
                        detail=f"{row.get('symbol')} spread {row.get('bid_ask_spread_pct')}% > {max_spread}%",
                    )
                )
        return breaches


__all__ = ["RiskEvaluator", "RiskBreach", "RiskLimitConfig"]
