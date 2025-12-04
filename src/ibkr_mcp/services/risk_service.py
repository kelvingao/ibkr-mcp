"""Risk evaluation and playbook helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ibkr_mcp.common.greeks import GreekCalculator, compute_concentration
from ibkr_mcp.common.playbooks import PlaybookContext, PlaybookEngine
from ibkr_mcp.common.positions import PositionLoader, PositionSource
from ibkr_mcp.common.rules import RiskEvaluator, RiskLimitConfig

from .base import IBService


@dataclass(slots=True)
class RiskService(IBService):
    """Encapsulates configuration and risk evaluation workflows."""

    def _load_positions(self, account: str):
        loader = PositionLoader(PositionSource(ib=self.ib))
        return loader.load(account)

    def _resolve_risk_config(
        self,
        config: Optional[Dict[str, Any]],
        config_path: str,
    ) -> tuple[RiskLimitConfig, Optional[Path]]:
        if config is not None:
            raw_config: Dict[str, Any] = config
            used_path: Optional[Path] = None
        else:
            effective_path = config_path or os.getenv("IBKR_MCP_RISK_CONFIG", "risk.yaml")
            path = Path(effective_path)
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    raw_config = yaml.safe_load(fh) or {}
                used_path = path
            else:
                raw_config = {}
                used_path = None

        risk_config = RiskLimitConfig(
            limits=raw_config.get("limits", {}),
            roll_rules=raw_config.get("roll_rules", {}),
        )
        return risk_config, used_path

    async def get_greeks_summary(self, account: str = "") -> Dict[str, Any]:
        self._ensure_connected()
        positions = self._load_positions(account)

        calculator = GreekCalculator(self.ib)
        greek_summary = calculator.compute(positions)
        concentration, concentration_series, concentration_totals = compute_concentration(positions)

        return {
            "greek_summary": {
                "per_symbol": greek_summary.per_symbol.to_dict(orient="records"),
                "totals": greek_summary.totals,
            },
            "concentration": {
                "per_symbol": concentration.to_dict(orient="records"),
                "series": concentration_series.to_dict(),
                "totals": concentration_totals,
            },
            "position_count": len(positions),
        }

    async def evaluate_portfolio_risk(
        self,
        account: str = "",
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "risk.yaml",
    ) -> Dict[str, Any]:
        self._ensure_connected()
        risk_config, used_path = self._resolve_risk_config(config, config_path)

        positions = self._load_positions(account)

        calculator = GreekCalculator(self.ib)
        greek_summary = calculator.compute(positions)
        concentration_df, concentration_series, concentration_totals = compute_concentration(positions)

        evaluator = RiskEvaluator(risk_config)
        breaches = evaluator.evaluate(
            greek_summary.per_symbol,
            greek_summary.totals,
            concentration_df,
            positions,
        )

        return {
            "ok": True,
            "account": account or None,
            "config_path": str(used_path) if used_path is not None else None,
            "position_count": int(len(positions)),
            "limits": risk_config.limits,
            "roll_rules": risk_config.roll_rules,
            "greeks": {
                "per_symbol": greek_summary.per_symbol.to_dict(orient="records"),
                "totals": greek_summary.totals,
            },
            "concentration": {
                "per_symbol": concentration_df.to_dict(orient="records"),
                "series": concentration_series.to_dict(),
                "totals": concentration_totals,
            },
            "breaches": [
                {
                    "metric": b.metric,
                    "symbol": b.symbol,
                    "value": b.value,
                    "limit": b.limit,
                    "detail": b.detail,
                }
                for b in breaches
            ],
        }

    async def generate_playbook_actions(
        self,
        account: str = "",
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "risk.yaml",
    ) -> Dict[str, Any]:
        self._ensure_connected()
        risk_config, used_path = self._resolve_risk_config(config, config_path)

        positions = self._load_positions(account)

        calculator = GreekCalculator(self.ib)
        greek_summary = calculator.compute(positions)
        concentration_df, _, _ = compute_concentration(positions)

        evaluator = RiskEvaluator(risk_config)
        breaches = evaluator.evaluate(
            greek_summary.per_symbol,
            greek_summary.totals,
            concentration_df,
            positions,
        )

        context = PlaybookContext(roll_rules=risk_config.roll_rules)
        engine = PlaybookEngine(context)
        actions = engine.generate(positions, breaches)

        return {
            "ok": True,
            "account": account or None,
            "config_path": str(used_path) if used_path is not None else None,
            "position_count": int(len(positions)),
            "limits": risk_config.limits,
            "roll_rules": risk_config.roll_rules,
            "actions": actions,
            "breaches": [
                {
                    "metric": b.metric,
                    "symbol": b.symbol,
                    "value": b.value,
                    "limit": b.limit,
                    "detail": b.detail,
                }
                for b in breaches
            ],
            "greeks": {
                "per_symbol": greek_summary.per_symbol.to_dict(orient="records"),
                "totals": greek_summary.totals,
            },
        }
