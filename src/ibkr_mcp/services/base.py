"""Shared helpers for IBKR service objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ib_async import IB


@dataclass(slots=True)
class IBService:
    """Base class for small service helpers that need the shared IB client."""

    ib: IB
    ensure_connected: Callable[[], None]

    def _ensure_connected(self) -> None:
        """Raise if the underlying IB client is disconnected."""
        self.ensure_connected()
