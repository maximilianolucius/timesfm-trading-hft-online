"""Risk management helpers used by the MT5 execution agent."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


@dataclass
class RiskParams:
    equity: float
    risk_percent: float
    atr: float
    entry_price: float
    stop_price: float
    min_lot: float = 0.01
    max_lot: Optional[float] = None
    contract_size: float = 100_000.0  # standard FX lot
    pip_value: float = 10.0


def compute_lot_size(params: RiskParams) -> float:
    risk_amount = params.equity * params.risk_percent / 100.0
    stop_distance = abs(params.entry_price - params.stop_price)
    if stop_distance <= 0:
        return params.min_lot
    per_lot_loss = stop_distance * (params.pip_value / 0.0001)
    if per_lot_loss <= 0:
        return params.min_lot
    lots = risk_amount / per_lot_loss
    lots = max(params.min_lot, lots)
    if params.max_lot:
        lots = min(lots, params.max_lot)
    return round(lots, 2)


@dataclass
class CalibrationGuard:
    coverage_threshold: float
    lookback: int
    history: Iterable[float] = field(default_factory=list)

    def update(self, covered: float) -> None:
        hist = list(self.history)
        hist.append(covered)
        if len(hist) > self.lookback:
            hist = hist[-self.lookback :]
        self.history = hist

    def is_blocked(self) -> bool:
        if not self.history:
            return False
        return float(np.mean(self.history)) < self.coverage_threshold


def calendar_blocked(calendar: Iterable[str], now: Optional[dt.datetime] = None) -> bool:
    now = now or dt.datetime.utcnow()
    now_iso = now.replace(microsecond=0).isoformat()
    now_date = now.date().isoformat()
    return any(entry in (now_iso, now_date) for entry in calendar)


__all__ = ["RiskParams", "compute_lot_size", "CalibrationGuard", "calendar_blocked"]
