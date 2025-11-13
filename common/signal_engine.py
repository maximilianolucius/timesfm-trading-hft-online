"""Trading signal generation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class Signal:
    side: str
    size: float
    sl: float
    tp: float
    entry: float
    confidence: float
    comment: str = ""


EDGE_DIRECTION = {"BUY": 1.0, "SELL": -1.0, "FLAT": 0.0}


def forecast_to_signal(
    price: float,
    forecast: float,
    quantiles: Optional[Dict[float, float]],
    atr: Optional[float],
    edge_bps: float,
    cost_bps: float,
    atr_multiplier: float = 1.5,
) -> Signal:
    """Convert a numeric forecast into a directional trading decision."""

    threshold = edge_bps / 10_000.0 * price
    net_signal = forecast - np.sign(forecast) * (cost_bps / 10_000.0 * price)
    if net_signal > threshold:
        side = "BUY"
    elif net_signal < -threshold:
        side = "SELL"
    else:
        return Signal(side="FLAT", size=0.0, sl=price, tp=price, entry=price, confidence=0.0)

    if quantiles:
        keys = sorted(quantiles.keys())
        lower = quantiles[keys[0]]
        median = quantiles[keys[len(keys) // 2]]
        upper = quantiles[keys[-1]]
    else:
        lower = price - (atr or 0.0) * atr_multiplier
        median = price + forecast
        upper = price + (atr or 0.0) * atr_multiplier

    if side == "BUY":
        sl = min(lower, price - (atr or 0.0) * atr_multiplier)
        tp = max(upper, price + abs(forecast))
        confidence = max(0.0, (median - price) / (tp - sl + 1e-6))
    else:
        sl = max(upper, price + (atr or 0.0) * atr_multiplier)
        tp = min(lower, price - abs(forecast))
        confidence = max(0.0, (price - median) / (sl - tp + 1e-6))

    return Signal(
        side=side,
        size=0.0,
        sl=sl,
        tp=tp,
        entry=price,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        comment="quantile" if quantiles else "atr_fallback",
    )


__all__ = ["Signal", "forecast_to_signal", "EDGE_DIRECTION"]
