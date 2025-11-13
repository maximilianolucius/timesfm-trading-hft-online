"""Evaluation metrics for forecasting and trading."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def hit_rate_net(
    returns: np.ndarray,
    forecasts: np.ndarray,
    cost_bps: float = 0.0,
) -> float:
    """Net hit rate after subtracting transaction costs."""

    if returns.shape != forecasts.shape:
        raise ValueError("returns and forecasts must share shape")

    positions = np.sign(forecasts)
    gross = positions * returns
    net = gross - (np.abs(positions) * cost_bps / 10_000.0)
    hits = net > 0.0
    return float(np.mean(hits))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    denom = np.std(excess, ddof=1)
    if denom == 0:
        return 0.0
    return float(np.mean(excess) / denom * math.sqrt(252))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    downside = excess[excess < 0.0]
    denom = np.std(downside, ddof=1) if len(downside) > 0 else 0.0
    if denom == 0.0:
        return 0.0
    return float(np.mean(excess) / denom * math.sqrt(252))


def coverage_rate(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    inside = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(inside))


def pinball_loss(y_true: np.ndarray, quantiles: Dict[float, np.ndarray]) -> Dict[float, float]:
    losses: Dict[float, float] = {}
    for q, preds in quantiles.items():
        diff = y_true - preds
        losses[q] = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return losses


def pit_score(y_true: np.ndarray, quantile_levels: Iterable[float], quantiles: Iterable[np.ndarray]) -> float:
    """Compute a probability integral transform (PIT) uniformity score.

    Uses simple linear interpolation to approximate the implied CDF from
    available quantiles and measures the Kolmogorov-Smirnov distance to a
    uniform distribution.
    """

    levels = np.array(list(quantile_levels), dtype=np.float32)
    preds = np.stack(list(quantiles), axis=0)
    sorted_idx = np.argsort(levels)
    levels = levels[sorted_idx]
    preds = preds[sorted_idx]

    cdf_vals = np.interp(y_true, preds.T, levels)
    cdf_vals = np.clip(cdf_vals, 1e-4, 1 - 1e-4)
    cdf_vals = np.sort(cdf_vals)
    n = len(cdf_vals)
    uniform = (np.arange(1, n + 1) - 0.5) / n
    return float(np.max(np.abs(cdf_vals - uniform)))


def aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    returns: np.ndarray,
    quantile_levels: Optional[Iterable[float]] = None,
    quantile_preds: Optional[Dict[float, np.ndarray]] = None,
    cost_bps: float = 0.0,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "hit_rate_net": hit_rate_net(returns, y_pred, cost_bps=cost_bps),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
    }
    if quantile_levels and quantile_preds:
        q_levels = list(quantile_levels)
        low = min(q_levels)
        high = max(q_levels)
        metrics["coverage"] = coverage_rate(
            y_true,
            quantile_preds[low],
            quantile_preds[high],
        )
        metrics["pit_ks"] = pit_score(
            y_true,
            q_levels,
            [quantile_preds[level] for level in q_levels],
        )
        for level, loss in pinball_loss(y_true, quantile_preds).items():
            metrics[f"pinball_{level:.2f}"] = loss
    return metrics


__all__ = [
    "rmse",
    "mae",
    "hit_rate_net",
    "sharpe_ratio",
    "sortino_ratio",
    "coverage_rate",
    "pinball_loss",
    "pit_score",
    "aggregate_metrics",
]
