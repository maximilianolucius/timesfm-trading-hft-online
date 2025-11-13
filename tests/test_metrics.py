from __future__ import annotations

import numpy as np

from common.metrics import (
    aggregate_metrics,
    coverage_rate,
    hit_rate_net,
    mae,
    rmse,
    sharpe_ratio,
)


def test_basic_metrics() -> None:
    y_true = np.array([0.0, 0.2, -0.1, 0.3], dtype=np.float32)
    y_pred = np.array([0.1, 0.1, -0.05, 0.25], dtype=np.float32)
    returns = y_true.copy()
    quantiles = {
        0.1: y_pred - 0.05,
        0.5: y_pred,
        0.9: y_pred + 0.05,
    }

    metrics = aggregate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        returns=returns,
        quantile_levels=[0.1, 0.5, 0.9],
        quantile_preds=quantiles,
        cost_bps=2.0,
    )

    assert np.isclose(metrics["rmse"], rmse(y_true, y_pred))
    assert np.isclose(metrics["mae"], mae(y_true, y_pred))
    assert metrics["hit_rate_net"] <= 1.0
    assert metrics["coverage"] >= 0.0
    assert metrics["sharpe"] == sharpe_ratio(returns)


def test_coverage_rate() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    q_low = np.array([0.5, 1.5, 2.5])
    q_high = np.array([1.5, 2.5, 3.5])
    assert np.isclose(coverage_rate(y_true, q_low, q_high), 1.0)


def test_hit_rate_net_costs() -> None:
    returns = np.array([0.02, -0.01, 0.03])
    forecasts = np.array([0.03, -0.02, 0.01])
    hr = hit_rate_net(returns, forecasts, cost_bps=5)
    assert 0.0 <= hr <= 1.0
