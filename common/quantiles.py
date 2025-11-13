"""Quantile handling helpers."""
from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, List


def ensure_monotonic(levels: Iterable[float], predictions: np.ndarray) -> Dict[float, np.ndarray]:
    """Force quantile predictions to be non-decreasing across quantile levels."""

    levels_arr = np.array(list(levels))
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.ndim == 2:
        predictions = predictions[:, :, None]
    if predictions.shape[1] != len(levels_arr):
        raise ValueError("Prediction shape mismatch with quantile levels")

    order = np.argsort(levels_arr)
    levels_sorted = levels_arr[order]
    preds_sorted = predictions[:, order, :].copy()

    # Enforce monotonicity per horizon step
    for i in range(1, preds_sorted.shape[1]):
        preds_sorted[:, i, :] = np.maximum(preds_sorted[:, i, :], preds_sorted[:, i - 1, :])

    return {float(level): preds_sorted[:, idx, :] for idx, level in enumerate(levels_sorted)}


def quantiles_from_bootstrap(samples: np.ndarray, quantile_levels: List[float]) -> Dict[float, np.ndarray]:
    """Compute empirical quantiles from bootstrap samples."""

    if samples.ndim != 2:
        raise ValueError("Expected samples with shape (n_samples, batch)")
    results: Dict[float, np.ndarray] = {}
    for q in quantile_levels:
        results[float(q)] = np.quantile(samples, q, axis=0)
    return results


def combine_quantile_sources(
    primary: Dict[float, np.ndarray],
    fallback: Dict[float, np.ndarray],
    weight: float = 0.75,
) -> Dict[float, np.ndarray]:
    """Blend two quantile dictionaries to stabilise estimates."""

    combined: Dict[float, np.ndarray] = {}
    for key in primary.keys():
        if key in fallback:
            combined[key] = weight * primary[key] + (1 - weight) * fallback[key]
        else:
            combined[key] = primary[key]
    for key, value in fallback.items():
        combined.setdefault(key, value)
    return combined


__all__ = [
    "ensure_monotonic",
    "quantiles_from_bootstrap",
    "combine_quantile_sources",
]
