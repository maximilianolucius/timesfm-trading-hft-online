"""Purged walk-forward backtest job."""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ProjectConfig
from common.data_io_csv import build_sliding_windows, load_csv, split_train_validation, zscore_normalise
from common.fine_tune import TensorDataset, build_model, train as finetune_train
from common.logging_utils import get_logger
from common.metrics import aggregate_metrics
from common.signal_engine import Signal, forecast_to_signal
from common.walkforward import PurgedWalkForward

LOGGER = get_logger("job.backtest")


def _prepare_arrays(cfg: ProjectConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    workdir = cfg.resolve_path(cfg.training.checkpoint_dir)
    train_npz = workdir / f"{cfg.task.name}_train.npz"
    if train_npz.exists():
        with np.load(train_npz) as payload:
            return payload["contexts"], payload["targets"], payload["timestamps"]

    df = load_csv(
        cfg.resolve_path(cfg.data.train_csv),
        frequency=cfg.data.frequency,
        timezone=cfg.data.timezone,
        limit=cfg.data.limit,
        resample=cfg.data.resample,
        expect_no_gaps=cfg.data.expect_no_gaps,
    )
    if cfg.data.normalize:
        df_norm, _, _ = zscore_normalise(df, cfg.task.features)
    else:
        df_norm = df

    if cfg.data.val_csv:
        df_val = load_csv(
            cfg.resolve_path(cfg.data.val_csv),
            frequency=cfg.data.frequency,
            timezone=cfg.data.timezone,
            limit=cfg.data.limit,
            resample=cfg.data.resample,
            expect_no_gaps=cfg.data.expect_no_gaps,
        )
    else:
        df_norm, _ = split_train_validation(
            df_norm,
            validation_split=cfg.prep.validation_split,
            shuffle=cfg.prep.shuffle_before_split,
            seed=cfg.training.seed,
        )

    ds = build_sliding_windows(
        df_norm,
        cfg.task.context_len,
        cfg.task.horizon_len,
        cfg.task.features,
        cfg.task.target,
    )
    return ds.contexts, ds.targets, ds.timestamps


def _build_datasets(contexts: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(
        contexts=torch.from_numpy(contexts).float(),
        targets=torch.from_numpy(targets).float(),
    )


def _train_for_split(
    cfg: ProjectConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    contexts: np.ndarray,
    targets: np.ndarray,
    fold_dir: pathlib.Path,
) -> pathlib.Path:
    train_dataset = _build_datasets(contexts[train_idx], targets[train_idx])
    val_dataset = _build_datasets(contexts[val_idx], targets[val_idx])
    fold_dir.mkdir(parents=True, exist_ok=True)
    return finetune_train(
        cfg.training,
        train_dataset,
        val_dataset,
        context_len=cfg.task.context_len,
        horizon_len=cfg.task.horizon_len,
        feature_dim=contexts.shape[-1],
        workdir=fold_dir,
    )


def _split_train_val(train_idx: np.ndarray, val_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    n = len(train_idx)
    cutoff = max(1, int(n * (1 - val_fraction)))
    return train_idx[:cutoff], train_idx[cutoff:]


def run_backtest(cfg: ProjectConfig) -> pathlib.Path:
    if cfg.backtest is None:
        raise ValueError("Backtest configuration missing. Add backtest block to YAML")
    contexts, targets, timestamps = _prepare_arrays(cfg)
    splitter = PurgedWalkForward(len(contexts), cfg.backtest.walkforward)

    forecasts: List[float] = []
    actuals: List[float] = []
    returns: List[float] = []
    quantiles_low: List[float] = []
    quantiles_mid: List[float] = []
    quantiles_high: List[float] = []
    signal_rows: List[Dict[str, object]] = []
    price_idx = cfg.task.features.index(cfg.task.target)

    for split_id, split in enumerate(splitter, start=1):
        train_idx, val_idx = _split_train_val(split.train_idx)
        fold_dir = cfg.resolve_path(cfg.backtest.output_dir) / f"fold_{split_id}"
        checkpoint = _train_for_split(cfg, train_idx, val_idx, contexts, targets, fold_dir)

        model = build_model(
            cfg.training,
            cfg.task.context_len,
            cfg.task.horizon_len,
            contexts.shape[-1],
        )
        model.load_checkpoint(checkpoint)
        model.eval()

        test_dataset = _build_datasets(contexts[split.test_idx], targets[split.test_idx])
        with torch.inference_mode():
            preds, quant = model.predict(test_dataset.contexts)
        preds_price = preds[:, -1].detach().cpu().numpy()
        actual_price = test_dataset.targets[:, -1].detach().cpu().numpy()
        last_price = test_dataset.contexts[:, -1, price_idx].detach().cpu().numpy()
        forecast_return = preds_price - last_price
        actual_return = actual_price - last_price

        if quant:
            q_sorted = sorted(quant.items())
            q_low = q_sorted[0][1][:, -1].cpu().numpy() - last_price
            q_mid = q_sorted[len(q_sorted) // 2][1][:, -1].cpu().numpy() - last_price
            q_high = q_sorted[-1][1][:, -1].cpu().numpy() - last_price
        else:
            q_low = -np.abs(actual_return)
            q_mid = forecast_return
            q_high = np.abs(actual_return)

        forecasts.extend(forecast_return)
        actuals.extend(actual_return)
        returns.extend(actual_return)
        quantiles_low.extend(q_low)
        quantiles_mid.extend(q_mid)
        quantiles_high.extend(q_high)

        for idx, ts_idx in enumerate(split.test_idx):
            timestamp = timestamps[ts_idx]
            signal = forecast_to_signal(
                price=float(last_price[idx]),
                forecast=float(forecast_return[idx]),
                quantiles={0.1: q_low[idx], 0.5: q_mid[idx], 0.9: q_high[idx]},
                atr=None,
                edge_bps=cfg.backtest.edge_bps,
                cost_bps=cfg.backtest.cost_bps,
            )
            signal_rows.append(
                {
                    "timestamp": np.datetime_as_string(timestamp, timezone="UTC"),
                    "forecast": float(forecast_return[idx]),
                    "actual": float(actual_return[idx]),
                    "side": signal.side,
                    "confidence": signal.confidence,
                    "sl": signal.sl,
                    "tp": signal.tp,
                }
            )

    metrics = aggregate_metrics(
        y_true=np.array(actuals),
        y_pred=np.array(forecasts),
        returns=np.array(returns),
        quantile_levels=[0.1, 0.5, 0.9],
        quantile_preds={
            0.1: np.array(quantiles_low),
            0.5: np.array(quantiles_mid),
            0.9: np.array(quantiles_high),
        },
        cost_bps=cfg.backtest.cost_bps,
    )

    output_dir = cfg.resolve_path(cfg.backtest.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_path = output_dir / "signals.csv"
    metrics_path = output_dir / "metrics.json"
    equity_path = output_dir / "equity.png"

    import csv

    with signals_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "forecast", "actual", "side", "confidence", "sl", "tp"],
        )
        writer.writeheader()
        writer.writerows(signal_rows)

    metrics_path.write_text(json.dumps(metrics, indent=2))

    equity = np.cumsum(returns) + 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Trade")
    plt.ylabel("Equity (relative)")
    plt.tight_layout()
    plt.savefig(equity_path)
    plt.close()

    LOGGER.info("Backtest complete", extra={"signals": str(signals_path), "metrics": str(metrics_path)})
    return metrics_path


__all__ = ["run_backtest"]
